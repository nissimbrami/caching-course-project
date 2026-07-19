# Critical Analysis: SGLang and RadixAttention
## For Graduate Course Presentation on Caching Policies

---

## 1. Assumptions About Workload Characteristics

RadixAttention's effectiveness is predicated on several key assumptions about LLM workloads:

### Assumption 1: Significant Prefix Sharing Exists

The paper assumes that real-world LLM workloads exhibit substantial prefix overlap:
- System prompts shared across all requests to a given model
- Chat history shared within multi-turn sessions
- Few-shot examples shared across evaluation instances
- Branching paths in reasoning share common stems

**Validity:** This is well-supported for production deployments (74.1% hit rate on Vicuna-33B), but varies dramatically by application. A creative writing service with unique prompts would see near-zero benefit.

### Assumption 2: Prefill Cost Dominates for Short Outputs

The optimization primarily reduces prefill (prompt processing) time. The paper acknowledges that for long outputs (256-512 tokens/turn), "decoding time dominates" and speedup is minimal.

**Quantitative boundary:** If output length O and shared prefix length P satisfy:
```
Speedup ~ (P + O) / (O + miss_tokens)
```
When O >> P, speedup approaches 1.0 regardless of cache hit rate.

### Assumption 3: Exact Token-Level Prefix Matching is Sufficient

The system only reuses KV cache for exact token-level prefix matches. This assumes:
- Users/applications structure prompts with shared prefixes
- Tokenization is deterministic (same text -> same tokens)
- Minor prompt variations (spacing, punctuation) completely invalidate sharing

**Limitation:** Two prompts like "Summarize this document: [doc]" and "Please summarize this document: [doc]" share zero prefix despite near-identical semantics.

### Assumption 4: LRU Approximates Optimal for Online Workloads

The paper proves optimality for batch (offline) scheduling via DFS ordering, but the online setting uses LRU + longest-prefix-first as a heuristic.

**Gap:** No formal competitive ratio is established for the online algorithm. The 96% empirical approximation holds for tested workloads but has no worst-case guarantee.

### Assumption 5: Single-Machine Memory is Sufficient

The radix tree resides entirely in GPU memory of one machine. This assumes:
- The working set of active prefixes fits in GPU memory
- No need for distributed cache coordination
- The model itself plus running requests plus cache all fit

---

## 2. When RadixAttention Does NOT Help (Worst-Case Scenarios)

### Scenario 1: Unique Prompts (Zero Sharing)

**Workload:** Each request has a completely unique prompt with no overlap with any other request (e.g., independent document analysis with different documents).

**Performance:** The paper reports < 0.3% overhead in this case, meaning it doesn't *hurt*, but provides zero benefit. The radix tree degenerates into a list of disconnected paths.

### Scenario 2: Long-Output Dominated Workloads

**Workload:** Code generation, story writing, or any task where output tokens >> input tokens.

**Analysis:** Even with 100% prefix hit rate, if output is 2000 tokens and saved prefix is 500 tokens:
```
Time without cache:  prefill(500) + decode(2000)
Time with cache:     decode(2000)
Speedup = (prefill(500) + decode(2000)) / decode(2000)
         ~ (500*T_prefill + 2000*T_decode) / (2000*T_decode)
```
Given prefill is ~10x faster per token than decode (due to parallelism), this yields minimal improvement.

### Scenario 3: Adversarial Access Patterns (Cache Thrashing)

**Workload:** Many distinct prefixes that cyclically access the cache, each evicting the others before reuse.

**Example:** 1000 distinct system prompts, each used once per hour, with GPU memory sufficient for only 100. Every access is a miss. LRU provides zero hit rate.

**Formal condition:** When working set size W > cache capacity C, LRU hit rate drops precipitously. For uniform random access: hit rate = C/W.

### Scenario 4: Streaming/Real-Time Single-User Chat

**Workload:** One user chatting with one model, no concurrent requests.

**Analysis:** The cache of turn N is used exactly once (for turn N+1). After that, it's never accessed again. LRU works here, but the benefit is bounded by the history reuse for the single conversation. No inter-request sharing occurs.

### Scenario 5: Rapidly Changing Contexts (RAG with Diverse Documents)

**Workload:** Each request retrieves different documents and prepends them to the prompt.

**Analysis:** Documents change per query, so only the (typically short) system prompt prefix is shared. CacheBlend specifically addresses this gap.

### Scenario 6: High Cardinality Few-Shot (Different Examples Per Request)

**Workload:** Few-shot learning where different subsets of examples are sampled for each request.

**Analysis:** If examples are shuffled randomly, prefix sharing is limited to the instruction prefix only. Only when the same ordered set of examples is reused does RadixAttention help significantly.

---

## 3. How Frequency-Aware Eviction Would Improve RadixAttention

### Current Limitation of LRU

LRU evicts based solely on last access time, ignoring how often a prefix is accessed. Consider:

- Prefix A (system prompt): accessed 1000 times/minute, last access 2ms ago
- Prefix B (one-off long document): accessed once, last access 1ms ago

LRU keeps B over A if B was accessed more recently, even though A has vastly higher expected future value.

### Quantitative Argument for LFU/Frequency-Aware

Define the **expected hit rate** improvement under frequency-aware eviction:

Let prefix i have:
- Access frequency: f_i (requests/second)
- Size: s_i (tokens/pages)
- Recomputation cost: c_i (proportional to s_i for prefill)

Under LRU with cache capacity C tokens, items are evicted when not accessed within a time window T_evict. The probability of reuse before eviction depends on frequency:
```
P(hit | LRU) = 1 - e^(-f_i * T_evict)   (exponential interarrival model)
```

For low-frequency items (f_i << 1/T_evict): P(hit) ~ f_i * T_evict (approximately linear)
For high-frequency items (f_i >> 1/T_evict): P(hit) ~ 1

LFU explicitly maintains high-frequency items regardless of recency, providing:
```
Expected cost saved per unit cache = sum_i(f_i * c_i) for items in cache
```

The **optimal** eviction (cost-frequency aware) keeps items maximizing:
```
Value_i = f_i * c_i / s_i   (benefit per unit cache space)
```

### Concrete Example

Production Vicuna-33B with 74.1% hit rate under LRU:
- System prompt (1024 tokens): f=100/s, c=1024*T_prefill
- User-specific history (512 tokens): f=0.01/s, c=512*T_prefill

Value_system = 100 * 1024 / 1024 = 100
Value_history = 0.01 * 512 / 512 = 0.01

Under LRU during memory pressure, a recently-accessed one-off history could displace the system prompt. Under frequency-aware eviction, the system prompt would never be evicted while actively serving traffic.

### Estimated Improvement

In workloads with **bimodal frequency distribution** (few hot prefixes, many cold ones):
- LRU hit rate: ~74% (production observed)
- Frequency-aware (LFU or adaptive): estimated 80-85% hit rate
- Improvement: 5-10 percentage points on hit rate

The SLRU strategy in SGLang's current implementation partially addresses this by protecting frequently-accessed items in the protected segment.

---

## 4. How Cost-Aware Eviction Would Improve RadixAttention

### The Recomputation Cost Asymmetry

Not all cache misses are equally expensive:
- Missing a 32-token prefix: costs 32 * T_prefill to recompute
- Missing a 4096-token prefix: costs 4096 * T_prefill to recompute

LRU treats both equally. A cost-aware policy would preferentially retain expensive-to-recompute items.

### Formal Model: Weighted Caching

Define the **cost-weighted eviction** problem:
- Each item i has size s_i and miss cost c_i
- Goal: Minimize total miss cost (not just miss count)

This is the **weighted caching problem** (a generalization of the standard paging problem).

**Optimal offline policy (generalized Belady):** Evict the item with smallest c_i / s_i whose next access is furthest in the future.

**Online approximation:** Maintain priority = c_i * f_i / s_i (where f_i is estimated frequency).

### Quantitative Argument

For RadixAttention specifically, c_i = s_i * T_prefill (linear recomputation cost). So:
```
Priority_i = s_i * T_prefill * f_i / s_i = T_prefill * f_i
```

Interestingly, when cost is linear in size, cost-aware eviction reduces to **frequency-aware eviction** (since the size terms cancel). However, this changes when:

1. **Attention is quadratic**: For self-attention without FlashAttention, recomputation cost is O(s_i^2), making:
   ```
   Priority_i = s_i^2 * f_i / s_i = s_i * f_i
   ```
   Now larger prefixes are weighted MORE heavily, preferring to keep them.

2. **Batched computation amortizes cost**: If multiple requests share a prefix, effective cost per request decreases with batch size.

### Expected Improvement

For workloads with high variance in prefix length:
- Short system prompt (128 tokens) vs. long document prefix (4096 tokens)
- Cost of missing the document prefix is 32x higher
- If both accessed equally frequently, cost-aware keeps the document prefix

Estimated benefit: 10-20% reduction in total recomputation time for workloads with high prefix length variance, compared to pure LRU.

### Combined Cost-Frequency Policy

The ideal policy combines both dimensions:
```
Eviction_priority(node) = -1 * (frequency(node) * recompute_cost(node) / size(node))
```

Lower priority = evicted first. This is equivalent to a **benefit-per-byte** metric used in web caching (GreedyDual-Size algorithm).

---

## 5. Connection to Course Caching Policies (Exhaustive)

### 5.1 LRU (Least Recently Used)

**Direct connection:** RadixAttention's default eviction policy.

**Key insight:** LRU works well when temporal locality is strong (recent access predicts future access). In LLM serving, this holds for:
- Active conversations (will continue soon)
- System prompts during peak usage
- Few-shot examples during evaluation runs

**Weakness in this context:** Does not capture frequency or cost. A one-time accessed long document prefix evicts a frequently-used system prompt if accessed more recently.

### 5.2 LFU (Least Frequently Used)

**Connection:** Implemented in SGLang source code as alternative strategy.

**When superior to LRU:** Workloads with stable popularity distribution (e.g., shared system prompts that are consistently popular). LFU correctly retains the hottest prefixes even during temporary access to cold items.

**Weakness:** Slow to adapt to changing popularity (frequency counters accumulate inertia). A formerly popular prefix that is no longer used retains a high count.

### 5.3 SLRU (Segmented LRU)

**Connection:** Implemented in SGLang as an eviction strategy.

**Mechanism:** Two-segment approach where items start in probationary segment and promote to protected segment on second access. Eviction targets probationary first.

**Benefit for RadixAttention:** Protects system prompts and other frequently-reused prefixes (which quickly promote) while allowing one-off prefixes to be evicted from probationary segment.

### 5.4 Belady's MIN (Optimal Offline)

**Connection:** Theorem 3.1 establishes the optimal offline schedule for batch processing. The DFS ordering achieves optimal hit rate because it processes requests in an order that ensures each prefix is needed by an immediate successor.

**Distinction:** Belady's MIN looks at future access times. SGLang's DFS ordering achieves optimality for a different reason: the request tree structure guarantees that DFS order minimizes the maximum number of simultaneously needed prefixes.

### 5.5 ARC (Adaptive Replacement Cache)

**Connection:** Not implemented in SGLang, but the SLRU implementation is conceptually related.

**Potential improvement:** ARC dynamically adjusts the boundary between recency-favoring and frequency-favoring segments. This could adapt to workloads that shift between exploratory (many unique prefixes) and repetitive (few shared prefixes) phases.

### 5.6 Working Set Model

**Connection:** The "working set" in RadixAttention is the set of active prefixes needed by current and near-future requests.

**Application:** If the working set fits in GPU memory, hit rate approaches 100%. If it exceeds memory, thrashing occurs. The production hit rates (52-74%) suggest partial working set accommodation.

**Sizing guidance:** Working set size = sum of unique prefix lengths across active conversations + system prompts + common templates.

### 5.7 Cache Replacement with Variable-Size Items

**Connection:** RadixAttention nodes have variable sizes (1 to thousands of tokens). This maps directly to the **variable-size caching problem** studied in web caching.

**Classical results:** GreedyDual-Size (GDS) algorithm generalizes LRU to variable-size items with heterogeneous miss costs. It assigns priority H(item) = cost/size and evicts the minimum-priority item.

**Application to RadixAttention:**
```
H(node) = recompute_cost(node) / size(node) = T_prefill  (constant!)
```
When cost is linear in size, GDS reduces to standard LRU. This explains why LRU performs well as the default.

### 5.8 Inclusive vs. Exclusive Caching

**Connection:** RadixAttention is an **inclusive** hierarchical cache:
- The radix tree stores ALL cached prefixes (no separate levels with exclusive content)
- The optional host_value extension creates an inclusive L2 (CPU memory)
- CachedAttention extends this to exclusive GPU/CPU/disk tiers

### 5.9 Cache Coherence and Consistency

**Connection:** Multi-tenant SGLang deployments must handle:
- **Namespace isolation:** Different LoRA adapters have different KV caches for same tokens
- **Invalidation:** Model updates require full cache flush
- **No cross-instance sharing:** Each server has independent cache (consistency is trivial but utilization suffers)

### 5.10 Prefetching and Prediction

**Connection:** RadixAttention does NOT implement prefetching. It is purely reactive (match prefix on request arrival).

**Opportunity:** Predictive prefetching could warm the cache:
- For multi-turn chat: prefetch user's likely next turn based on conversation state
- For agent loops: prefetch template + common tool outputs
- For few-shot: prefetch example sets during low-utilization periods

### 5.11 Cache-Oblivious vs. Cache-Aware

**Connection:** RadixAttention is explicitly **cache-aware**: the scheduling algorithm knows the cache state and optimizes request ordering based on it.

**Trade-off:** Cache-aware scheduling adds O(n*m) overhead per scheduling round (n requests, m tree depth for matching). The paper reports this is negligible (<0.3%), but it could become significant at extreme scale.

### 5.12 Competitive Ratio Analysis

**Connection:** Online caching algorithms are analyzed by competitive ratio (worst-case performance vs. optimal offline).

**LRU competitive ratio:** k-competitive for cache size k (measured in pages). This is tight for adversarial sequences.

**RadixAttention's advantage:** Real workloads are not adversarial. The tree structure of LLM workloads provides natural locality that LRU exploits well. The 96% approximation ratio is empirical, not worst-case.

### 5.13 Cache Partitioning and Sharing

**Connection:** RadixAttention dynamically shares memory between:
- Active computation (running batch)
- Passive caching (stored prefixes)

This is analogous to **dynamic partitioning** in multi-level caches, where the partition boundary adapts to workload pressure.

**Policy:** Evict cache entries to make room for active computation. This always prioritizes throughput over hit rate, which is correct: a cache hit only helps if the request can actually run.

### 5.14 Write Policies (Write-Back, Write-Through, Write-Allocate)

**Connection:** When a request completes:
- **Write-allocate:** The completed KV cache is always inserted into the tree (allocate on miss equivalent: generate on miss)
- **Write-back equivalent:** The tree is the final store; eviction = data loss (must recompute)
- No "write-through" needed since there's no backing store in the basic design

CachedAttention adds write-through to CPU/disk, creating a true memory hierarchy.

### 5.15 Temporal vs. Spatial Locality

**Connection:**
- **Temporal locality:** Recently accessed prefixes likely accessed again (LRU exploits this)
- **Spatial locality:** The radix tree structure itself captures spatial locality - if you access a node, you're accessing all ancestor nodes (the full prefix path)

The tree structure provides FREE spatial locality exploitation: accessing any node implicitly validates all ancestor nodes' utility.

### 5.16 Cold Start and Compulsory Misses

**Connection:** Every unique prefix encounters exactly one compulsory miss (first computation). RadixAttention cannot eliminate compulsory misses.

**Impact:** For workloads with many unique users starting new conversations, the cold start period before the system prompt cache is populated represents irreducible overhead.

### 5.17 Capacity and Conflict Misses

**Connection:**
- **Capacity misses:** When total unique prefix set > GPU memory (requires eviction)
- **No conflict misses:** The radix tree uses direct addressing (hash of token sequence), not set-associative lookup. Any prefix can reside anywhere, eliminating conflicts.

---

## 6. Proposed Improvements

### 6.1 Hybrid Frequency-Recency Policy

Combine LRU and LFU using an ARC-like adaptive mechanism:
```
priority(node) = alpha * recency_score(node) + (1-alpha) * frequency_score(node)
```
Where alpha adapts based on which signal better predicts future access.

**Expected benefit:** 5-15% hit rate improvement on mixed workloads.

### 6.2 Cost-Aware Eviction for Quadratic Attention

For models without FlashAttention, or for very long sequences:
```
priority(node) = frequency * prefix_length^2 / node_size
```
This quadratic term preferentially retains long prefixes whose recomputation is disproportionately expensive.

### 6.3 Semantic Prefix Matching

Instead of exact token matching, allow approximate matching:
- Use embedding similarity for prompt variations
- Share KV caches for prompts differing only in minor ways
- CacheBlend's selective recomputation approach addresses this partially

### 6.4 Predictive Prefetching

Use workload patterns to predict future prefix needs:
- Time-of-day models for traffic patterns
- Session-based prediction for multi-turn chat
- Pipeline-aware prefetching for agent loops

### 6.5 Distributed Radix Tree

Extend the radix tree across multiple GPUs/machines:
- Shard by prefix hash (root-level split)
- Replicate hot prefixes
- MemServe's MemPool architecture addresses this direction

---

## 7. Summary of Critical Findings

| Aspect | SGLang Strength | SGLang Weakness |
|--------|----------------|-----------------|
| Hit rate | 52-99% on supported workloads | 0% on unique-prompt workloads |
| Overhead | <0.3% on non-reusable workloads | Scheduling adds latency |
| Eviction | LRU simple and effective | Not frequency or cost aware |
| Scheduling | Proven optimal for batch | No fairness guarantees |
| Scope | Single machine, efficient | No distributed support |
| Matching | Exact, zero-overhead | No semantic/approximate matching |
| Workloads | Tree-structured, shared prefixes | Long outputs, unique prompts |
| Theory | Optimal offline (Thm 3.1) | No online competitive ratio |

### Key Takeaway for Presentation

RadixAttention is a **workload-aware** caching system that exploits the specific structure of LLM serving workloads (tree-structured prefix sharing). Its effectiveness is bounded by:
1. The fraction of computation in prefill vs. decode
2. The degree of prefix sharing in the workload
3. The ratio of working set to cache capacity

The most impactful improvement would be **frequency-cost-aware eviction** (combining access frequency with recomputation cost), which would address the main weakness of the default LRU policy without sacrificing the elegant radix tree structure.
