# SGLang: Efficient Execution of Structured Language Model Programs
## Deep Technical Read for Graduate Course Presentation

**Paper:** "SGLang: Efficient Execution of Structured Language Model Programs"
**Authors:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
**Venue:** arXiv:2312.07104 (Dec 2023, revised Jun 2024)
**Code:** https://github.com/sgl-project/sglang

---

## 1. Problem Statement and Motivation

Modern LLM applications are no longer single-prompt-in, single-response-out systems. They involve:
- Multi-turn conversations with shared history
- Few-shot prompting with shared exemplars across many queries
- Tree-of-thought reasoning with branching paths sharing common prefixes
- Agent loops (ReAct) that iteratively extend prompts
- Structured output generation (JSON, SQL) with constrained decoding

Existing systems (vLLM, llama.cpp, HuggingFace Transformers) treat each generation call independently, recomputing KV caches from scratch even when significant prefix overlap exists between requests. SGLang addresses this with a co-designed frontend language and runtime system.

---

## 2. System Architecture Overview

SGLang consists of three layers:

1. **Frontend DSL** (Python-embedded): Primitives for generation, selection, forking, and structured output
2. **Runtime Interpreter/Compiler**: Manages prompt state, issues generation calls, handles synchronization
3. **Backend Server**: Implements RadixAttention, cache-aware scheduling, and compressed FSM decoding

### Frontend Primitives
| Primitive | Description |
|-----------|-------------|
| `gen(name, regex=...)` | Generate text, optionally constrained by regex |
| `select(name, options)` | Choose highest-probability option from list |
| `extend(text)` / `+=` | Append text to prompt state |
| `fork(n)` | Create n parallel copies of current state |
| `join()` | Synchronize forked branches |
| `[var_name]` | Retrieve generation result (blocks until ready) |
| `image(path)` / `video(path)` | Multimodal inputs |

---

## 3. RadixAttention: Full Algorithm Description

### 3.1 Data Structure: The Radix Tree

RadixAttention uses a **radix tree** (Patricia trie) where:
- **Edges** are labeled with sequences of token IDs (variable length)
- **Nodes** store pointers to corresponding KV cache pages in GPU memory
- **The root** represents the empty prefix
- **Each path from root to a node** represents a complete token prefix whose KV cache is stored

Key properties of radix trees exploited:
- Edges can store sequences (not just single tokens), enabling compact representation
- Prefix sharing is implicit in the tree structure
- Lookup, insertion, and deletion are O(k) where k is key length

### 3.2 TreeNode Structure (from source code)

```python
class TreeNode:
    children: Dict[token_sequence, TreeNode]  # child edges
    parent: TreeNode                           # parent pointer
    key: RadixKey                              # token sequence on edge to this node
    value: torch.Tensor                        # KV cache indices (GPU memory pages)
    lock_ref: int                              # reference counter (>0 = protected)
    last_access_time: float                    # for LRU eviction
    creation_time: float                       # for FIFO eviction
    hit_count: int                             # for LFU eviction
    priority: float                            # for priority-based eviction
    hash_value: List[bytes]                    # SHA256 per page for dedup
    host_value: Optional[torch.Tensor]         # CPU-side backup (optional)
    host_ref_counter: int                      # CPU backup reference count
```

### 3.3 Prefix Matching Algorithm (Pseudocode)

```
function MATCH_PREFIX(root, token_sequence):
    current_node = root
    position = 0
    matched_indices = []

    while position < len(token_sequence):
        # Find child whose edge starts with token_sequence[position]
        child = find_child(current_node, token_sequence[position:])

        if child is None:
            break

        edge_key = child.key
        # Compare edge_key with remaining token_sequence
        match_len = common_prefix_length(edge_key, token_sequence[position:])

        if match_len < len(edge_key):
            # Partial match: split the node
            SPLIT_NODE(current_node, child, match_len)
            matched_indices.extend(child.value[:match_len])
            position += match_len
            break
        else:
            # Full edge match: continue traversal
            matched_indices.extend(child.value)
            position += len(edge_key)
            current_node = child

    # Update access metadata
    update_access_time(current_node)
    increment_hit_count(current_node)

    return matched_indices, current_node
```

### 3.4 Node Splitting

When a prefix match terminates mid-edge, the node is split:

```
function SPLIT_NODE(parent, child, split_position):
    # Create new intermediate node
    new_node = TreeNode()
    new_node.key = child.key[:split_position]
    new_node.value = child.value[:split_position]
    new_node.priority = child.priority

    # Update child to represent remainder
    child.key = child.key[split_position:]
    child.value = child.value[split_position:]

    # Rewire tree
    parent.children[new_node.key] = new_node
    new_node.children[child.key] = child
    new_node.parent = parent
    child.parent = new_node

    # Split hash values accordingly
    new_node.hash_value = child.hash_value[:pages_for(split_position)]
    child.hash_value = child.hash_value[pages_for(split_position):]
```

### 3.5 Insertion Algorithm

```
function INSERT(root, token_sequence, kv_indices):
    # First, match existing prefix
    matched_len, terminal_node = MATCH_PREFIX(root, token_sequence)

    if matched_len == len(token_sequence):
        return  # Already fully cached

    # Insert remaining tokens as new edge
    remaining_tokens = token_sequence[matched_len:]
    remaining_kv = kv_indices[matched_len:]

    new_node = TreeNode()
    new_node.key = remaining_tokens
    new_node.value = remaining_kv
    new_node.last_access_time = current_time()
    new_node.creation_time = current_time()

    terminal_node.children[remaining_tokens] = new_node
    new_node.parent = terminal_node

    # Propagate priority upward (max along path)
    propagate_priority(new_node)

    # Update evictable size tracking
    evictable_size += len(remaining_kv)
    add_to_evictable_leaves(new_node)
```

### 3.6 Cache-Aware Scheduling (Algorithm 1 from Paper)

```
function SCHEDULE_BATCH(waiting_queue, radix_tree, available_memory):
    # Step 1: Match all waiting requests against radix tree
    for request in waiting_queue:
        request.matched_prefix_len = MATCH_PREFIX(radix_tree, request.tokens).length

    # Step 2: Sort by longest-shared-prefix-first
    waiting_queue.sort(key=lambda r: r.matched_prefix_len, descending=True)

    # Step 3: Select requests fitting in memory
    batch = []
    memory_used = 0
    for request in waiting_queue:
        new_tokens = len(request.tokens) - request.matched_prefix_len
        if memory_used + new_tokens <= available_memory:
            batch.append(request)
            memory_used += new_tokens
            # Lock matched prefix nodes (prevent eviction)
            INC_LOCK_REF(request.matched_nodes)
        else:
            break

    # Step 4: If memory insufficient, evict LRU leaves
    if need_more_memory:
        EVICT(radix_tree, required_pages)

    return batch

function ON_REQUEST_COMPLETE(request, radix_tree):
    # Decrement lock references
    DEC_LOCK_REF(request.matched_nodes)
    # Insert completed KV cache into tree
    INSERT(radix_tree, request.full_tokens, request.kv_indices)
```

### 3.7 Theoretical Optimality (Theorem 3.1)

**Theorem:** For a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in depth-first search order, with a cache size >= the maximum request length.

**Key insight:** The longest-shared-prefix-first ordering is equivalent to DFS traversal of the request tree, which ensures each unique edge is computed exactly once.

**Proof sketch:** DFS ensures that when processing a request, its parent prefix is always in cache (just used by the previous sibling request). This means each edge e in the tree is computed only once, achieving the theoretical lower bound on computation.

**Practical approximation:** The cache-aware scheduler achieves approximately 96% of the optimal hit rate on average across benchmarks.

---

## 4. Eviction Policy: Detailed Analysis

### 4.1 Granularity

Eviction operates at the **node level** (variable-length token sequences), not at fixed-size blocks or individual tokens. Each node corresponds to an edge in the radix tree, which can represent anywhere from 1 to hundreds of tokens.

However, the implementation supports **page-aligned** operations where pages are configurable in size (default: 1 token per page in the original paper). The actual memory freed corresponds to the KV cache pages stored at that node.

### 4.2 Default Policy: LRU

The paper uses **LRU (Least Recently Used)** as the default eviction strategy:
- Each node tracks `last_access_time`
- Eviction targets **leaf nodes** first (bottom-up)
- When all children of a node are evicted, the parent becomes a leaf and is eligible
- Nodes with `lock_ref > 0` are never evicted (actively serving requests)

### 4.3 Eviction Mechanism (from source code)

```python
def evict(self, num_tokens_to_evict):
    # Build priority heap from evictable leaves
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node)
        for node in self.evictable_leaves
    ]
    heapq.heapify(eviction_heap)

    tokens_evicted = 0
    while tokens_evicted < num_tokens_to_evict and eviction_heap:
        priority, node = heapq.heappop(eviction_heap)

        # Free KV cache pages
        freed_pages = release_kv_pages(node.value)
        tokens_evicted += len(node.value)

        # Record event
        emit_event(BlockRemoved, node.hash_value)

        # Remove from tree
        parent = node.parent
        del parent.children[node.key]

        # Check if parent becomes evictable leaf
        if not parent.children and parent.lock_ref == 0 and parent != self.root:
            heapq.heappush(eviction_heap,
                (self.eviction_strategy.get_priority(parent), parent))

    self.evictable_size -= tokens_evicted
    return tokens_evicted
```

### 4.4 All Supported Eviction Strategies (Current Implementation)

| Strategy | Priority Key | Description |
|----------|-------------|-------------|
| LRU | `last_access_time` (ascending) | Evict least recently accessed |
| LFU | `hit_count` (ascending) | Evict least frequently used |
| FIFO | `creation_time` (ascending) | Evict oldest created first |
| MRU | `-last_access_time` (descending) | Evict most recently used |
| FILO | `-creation_time` (descending) | Evict newest first |
| Priority | `priority` (ascending) | Evict lowest priority first |
| SLRU | Segmented tiers | Protected/probationary segments |

The paper originally describes only LRU; the additional strategies were added in subsequent development.

### 4.5 Memory Protection: Lock Reference Counting

```
INC_LOCK_REF(node):
    node.lock_ref += 1
    if node was in evictable_leaves:
        remove from evictable_leaves
        evictable_size -= len(node.value)
        protected_size += len(node.value)

DEC_LOCK_REF(node):
    node.lock_ref -= 1
    if node.lock_ref == 0 and node is leaf:
        add to evictable_leaves
        protected_size -= len(node.value)
        evictable_size += len(node.value)
```

---

## 5. Memory Management Details

### 5.1 Dynamic Memory Sharing

The system dynamically shares GPU memory between:
- **Running requests** (actively being computed)
- **Cached KV entries** (stored in radix tree for future reuse)

There is no fixed partition. When a new batch needs memory, cached entries are evicted. When requests complete, their KV caches are inserted into the tree for future reuse.

### 5.2 Page-Based Allocation

KV cache memory is managed in pages (inspired by vLLM's PagedAttention):
- Non-contiguous allocation avoids fragmentation
- Pages can be shared across requests (copy-on-write semantics implicit via tree structure)
- Each node's `value` tensor stores indices into a global KV cache page pool

### 5.3 Size Tracking

The system maintains:
- `evictable_size_`: Total tokens in unlocked leaf nodes (available for eviction)
- `protected_size_`: Total tokens in locked nodes (serving active requests)
- Total cache usage = evictable_size + protected_size

### 5.4 Host (CPU) Memory Extension

The implementation supports optional CPU-side backup:
- `host_value`: CPU copy of KV cache pages
- `host_ref_counter`: Reference count for CPU copies
- Enables hierarchical caching (GPU -> CPU -> evict)

---

## 6. Compressed Finite State Machine (Structured Output)

### 6.1 Technique

For constrained decoding (e.g., JSON schema compliance):
1. Build FSM from regex/grammar specification
2. Identify **singular transitions**: states where only one valid next character exists
3. **Compress** consecutive singular transitions into multi-token edges
4. Decode multiple tokens simultaneously when traversing compressed edges

### 6.2 Performance Impact

- 1.6x throughput increase on JSON decoding
- 2.4x reduction in preprocessing overhead when amortized across batches
- No accuracy loss (output is identical to token-by-token constrained decoding)

---

## 7. Benchmark Results (Complete)

### 7.1 Hardware Configuration

| Setup | GPU | Memory | Use Case |
|-------|-----|--------|----------|
| Primary | NVIDIA A10G | 24GB | 7B models, single GPU |
| Extended | NVIDIA A10G x4 | 96GB | 70B models, tensor parallelism |
| Supplementary | NVIDIA A100 | 80GB | Larger batch experiments |
| Platform | AWS EC2 G5 instances | - | All experiments |

### 7.2 Models Evaluated

- **Llama-2-7B**: Primary model for most benchmarks
- **Llama-2-70B**: Tensor parallel evaluation
- **Mixtral-8x7B**: Sparse MoE architecture
- **LLaVA-v1.5-7B**: Image-language model
- **LLaVA-NeXT-34B**: Video-language model (production)
- **Vicuna-33B**: Production chatbot (Chatbot Arena)
- **GPT-3.5**: API model (speculative execution benchmarks)

### 7.3 Baselines

- **Guidance v0.1.8** with llama.cpp backend
- **vLLM v0.2.5** (before RadixAttention integration)
- **LMQL v0.7.3** with HuggingFace Transformers backend

### 7.4 Workloads and Results

| Workload | Description | Speedup | Cache Hit Rate |
|----------|-------------|---------|----------------|
| **Multi-turn chat (short)** | 4 turns, 4-8 tokens/turn | Up to 5x | High (history reuse) |
| **Multi-turn chat (long)** | 4 turns, 256-512 tokens/turn | Minimal | High (but decoding dominates) |
| **Few-shot MMLU** | 5-shot multiple choice | ~4x | ~99% (shared exemplars) |
| **Few-shot HellaSwag** | 20-shot with two-level sharing | ~5x | ~99% (two-level sharing) |
| **Tree-of-thought (GSM-8K)** | Branching reasoning paths | ~4-6x | 50-90% (branch-dependent) |
| **Skeleton-of-thought** | Parallel outline generation | ~3x | Moderate |
| **ReAct agent** | Iterative tool-use loop | ~5x | High (template + history) |
| **Generative agents** | Multi-agent simulation | ~4x | Moderate-High |
| **JSON decoding** | Schema-constrained output | ~1.6x (FSM) | N/A (FSM optimization) |
| **DSPy RAG pipeline** | Retrieval-augmented generation | ~3x | Moderate |
| **LLaVA image QA** | Multimodal with prefix | ~3x | High (system prompt) |

### 7.5 Key Quantitative Results

- **Maximum throughput improvement:** 6.4x over baselines
- **Maximum latency reduction:** 3.7x
- **Cache-aware scheduling:** Achieves 96% of optimal hit rate
- **Overhead on non-reusable workloads:** < 0.3% (negligible)
- **Production (Vicuna-33B):** 74.1% cache hit rate, 1.7x first-token latency reduction
- **Production (LLaVA-NeXT-34B):** 52.4% cache hit rate, ~1.7x first-token latency reduction

### 7.6 Metrics Reported

- **Throughput:** Programs executed per second (saturated batch)
- **Latency:** Average time per program execution (no batching)
- **Cache hit rate:** Fraction of tokens reused from cache vs. recomputed

---

## 8. Implementation Details from Source Code

### 8.1 RadixKey Class

```python
class RadixKey:
    token_ids: Tuple[int, ...]     # Raw token IDs
    extra_key: Optional[Any]        # Namespace isolation (LoRA ID, version)
    bigram_mode: bool               # For speculative decoding (EAGLE)
    page_size: int                  # Alignment granularity

    def length(self):
        if self.bigram_mode:
            return max(0, len(self.token_ids) - 1)
        return len(self.token_ids)

    def page_aligned_length(self):
        return (self.length() // self.page_size) * self.page_size
```

### 8.2 Namespace Isolation

The `extra_key` field enables isolation between:
- Different LoRA adapters (same base tokens, different KV caches)
- Different cache versions (invalidation without tree rebuild)
- Multi-tenant scenarios (preventing cross-user cache pollution)

### 8.3 Event Recording System

The implementation tracks KV cache lifecycle events:
- `BlockStored(hash, page_index)`: New page cached
- `BlockRemoved(hash, page_index)`: Page evicted
- `AllBlocksCleared`: Full cache reset

This enables external monitoring, debugging, and potential distributed coordination.

### 8.4 Integration with Continuous Batching

RadixAttention integrates with continuous batching:
1. New tokens from running requests are periodically "committed" to the tree
2. `cache_finished_req()`: Inserts completed request's full KV cache
3. `cache_unfinished_req()`: Partially caches interrupted requests (preemption support)

### 8.5 SLRU (Segmented LRU) Implementation

The SLRU strategy maintains two segments:
- **Protected segment**: Nodes that have been accessed at least twice
- **Probationary segment**: Newly inserted nodes
- On hit in probationary: promote to protected
- Eviction targets probationary first, then protected

---

## 9. The Complete SGLang Optimization Stack

| Optimization | Layer | Impact |
|-------------|-------|--------|
| RadixAttention | Runtime/Backend | KV cache reuse across requests |
| Cache-aware scheduling | Scheduler | Optimal request ordering |
| Compressed FSM | Backend | Multi-token constrained decoding |
| API speculative execution | Frontend | Reduces API costs by ~3x |
| Fork/join parallelism | Frontend | Concurrent branch execution |
| Prefix hints | Frontend-Runtime | Simplifies scheduling |

---

## 10. Known Limitations

### 10.1 Limitations Acknowledged in the Paper

1. **Starvation risk**: Greedy longest-prefix-first scheduling can indefinitely delay short-prefix requests under heavy load. No fairness guarantees are provided.

2. **Exact token matching only**: RadixAttention requires exact token-level prefix matches. Semantically equivalent but differently tokenized prompts receive no reuse benefit.

3. **Single-machine scope**: The paper's radix tree is local to one server instance. No distributed KV cache sharing across machines.

4. **Memory hierarchy unexplored**: The paper does not exploit DRAM/SSD tiers for larger effective cache sizes (later addressed by CachedAttention).

5. **Fairness-scheduling integration**: Combining cache optimization with fairness/SLO scheduling is left as future work.

6. **Long-output workloads**: When generation length dominates prefill time (e.g., multi-turn chat with long responses), cache reuse provides minimal end-to-end benefit.

### 10.2 Limitations from External Analysis

7. **Cold start**: First request for any prefix always pays full computation cost. No predictive prefetching.

8. **Fragmentation in variable-length nodes**: Unlike fixed-block approaches (vLLM), variable-length edges can lead to suboptimal memory utilization after many splits.

9. **No cost-aware eviction**: LRU does not account for the recomputation cost of evicted prefixes (longer prefixes are more expensive to regenerate).

10. **No frequency-aware default**: The default LRU does not consider access frequency, potentially evicting frequently-used but not most-recently-used prefixes.

---

## 11. Follow-Up and Related Work

### 11.1 Direct Follow-Up: vLLM Automatic Prefix Caching

vLLM integrated SGLang-inspired prefix caching as an experimental feature. Their approach uses **hash-based block matching** rather than a radix tree, operating at fixed block granularity. Trade-off: simpler implementation but less efficient for variable-length sharing.

### 11.2 CacheGen (SIGCOMM 2024)

- **Problem:** KV cache transmission bottleneck across network
- **Solution:** Custom tensor encoder compressing KV caches (3.5-4.3x size reduction)
- **Relevance:** Complementary to RadixAttention; could enable distributed cache sharing

### 11.3 ChunkAttention (ACL 2024)

- **Problem:** Prefix sharing across requests with shared system prompts
- **Solution:** Chunked KV cache structured into prefix tree + two-phase partition algorithm
- **Performance:** 3.2-4.8x speedup for system prompts (1024-4096 tokens)
- **Difference from SGLang:** Focuses on kernel-level attention optimization within shared prefixes

### 11.4 CachedAttention (USENIX ATC 2024)

- **Problem:** Multi-turn conversation KV cache reuse across hierarchical memory
- **Solution:** Layer-wise pre-loading, async saving, scheduler-aware fetching/eviction
- **Performance:** Up to 87% TTFT reduction, 7.8x prefill throughput improvement
- **Improvement over SGLang:** Hierarchical memory (GPU/CPU/disk), positional encoding decoupling

### 11.5 CacheBlend (2024)

- **Problem:** KV caches unusable when text is not a prefix (RAG scenarios)
- **Solution:** Selective recomputation of subset of tokens in pre-computed KV caches
- **Performance:** 2.2-3.3x TTFT reduction, 2.8-5x throughput improvement
- **Improvement over SGLang:** Works for non-prefix reuse (arbitrary document chunks)

### 11.6 Prompt Cache (2023)

- **Problem:** Reuse attention states for recurring prompt segments
- **Solution:** "Prompt modules" with schema ensuring positional accuracy
- **Performance:** 8x GPU latency reduction, 60x CPU latency reduction
- **Difference from SGLang:** Explicit module definition vs. automatic prefix detection

### 11.7 MemServe (2024)

- **Problem:** Combining inter-request and intra-request memory optimizations
- **Solution:** MemPool - elastic distributed memory pool with global prompt tree scheduling
- **Improvement over SGLang:** Distributed setting, combining disaggregated inference with caching

### 11.8 Infinite-LLM (2024)

- **Problem:** Long-context serving (up to 2M tokens)
- **Solution:** DistAttention + distributed KV cache pool across cluster
- **Performance:** 1.35-3.4x throughput improvement on 32 A100s
- **Complementary to SGLang:** Addresses the scale dimension rather than prefix sharing

---

## 12. Production Deployment Results

SGLang has been deployed at scale:
- **Chatbot Arena (LMSYS):** Serving multiple models including Vicuna-33B and LLaVA-NeXT-34B
- **Adoption:** Over 400,000 GPUs worldwide run SGLang in production
- **Organizations:** xAI, AMD, NVIDIA, Google Cloud, major academic institutions
- **Repository:** 27,000+ stars, 5,700+ forks, 12,000+ commits

Production observations:
- Vicuna-33B: 74.1% cache hit rate (dominated by system prompt + chat history reuse)
- LLaVA-NeXT-34B: 52.4% cache hit rate (lower due to diverse image inputs)
- Both achieve ~1.7x reduction in first-token latency

---

## 13. Connection to Broader Caching Theory

RadixAttention can be understood through classical caching theory:

| Concept | Classical | RadixAttention |
|---------|-----------|---------------|
| Cache item | Fixed-size block | Variable-length node (token sequence) |
| Cache key | Address | Token prefix sequence |
| Hit | Address match | Prefix match in radix tree |
| Miss | Address not found | No matching prefix |
| Eviction | Block replacement | Leaf node removal (bottom-up) |
| Optimal | Belady's MIN | DFS ordering (Theorem 3.1) |
| Working set | Active pages | Active prefix set |

The system fundamentally exploits **temporal and structural locality** in LLM workloads:
- **Temporal locality:** Recently used prefixes likely reused (LRU effective)
- **Structural locality:** Tree-structured sharing among concurrent requests (DFS scheduling optimal)
