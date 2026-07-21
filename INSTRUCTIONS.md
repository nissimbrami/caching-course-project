# Caching in LLMs — Course Project Instructions (Canonical)

**Owner:** Nissim Brami — `nissimbrami@post.bgu.ac.il` — GitHub: `nissimbrami`
**Course:** Caching in LLMs, Ben-Gurion University
**Instructor:** Prof. Gil Einziger
**Repo root:** `C:\Users\I763940\caching-course-project\`
**Last consolidated:** 2026-07-21

This document is the single source of truth for what has to be delivered and
under what rules. Everything else in the repo is implementation. If any note
in another file contradicts this one, this one wins.

---

## 1. The course has two deliverables

### Task 1 — Paper presentation (in-class talk)

- Pick **one paper on caching in the context of LLMs**.
- The paper must be from a **good AI conference in the last 1–2 years**
  (ICML, ICLR, NeurIPS, ACL, EMNLP, NAACL, MLSys, ASPLOS, SOSP, OSDI, etc.).
- **Selected paper (locked 2026-07-21):**
  *Efficient Streaming Language Models with Attention Sinks*
  Xiao, Tian, Chen, Han, Lewis — ICLR 2024 — arXiv:2309.17453
  Official code: https://github.com/mit-han-lab/streaming-llm
  Chosen because: (i) it is at a good AI conference within the recency
  window, (ii) it is the lightest-weight paper on the shortlist to
  present (three slides, one softmax-intuition equation, three iconic
  figures, no CUDA and no probability theory), and (iii) its eviction
  rule is purely positional, which lets Task 2 (cost-aware GDSF eviction
  on GPTCache) plug in as the natural next step in the same story.
- Deliverable: a slide deck + speaker notes. Presented in class.
- Delivery method: same repo as Task 2 (git / online repo). No separate
  Moodle upload is required beyond the repo link.
- **Reference decks (mandatory style match).** The following decks live
  under `task1-presentation/reference-decks/`. The Task 1 deck must
  match their visual and structural conventions:
  - `Chapter1 - Cache (1).pptx` — **Prof. Gil Einziger's own Chapter 1
    lecture ("Adaptive W-TinyLFU")**. 43 slides at 10×7.5 in (4:3).
    Vocabulary: LRU, LFU, ARC, LIRS, Hyperbolic, FRD, W-TinyLFU,
    admission vs. eviction, hit-ratio evaluation, hill climbing,
    indicator. This is the professor's own aesthetic — plain background,
    one crisp idea per slide, workload names ("gradle", "S3", "OLTP",
    "F1", "DS1", "WS1") called out on comparison slides. Ends with an
    "Open Source" slide and a "The End" slide.
  - `SCALM.pptx` — **A prior student deck (Olga Oznovich) on SCALM**.
    22 slides at 13.33×7.5 in (16:9 widescreen). Structure:
    Title → Agenda → Basics → The Experiment → The Result → Implementation
    → What's next → Conclusion → Thank You. Uses course vocabulary (LFU)
    to introduce the paper. **This is the closest previous student
    deliverable to what we owe the professor.**
  - `Intro.pptx` — the course intro deck.
  - `Extended pipeline cache.pptx` — another chapter deck.
  Style rules extracted from these:
  1. 16:9 widescreen (13.33 × 7.5 in) — matches the recent student deck.
  2. One idea per slide. Concise bullets, not paragraphs.
  3. Open with title slide → agenda → basics → the paper → results →
     limits → what I'd change → conclusion → thank-you.
  4. Introduce the paper by anchoring to a concept the class already
     knows (LFU / LRU / admission vs eviction).
  5. End with an explicit bridge to my final project (Task 2 — GDSF on
     GPTCache).
  6. Every slide's numeric claim must be traceable to `deep-read.md`.

### Task 2 — Research mini-paper (final project)

- Pick a published result in the caching-for-LLMs literature.
- **Improve it** — algorithmically, systemically, or empirically.
- Support the improvement with **your own experiments**.
- Write it up as a short research paper (report PDF).
- Ship the **code** that produced the numbers, in a git repository.
- A **small bonus** is awarded for making the contribution an open-source
  patch against a real project (e.g. a GPTCache / vLLM / SGLang PR).
- Delivery: git / online repo link.

### What "improve a result" means in practice (from the professor)

The professor's exact framing is: take a paper, run its experiment or a
faithful reconstruction of it, and then show — with numbers — that a
modification you propose does better on at least one meaningful axis. It
does **not** have to be state-of-the-art. It has to be *honest, measured,
and reproducible*.

---

## 2. Concrete constraints and hard rules (aggregated across all messages)

These are the rules the user has laid down over the course of the conversation.
Every future action must respect them.

1. **Never fabricate numbers.** Every number that appears in a report,
   slide, or table must be traceable to a specific row of the benchmark
   JSON, or to code that can be re-executed.
2. **Never fabricate paper claims.** Every author name, algorithm detail,
   dataset name, and headline number in a presentation must be reproducible
   from the paper's actual text.
2b. **Task 1 paper must be about *caching in LLMs*, not about LLMs
    generally.** The paper's central contribution must be a cache
    mechanism (KV cache, semantic cache, response cache, prompt cache,
    prefix cache, embedding cache, or the eviction / admission /
    compression policy of one of those). A generic LLM paper on model
    architecture, training, or alignment does not qualify no matter how
    prestigious the venue. StreamingLLM (ICLR 2024) qualifies because
    its central object is the KV cache — see §3.2 "Rolling KV Cache
    with Attention Sinks" and Fig.4 "The KV cache of StreamingLLM."
3. **Any statistical claim requires code.** If the report says "paired
   t-test, Bonferroni-corrected, BCa bootstrap 95% CI", the script that
   computes those values must live under `scripts/` and must run on the
   benchmark JSON. No hand-typed intervals.
4. **Do not optimise for token cost. Ever.** The user has explicitly and
   repeatedly rescinded any earlier instruction to be terse. Use whatever
   tools, agents, passes, PDF reads, benchmark re-runs, and audit rounds
   are needed to reach a *correct* result. Never truncate a paper read,
   never skip a verification pass, never shorten a report to save
   context, never skip spawning an audit agent because "it might be
   fine." Terseness is a bug in this project, not a feature. This rule
   overrides any earlier or default token-saving heuristic.
5. **When in doubt, spawn parallel agents.** For audits, cross-checks,
   paper searches, or independent verification passes, launch multiple
   Agent tool calls in a single message.
6. **Never `git push --force` to `main`.** Never bypass hooks. Never
   auto-commit; only commit when explicitly asked.
7. **Never commit secrets.** The GitHub PAT that appeared in chat must be
   treated as leaked — the user must rotate it before we push again.
8. **Windows paths + `bash` shell.** The environment is Git Bash on
   Windows 11. Use forward slashes and `/dev/null` in commands.
9. **PDF backend is reportlab.** `weasyprint` does not work on this
   Windows box; `scripts/build_report_pdf.py` uses reportlab.
10. **PowerPoint files may be locked.** `~$*.pptx` lock files are in
    `.gitignore`. Never delete `~$*.pptx` while PowerPoint is open.

11. **GitHub push is the last thing, not a blocker.** Every verification
    gate in §9 must be green for *both* tasks before we push. No partial
    pushes, no "push and fix later".
12. **No AI-trace.** Nothing about the writing, code comments, commit
    messages, slide notes, or file headers may look machine-generated.
    No em-dash-heavy sentences, no "As an AI", no "Certainly!", no
    boilerplate hedges, no signature footers like "generated by".
    Every artifact must read as if a human PhD student wrote it after a
    week of thought.
13. **Verify every stage against the instructions.** After each stage in
    §9, we re-open INSTRUCTIONS.md §1–§5 and check off — in writing —
    which rules the stage satisfied. Nothing is "done" until that
    checklist is green.

---

## 3. Original paper list (context only — most is off-limits)

The registration form the user was shown originally lists ~34 papers with
students already assigned. The three that were left unclaimed on the form
were:

- **LeanKV** — turned out to have been renamed **DiffKV** and accepted at
  **SOSP 2025**. Still viable if we want it.
- **vAttention: Dynamic Memory Management for Serving LLMs without
  PagedAttention** — real, **ASPLOS 2025**. Viable.
- **"Hybrid KV Compression for Extending Context Length in vLLM"** — audit
  found **no such paper exists** under that exact title. Off the table.

The professor then said in email that the entire list is stale anyway and
that any 2024–2026 top-AI-venue caching paper is fine. So the list is
informational, not binding.

---

## 4. Task 1 — Presentation: current state and required fixes

### Current state (as of 2026-07-21)

- Present artifacts on disk:
  - `task1-presentation/build_scalm_deck.py`
  - `task1-presentation/SCALM_Presentation.pptx`
  - `task1-presentation/RadixAttention_Presentation_v2.pptx` (older draft)
  - `task1-presentation/notes/speaker-notes.md` (about SCALM)
  - `task1-presentation/papers/sglang-deep-read.md`
  - `task1-presentation/papers/sglang-critical-analysis.md`
- The **SCALM deck and speaker-notes are factually wrong** on nearly every
  hard claim:
  - Wrong author list (real authors: Jiaxing Li, Chi Xu, Feng Wang,
    Isaac M. von Riedemann, Cong Zhang, Jiangchuan Liu)
  - Wrong clustering algorithm (real: DBSCAN — not streaming k-means with
    EMA decay λ)
  - Wrong eviction description (real: rank-weighted LFU with 3/2/1
    weights — not `freq × cost / age`)
  - Wrong datasets (real: LMSYS-Chat + MOSS — not "enterprise assistant")
  - Wrong cache-size range (real: 20 – 200 — not 10,000)
  - Wrong headline hit-rate numbers
- SCALM is also **arXiv-only** (arXiv:2406.00025) — it is not published at
  a conference — so under the professor's rule "good AI conference in the
  last 1-2 years", it does not qualify.

### Required action

**Retire SCALM.** Rebuild Task 1 around one of these conference-published
caching-for-LLMs papers from 2024–2025 (all confirmed real by audit):

| Paper                                     | Venue         | Why it's a good pick                              |
|-------------------------------------------|---------------|---------------------------------------------------|
| Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | ICML 2024     | KV-cache pruning; clean numbers; easy to present  |
| FastGen (Adaptive KV Cache Compression)   | ICLR 2024     | Per-head compression policy; strong ablation      |
| Layer-Condensed KV Cache                  | ACL 2024      | Cross-layer sharing; one core idea; small slides  |
| vAttention                                | ASPLOS 2025   | Systems-flavoured; PagedAttention alternative     |
| StreamingLLM                              | ICLR 2024     | Attention-sink insight; iconic figure             |
| Keyformer                                 | MLSys 2024    | Token importance for KV eviction                  |
| DiffKV (a.k.a. LeanKV)                    | SOSP 2025     | Differentiated key/value quantisation             |

Recommended default: **Quest (ICML 2024)** — smallest surface area, clean
figures, one central mechanism, straightforward to align with the course
vocabulary.

Once the paper is chosen:

1. Delete `build_scalm_deck.py`, `SCALM_Presentation.pptx`, and
   `notes/speaker-notes.md`.
2. Read the chosen paper end-to-end.
3. Rebuild `task1-presentation/build_<paper>_deck.py` and
   `notes/speaker-notes.md` from the paper text — every author name,
   dataset, number, and figure caption must be re-verified against the
   paper.
4. Regenerate the `.pptx`.

---

## 5. Task 2 — Final project: current state and required fixes

### One-line thesis

Replace GPTCache's LRU eviction with a full **Greedy-Dual-Size-Frequency
(GDSF)** cost-aware eviction policy and show, via 30-run benchmarks, that
it wins on cost-heterogeneous LLM workloads.

Priority function:

    Priority(i) = Clock + freq(i)^α · cost(i)^β / size(i)

where `Clock` is the last-eviction priority (GDSF aging), `cost` is the
dollarised token cost of regenerating the entry, and `size` is bytes.

### Layout (`task2-final-project/code/`)

- `src/cost_aware_eviction/` — the GDSF plugin proper (indexed min-heap
  implementation, tested).
- `benchmarks/policies.py` — a **second** GDSF implementation (tombstone
  heap) that the benchmark harness actually uses.
- `benchmarks/workloads.py` — 6 synthetic workloads:
  `uniform_cost`, `high_variance_cost`, `zipf_variable_cost`, `bursty`,
  `adversarial_lru`, `size_varying`.
- `benchmarks/run_all.py` — top-level driver.
- `benchmarks/runner.py` — per-experiment runner.
- `tests/` — pytest + Hypothesis property tests.
- `results/benchmark_results_20260719_183822.json` — the real 30-run
  output (3,600 experiments = 5 policies × 6 workloads × 4 cache sizes ×
  30 seeds).
- `results/plots/fig1..8_*.png` — plots referenced by the report.
- `docs/report-draft.md` — the paper source.
- `docs/report.pdf` — rendered PDF (via `scripts/build_report_pdf.py`).
- `scripts/build_report_pdf.py` — reportlab-based Markdown-to-PDF
  compiler; supports headings, lists, tables, code, images, LaTeX-lite
  math stripping, `[Figure N: caption]` figure references.
- `docker-compose.yml`, `Dockerfile`, `Makefile`, `pyproject.toml`,
  `requirements.txt`, `README.md` — reproducibility surface.

### Known problems (from parallel audit — must fix before submission)

1. **Report Section 5 tables do not match the JSON.** The CWHR / hit-rate
   / dollar-savings numbers currently printed in `docs/report-draft.md`
   were mean-aggregated in a way that does not reproduce from any single
   `cache_size` row. Fix: rewrite Section 5 to report per-`cache_size`
   values that come directly from `benchmark_results_20260719_183822.json`.
2. **Two GDSF implementations disagree.** `src/cost_aware_eviction/` uses
   an indexed min-heap; `benchmarks/policies.py` uses a tombstone heap.
   The benchmark measures the tombstone version, the tests test the
   indexed version. Fix: unify.
3. **Duplicate-put-larger-size bug** in
   `src/cost_aware_eviction/eviction_manager.py` around lines 113–126:
   updating an existing key with a larger `size` does not re-enforce
   capacity.
4. **Statistical methodology is claimed but not implemented.** The report
   claims BCa bootstrap 95% CIs and paired t-tests with Bonferroni
   correction. No script produces those. Fix: add
   `scripts/compute_statistics.py` that reads the JSON and emits a
   deterministic stats table.
5. **Workload descriptions in report Section 4.1 are wrong.** The report
   describes `high_variance_cost` as log-normal — the code uses a
   trimodal mixture. The report says `size_varying` spans 10× — the code
   spans 1000×.
6. **Cache-size unit confusion.** `run_all.py` defaults to bytes
   (50000/100000/250000/500000); `runner.py` defaults to entry counts
   (100/500/1000/5000). Report copy sometimes says "entries", sometimes
   "bytes". Pick one and audit every reference.

### After the six fixes above

- Re-run the full benchmark suite (30 seeds × 6 workloads × 4 sizes × 5
  policies).
- Regenerate all figures (`scripts/plot_*.py`) and the PDF.
- Commit atomically. Do not force-push.

---

## 6. Delivery / git

**GitHub push is the LAST step, not the first.** It is not a blocker for
any earlier stage. Both Task 1 and Task 2 must be planned, executed, and
verified end-to-end (see §9) *before* we push anything anywhere. Only
after every verification gate is green do we rotate the token and push.

- Repo already initialised at
  `C:\Users\I763940\caching-course-project\` with a single commit
  (`bda52d4`) authored by `Nissim Brami <nissimbrami@post.bgu.ac.il>`.
- `.gitignore` at repo root excludes venvs, pytest cache, Office lock
  files, editor scratch, and the older RadixAttention draft.
- The PAT that was pasted earlier is treated as leaked and will not be
  used. A fresh token will be requested at the very end, right before push.
- Target remote: `https://github.com/nissimbrami/caching-course-project`
  (name TBD).
- Submission = single URL to that repo, containing both `task1-presentation/`
  and `task2-final-project/`.

---

## 7. Do I understand the two tasks? (Short answer)

**Yes.**

- **Task 1** is a presentation on a *caching-in-LLMs* paper from a good AI
  conference in the last 1–2 years. It is not on the stale registration
  list; we pick one from the audited shortlist (default: Quest, ICML 2024).
- **Task 2** is a research mini-paper that improves a published caching
  result (in our case: replacing GPTCache's LRU with cost-aware GDSF),
  supported by our own reproducible benchmarks, delivered as a report PDF
  plus code in the same git repo, with an optional open-source-contribution
  bonus.

---

## 8. Immediate next actions (ordered)

1. **User picks the Task 1 paper** from §4's shortlist (default: Quest).
2. Nuke the SCALM artifacts, rebuild the deck from the chosen paper.
3. Fix the six Task 2 audit findings listed in §5.
4. Regenerate the PDF, sanity-check every table cell against JSON.
5. Run the §9 verification gates for both tasks.
6. **Only then**: rotate the PAT, push to GitHub, submit the link.

---

## 9. Staged execution plan with verification gates

The rule is: **no stage may start until the previous stage's verification
gate is green.** Verification is not "I looked at it and it seems fine";
it is a concrete, mechanical check that either passes or fails.

### Stage 0 — Freeze the plan (this file)

- INSTRUCTIONS.md exists at repo root and covers everything both tasks
  must do.
- Verification gate:
  - [ ] User has read §1–§8 and explicitly signed off.
  - [ ] User has picked the Task 1 paper (default: Quest, ICML 2024).

### Stage 1 — Task 1 paper acquisition and deep-read

- Download the chosen paper's PDF into `task1-presentation/papers/`.
- Read it end-to-end (Explore agent, thorough).
- Produce `task1-presentation/papers/<paper>-deep-read.md` — a
  section-by-section faithful summary with page/line references. This is
  the *ground-truth artifact*; every later slide claim must trace back to
  a paragraph in this file.
- Produce `task1-presentation/papers/<paper>-critical-analysis.md` — our
  own critique: what the paper does well, where it's weak, what we would
  change. This is what feeds the "improvements I'd propose" slide.
- Verification gate:
  - [ ] PDF is on disk.
  - [ ] Deep-read covers every section of the paper, not just abstract.
  - [ ] Every headline number in the deep-read has a page reference.
  - [ ] Independent audit-agent run confirms deep-read matches paper text
        with **zero** invented details.

### Stage 2 — Task 1 deck build

- Delete SCALM artifacts:
  - `task1-presentation/build_scalm_deck.py`
  - `task1-presentation/SCALM_Presentation.pptx`
  - `task1-presentation/notes/speaker-notes.md`
- Also decide the fate of `RadixAttention_Presentation_v2.pptx` (likely
  delete — it's an unrelated older draft).
- Build `task1-presentation/build_<paper>_deck.py` using python-pptx.
- Build `task1-presentation/notes/speaker-notes.md` from the deep-read.
- Generate `task1-presentation/<Paper>_Presentation.pptx`.
- Verification gate:
  - [ ] Every author name in the deck matches the paper's title page.
  - [ ] Every number in the deck matches the deep-read.
  - [ ] Every figure caption cites a figure/table number in the paper.
  - [ ] No SCALM artifact remains on disk.
  - [ ] Speaker notes do not read as AI-generated (no em-dash pileups,
        no "Firstly/Secondly/Finally", no boilerplate hedges).
  - [ ] Independent audit-agent reads the deck + notes cold and reports
        zero factual mismatches vs the paper.

### Stage 3 — Task 2 code correctness fixes

Fix, in this order:

1. Unify the two GDSF implementations. Keep the indexed min-heap version
   in `src/cost_aware_eviction/`; make `benchmarks/policies.py` import
   from it. Delete the tombstone copy.
2. Fix the duplicate-put-larger-size bug in
   `src/cost_aware_eviction/eviction_manager.py:113-126`.
3. Reconcile `run_all.py` vs `runner.py` cache-size defaults. Decide
   once: cache size is in **bytes**. Rewrite `runner.py` defaults to
   match. Grep the whole tree for the word "entries" and audit.
4. Rewrite `benchmarks/workloads.py` docstrings so each workload's
   *code* matches its report description (trimodal, 1000× range, etc.),
   or vice versa — pick one and align.

- Verification gate:
  - [ ] `pytest tests/` passes fully.
  - [ ] Hypothesis property tests pass with `--hypothesis-seed=0..9`.
  - [ ] `grep -R "entries" task2-final-project/code/` returns only
        intentional matches (workload comments, not size claims).
  - [ ] Only one GDSF class exists in the tree.
  - [ ] Independent audit-agent diffs the code against the report claims
        and reports zero mismatches.

### Stage 4 — Task 2 benchmark re-run

- Re-run the full 30-seed × 6-workload × 4-cache-size × 5-policy suite
  using the unified GDSF.
- Emit a new `results/benchmark_results_<timestamp>.json`.
- Emit a new `results/stats_<timestamp>.json` from
  `scripts/compute_statistics.py` (paired t-tests + Bonferroni + BCa
  bootstrap CIs — actually implemented this time).
- Regenerate all 8 figures.
- Verification gate:
  - [ ] Benchmark JSON has exactly `5 × 6 × 4 × 30 = 3600` rows.
  - [ ] Stats JSON exists and every claim in the report references a key
        in it.
  - [ ] All 8 PNGs regenerate without error.

### Stage 5 — Task 2 report rewrite

- Rewrite Section 5 of `docs/report-draft.md` so **every** number, CI,
  p-value, and Δ comes from the JSON files produced in Stage 4. No
  hand-typed intervals.
- Fix Section 4.1 workload descriptions to match `workloads.py`.
- Regenerate `docs/report.pdf` via `scripts/build_report_pdf.py`.
- Verification gate:
  - [ ] Every table cell in Section 5 is reproducible by a documented
        query against the benchmark JSON.
  - [ ] Every "p = ..." and "95% CI = [..., ...]" resolves to a key in
        the stats JSON.
  - [ ] PDF renders; all 8 figures embedded; no `[Figure N missing]`
        placeholders.
  - [ ] Independent audit-agent reads the PDF cold, cross-checks every
        numeric claim against JSON, reports zero mismatches.
  - [ ] Report prose does not read AI-generated. In particular:
        no em-dash chains, no "In summary", no "It is worth noting",
        no signature footer.

### Stage 6 — Cross-task consistency + full re-audit

- Read INSTRUCTIONS.md §1–§8 line by line and check every rule.
- Launch parallel audit agents:
  1. Task 1 deck vs paper (as in Stage 2 gate).
  2. Task 2 code vs report (as in Stage 3 gate).
  3. Task 2 PDF vs JSON (as in Stage 5 gate).
  4. Full-repo AI-trace scan (looks for em-dash pileups, boilerplate,
     signature footers, "As an AI", "Certainly!", etc.).
- Verification gate:
  - [ ] All four audit agents return clean.
  - [ ] Every checkbox in §9 Stages 0–5 is ticked.

### Stage 7 — GitHub push (LAST STAGE, only after §9.6 is green)

- User rotates the leaked PAT and pastes a fresh one.
- Verify token: `curl -sS -H "Authorization: Bearer <PAT>" https://api.github.com/user`.
- Add remote, push `main`.
- Verify remote reflects local `HEAD`.
- User submits the repo URL.
- Verification gate:
  - [ ] `git ls-remote origin main` matches local `HEAD`.
  - [ ] Repo is browsable in a browser.
  - [ ] Both `task1-presentation/` and `task2-final-project/` are visible
        with all expected artifacts.

---
