# CLAUDE_Agentic.md

Changes and context for the agentic retrieval layer build.
See also: `CHANGES_Review2.md` for the design-fix foundation these build on.

---

## Foundation Fixes (pre-agentic, from design review)

### `retrieval/result.py` (new)
Domain type for retrieval results. Qdrant-agnostic `RetrievalResult` dataclass.
- `from_scored_point(point)` — single canonical conversion from `ScoredPoint`
- `to_dict()` — compatible with `crossencoder_rerank()` input (`"text"` key)

### `retrieval/retrievers/dispatcher.py` (new)
Unified entry point: `retrieve(query, strategy, top_k) -> list[RetrievalResult]`.
Routes to dense / sparse / hybrid, converts at the boundary.
```python
from retrieval.retrievers.dispatcher import retrieve
results = retrieve(query, strategy="hybrid", top_k=10)
```

### `retrieval/__init__.py` (edited)
Re-exports `RetrievalResult` at package level:
```python
from retrieval import RetrievalResult
```

### `retrieval/agent/__init__.py` (new)
Establishes the `retrieval/agent/` package for all agentic components.

---

## Phase 1 — Query Rewriter

### `retrieval/agent/query_rewriter.py` (new)
Calls gpt-4o-mini to generate 3 retrieval-optimised variants of a user query.
Slots in before the `retrieve()` call.

**Public API:**
```python
from retrieval.agent.query_rewriter import rewrite_query

variants = rewrite_query("What is Gemma?")
# ["What is the Gemma language model by Google?",
#  "Gemma open-source AI model capabilities",
#  "Google lightweight LLM Gemma features"]
```

**Behaviour:**
- Returns `list[str]` of length `num_variants` (default 3)
- Falls back to `[query]` on any API or parse error — caller always gets a usable list
- Parses response as plain text, one variant per line (no `response_format`)
- Debug log of raw response before parsing

**Fixes applied during build:**
- `max_tokens` → `max_completion_tokens` (gpt-5-nano requirement, carried forward)
- Removed `response_format={"type": "json_object"}` — not supported by gpt-5-nano
- Switched model gpt-5-nano → gpt-4o-mini (gpt-5-nano returned empty responses)

**Caller pattern (Phase 2/3):**
```python
variants = rewrite_query(query)
all_queries = [query] + variants
results = [retrieve(q, strategy="hybrid") for q in all_queries]
# then merge/dedup → self-evaluator
```

### `retrieval/agent/config.yaml` (new)
```yaml
query_rewriter:
  model: "gpt-4o-mini"
  num_variants: 3
  max_tokens: 512
```

---

## Phase 2 — Self-Evaluator

### `retrieval/agent/self_evaluator.py` (new)
Scores the relevance of retrieved chunks to a query using gpt-4o-mini.
Returns a `ScoredResult` per chunk with an integer score (1–5) and a one-sentence reasoning.
Used by Phase 3 to decide whether results are good enough or a strategy switch is needed.

**New type:**
```python
@dataclass
class ScoredResult:
    result: RetrievalResult
    score: int        # 1–5
    reasoning: str
```

**Public API:**
```python
from retrieval.agent.self_evaluator import evaluate_results, is_good_enough

scored = evaluate_results(query, results)       # list[ScoredResult]
good   = is_good_enough(scored)                 # bool, uses score_threshold from config
```

**Behaviour:**
- All chunks evaluated in a **single** API call to keep latency and cost low
- Uses `response_format={"type": "json_object"}` — gpt-4o-mini supports it reliably
- Response aligned back to input by `index` key — a missing entry produces `score=0`
  rather than a misaligned list
- Falls back to `score=0, reasoning="api error"` per chunk on any failure
- `text_preview_chars=300` caps prompt tokens regardless of chunk size
- Debug log of raw response before parsing

**Phase 3 usage pattern:**
```python
for strategy in ("hybrid", "dense", "sparse"):
    results = retrieve(query, strategy=strategy, top_k=10)
    scored  = evaluate_results(query, results)
    if is_good_enough(scored):
        break
```

### `retrieval/agent/config.yaml` (updated)
```yaml
self_evaluator:
  model: "gpt-4o-mini"
  max_completion_tokens: 1024
  text_preview_chars: 300   # chars of chunk text shown to the model per result
  score_threshold: 3        # minimum score for is_good_enough()
```

---

## Phase 3 — Retry Loop

### `retrieval/agent/retry_loop.py` (new)
Wires Phase 1 and Phase 2 together with strategy switching.
Tries strategies in order (`hybrid → dense → sparse`) until `is_good_enough()` returns True
or all strategies are exhausted.

**New type:**
```python
@dataclass
class AgentResult:
    query: str
    variants: list[str]               # rewritten variants used
    strategy_used: str                # strategy that produced the final results
    attempts: int                     # number of strategies tried
    scored_results: list[ScoredResult]
    good_enough: bool
    # property:
    top_results: list[RetrievalResult]  # scored_results sorted by LLM score desc
```

**Public API:**
```python
from retrieval.agent.retry_loop import run

result = run("What is Gemma?")
# result.strategy_used  → "hybrid" | "dense" | "sparse"
# result.good_enough    → True/False
# result.top_results    → RetrievalResult list, best first
```

**Flow:**
1. `rewrite_query(query)` → variants (Phase 1)
2. For each strategy in `["hybrid", "dense", "sparse"]`:
   - Retrieve on original query + all variants
   - Merge and deduplicate results by `doc_id` (keep highest retrieval score)
   - `evaluate_results(query, merged)` (Phase 2)
   - If `is_good_enough()` → log accepted strategy and return
   - Else → log best score and reason for switching, continue
3. If all strategies exhausted → return last attempt with `good_enough=False`

**Logging (INFO level):**
- Variants generated
- Each attempt: strategy name, attempt number
- Unique chunk count after merge
- On accept: strategy, attempt, best score
- On reject: best score and next strategy to try
- On exhaustion: final fallback message

**Internal helper — `_merge_results`:**
Deduplicates across multiple per-query result lists by `doc_id`, retaining the
entry with the highest retrieval score. Preserves first-occurrence order.
No external dependency — avoids the missing merge utility (Gap 5).

### `retrieval/agent/config.yaml` (updated)
```yaml
retry_loop:
  strategies: ["hybrid", "dense", "sparse"]
  top_k: 10
```

---

## Conventions

- API key: `OPENAI_API_KEY` from environment (never hardcode)
- Config: `retrieval/agent/config.yaml` via `yaml.safe_load()`
- Errors: logged with `[ERROR]` prefix, never swallowed
- All agent components live in `retrieval/agent/`

---

## Remaining Phases

- [x] Phase 2: `self_evaluator.py` — score chunk relevance 1–5 (gpt-4o-mini)
- [x] Phase 3: `retry_loop.py` — strategy switching with quality gate, uses dispatcher
- [ ] Phase 4: Final evaluation on `data/eval/test/multi_doc.json` vs hybrid baseline

**Target:** beat hybrid MRR 0.9084 on `dev/multi_doc.json`
