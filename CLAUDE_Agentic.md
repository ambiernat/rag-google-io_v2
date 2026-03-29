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
Calls GPT-5-nano to generate 3 retrieval-optimised variants of a user query.
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
- Uses `response_format={"type": "json_object"}` for reliable parsing

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

## Conventions

- API key: `OPENAI_API_KEY` from environment (never hardcode)
- Config: `retrieval/agent/config.yaml` via `yaml.safe_load()`
- Errors: logged with `[ERROR]` prefix, never swallowed
- All agent components live in `retrieval/agent/`

---

## Remaining Phases

- [ ] Phase 2: `self_evaluator.py` — score chunk relevance 1–5 (GPT-5-nano or reranker)
- [ ] Phase 3: `retry_loop.py` — strategy switching with quality gate, uses dispatcher
- [ ] Phase 4: Final evaluation on `data/eval/test/multi_doc.json` vs hybrid baseline

**Target:** beat hybrid MRR 0.9084 on `dev/multi_doc.json`
