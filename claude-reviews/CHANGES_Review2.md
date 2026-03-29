# CHANGES_Review2.md

Design fixes applied following code review for agentic retrieval readiness.
Date: 2026-03-29

---

## Changes

### `retrieval/result.py` (new)

Introduced `RetrievalResult` dataclass as the canonical domain type for retrieval
output. Provides a Qdrant-agnostic representation and the single authoritative
conversion from `ScoredPoint` objects.

Key methods:
- `from_scored_point(point)` — converts a Qdrant `ScoredPoint`, centralising all
  payload key access (`doc_id`, `text`, `title`, `video_id`) in one place.
- `to_dict()` — returns a plain dict compatible with `crossencoder_rerank()` input
  (which expects a `"text"` key).

Fixes **Gap 3** (no domain type) and **Gap 2** (scattered `ScoredPoint` → dict
conversion). Previously, every caller duplicated `.payload.get("doc_id", str(hit.id))`
and friends inline.

---

### `retrieval/retrievers/dispatcher.py` (new)

Unified retrieval entry point: `retrieve(query, strategy, top_k) -> list[RetrievalResult]`.

Routes to `retrieve_dense`, `retrieve_sparse`, or `retrieve_hybrid` by strategy,
then converts at the boundary using `RetrievalResult.from_scored_point`. Callers
(agent, evaluation harness) receive typed results and never touch `ScoredPoint`.

Also exports `Strategy = Literal["dense", "sparse", "hybrid"]` for use in type
annotations elsewhere.

Fixes **Gap 1** (no unified dispatcher). The Phase 3 retry loop can now be written as:

```python
for strategy in ("hybrid", "dense", "sparse"):
    results = retrieve(query, strategy=strategy, top_k=10)
    if is_good_enough(results):
        break
```

---

### `retrieval/agent/__init__.py` (new)

Establishes the `retrieval/agent` package. Empty aside from comments indicating
the planned modules:

- `query_rewriter.py` — Phase 1
- `self_evaluator.py` — Phase 2
- `retry_loop.py` — Phase 3

Fixes **Gap 6** (no designated location for the agentic component).

---

### `retrieval/__init__.py` (edited)

Re-exports `RetrievalResult` at package level so callers can use
`from retrieval import RetrievalResult` rather than the internal path.

---

## What was not changed

- `retrieve_dense.py`, `retrieve_sparse.py`, `retrieve_hybrid.py` — untouched;
  still return `List[ScoredPoint]` internally.
- `crossencoder_reranker.py` — untouched; `RetrievalResult.to_dict()` is already
  compatible with its `doc.get("text", ...)` interface.
- `api/routers/search.py` — untouched; migration to `dispatcher.py` is out of scope.
- Evaluation scripts — untouched; importability refactor is a separate task (Gap 4).

---

## Known pre-existing issue noted

The retriever modules execute `[INFO]` print statements at import time (module-level
side effects). These will appear whenever `dispatcher.py` is imported, which will
be noisy inside the agent loop. Recommend moving initialisation prints behind a
`if __name__ == "__main__"` guard or converting to `logging.debug` before Phase 1.
