# Code Review — Changes & Remaining TODOs

Review date: 2026-03-25

---

## Critical Fixes

### 1. Broken `SparseRetriever` import removed
**Why:** `api/routers/search.py` imported and instantiated `SparseRetriever`, a class whose entire body had been commented out in `retrieve_sparse.py`. This caused an `ImportError` or `AttributeError` on API startup. The actual retrieval logic exists as the module-level function `retrieve_sparse()`, which is what the router now calls directly.

**Affected files:**
- `api/routers/search.py`

---

## Refactoring

### 2. Extracted `get_latest_ground_truth` to shared utility
**Why:** The function was copy-pasted verbatim across four files. Any change to the selection logic (e.g. sorting by name instead of mtime) would have required four edits with no guarantee of consistency.

**New file:** `evaluation/utils.py`

**Affected files (duplicate removed, import added):**
- `evaluation/evaluate_dense.py`
- `evaluation/evaluate_sparse.py`
- `evaluation/evaluate_hybrid.py`
- `retrieval/hpo/hybrid_rerank_hpo.py`

---

### 3. Removed duplicate metric function definitions
**Why:** `recall_at_k`, `precision_at_k`, and `mrr` were already canonically defined in `retrieval/evaluation/metrics.py` but were re-implemented inline in each evaluation script. Divergence between copies was a silent bug risk. Scripts now import from the single canonical source.

**Affected files (local definitions removed, import added):**
- `evaluation/evaluate_dense.py`
- `evaluation/evaluate_sparse.py`
- `evaluation/evaluate_hybrid.py`

---

### 4. Consolidated embedding model lazy-load to `api/models.py`
**Why:** The `_embedding_model` global and `embed_query()` function were duplicated in three places. If the model name or normalisation settings changed, all three copies had to be kept in sync. The retriever modules now import `embed_query` from `api/models.py`.

**Affected files (local definition removed, import added):**
- `retrieval/retrievers/retrieve_dense.py`
- `retrieval/retrievers/retrieve_hybrid.py`

---

### 5. Extracted `log_manifest` to shared ingestion utility
**Why:** The function was copy-pasted between two ingestion scripts. A single shared implementation prevents the manifest format drifting between pipeline stages.

**New file:** `ingestion/utils.py`

**Affected files (local definition removed, import added):**
- `ingestion/fetch.py`
- `ingestion/chunk.py`

---

### 6. Replaced deprecated `datetime.utcnow()`
**Why:** `datetime.utcnow()` is deprecated since Python 3.12 and raises a `DeprecationWarning`. It returns a naïve datetime with no timezone info, which is ambiguous. `datetime.now(timezone.utc)` returns an aware datetime and is the documented replacement.

**Affected files:**
- `retrieval/retrievers/retrieve_dense.py`
- `retrieval/retrievers/retrieve_sparse.py`
- `evaluation/generate_ground_truth_llm.py`

---

### 7. Implemented real Qdrant connectivity check in `/ready` endpoint
**Why:** The readiness probe unconditionally returned `{"ready": True}`, making it indistinguishable from the liveness probe. Container orchestrators (ECS health checks, Kubernetes) rely on the readiness probe to withhold traffic until dependencies are available. The endpoint now attempts `client.get_collections()` and returns HTTP 503 if Qdrant is unreachable.

**Affected files:**
- `api/routers/health.py`

---

### 8. Replaced hardcoded Qdrant URL with environment variable
**Why:** Hardcoding `http://localhost:6333` means tests and evaluation scripts cannot target a remote or Dockerised Qdrant instance without modifying source code. `QDRANT_URL` env var now takes precedence, with the hardcoded value as a fallback.

**Affected files:**
- `tests/conftest.py`
- `evaluation/evaluate_dense.py`
- `evaluation/evaluate_sparse.py`
- `evaluation/evaluate_hybrid.py`

---

### 9. Added `top_k` bounds validation to API request schema
**Why:** An unbounded integer field allowed callers to pass negative values or arbitrarily large numbers (e.g. `top_k=100000`), which would either cause a Qdrant error or exhaust memory silently. Pydantic `Field(ge=1, le=100)` rejects invalid values at the schema layer before any retrieval logic runs.

**Affected files:**
- `api/schemas.py`

---

### 10. Replaced `print()` with `logging` across ingestion and evaluation scripts
**Why:** `print()` output cannot be filtered by level, redirected to log aggregators (CloudWatch, Loki), or silenced in tests. Structured logging with `logging.getLogger(__name__)` gives per-module control and is consistent with how the API layer already logs.

**Affected files:**
- `ingestion/fetch.py`
- `ingestion/chunk.py`
- `ingestion/canonicalize.py`
- `ingestion/orchestrator.py`
- `evaluation/evaluate_dense.py`
- `evaluation/evaluate_sparse.py`
- `evaluation/evaluate_hybrid.py`
- `evaluation/generate_ground_truth_llm.py`
- `retrieval/rerankers/crossencoder_reranker.py`

---

### 11. Added specific exception types to ingestion error handlers
**Why:** Bare `except Exception` hides the nature of failures and makes debugging harder. JSON parse failures and file I/O errors are now caught as `json.JSONDecodeError` and `OSError` respectively, with appropriate log levels. `logger.exception()` is used for unexpected errors so the full traceback is preserved.

**Affected files:**
- `ingestion/chunk.py`
- `ingestion/canonicalize.py`

---

### 12. Added missing return type hints to key functions
**Why:** Functions without return types make static analysis tools (mypy, Pyright) less effective and force readers to trace through the body to understand the contract. Only functions with no existing annotations were updated.

**Affected files:**
- `api/routers/search.py`
- `api/routers/health.py`
- `retrieval/rerankers/crossencoder_reranker.py`

---

## Intentionally Left Unchanged

- **`evaluation/evaluate_retrieval.py`** — file is broken (imports non-existent paths, incomplete `ndcg_at_k` implementation) but is kept as a work-in-progress placeholder. Should be fixed or deleted before production use.

---

## Remaining TODOs

### High Priority

- **Test coverage** — only ~20% of files have tests. Critical paths with no coverage include `api/routers/search.py` (the core endpoint), all `ingestion/` modules, and `retrieval/rerankers/crossencoder_reranker.py`. Unit tests for the search router should mock the retriever functions; ingestion tests should use temporary directories.

- **Config fragmentation** — there are three separate `config.yaml` files (`retrieval/`, `evaluation/`, `vector_store/`) that independently declare the Qdrant URL with different values (`qdrant:6333` vs `localhost:6333`). A single root-level config or a dedicated config loader that merges env vars would remove this ambiguity.

- **`evaluation/evaluate_retrieval.py` cleanup** — imports reference non-existent module paths and `ndcg_at_k` returns nothing. Either wire it up to the current retriever interface or delete it.

### Medium Priority

- **Qdrant request timeouts** — Qdrant client calls in the API have no timeout. A slow or hung Qdrant instance will hold FastAPI worker threads indefinitely. Set `timeout` on the `QdrantClient` constructor.

- **API input validation for `query`** — `query` is an unconstrained `str`. An empty string or a multi-megabyte payload will pass schema validation. Add `min_length=1` and a reasonable `max_length`.

- **Remaining `datetime.utcnow()` instances** — run `grep -r "utcnow" .` periodically as new code is added; the deprecation warning will become an error in Python 3.14.

- **Model cold start** — the `Dockerfile` has a commented-out model preload step. First-request latency can exceed 30 seconds while SentenceTransformers and CrossEncoder are downloaded. Uncomment or implement model pre-warming.

- **Qdrant persistence on ECS** — current deployment uses local `qdrant_storage/` which is wiped on container restart. Mount an EFS volume or switch to Qdrant Cloud before running in production.

### Low Priority

- **Docstrings on public functions** — most functions have no docstring. At minimum, the API endpoint, retriever functions, and metric functions would benefit from one-line descriptions of parameters and return values.

- **Type hints across remaining modules** — `evaluation/` LLM scripts and several `vector_store/` ingest scripts have no type annotations. Incremental adoption via mypy `--ignore-missing-imports` is a low-friction starting point.

- **Prometheus / metrics endpoint** — no `/metrics` endpoint exists. Adding `prometheus-fastapi-instrumentator` would expose request latency and error rates with minimal code and allow alerting on SLO breaches.

- **Data versioning in ingestion** — chunks and canonical documents are written without checksums or version identifiers. A failed mid-run cannot be safely resumed. Consider adding a content hash to each manifest entry and skipping already-processed items.
