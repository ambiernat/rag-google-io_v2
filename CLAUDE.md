# CLAUDE.md — rag-google-io

## Project Overview
Production-grade RAG system built on Google I/O 2025 YouTube transcripts.
Supports dense, sparse, and hybrid retrieval strategies with offline evaluation
and a deployed FastAPI service on AWS ECS Fargate.

**Current focus:** Adding agentic retrieval layer on top of existing system.

---

## Project Structure
```
~/Documents/rag-google-io/
├── api/                    # FastAPI service (main.py, routers, schemas)
├── retrieval/              # Retrieval strategies
│   ├── retrievers/         # retrieve_dense.py, retrieve_sparse.py, retrieve_hybrid.py
│   ├── rerankers/          # CrossEncoder reranker
│   └── hpo/                # Optuna HPO scripts
├── evaluation/             # Evaluation scripts and ground truth generation
│   ├── evaluate_sparse.py
│   ├── evaluate_dense.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_retrieval.py   # ⚠️ ABANDONED — do not use or modify
│   ├── ground_truth_llm_synthetic.py
│   ├── ground_truth_llm_paraphrase.py
│   └── ground_truth_llm_multi-doc.py
├── ingestion/              # Transcript fetch, canonicalize, chunk, orchestrate
├── vector_store/           # Qdrant ingestion scripts (ingest_dense/sparse/hybrid)
├── build/                  # BM25 preparation
├── data/
│   ├── raw/                # Original Google I/O transcripts — never modify
│   ├── canonical/          # Cleaned transcripts (all_documents.json)
│   ├── chunked/            # Chunked documents
│   └── eval/
│       ├── all/            # Full unsplit ground truth — source of truth
│       ├── dev/            # 60% — use for tuning, HPO, prompt engineering
│       ├── val/            # 20% — periodic sanity checks only
│       ├── test/           # 20% — LOCKED, do not touch until final evaluation
│       └── split_indices.json  # Reproducibility record — never modify
├── configs/                # ingestion.yaml
├── mlruns/                 # MLflow experiment tracking
└── qdrant_storage/         # Local Qdrant data
```

---

## Environment
- **OS:** Ubuntu (VM, username: al)
- **Python:** miniconda, env: `rag`
- **Activate:** `conda activate rag`
- **Run scripts:** always prefix with `QDRANT_URL=http://localhost:6333 PYTHONPATH=.`
- **Qdrant:** runs locally at `http://localhost:6333`
- **MLflow:** local tracking, `mlruns/` directory

---

## Qdrant Collections
| Collection | Strategy |
|---|---|
| `google-io-transcripts-dense` | Dense (sentence-transformers/all-MiniLM-L6-v2) |
| `google-io-transcripts-sparse` | Sparse (BM25) |
| `google-io-transcripts-hybrid` | Hybrid (dense + sparse combined) |

---

## Evaluation Discipline — IMPORTANT
Ground truth is split into dev/val/test. Respect these boundaries at all times:

| Split | Location | Purpose |
|---|---|---|
| dev | `data/eval/dev/` | All tuning — Optuna HPO, prompt engineering |
| val | `data/eval/val/` | Sanity checks only — not for tuning decisions |
| test | `data/eval/test/` | 🔒 LOCKED — touch only once, at final evaluation |

**Primary evaluation file:** `data/eval/dev/multi_doc.json`
**Never run Optuna or agentic loop tuning against val or test.**

### Baseline Metrics (dev/multi_doc.json)
| Method | Recall@K | MRR | Precision@K |
|---|---|---|---|
| Dense | 0.9251 | 0.7954 | 0.3989 |
| Sparse | 0.9786 | 0.8825 | 0.4064 |
| **Hybrid** | **1.0000** | **0.9084** | **0.4513** |

**Hybrid is the best baseline** — this is what the agentic loop must beat.

---

## Ground Truth Chain
Generated in this order — do not modify generated files, regenerate if needed:
```
ground_truth_llm_synthetic.py    → gt_gpt-5-nano_synthetic_{timestamp}.json
        ↓
ground_truth_llm_paraphrase.py   → gt_gpt-5-nano_paraphrased_{timestamp}.json
        ↓
ground_truth_llm_multi-doc.py    → gt_llm_multi-doc_{timestamp}.json
        ↓
split_ground_truth.py            → dev/ val/ test/ + split_indices.json
```

**Known limitation:** multi-doc enrichment ran against full Qdrant index before
split was introduced (Option C leakage). Labels are slightly biased toward dense
retrieval behaviour. Acceptable for portfolio/production use; not suitable for
academic publication without re-running enrichment post-split.

---

## AWS Deployment
- **Region:** us-east-1
- **ECR repo:** `<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag`
- **ECS cluster:** `rag-cluster`
- **ECS service:** `rag-task3`
- **Runtime:** Fargate (serverless) — 2 containers: FastAPI (3GB) + Qdrant (1GB)
- **Restart service:**
```bash
aws ecs update-service \
  --cluster rag-cluster \
  --service rag-task3 \
  --desired-count 1 \
  --region us-east-1
```

---

## Coding Conventions
- All retrieval functions return Qdrant `ScoredPoint` objects
- Error handling: always log with `[ERROR]` prefix, never silently swallow
- Timestamps: always use UTC (`datetime.now(UTC)`)
- Config: always load from `config.yaml` via `yaml.safe_load()`
- Tests: `pytest tests/` — never `python -m unittest`
- Ground truth loading: use `get_latest_ground_truth()` for timestamped files,
  direct path for split files (no timestamp)

---

## Current Agentic Retrieval Build Plan
Building agentic retrieval layer in phases — all tuning against `dev/multi_doc.json`:

- [ ] Phase 1: Query rewriter (Claude API — generate 2-3 retrieval-optimized variants)
- [ ] Phase 2: Self-evaluator (Claude API — score chunk relevance 1-5)
- [ ] Phase 3: Retry loop with strategy switching (hybrid → dense → sparse)
- [ ] Phase 4: Final evaluation vs baseline on `test/multi_doc.json`

**Target:** beat hybrid baseline MRR of 0.9084 on dev/multi_doc.json

---

## Known Issues & Limitations
- `evaluate_retrieval.py` — **abandoned**, do not use or modify
- `split_indices.json` — never modify, reproducibility depends on it
- Option C leakage in multi-doc ground truth (documented above)
- Existing Optuna HPO ran against full dataset pre-split —
  TODO: rerun cleanly against `data/eval/dev/multi_doc.json` before final reporting
- `.claude/` is in `.gitignore` — do not stage Claude Code internal files

---

## Git Rules
- **Never push to git without explicit approval**
- Always run `git diff` and summarise changes before any commit
- `.claude/` must remain in `.gitignore`
- Never stage `qdrant_storage/`, `mlruns/`, or `data/` directories

---

## Credentials
Stored in KeePassXC. Never hardcode API keys. Load from environment:
```bash
export ANTHROPIC_API_KEY=...   # from KeePassXC
export OPENAI_API_KEY=...      # from KeePassXC
```
