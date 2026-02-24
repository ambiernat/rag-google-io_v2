# 📄 RAG Retrieval, Evaluation & Production Deployment

## Project Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** for Google I/O transcripts, supporting **dense, sparse, and hybrid retrieval**, **reranking**, **offline evaluation**, and a **deployed FastAPI service on AWS**.

It combines:

- Research-grade evaluation pipelines  
- Config-driven retrieval experimentation  
- A production-ready FastAPI API  
- Containerized cloud deployment using **AWS ECS + Fargate**

---

## 🧠 Core Capabilities

### Retrieval
- **Sparse retrieval** — BM25  
- **Dense retrieval** — SentenceTransformers embeddings  
- **Hybrid retrieval** — Dense + sparse fusion  
- **Vector store** — Qdrant  

### Reranking
- CrossEncoder-based reranking  
- Hyperparameter optimization (HPO)  
- Offline comparison of reranking strategies  

### Evaluation
- Recall@K, MRR, Precision@K  
- Offline A/B testing  
- Experiment tracking artifacts  

### Production
- FastAPI search service  
- Dockerized deployment  
- AWS ECS + Fargate  
- CloudWatch logging & monitoring  

---

## 📂 Repository Structure (Actual)

```text
.
├── api/                    # FastAPI app (production)
│   ├── main.py
│   ├── routers/
│   │   ├── health.py
│   │   └── search.py
│   ├── models.py
│   └── schemas.py
│
├── retrieval/              # Retrieval logic
│   ├── retrievers/
│   │   ├── retrieve_dense.py
│   │   ├── retrieve_sparse.py
│   │   └── retrieve_hybrid.py
│   └── rerankers/
│
├── vector_store/           # Qdrant ingestion
│   ├── ingest_dense.py
│   ├── ingest_sparse.py
│   └── ingest_hybrid.py
│
├── evaluation/             # Offline evaluation
│   ├── evaluate_dense.py
│   ├── evaluate_sparse.py
│   ├── evaluate_hybrid.py
│   └── evaluate_rerank_post_hpo.py
│
├── ingestion/              # Data ingestion & preprocessing
├── data/                   # Raw, chunked, evaluation data
├── qdrant_storage/         # Local Qdrant persistence (dev)
├── tests/                  # Unit, integration & E2E tests
│
├── Dockerfile               # Production image
├── docker-compose.yml       # Local dev stack
├── docker-compose_prod.yml  # Production-like stack
├── requirements.api.txt
└── README.md

## ⚙️ Configuration

Configuration is **YAML-driven** across ingestion, retrieval, and evaluation.

**Example (`retrieval/config.yaml`):**

```yaml
qdrant:
  url: "http://localhost:6333"

collections:
  dense: "google-io-transcripts-dense"
  sparse: "google-io-transcripts-sparse"
  hybrid: "google-io-transcripts-hybrid"

retrieval:
  top_k: 5

## ▶️ Running Locally

### Docker (Recommended)

```bash
docker-compose up

## Services

### Example API Call

```bash
curl "http://localhost:8000/search?query=large language models&top_k=5"

## 🧪 Offline Evaluation
### Run retrieval benchmarks locally:

```bash
python evaluation/evaluate_dense.py
python evaluation/evaluate_sparse.py
python evaluation/evaluate_hybrid.py
python evaluation/evaluate_rerank_post_hpo.py

### Outputs are written to:

```text
data/eval/results/

## 🧠 Testing

- **Unit tests** — retrievers, rerankers, embeddings  
- **Integration tests** — Qdrant connectivity, collections  
- **End-to-end tests** — FastAPI search endpoint  

Run all tests with:

```bash
pytest

## 🚀 Production Deployment (AWS)

This project is fully deployed on AWS using serverless containers.

### 🐳 Docker
Production Docker image bundles:

- FastAPI API

- Retrieval logic

- Model dependencies

**Image size**: ~550MB

## 📦 ECR — Elastic Container Registry

- Private image registry

- Repository: fastapi-rag

**Image URI:**

```text
886166401772.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag:latest

## 🎯 ECS — Elastic Container Service

- **Cluster:** `rag-cluster`

### Task Definition
- **2 containers:**
  - FastAPI (3 GB RAM)
  - Qdrant (1 GB RAM)
- **Total resources:** 1 vCPU, 4 GB RAM

The ECS service keeps tasks alive (desired count configurable).

---

## ⚡ Fargate (Serverless Compute)

- No servers to manage  
- Pay only when tasks are running  

**Approximate cost at 1 running task:**
- ~$42/month  
- **$0 when desired count = 0**

---

## 🌐 Networking

- Default VPC  
- Public IP assigned per task  

**Security Group:**
- Inbound TCP 8000 (FastAPI)

**Example access:**
```text
http://<public-ip>:8000

## 📊 CloudWatch Logs

- **Log group:** `/ecs/rag-task`
- Separate streams per container:
  - `fastapi`
  - `qdrant`

**Used to debug:**
- Startup failures
- Missing models
- Misconfigured environment variables

---

## 🔐 IAM

- **ECS Task Execution Role:**
  - Pull images from ECR
  - Write logs to CloudWatch
- CLI user created for deployments
- Principle of least privilege applied

---

## 💻 EC2 (Temporary)

Used only once when CloudShell ran out of disk.

**Purpose:**
- Build Docker image
- Push image to ECR
- Instance terminated after use → no ongoing cost

---

## 📈 Offline & Online Experimentation

### Offline
- Metric comparison across retrieval strategies
- Reranking effectiveness
- Hyperparameter optimization

### Online (Foundation in Place)
The API can be extended to log:
- Queries
- Retrieved documents
- Clicks
- Experiment group

This enables production **A/B testing**.

---

## 🔮 Future Enhancements

- Persistent Qdrant storage via EFS
- Autoscaling ECS services
- Authentication & rate limiting
- Query analytics dashboard
- Multi-language retrieval
- Online learning from user feedback

---

## ✅ Summary

This repository implements a **complete RAG system lifecycle**:

- ✔ Research & evaluation
- ✔ Retrieval + reranking experimentation
- ✔ Production FastAPI service
- ✔ Dockerized deployment
- ✔ Serverless AWS infrastructure

It bridges the gap between **ML research code** and **real-world production deployment**.