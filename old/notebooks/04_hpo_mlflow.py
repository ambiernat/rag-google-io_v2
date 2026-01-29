#!/usr/bin/env python
# coding: utf-8

# # Set up and imports

# In[12]:


# Add project root to path
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)


# In[13]:


import json
import time
import numpy as np
from tqdm import tqdm

import optuna
import mlflow
import mlflow.optuna

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval.rerankers.cross_encoder import CrossEncoderReranker


# In[14]:


from retrieval.evaluation.pipelines import (
    dense_retrieve,
    evaluate_reranking,
)

from retrieval.rerankers.cross_encoder import CrossEncoderReranker


# # MLflow Setup

# In[15]:


mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("dense_rerank_hpo")


# In[16]:


mlflow.get_tracking_uri() #check


# # Load Data & Models

# In[17]:


# Load ground truth
with open("../data/eval/ground_truth_gpt5nano.json", "r") as f:
    ground_truth = json.load(f)

print(f"Loaded {len(ground_truth)} evaluation queries")


# In[19]:


# Qdrant
q_client = QdrantClient(url="http://localhost:6333")
q_client = QdrantClient(url="http://localhost:6333")
print(q_client.get_collections()) #check
COLLECTION = "hybrid_collection"

# Embedding model (fixed for HPO)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(text: str):
    return embedding_model.encode(text).tolist()


# # Evaluation Wrapper (Key for HPO)

# In[20]:


def evaluate_config(
    retrieve_k: int,
    rerank_k: int,
    rerank_model: str,
):
    reranker = CrossEncoderReranker(rerank_model)

    recalls = []
    mrrs = []
    latencies = []

    for item in ground_truth:
        start = time.time()

        results = retrieve_and_rerank(
            query=item["query"],
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
            reranker=reranker,
        )

        latencies.append(time.time() - start)

        relevant_ids = item["relevant_doc_ids"]
        recalls.append(recall_at_k(results, relevant_ids, 5))
        mrrs.append(mrr(results, relevant_ids))

    return {
        "recall@5": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "latency_avg": float(np.mean(latencies)),
    }


# # Optuna Objective (with MLflow logging)

# In[21]:


def objective(trial):

    retrieve_k = trial.suggest_categorical("retrieve_k", [20, 30, 50, 75, 100])
    rerank_k = trial.suggest_categorical("rerank_k", [5, 10])
    rerank_model = trial.suggest_categorical(
        "rerank_model",
        [
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "BAAI/bge-reranker-base",
        ],
    )

    reranker = CrossEncoderReranker(rerank_model)

    metrics = evaluate_reranking(
        ground_truth=ground_truth,
        retrieve_fn=lambda q, k: dense_retrieve(
            q_client, embed_query, COLLECTION, q, k
        ),
        reranker=reranker,
        retrieve_k=retrieve_k,
        rerank_k=rerank_k,
    )

    cost_penalty = (retrieve_k / 100) * 0.1
    return metrics["recall@5"] - cost_penalty


# # Run the Study

# In[22]:


study = optuna.create_study(
    direction="maximize",
    study_name="dense_rerank_hpo",
)

study.optimize(objective, n_trials=20)


# # Results Summary

# In[23]:


print("Best trial:")
trial = study.best_trial

print("  Value:", trial.value)
print("  Params:")
for k, v in trial.params.items():
    print(f"    {k}: {v}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




