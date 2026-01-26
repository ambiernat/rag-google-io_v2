#!/usr/bin/env python
# coding: utf-8
"""
dense_rerank_hpo.py

Hyperparameter Optimization (HPO) for Dense Retrieval + CrossEncoder Rerank
using Optuna and MLflow (manual logging, MLflow 2.x compatible).

Author: Your Name
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

import optuna
import mlflow

# -----------------------------
# Project Root
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-google-io/
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# HPO Config
# -----------------------------
import yaml

CONFIG_PATH = PROJECT_ROOT / "retrieval" / "hpo" / "config_dense_rerank.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=cfg.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger(__name__)

# -----------------------------
# Imports: Qdrant & Models
# -----------------------------
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.rerankers.crossencoder_reranker import crossencoder_rerank  # function-based reranker

# -----------------------------
# Ground Truth Loading
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir} with prefix {prefix}")
    return files[0]

gt_dir = PROJECT_ROOT / cfg["data"]["ground_truth"]["dir"]
gt_prefix = cfg["data"]["ground_truth"]["prefix"]
gt_file = get_latest_ground_truth(gt_dir, gt_prefix)

with open(gt_file, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

logger.info(f"Loaded {len(ground_truth)} ground truth queries from {gt_file.name}")

# -----------------------------
# Qdrant Client
# -----------------------------
q_client = QdrantClient(url="http://localhost:6333")
collection_name = cfg["retrieval"]["collection_name"]
logger.info(f"Dense retriever initialized | Collection: {collection_name}")

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = embedding_model.get_sentence_embedding_dimension()
logger.info(f"Embedding model: {embedding_model.__class__.__name__} | Dim: {embedding_dim}")

def embed_query(text: str):
    return embedding_model.encode(text).tolist()

# -----------------------------
# Optuna Objective
# -----------------------------

def dense_wrapper(query, top_k):
    hits = retrieve_dense(query)  # whatever your current signature is
    return hits[:top_k]

def objective(trial):
    retrieve_k = trial.suggest_categorical("retrieve_k", cfg["retrieval"]["retrieve_k"]["choices"])
    rerank_k = trial.suggest_categorical("rerank_k", cfg["reranker"]["rerank_k"]["choices"])
    rerank_model = trial.suggest_categorical("rerank_model", cfg["reranker"]["models"])

    primary_metric = cfg["objective"].get("primary_metric", "recall@5")
    cost_penalty_cfg = cfg["objective"].get("cost_penalty", {})
    cost_penalty_enabled = cost_penalty_cfg.get("enabled", False)
    retrieve_k_weight = cost_penalty_cfg.get("retrieve_k_weight", 0.1)

    metrics_list = []

    # Start nested MLflow run for this trial
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "retrieve_k": retrieve_k,
            "rerank_k": rerank_k,
            "rerank_model": rerank_model
        })

        for item in tqdm(ground_truth, desc="Evaluating queries", leave=False):
            query = item["query"]
            relevant_ids = item["relevant_doc_ids"]

            # -----------------------------
            # Dense Retrieval
            # -----------------------------
            hits = dense_wrapper(query, retrieve_k)
            retrieved_docs = [{"id": h.id, **h.payload} for h in hits]

            # -----------------------------
            # CrossEncoder Rerank
            # -----------------------------
            reranked_docs = crossencoder_rerank(query, retrieved_docs, rerank_model, rerank_k)

            retrieved_ids = [doc["id"] for doc in reranked_docs]

            # Compute metrics
            recall = int(any(rid in relevant_ids for rid in retrieved_ids[:rerank_k]))
            mrr = 0.0
            for i, rid in enumerate(retrieved_ids, start=1):
                if rid in relevant_ids:
                    mrr = 1.0 / i
                    break
            precision = sum(1 for rid in retrieved_ids[:rerank_k] if rid in relevant_ids) / rerank_k

            metrics_list.append({
                "recall@k": recall,
                "mrr": mrr,
                "precision@k": precision
            })

        # Metric mapping for MLflow
        metric_map = {
            "recall@k": "recall_k",
            "mrr": "mrr",
            "precision@k": "precision_k"
        }

        # Compute average metrics
        avg_metrics = {metric_map[k]: float(np.mean([m.get(k, 0) for m in metrics_list])) for k in metric_map}

        # Apply cost penalty
        objective_value = avg_metrics.get(primary_metric, 0.0)
        if cost_penalty_enabled:
            objective_value -= (retrieve_k / 100) * retrieve_k_weight

        # Log metrics
        for metric_name, metric_value in avg_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    return objective_value

# -----------------------------
# MLflow & Optuna Setup
# -----------------------------
mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
mlflow.set_experiment(cfg["experiment"]["name"])
logger.info(f"MLflow experiment: {cfg['experiment']['name']} | Tracking URI: {cfg['experiment']['tracking_uri']}")

study = optuna.create_study(direction=cfg["experiment"].get("direction", "maximize"))

# -----------------------------
# Run HPO
# -----------------------------
n_trials = cfg["experiment"].get("n_trials", 20)
logger.info(f"Starting Optuna study for {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

# -----------------------------
# Save Best Params, Results, and Study Summary
# -----------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
output_dir = PROJECT_ROOT / cfg["data"]["ground_truth"]["results_dir"]
output_dir.mkdir(parents=True, exist_ok=True)

# Best params
if cfg["artifacts"].get("save_best_params", True):
    best_params_file = output_dir / f"best_params_{timestamp}.json"
    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best params to {best_params_file}")

# Best results
if cfg["artifacts"].get("save_best_results", True):
    best_results_file = output_dir / f"best_results_{timestamp}.json"
    best_trial = study.best_trial
    best_retrieve_k = best_trial.params["retrieve_k"]
    best_rerank_k = best_trial.params["rerank_k"]
    best_rerank_model = best_trial.params["rerank_model"]

    best_results = []
    for item in tqdm(ground_truth, desc="Computing best trial results"):
        query = item["query"]
        relevant_ids = item["relevant_doc_ids"]

        hits = dense_wrapper(query, best_retrieve_k)
        retrieved_docs = [{"id": h.id, **h.payload} for h in hits]

        reranked_docs = crossencoder_rerank(query, retrieved_docs, best_rerank_model, best_rerank_k)
        best_results.append({
            "query": query,
            "reranked_docs": reranked_docs,
            "relevant_doc_ids": relevant_ids
        })

    with open(best_results_file, "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    logger.info(f"Saved best reranked results to {best_results_file}")

# Study summary
if cfg["artifacts"].get("save_study_summary", True):
    study_summary_file = output_dir / f"study_summary_{timestamp}.json"
    df_summary = study.trials_dataframe()
    df_summary.to_json(study_summary_file, orient="records", indent=2)
    logger.info(f"Saved study summary to {study_summary_file}")

logger.info("HPO completed successfully.")
