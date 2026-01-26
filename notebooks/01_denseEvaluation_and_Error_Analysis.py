#!/usr/bin/env python
# coding: utf-8

# # "From First Principles"

# ## Settings

# In[10]:


ground_truth_path = '../data/eval/ground_truth_gpt5nano.json'
documents_path = '../data/canonical/all_documents.json'


# In[11]:


# For loading in modules pre-defined in scripts
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parents[0]  # rag-google-io
sys.path.insert(0, str(PROJECT_ROOT))

print("Added to PYTHONPATH:", PROJECT_ROOT)


# In[13]:


#check
import os

print("CWD:", os.getcwd())
print("Notebook path:", Path.cwd())


# ## Load Datasets

# In[14]:


import json

# Open the file for reading ('r')
with open(ground_truth_path, 'r') as file:
    ground_truth_dataset = json.load(file)


# In[15]:


type(ground_truth_dataset)


# In[16]:


ground_truth_dataset[0]


# In[17]:


# Open the file for reading ('r')
with open(documents_path, 'r') as file:
    documents = json.load(file)


# In[18]:


type(documents)


# In[19]:


documents[0]


# ##### check if there are documents in the ground truth that have more than 1 revelant document / chunk

# In[20]:


extract_relevant_doc_ids = []
for i in ground_truth_dataset:
    doc_id=i['relevant_doc_ids']
    print(doc_id)
    extract_relevant_doc_ids.append(doc_id)


# In[21]:


type(extract_relevant_doc_ids)


# ## Evaluation - Main Part
# using Qdrant

# #### Assumptions:
# * You already have a running Qdrant instance
# 
# * Your collection name is known (e.g. "google-io-transcripts")
# 
# * Your payload contains "doc_id" (as in your build_points function)
# 
# * You are using cosine similarity with normalized embeddings

# In[22]:


from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np


# In[23]:


# Qdrant connection
qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

COLLECTION_NAME = "google-io-transcripts"

# Embedding model (must match indexing time)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# In[24]:


from retrieval.vector_search0_2 import vector_search


# In[25]:


all_results = []  # store all queries' retrieved docs, cosine, etc.

TOP_K = 5

from tqdm import tqdm
import numpy as np

for q in tqdm(ground_truth_dataset):
    query = q["query"]
    relevant_doc_ids = set(q["relevant_doc_ids"])

    # Retrieve top-K from Qdrant
    retrieved_docs = vector_search(query, top_k=TOP_K, with_vectors=True)

    retrieved_doc_ids = []
    relevance = []
    cosine_scores = []

    # Encode query once
    query_vec = embedding_model.encode(query, normalize_embeddings=True)

    for hit in retrieved_docs:
        doc_id = hit.payload["doc_id"]
        retrieved_doc_ids.append(doc_id)
        relevance.append(doc_id in relevant_doc_ids)

        # Per-doc cosine similarity
        doc_vec = np.array(hit.vector)
        cosine = float(np.dot(query_vec, doc_vec))
        cosine_scores.append(cosine)

    all_results.append({
        "query": query,
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevance": relevance,
        "cosine_scores": cosine_scores,
        "relevant_doc_ids": list(relevant_doc_ids)
    })


# In[26]:


# Example for the first query
first_query_results = all_results[0]

print(first_query_results["query"])
print(first_query_results["retrieved_doc_ids"])
print(first_query_results["relevance"])
print(first_query_results["cosine_scores"])


# In[27]:


from tqdm import tqdm

TOP_K = 5
relevance_total = []

for q in tqdm(ground_truth_dataset):
    query = q["query"]
    relevant_doc_ids = set(q["relevant_doc_ids"])

    # Retrieve top-K from Qdrant
    results = vector_search(query, top_k=TOP_K, with_vectors=True)

    # Build relevance vector: [True/False per rank]
    relevance = [
        hit.payload.get("doc_id") in relevant_doc_ids
        for hit in results
    ]

    relevance_total.append(relevance)


# In[28]:


relevance_total


# #### **Recall@k**

# In[29]:


import numpy as np

recall_at_k = np.mean([any(row) for row in relevance_total])


# In[30]:


recall_at_k


# #### **MRR@k**

# In[31]:


def rr(row):
    for i, val in enumerate(row):
        if val:
            return 1.0 / (i + 1)
    return 0.0

mrr = np.mean([rr(row) for row in relevance_total])


# In[32]:


mrr


# #### **NDCG score**

# In[33]:


import numpy as np
from sklearn.metrics import ndcg_score

y_true = np.array(relevance_total, dtype=int)

y_scores = np.tile(
    np.linspace(1, 0.1, TOP_K),
    (y_true.shape[0], 1)
)

ndcg = ndcg_score(y_true, y_scores, k=TOP_K)
print(f"NDCG@{TOP_K}: {ndcg:.3f}")


# # Building prompt and LLM

# In[34]:


retrieved_docs


# ### Build prompt for RAG

# In[35]:


def build_prompt(query, retrieved_docs):
    """
    Create a RAG-style prompt using top-K retrieved documents.
    """
    context = "\n---\n".join([hit.payload["text"][:1000] for hit in retrieved_docs])

    prompt = f"""
You're a helpful assistant. Answer the QUESTION using ONLY the CONTEXT from the retrieved documents.

QUESTION:
{query}

CONTEXT:
{context}
""".strip()
    return prompt


# ### Generate answer using LLM

# In[36]:


import getpass

try:
    API_KEY = getpass.getpass()
except Exception as error:
    print('ERROR', error)
else:
    print('API_KEY entered')


# In[37]:


from openai import OpenAI

client = OpenAI(api_key=API_KEY)  # Make sure OPENAI_API_KEY is set

def rag_answer(query, retrieved_docs, model="gpt-5-nano"):
    prompt = build_prompt(query, retrieved_docs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0  # ❌ remove this line for gpt-5-nano
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("RAG LLM error:", e)
        return ""


# In[38]:


query = "What were the main topics and components discussed in Google's AI Stack for Developers session at Google I/O, including foundation models and the developer frameworks mentioned?"


# In[39]:


answer = rag_answer(query, retrieved_docs, model="gpt-5-nano")


# In[40]:


answer[:1000]


# In[ ]:





# ## Final Evaluations

# In[41]:


def compute_cosine(query_vec, doc_vec):
    """Compute cosine similarity between two vectors"""
    return float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)))


# In[42]:


def compute_avg_cosine(query, results):
    """Average cosine similarity of top-K retrieved docs"""
    query_vec = embedding_model.encode(query, normalize_embeddings=True)
    doc_vecs = [np.array(hit.vector) for hit in results if hasattr(hit, "vector")]
    if not doc_vecs:
        return 0.0
    return float(np.mean([compute_cosine(query_vec, v) for v in doc_vecs]))


# #### LLM judge

# In[46]:


def llm_judge(query, retrieved_docs, model="gpt-5-nano"):
    """
    Ask the LLM to rate the quality of retrieved documents for the question.
    Returns 1-5
    """
    retrieved_texts = "\n---\n".join([hit.payload["text"][:1000] for hit in retrieved_docs if "text" in hit.payload])
    prompt = f"""
    You are an expert evaluating document retrieval.
    Question: {query}

    Retrieved documents:
    {retrieved_texts}

    Rate how well these documents answer the question on a scale from 1 (very poor) to 5 (excellent).
    Only respond with a single integer.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            #temperature=0 # ❌ remove this line for gpt-5-nano
        )
        score = int(response.choices[0].message.content.strip())
        return max(1, min(score, 5))
    except Exception as e:
        print("LLM scoring error:", e)
        return None


# In[44]:


# # Loop over ground truth
# results_all = []

# for q in tqdm(ground_truth_dataset):
#     query = q["query"]
#     relevant_doc_ids = set(q["relevant_doc_ids"])

#     # Retrieve top-K docs
#     retrieved_docs = vector_search(query, top_k=TOP_K)

#     # Compute Recall@K, MRR, NDCG (you already did)

#     # Generate RAG answer
#     answer = rag_answer(query, retrieved_docs)

#     # Ask LLM to judge retrieval quality
#     llm_score = llm_judge(query, retrieved_docs)

#     results_all.append({
#         "query": query,
#         "relevant_doc_ids": list(relevant_doc_ids),
#         "retrieved_doc_ids": [hit.payload["doc_id"] for hit in retrieved_docs],
#         "rag_answer": answer,
#         "llm_score": llm_score
#     })


# In[102]:


# Evaluation loop - loop over ground truth
# -----------------------------
results_all = []

for item in tqdm(ground_truth_dataset, desc="Evaluating queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])

    # Retrieve top-K docs
    retrieved_docs = vector_search(query, top_k=TOP_K, with_vectors=True)

    # Compute per-query cosine similarity
    avg_cos = compute_avg_cosine(query, retrieved_docs)

    # Prepare text for LLM judge (first 1000 chars per doc)
    retrieved_texts = "\n---\n".join([hit.payload["text"][:1000] for hit in retrieved_docs if "text" in hit.payload])

    # Get LLM score
    llm_score = llm_judge(query, retrieved_docs)

    # Recall@K
    retrieved_ids = [hit.payload["doc_id"] for hit in retrieved_docs if "doc_id" in hit.payload]
    recall_at_k = int(any(rid in relevant_ids for rid in retrieved_ids))



    # intermediate calcs for MRR and NDCG:
    retrieved_ids = [hit.payload["doc_id"] for hit in retrieved_docs if "doc_id" in hit.payload]
    retrieved_ids = []
    relevance = []

    for hit in retrieved_docs:
        doc_id = hit.payload.get("doc_id")
        retrieved_ids.append(doc_id)
        relevance.append(int(doc_id in relevant_ids))  # True or False -> 1 or 0

    def reciprocal_rank(relevance):
        for i, rel in enumerate(relevance, start=1):
            if rel:
                return 1.0 / i
        return 0.0


    # MRR@K
    mrr_at_k = reciprocal_rank(relevance)

    # NDCG@K
    y_true = np.array([relevance])
    y_scores = np.array(
        [np.linspace(1, 0.1, TOP_K)]
    )
    ndcg_at_k = ndcg_score(y_true, y_scores, k=TOP_K)

    # Save results
    results_all.append({
        "query": query,
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids,
        "recall_at_5": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
        "avg_cosine": avg_cos,
        "llm_score": llm_score,
    })


# In[ ]:


# should save results_all and then just call it for visualisations


# In[103]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[104]:


# -----------------------------
# Analysis & Plots
# -----------------------------
df = pd.DataFrame(results_all)


# In[105]:


# Cosine similarity distribution
plt.figure(figsize=(8,5))
sns.histplot(df["avg_cosine"], kde=True, bins=20)
plt.title("Distribution of average cosine similarity per query")
plt.xlabel("Avg cosine similarity")
plt.ylabel("Count")
plt.show()


# In[106]:


sns.distplot(df['avg_cosine'])


# In[107]:


# LLM score distribution
plt.figure(figsize=(8,5))
sns.histplot(df["llm_score"], kde=True, bins=5, discrete=True)
plt.title("Distribution of LLM judge scores per query")
plt.xlabel("LLM score (1-5)")
plt.ylabel("Count")
plt.show()


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your results (if not already loaded)
df = df#pd.read_json("data/eval/retrieval_llm_eval.json")

# Optional: normalize cosine similarity to 0-5 scale for better comparison
cos_min, cos_max = df["avg_cosine"].min(), df["avg_cosine"].max()
df["cosine_scaled"] = 1 + 4 * (df["avg_cosine"] - cos_min) / (cos_max - cos_min)  # scale to [1,5]

# Sort queries by LLM score or any other metric if desired
df_sorted = df.sort_values("llm_score", ascending=False)

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="llm_score", marker="o", label="LLM score")
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="cosine_scaled", marker="x", label="Cosine similarity (scaled)")
plt.xlabel("Query index")
plt.ylabel("Score / Similarity (scaled)")
plt.title("LLM Judge Score vs Cosine Similarity per Query")
plt.legend()
plt.show()


# In[109]:


df


# In[110]:


df['cosine_scaled_llm_score__diff'] = df['cosine_scaled'] - df['llm_score']


# In[111]:


# High cosine, but low LLM => retrieval may be semantically similar, but not answering the question
plt.figure(figsize=(8,5))
sns.histplot(df["cosine_scaled_llm_score__diff"], kde=True, bins=5, discrete=True)
plt.title("Distribution of the Difference between a cosine vs llm score")
plt.xlabel("score (1-5)")
plt.ylabel("Count")
plt.show()


# In[112]:


# Sort queries by LLM score or any other metric if desired
df_sorted = df#.sort_values("cosine_scaled_llm_score__diff", ascending=False)

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="cosine_scaled_llm_score__diff", marker="o", label="Cosine-LLM score Difference")
plt.xlabel("Query index")
plt.ylabel("Score / Similarity (scaled)")
plt.title("Cosine Similarity - LLM Judge score Difference per Query")
plt.legend()
plt.show()


# In[76]:


df.iloc[4]


# In[77]:


df.iloc[4]['query']


# In[93]:


filtered_docs = [
    d for d in documents
    if "4TE-KFXvhAk__chunk_002" in d["id"]
]

relevant_doc_text = filtered_docs[0]['text']
print(relevant_doc_text.replace("\\n", "\n"))


# In[91]:





# In[92]:





# In[113]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your results (if not already loaded)
df = df#pd.read_json("data/eval/retrieval_llm_eval.json")

# Optional: normalize cosine similarity to 0-5 scale for better comparison
ndcg_min, ndcg_max = df["ndcg_at_k"].min(), df["ndcg_at_k"].max()
df["ndcg_scaled"] = 1 + 4 * (df["ndcg_at_k"] - ndcg_min) / (ndcg_max - ndcg_min)  # scale to [1,5]

# Sort queries by LLM score or any other metric if desired
df_sorted = df.sort_values("llm_score", ascending=False)

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="llm_score", marker="o", label="LLM score")
sns.lineplot(data=df_sorted, x=range(len(df_sorted)), y="ndcg_scaled", marker="x", label="NDCG (scaled)")
plt.xlabel("Query index")
plt.ylabel("Score / Similarity (scaled)")
plt.title("LLM Judge Score vs NDCG Similarity per Query")
plt.legend()
plt.show()


# In[ ]:




