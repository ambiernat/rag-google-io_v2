#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports & Setup
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


# In[2]:


# import pandas as pd
# # Load documents:
# documents = pd.read_json("../data/canonical/all_documents.json")


# Read as string (keeps JSON format)
with open('../data/canonical/all_documents.json', 'r') as f:
    documents = f.read()

#parse JSON into Python objects / deserialize
import json

documents = json.loads(documents)


# ## Initialize Embedding Model (Baseline, i.e. no tuning yet)

# In[4]:


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()


# ## Connecting to Qdrant:

# In[5]:


q_client = QdrantClient(
    url="http://localhost:6333"
)


# In[6]:


COLLECTION_NAME = "google-io-transcripts-dense"


# In[7]:


q_client.delete_collection(collection_name=COLLECTION_NAME)


# ## Create Collection (This Defines the Index)
# This is the index creation step in vector-DB terms.

# In[8]:


# if not q_client.collection_exists(COLLECTION_NAME):
q_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIM,
        distance=models.Distance.COSINE
    )
)
    #raise ValueError("Collection was not created successfully.")
#     print(f"Collection '{COLLECTION_NAME}' created successfully.")
# else:
#     print(f"Collection '{COLLECTION_NAME}' already exists.")


# ## prepare points for insertion

# In[9]:


from qdrant_client.http import models

def build_points(documents):

    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True #Most modern retrieval pipelines assume normalized vectors when using cosine distance.
    )

    points = []
    for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
        points.append(
            models.PointStruct(
                id=idx,  # 👈 Qdrant technical ID
                vector=vector.tolist(),
                payload={
                    "doc_id": doc["id"],  # 👈 canonical ID
                    "video_id": doc["video_id"],
                    "title": doc["title"],
                    "timestamp_start": doc["timestamp_start"],
                    "timestamp_end": doc["timestamp_end"],
                    "text": doc["text"],
                    "source": doc["source"],
                    "speaker": doc["speaker"]
                }
            )
        )
    return points


# ## Upsert Documents (Batching is Important)

# In[10]:


from math import ceil

BATCH_SIZE = 64

points = build_points(documents)

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )


# In[11]:


documents


# At this point:
# 
# Your **vector index exists**
# 
# Your **data is searchable**

# In[28]:


documents


# In[ ]:




