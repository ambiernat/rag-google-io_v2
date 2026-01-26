# Data Schemas & Contracts

This document defines the data contracts used across the ingestion, retrieval, and evaluation pipeline.  
Schemas are **versioned** to support re-ingestion and experimentation without breaking downstream components.

The pipeline flow is:

**raw → chunked → canonical → indexed → retrieval/evaluation**

---

## 1. Raw Transcript Schema (`raw_v1`)

**Purpose**  
Represents the original YouTube transcript as fetched from the source, with minimal normalization.

**Produced by**  
`ingestion/fetch.py`

**Used by**  
`ingestion/chunk.py`

### Schema
```json
{
  "schema_version": "raw_v1",
  "video_id": "string",
  "language": "string",
  "segments": [
    {
      "text": "string",
      "start": "float",
      "duration": "float"
    }
  ]
}
