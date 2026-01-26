from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval results.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace model name
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (very fast)
                - BAAI/bge-reranker-base (high quality)
                - BAAI/bge-reranker-v2-m3 (state-of-the-art)
        """
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document dicts with 'text' or 'payload' field
            top_k: Number of top documents to return
            return_scores: Whether to add rerank scores to results
            
        Returns:
            Reranked list of documents (top_k)
        """
        if not documents:
            return []
        
        # Extract text from documents
        # Handle both formats: documents with 'payload' or direct 'text'
        texts = []
        for doc in documents:
            if 'payload' in doc and 'text' in doc['payload']:
                texts.append(doc['payload']['text'])
            elif 'text' in doc:
                texts.append(doc['text'])
            else:
                # Fallback: try to get any text field
                texts.append(str(doc))
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Rerank documents
        reranked_docs = []
        for idx in sorted_indices[:top_k]:
            doc = documents[idx].copy()
            if return_scores:
                doc['rerank_score'] = float(scores[idx])
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def score_pairs(self, query_doc_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Score query-document pairs directly.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            
        Returns:
            Array of relevance scores
        """
        return self.model.predict(query_doc_pairs)


class CohereReranker:
    """
    Cohere Rerank API wrapper (requires API key).
    """
    
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Cohere rerank model
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")
        
        self.client = cohere.Client(api_key)
        self.model = model
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """Rerank using Cohere API"""
        if not documents:
            return []
        
        # Extract texts
        texts = []
        for doc in documents:
            if 'payload' in doc and 'text' in doc['payload']:
                texts.append(doc['payload']['text'])
            elif 'text' in doc:
                texts.append(doc['text'])
            else:
                texts.append(str(doc))
        
        # Call Cohere API
        results = self.client.rerank(
            query=query,
            documents=texts,
            top_n=top_k,
            model=self.model
        )
        
        # Reorder documents
        reranked_docs = []
        for result in results.results:
            doc = documents[result.index].copy()
            if return_scores:
                doc['rerank_score'] = result.relevance_score
            reranked_docs.append(doc)