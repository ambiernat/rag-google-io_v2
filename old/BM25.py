from rank_bm25 import BM25Okapi
from qdrant_client.models import SparseVector

class BM25Encoder:
    def __init__(self, tokenized_corpus):
        """
        tokenized_corpus: List[List[str]] - each doc is a list of tokens
        """
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.vocab = self._build_vocab(tokenized_corpus)
    
    def _build_vocab(self, tokenized_corpus):
        """Build vocabulary mapping: token -> index"""
        vocab = {}
        idx = 0
        for doc in tokenized_corpus:
            for token in doc:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab
    
    def encode_documents(self, tokenized_corpus):
        """Encode all documents as sparse vectors"""
        sparse_vectors = []
        for doc_tokens in tokenized_corpus:
            indices = []
            values = []
            for token in set(doc_tokens):  # unique tokens in doc
                if token in self.vocab:
                    idx = self.vocab[token]
                    # Use term frequency or BM25 score for this term
                    score = doc_tokens.count(token)  # simple TF
                    if score > 0:
                        indices.append(idx)
                        values.append(float(score))
            
            sparse_vectors.append(
                SparseVector(indices=indices, values=values)
            )
        return sparse_vectors
    
    def encode_query(self, query_tokens):
        """Encode query as sparse vector"""
        indices = []
        values = []
        for token in set(query_tokens):
            if token in self.vocab:
                idx = self.vocab[token]
                # Get BM25 score for this query term
                score = self.bm25.get_scores([token])
                avg_score = float(score.mean()) if len(score) > 0 else 0.0
                if avg_score > 0:
                    indices.append(idx)
                    values.append(avg_score)
        
        return SparseVector(indices=indices, values=values)