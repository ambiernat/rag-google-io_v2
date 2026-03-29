"""
retrieval/result.py

Domain type for retrieval results. Provides a Qdrant-agnostic representation
and the single canonical conversion from ScoredPoint objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    text: str
    title: str | None = None
    video_id: str | None = None

    @classmethod
    def from_scored_point(cls, point) -> RetrievalResult:
        """Convert a Qdrant ScoredPoint to a RetrievalResult."""
        return cls(
            doc_id=point.payload.get("doc_id", str(point.id)),
            score=point.score,
            text=point.payload.get("text", ""),
            title=point.payload.get("title"),
            video_id=point.payload.get("video_id"),
        )

    def to_dict(self) -> dict:
        """
        Return a plain dict representation.
        Compatible with crossencoder_rerank() which expects a 'text' key.
        """
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "text": self.text,
            "title": self.title,
            "video_id": self.video_id,
        }
