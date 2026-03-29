#!/usr/bin/env python
# coding: utf-8
"""
retrieval/agent/self_evaluator.py

Phase 2 — Self-evaluator.
Scores the relevance of retrieved chunks to a query using the OpenAI API.
Returns a score (1-5) and a brief reasoning string for each result.
Used by Phase 3 to decide whether to accept results or switch strategy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI, APIError

from retrieval.result import RetrievalResult

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Config
# -------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Agent config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

_eval_cfg = _config["self_evaluator"]
_MODEL = _eval_cfg["model"]
_MAX_TOKENS = int(_eval_cfg.get("max_completion_tokens", 1024))
_TEXT_PREVIEW = int(_eval_cfg.get("text_preview_chars", 300))
_SCORE_THRESHOLD = int(_eval_cfg.get("score_threshold", 3))

# -------------------------------------------------
# OpenAI client (API key from OPENAI_API_KEY env var)
# -------------------------------------------------
_client = OpenAI()

# -------------------------------------------------
# Domain type
# -------------------------------------------------
@dataclass
class ScoredResult:
    result: RetrievalResult
    score: int        # 1–5
    reasoning: str


# -------------------------------------------------
# Prompts
# -------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a relevance evaluator for a retrieval system covering "
    "Google I/O 2025 conference talks.\n\n"
    "You will receive a user query and a list of retrieved text chunks. "
    "For each chunk, assign a relevance score from 1 to 5:\n"
    "  5 — directly and fully answers the query\n"
    "  4 — highly relevant, covers most of the query\n"
    "  3 — partially relevant, touches on the topic\n"
    "  2 — loosely related, unlikely to help\n"
    "  1 — irrelevant\n\n"
    "Return a JSON object with a single key 'evaluations', containing an array "
    "of objects in the same order as the input chunks. Each object must have:\n"
    '  "index": <int>,\n'
    '  "score": <int 1-5>,\n'
    '  "reasoning": <one sentence>\n\n'
    "No extra keys, no markdown, no other text."
)

_USER_TEMPLATE = (
    "Query: {query}\n\n"
    "Chunks:\n"
    "{chunks}"
)


def _format_chunks(results: list[RetrievalResult]) -> str:
    lines = []
    for i, r in enumerate(results):
        preview = r.text[:_TEXT_PREVIEW].replace("\n", " ")
        if len(r.text) > _TEXT_PREVIEW:
            preview += "..."
        lines.append(f"[{i}] {r.doc_id}\n{preview}")
    return "\n\n".join(lines)


# -------------------------------------------------
# Public API
# -------------------------------------------------
def evaluate_results(
    query: str,
    results: list[RetrievalResult],
) -> list[ScoredResult]:
    """
    Score each retrieved chunk for relevance to the query.

    All chunks are evaluated in a single API call.

    Args:
        query:   The user query.
        results: Retrieved chunks to evaluate.

    Returns:
        List of ScoredResult in the same order as results.
        Falls back to score=0 with an error note on any API or parse failure.
    """
    if not results:
        return []

    user_message = _USER_TEMPLATE.format(
        query=query,
        chunks=_format_chunks(results),
    )

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            max_completion_tokens=_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        text = response.choices[0].message.content.strip()
        logger.debug("[DEBUG] evaluate_results raw response: %r", text)

        data = json.loads(text)
        evaluations = data.get("evaluations", [])

        # Build a lookup by index for safe alignment
        eval_by_index = {e["index"]: e for e in evaluations if "index" in e}

        scored = []
        for i, result in enumerate(results):
            entry = eval_by_index.get(i)
            if entry:
                scored.append(ScoredResult(
                    result=result,
                    score=int(entry.get("score", 0)),
                    reasoning=str(entry.get("reasoning", "")),
                ))
            else:
                logger.error("[ERROR] Missing evaluation for chunk index %d", i)
                scored.append(ScoredResult(result=result, score=0, reasoning="missing evaluation"))

        return scored

    except APIError as exc:
        logger.error("[ERROR] OpenAI API call failed in evaluate_results: %s", exc)
        return [ScoredResult(result=r, score=0, reasoning="api error") for r in results]
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.error("[ERROR] Failed to parse evaluate_results response: %s", exc)
        return [ScoredResult(result=r, score=0, reasoning="parse error") for r in results]


def is_good_enough(scored_results: list[ScoredResult], threshold: int = _SCORE_THRESHOLD) -> bool:
    """
    Return True if at least one result meets the score threshold.
    Used by the Phase 3 retry loop to decide whether to stop or switch strategy.
    """
    return any(s.score >= threshold for s in scored_results)


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    # Minimal synthetic results for smoke testing
    dummy_results = [
        RetrievalResult(
            doc_id="abc__chunk_000",
            score=0.9,
            text="Gemma is a family of lightweight open models built by Google DeepMind.",
        ),
        RetrievalResult(
            doc_id="abc__chunk_001",
            score=0.4,
            text="The weather in San Francisco is often foggy in the summer months.",
        ),
    ]

    scored = evaluate_results("What is Gemma?", dummy_results)
    for s in scored:
        print(f"\n{s.result.doc_id}")
        print(f"  Score:     {s.score}/5")
        print(f"  Reasoning: {s.reasoning}")

    print(f"\nGood enough (threshold={_SCORE_THRESHOLD}):", is_good_enough(scored))
