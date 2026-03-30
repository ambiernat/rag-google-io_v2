#!/usr/bin/env python
# coding: utf-8
"""
retrieval/agent/retry_loop.py

Phase 3 — Retry loop.
Wires query_rewriter and self_evaluator together with strategy switching.

The self-evaluator acts as a circuit breaker only — not a re-ranker.
Results are always returned in original retrieval score order.
A strategy switch is triggered only on genuine failure (best LLM score ≤ 2).
Scores of 3–5 are treated as acceptable; the first non-failing strategy wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from retrieval.result import RetrievalResult
from retrieval.retrievers.dispatcher import retrieve
from retrieval.agent.query_rewriter import rewrite_query
from retrieval.agent.self_evaluator import ScoredResult, evaluate_results

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Config
# -------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Agent config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

_loop_cfg = _config["retry_loop"]
_STRATEGIES: list[str] = _loop_cfg["strategies"]   # ["hybrid", "dense", "sparse"]
_TOP_K: int = int(_loop_cfg.get("top_k", 10))
_USE_SELF_EVAL: bool = bool(_loop_cfg.get("use_self_eval", True))

# Circuit-breaker threshold: switch strategy only when best score ≤ this value.
# Scores 1–2 = genuine retrieval failure; 3–5 = acceptable, keep results as-is.
_FAILURE_THRESHOLD: int = 2


# -------------------------------------------------
# Domain type
# -------------------------------------------------
@dataclass
class AgentResult:
    query: str
    variants: list[str]                          # rewritten variants used
    strategy_used: str                           # strategy that produced the final results
    attempts: int                                # number of strategies tried
    scored_results: list[ScoredResult]           # evaluated results from winning strategy
    good_enough: bool                            # whether is_good_enough() returned True

    @property
    def top_results(self) -> list[RetrievalResult]:
        """Results in original retrieval score order (self-evaluator is circuit breaker only)."""
        return [s.result for s in self.scored_results]


# -------------------------------------------------
# Internal helpers
# -------------------------------------------------
def _merge_results(result_lists: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """
    Merge multiple result lists, deduplicating by doc_id.
    When the same doc appears in multiple lists, keep the highest retrieval score.
    Preserves relative order of first occurrence.
    """
    seen: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for r in results:
            if r.doc_id not in seen or r.score > seen[r.doc_id].score:
                seen[r.doc_id] = r
    return list(seen.values())


# -------------------------------------------------
# Public API
# -------------------------------------------------
def run(query: str) -> AgentResult:
    """
    Run the full agentic retrieval pipeline for a query.

    Steps:
      1. Rewrite query into variants (Phase 1).
      2. For each strategy in config order:
         a. Retrieve using the original query and all variants.
         b. Merge and deduplicate results.
         c. Evaluate relevance with the self-evaluator (Phase 2).
         d. If best score > _FAILURE_THRESHOLD → accept and return results
            in original retrieval order (circuit breaker passed).
         e. Else → genuine failure, log and try next strategy.
      3. If all strategies exhausted, return the last attempt's results.

    Args:
        query: The original user query.

    Returns:
        AgentResult with the strategy used, scored results, and outcome flag.
        Results are always in original retrieval score order.
    """
    # Phase 1: rewrite
    variants = rewrite_query(query)
    all_queries = [query] + variants
    logger.info("[INFO] Variants generated: %s", variants)

    # Fast path: no self-evaluation, single hybrid pass
    if not _USE_SELF_EVAL:
        strategy = _STRATEGIES[0]
        logger.info("[INFO] use_self_eval=False — single pass, strategy: %s", strategy)
        result_lists = [retrieve(q, strategy=strategy, top_k=_TOP_K) for q in all_queries]
        merged = _merge_results(result_lists)
        logger.info("[INFO] Retrieved %d unique chunks across %d queries", len(merged), len(all_queries))
        # Wrap in ScoredResult with score=0 (no LLM scoring); top_results preserves merge order
        scored = [ScoredResult(result=r, score=0, reasoning="skipped") for r in merged]
        return AgentResult(
            query=query,
            variants=variants,
            strategy_used=strategy,
            attempts=1,
            scored_results=scored,
            good_enough=True,
        )

    scored: list[ScoredResult] = []
    strategy_used = _STRATEGIES[0]

    for attempt, strategy in enumerate(_STRATEGIES, start=1):
        strategy_used = strategy
        logger.info("[INFO] Attempt %d/%d — strategy: %s", attempt, len(_STRATEGIES), strategy)

        # Retrieve on every query variant and merge
        result_lists = [retrieve(q, strategy=strategy, top_k=_TOP_K) for q in all_queries]
        merged = _merge_results(result_lists)
        logger.info("[INFO] Retrieved %d unique chunks across %d queries", len(merged), len(all_queries))

        # Phase 2: circuit-breaker evaluation (does not re-rank)
        scored = evaluate_results(query, merged)
        best_score = max((s.score for s in scored), default=0)

        if best_score > _FAILURE_THRESHOLD:
            logger.info(
                "[INFO] Strategy '%s' accepted on attempt %d (best score: %d)",
                strategy,
                attempt,
                best_score,
            )
            return AgentResult(
                query=query,
                variants=variants,
                strategy_used=strategy,
                attempts=attempt,
                scored_results=scored,
                good_enough=True,
            )

        next_strategy = _STRATEGIES[attempt] if attempt < len(_STRATEGIES) else "none"
        logger.info(
            "[INFO] Strategy '%s' failed circuit breaker (best score: %d ≤ %d). "
            "Switching to '%s'.",
            strategy,
            best_score,
            _FAILURE_THRESHOLD,
            next_strategy,
        )

    # All strategies exhausted
    logger.info(
        "[INFO] All strategies exhausted after %d attempts. Returning best available results.",
        len(_STRATEGIES),
    )
    return AgentResult(
        query=query,
        variants=variants,
        strategy_used=strategy_used,
        attempts=len(_STRATEGIES),
        scored_results=scored,
        good_enough=False,
    )


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    test_query = "What is Gemma?"
    result = run(test_query)

    print(f"\nQuery:          {result.query}")
    print(f"Variants:       {result.variants}")
    print(f"Strategy used:  {result.strategy_used}")
    print(f"Attempts:       {result.attempts}/{len(_STRATEGIES)}")
    print(f"Good enough:    {result.good_enough}")
    print(f"\nTop results (retrieval order):")
    for s in result.scored_results[:3]:
        print(f"  [{s.score}/5] {s.result.doc_id} — {s.reasoning}")
