#!/usr/bin/env python
# coding: utf-8
"""
retrieval/agent/query_rewriter.py

Phase 1 — Query rewriter.
Calls the OpenAI API to generate retrieval-optimised variants of a user query.
Slots in before the retrieve() call; callers pass all variants to the dispatcher.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from openai import OpenAI, APIError

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Config
# -------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Agent config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

_rewriter_cfg = _config["query_rewriter"]
_MODEL = _rewriter_cfg["model"]
_NUM_VARIANTS = int(_rewriter_cfg.get("num_variants", 3))
_MAX_TOKENS = int(_rewriter_cfg.get("max_tokens", 512))

# -------------------------------------------------
# OpenAI client (API key from OPENAI_API_KEY env var)
# -------------------------------------------------
_client = OpenAI()

# -------------------------------------------------
# Prompts
# -------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a search query optimizer for a retrieval system covering "
    "Google I/O 2025 conference talks. The system uses both keyword (BM25) "
    "and semantic (dense vector) retrieval.\n\n"
    "When given a user query, generate {n} retrieval-optimized variants that:\n"
    "- Use different but semantically related vocabulary to improve keyword retrieval\n"
    "- Are more concrete or specific to improve semantic retrieval\n"
    "- Approach the same information need from distinct angles\n\n"
    "Return ONLY {n} variants, one per line, with no numbering, bullets, or extra text."
)

_USER_TEMPLATE = "User query: {query}"


# -------------------------------------------------
# Public API
# -------------------------------------------------
def rewrite_query(query: str) -> list[str]:
    """
    Generate retrieval-optimised variants of a user query via the OpenAI API.

    Args:
        query: The original user query.

    Returns:
        List of variant strings (length == config.num_variants).
        Falls back to [query] on any error so the caller can always proceed.
    """
    system = _SYSTEM_PROMPT.format(n=_NUM_VARIANTS)

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            max_completion_tokens=_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _USER_TEMPLATE.format(query=query)},
            ],
        )

        text = response.choices[0].message.content.strip()
        logger.debug("[DEBUG] rewrite_query raw response: %r", text)

        variants = [line.strip() for line in text.splitlines() if line.strip()]

        if not variants:
            raise ValueError(f"No variants parsed from response: {text!r}")

        return variants[:_NUM_VARIANTS]

    except APIError as exc:
        logger.error("[ERROR] OpenAI API call failed in rewrite_query: %s", exc)
        return [query]
    except ValueError as exc:
        logger.error("[ERROR] Failed to parse rewrite_query response: %s", exc)
        return [query]


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    test_queries = [
        "What is Gemma?",
        "How does Google handle on-device AI?",
        "What new features were announced for Android?",
    ]

    for q in test_queries:
        print(f"\n=== Original: {q} ===")
        variants = rewrite_query(q)
        for i, v in enumerate(variants, 1):
            print(f"  Variant {i}: {v}")
