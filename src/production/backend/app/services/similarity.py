"""Cosine-similarity retrieval over detection embeddings.

Used to find past detections whose model embedding is closest to a given
detection, so a reviewer can compare a new/uncertain detection against
precedent instead of judging it from confidence alone. Pure, dependency-free
logic (no Mongo, no model) so it can be unit tested without a live database
or a loaded checkpoint - see docs/team-guides/TDD_Guide.md for the pattern
this follows.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

NOVEL_SIMILARITY_THRESHOLD = 0.5


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding length mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def top_k_similar(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    k: int = 5,
    exclude_id: Optional[str] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    """candidates: list of dicts, each with an "embedding" key and an "id" key."""
    scored = []
    for candidate in candidates:
        if exclude_id is not None and candidate.get("id") == exclude_id:
            continue
        embedding = candidate.get("embedding")
        if not embedding:
            continue
        score = cosine_similarity(query_embedding, embedding)
        scored.append((score, candidate))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


def flag_result(
    top_k_results: List[Tuple[float, Dict[str, Any]]],
    novel_threshold: float = NOVEL_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    if not top_k_results:
        return {
            "ambiguous": False,
            "novel": True,
            "max_similarity": 0.0,
            "species_in_top_k": [],
        }

    species_in_top_k = sorted({candidate.get("species") for _, candidate in top_k_results})
    max_similarity = top_k_results[0][0]

    return {
        "ambiguous": len(species_in_top_k) > 1,
        "novel": max_similarity < novel_threshold,
        "max_similarity": max_similarity,
        "species_in_top_k": species_in_top_k,
    }
