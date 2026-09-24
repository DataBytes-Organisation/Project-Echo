"""Cosine-similarity retrieval over detection embeddings.

Used to find past detections whose model embedding is closest to a given
detection, so a reviewer can compare a new/uncertain detection against
precedent instead of judging it from confidence alone. Pure, dependency-free
logic (no Mongo, no model) so it can be unit tested without a live database
or a loaded checkpoint - see docs/team-guides/TDD_Guide.md for the pattern
this follows.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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


def is_valid_embedding(embedding: Any) -> bool:
    """True if embedding is a non-empty list of finite numbers.

    Used to reject malformed embeddings (wrong type, NaN/Inf values from a
    bad inference run) before they ever reach cosine_similarity, since a
    single bad value stored in the database should not be able to crash a
    similarity lookup involving many other, valid candidates.
    """
    if not embedding:
        return False
    try:
        return all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in embedding
        )
    except TypeError:
        return False


def top_k_similar(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    k: int = 5,
    exclude_id: Optional[str] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    """candidates: list of dicts, each with an "embedding" key and an "id" key.

    Candidates with a missing, malformed (non-finite values), or
    dimension-mismatched embedding are skipped rather than raising, since one
    bad stored embedding (e.g. left over from a different model version)
    should not fail the whole request for every other, valid candidate. Each
    skip is logged (not just silently dropped), since a candidate that keeps
    getting skipped is itself a signal of a real data-quality problem, such
    as detections stored by an old, incompatible model version.
    """
    scored = []
    for candidate in candidates:
        if exclude_id is not None and candidate.get("id") == exclude_id:
            continue
        embedding = candidate.get("embedding")
        candidate_id = candidate.get("id")
        if not is_valid_embedding(embedding):
            logger.warning(
                "Skipping candidate %s in similarity lookup: embedding is missing or contains non-finite values",
                candidate_id,
            )
            continue
        try:
            score = cosine_similarity(query_embedding, embedding)
        except ValueError:
            # Dimension mismatch against the query embedding, most likely a
            # candidate stored by a different model version. Skip it rather
            # than fail the whole lookup for every other valid candidate.
            logger.warning(
                "Skipping candidate %s in similarity lookup: embedding length %d does not match query length %d",
                candidate_id,
                len(embedding),
                len(query_embedding),
            )
            continue
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
