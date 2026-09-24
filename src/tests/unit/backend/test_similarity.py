"""Unit tests for app/services/similarity.py's cosine-similarity retrieval.

Pure logic, no Mongo/model needed - see docs/team-guides/TDD_Guide.md for
the pattern this follows. The embedding-collapse investigation that
motivated this feature is documented in
Sprint2_HD_Feature_Report_Nolan_Nguyen.md.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[3] / "src" / "production" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import math  # noqa: E402

from app.services.similarity import (  # noqa: E402
    cosine_similarity,
    top_k_similar,
    flag_result,
    is_valid_embedding,
)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_zero_vector_returns_zero_instead_of_dividing_by_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_length_raises(self):
        import pytest

        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestTopKSimilar:
    def _candidate(self, id_, species, embedding):
        return {"id": id_, "species": species, "embedding": embedding}

    def test_returns_closest_first(self):
        candidates = [
            self._candidate("a", "SpeciesA", [1.0, 0.0]),
            self._candidate("b", "SpeciesB", [0.0, 1.0]),
            self._candidate("c", "SpeciesA", [0.9, 0.1]),
        ]
        results = top_k_similar([1.0, 0.0], candidates, k=2)
        assert [c["id"] for _, c in results] == ["a", "c"]

    def test_excludes_the_query_detection_itself(self):
        candidates = [
            self._candidate("query", "SpeciesA", [1.0, 0.0]),
            self._candidate("b", "SpeciesA", [0.99, 0.01]),
        ]
        results = top_k_similar([1.0, 0.0], candidates, k=5, exclude_id="query")
        assert [c["id"] for _, c in results] == ["b"]

    def test_candidates_missing_an_embedding_are_skipped_not_crashed(self):
        candidates = [
            {"id": "no-embedding", "species": "SpeciesA"},
            self._candidate("has-embedding", "SpeciesA", [1.0, 0.0]),
        ]
        results = top_k_similar([1.0, 0.0], candidates, k=5)
        assert [c["id"] for _, c in results] == ["has-embedding"]

    def test_a_mixed_dimension_candidate_is_skipped_not_crashed(self):
        # Regression test: a candidate stored with a different embedding size
        # (e.g. left over from an earlier model version) used to raise
        # ValueError out of cosine_similarity and crash the whole lookup,
        # failing every other, valid candidate along with it.
        candidates = [
            self._candidate("wrong-dimension", "SpeciesA", [1.0, 0.0, 0.0]),
            self._candidate("has-embedding", "SpeciesA", [1.0, 0.0]),
        ]
        results = top_k_similar([1.0, 0.0], candidates, k=5)
        assert [c["id"] for _, c in results] == ["has-embedding"]

    def test_a_malformed_candidate_embedding_is_skipped_not_crashed(self):
        # Regression test: NaN/Inf values (e.g. from a bad inference run)
        # would otherwise propagate into the cosine similarity computation
        # and produce a NaN score, corrupting the ranking silently instead
        # of just being excluded.
        candidates = [
            self._candidate("has-nan", "SpeciesA", [float("nan"), 0.0]),
            self._candidate("has-inf", "SpeciesA", [float("inf"), 0.0]),
            self._candidate("has-embedding", "SpeciesA", [1.0, 0.0]),
        ]
        results = top_k_similar([1.0, 0.0], candidates, k=5)
        assert [c["id"] for _, c in results] == ["has-embedding"]


class TestIsValidEmbedding:
    def test_normal_embedding_is_valid(self):
        assert is_valid_embedding([1.0, 0.0, -0.5]) is True

    def test_empty_embedding_is_invalid(self):
        assert is_valid_embedding([]) is False
        assert is_valid_embedding(None) is False

    def test_nan_or_inf_values_are_invalid(self):
        assert is_valid_embedding([1.0, float("nan")]) is False
        assert is_valid_embedding([1.0, float("inf")]) is False
        assert is_valid_embedding([1.0, float("-inf")]) is False

    def test_non_numeric_values_are_invalid(self):
        assert is_valid_embedding([1.0, "not-a-number"]) is False
        assert is_valid_embedding("not-a-list") is False

    def test_boolean_values_are_invalid(self):
        # bool is a subclass of int in Python, so True/False would otherwise
        # silently pass an isinstance(x, (int, float)) check.
        assert is_valid_embedding([1.0, True]) is False


class TestFlagResult:
    def test_single_species_top_k_is_not_ambiguous(self):
        results = [(0.9, {"species": "SpeciesA"}), (0.8, {"species": "SpeciesA"})]
        flags = flag_result(results)
        assert flags["ambiguous"] is False

    def test_mixed_species_top_k_is_ambiguous(self):
        results = [(0.9, {"species": "SpeciesA"}), (0.85, {"species": "SpeciesB"})]
        flags = flag_result(results)
        assert flags["ambiguous"] is True
        assert flags["species_in_top_k"] == ["SpeciesA", "SpeciesB"]

    def test_low_max_similarity_is_flagged_novel(self):
        results = [(0.2, {"species": "SpeciesA"})]
        flags = flag_result(results, novel_threshold=0.5)
        assert flags["novel"] is True

    def test_high_max_similarity_is_not_novel(self):
        results = [(0.9, {"species": "SpeciesA"})]
        flags = flag_result(results, novel_threshold=0.5)
        assert flags["novel"] is False

    def test_empty_results_is_novel_and_not_ambiguous(self):
        flags = flag_result([])
        assert flags == {
            "ambiguous": False,
            "novel": True,
            "max_similarity": 0.0,
            "species_in_top_k": [],
        }
