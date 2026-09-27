"""Tests for conditional GET and ETag behaviour (B4.1)."""

from app.http_cache import if_none_match_matches, make_etag


def test_etag_is_stable_across_dictionary_order():
    assert make_etag({"species": "Koala", "confidence": 91}) == make_etag(
        {"confidence": 91, "species": "Koala"}
    )


def test_etag_changes_when_the_representation_changes():
    assert make_etag({"confidence": 91}) != make_etag({"confidence": 92})


def test_exact_if_none_match_value_matches():
    etag = make_etag({"id": "one"})
    assert if_none_match_matches(etag, etag)


def test_weak_validator_matches_for_get_revalidation():
    etag = make_etag({"id": "one"})
    assert if_none_match_matches("W/" + etag, etag)


def test_validator_list_and_wildcard_are_supported():
    etag = make_etag({"id": "one"})
    assert if_none_match_matches('"other", ' + etag, etag)
    assert if_none_match_matches("*", etag)


def test_missing_or_different_validator_does_not_match():
    etag = make_etag({"id": "one"})
    assert not if_none_match_matches(None, etag)
    assert not if_none_match_matches('"different"', etag)
