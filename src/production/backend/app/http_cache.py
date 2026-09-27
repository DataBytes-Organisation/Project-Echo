"""Helpers for HTTP entity tags and conditional GET requests."""

import hashlib
import json

from fastapi.encoders import jsonable_encoder


def make_etag(value):
    """Return a deterministic strong ETag for a JSON-compatible value."""
    canonical = json.dumps(
        jsonable_encoder(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return '"{}"'.format(hashlib.sha256(canonical).hexdigest())


def if_none_match_matches(header_value, current_etag):
    """Return whether ``If-None-Match`` matches the current representation.

    GET cache validation uses the weak comparison rule, so ``W/\"tag\"`` and
    ``\"tag\"`` represent the same cached version.  A client may also send a
    comma-separated list or the wildcard value.
    """
    if not header_value:
        return False

    def normalise(tag):
        tag = tag.strip()
        if tag[:2].lower() == "w/":
            tag = tag[2:].strip()
        return tag

    expected = normalise(current_etag)
    for candidate in header_value.split(","):
        candidate = candidate.strip()
        if candidate == "*" or normalise(candidate) == expected:
            return True
    return False
