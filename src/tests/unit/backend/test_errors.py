"""First unit tests for app/errors.py's error_body() / STATUS_CODES.

Pure function, no Mongo/Redis/network needed - see docs/team-guides/TDD_Guide.md
for the TDD (red-green) story behind test_locked_status_maps_to_locked_code.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[4] / "src" / "production" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.errors import error_body, STATUS_CODES  # noqa: E402


class TestErrorBody:
    def test_known_status_codes_map_to_documented_names(self):
        assert error_body(400, "msg")["error"]["code"] == "BAD_REQUEST"
        assert error_body(404, "msg")["error"]["code"] == "RESOURCE_NOT_FOUND"
        assert error_body(500, "msg")["error"]["code"] == "INTERNAL_ERROR"

    def test_message_and_details_pass_through(self):
        body = error_body(400, "Bad input", details={"field": "email"})
        assert body["error"]["message"] == "Bad input"
        assert body["error"]["details"] == {"field": "email"}

    def test_details_defaults_to_none_when_not_given(self):
        body = error_body(404, "Not found")
        assert body["error"]["details"] is None

    def test_unmapped_status_code_falls_back_to_generic_code(self):
        # 418 isn't (and doesn't need to be) in STATUS_CODES.
        assert error_body(418, "teapot")["error"]["code"] == "REQUEST_FAILED"

    def test_locked_status_maps_to_locked_code(self):
        """TDD example: this test was written first (and failed - the code
        fell back to "REQUEST_FAILED") before 423 was added to STATUS_CODES.
        423 isn't raised by any route yet, but the two_factor auth flow could
        plausibly need an account-lockout response in future; adding the
        mapping now costs nothing and means error_body already does the
        right thing whenever a route needs it."""
        assert error_body(423, "Account locked")["error"]["code"] == "LOCKED"

    def test_all_status_codes_map_to_non_empty_strings(self):
        for code, name in STATUS_CODES.items():
            assert isinstance(code, int)
            assert isinstance(name, str) and name
