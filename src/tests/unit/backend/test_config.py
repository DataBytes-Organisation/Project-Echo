"""Environment variable fallback tests for app.config.

Task C1.5. Each setting is checked in three situations:

* the variable is set to a valid value  -> the value is used
* the variable is not set               -> a required setting raises
                                           ConfigError, an optional setting
                                           returns its documented default
* the variable is set to something bad  -> the value is rejected rather than
                                           partially accepted

``monkeypatch`` is a pytest fixture that changes the environment for the
duration of one test and puts it back afterwards, so the tests do not leak
into each other and do not depend on what happens to be set on the machine
running them.
"""

import pytest

from app import config


# ---------------------------------------------------------------------------
# MONGODB_URI - required secret, no fallback
# ---------------------------------------------------------------------------


def test_mongodb_uri_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://user:pass@db:27017/EchoNet")
    assert config.mongodb_uri() == "mongodb://user:pass@db:27017/EchoNet"


def test_mongodb_uri_accepts_srv_scheme(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://user:pass@cluster.mongodb.net/EchoNet")
    assert config.mongodb_uri().startswith("mongodb+srv://")


def test_mongodb_uri_raises_when_unset(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(config.ConfigError) as excinfo:
        config.mongodb_uri()
    # The message must name the variable, otherwise a missing setting in a
    # deployed container is very hard to diagnose from the logs.
    assert "MONGODB_URI" in str(excinfo.value)


def test_mongodb_uri_raises_when_blank(monkeypatch):
    # `docker run -e MONGODB_URI=` sets the variable to an empty string rather
    # than leaving it out. That must be treated the same as unset.
    monkeypatch.setenv("MONGODB_URI", "   ")
    with pytest.raises(config.ConfigError):
        config.mongodb_uri()


def test_mongodb_uri_raises_when_malformed(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "not-a-connection-string")
    with pytest.raises(config.ConfigError) as excinfo:
        config.mongodb_uri()
    assert "MONGODB_URI" in str(excinfo.value)


def test_mongodb_uri_does_not_fall_back_to_a_hardcoded_credential(monkeypatch):
    """Regression test for audit finding F-01.

    The previous implementation passed a working connection string as the
    default argument of os.getenv, so an unset variable silently produced a
    real credential. This asserts that never happens again.
    """
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(config.ConfigError):
        config.mongodb_uri()


# ---------------------------------------------------------------------------
# USER_MONGODB_URI - required secret, no fallback
# ---------------------------------------------------------------------------


def test_user_mongodb_uri_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("USER_MONGODB_URI", "mongodb://u:p@db/UserSample?authSource=admin")
    assert config.user_mongodb_uri().endswith("authSource=admin")


def test_user_mongodb_uri_raises_when_unset(monkeypatch):
    monkeypatch.delenv("USER_MONGODB_URI", raising=False)
    with pytest.raises(config.ConfigError) as excinfo:
        config.user_mongodb_uri()
    assert "USER_MONGODB_URI" in str(excinfo.value)


def test_user_mongodb_uri_raises_when_malformed(monkeypatch):
    monkeypatch.setenv("USER_MONGODB_URI", "postgres://u:p@db/UserSample")
    with pytest.raises(config.ConfigError):
        config.user_mongodb_uri()


def test_user_mongodb_uri_does_not_fall_back_to_a_hardcoded_credential(monkeypatch):
    """Regression test for audit finding F-02 (the root:root_password default)."""
    monkeypatch.delenv("USER_MONGODB_URI", raising=False)
    with pytest.raises(config.ConfigError):
        config.user_mongodb_uri()


# ---------------------------------------------------------------------------
# API_BASE_URL - optional, safe default
# ---------------------------------------------------------------------------


def test_api_base_url_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://ts-api-cont:9000")
    assert config.api_base_url() == "http://ts-api-cont:9000"


def test_api_base_url_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert config.api_base_url() == "http://localhost:9000"


def test_api_base_url_falls_back_when_blank(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "")
    assert config.api_base_url() == "http://localhost:9000"


def test_api_base_url_strips_trailing_slash(monkeypatch):
    # Callers append paths like "/engine/algorithms_data". Without stripping,
    # a trailing slash produces a double slash in the request URL.
    monkeypatch.setenv("API_BASE_URL", "http://ts-api-cont:9000/")
    assert config.api_base_url() == "http://ts-api-cont:9000"


# ---------------------------------------------------------------------------
# CORS_ALLOWED_ORIGINS - optional list, safe default
# ---------------------------------------------------------------------------


def test_cors_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert config.cors_allowed_origins() == ["http://localhost:8080"]


def test_cors_parses_a_single_origin(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://echo.example.org")
    assert config.cors_allowed_origins() == ["https://echo.example.org"]


def test_cors_parses_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://echo.example.org,http://localhost:8080",
    )
    assert config.cors_allowed_origins() == [
        "https://echo.example.org",
        "http://localhost:8080",
    ]


def test_cors_ignores_surrounding_whitespace_and_blank_entries(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", " https://a.example , , http://b.example ")
    assert config.cors_allowed_origins() == ["https://a.example", "http://b.example"]


def test_cors_falls_back_when_every_entry_is_blank(monkeypatch):
    # An empty allow-list would block all browser traffic while looking as
    # though the setting had been configured. Falling back is safer.
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", " , , ")
    assert config.cors_allowed_origins() == ["http://localhost:8080"]


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------


def test_get_required_strips_whitespace(monkeypatch):
    monkeypatch.setenv("ECHO_TEST_VALUE", "  spaced  ")
    assert config.get_required("ECHO_TEST_VALUE") == "spaced"


def test_get_optional_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("ECHO_TEST_VALUE", raising=False)
    assert config.get_optional("ECHO_TEST_VALUE", "fallback") == "fallback"


def test_get_optional_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("ECHO_TEST_VALUE", "provided")
    assert config.get_optional("ECHO_TEST_VALUE", "fallback") == "provided"


def test_get_list_does_not_share_the_default_object(monkeypatch):
    """Mutating one caller's result must not corrupt the default for the next."""
    monkeypatch.delenv("ECHO_TEST_LIST", raising=False)
    default = ["http://localhost:8080"]
    first = config.get_list("ECHO_TEST_LIST", default)
    first.append("http://injected.example")
    second = config.get_list("ECHO_TEST_LIST", default)
    assert second == ["http://localhost:8080"]
