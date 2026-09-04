"""Environment variable fallback tests for the boolean settings.

Extends task C1.5 to cover SECURITY_HSTS_ENABLED, added for the security
header middleware. Same three situations as the other settings: set, unset,
and set to something invalid.
"""

import pytest

from app import config


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_get_bool_accepts_true_spellings(monkeypatch, value):
    monkeypatch.setenv("ECHO_TEST_FLAG", value)
    assert config.get_bool("ECHO_TEST_FLAG", False) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_get_bool_accepts_false_spellings(monkeypatch, value):
    monkeypatch.setenv("ECHO_TEST_FLAG", value)
    assert config.get_bool("ECHO_TEST_FLAG", True) is False


def test_get_bool_returns_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("ECHO_TEST_FLAG", raising=False)
    assert config.get_bool("ECHO_TEST_FLAG", True) is True
    assert config.get_bool("ECHO_TEST_FLAG", False) is False


def test_get_bool_returns_the_default_when_blank(monkeypatch):
    monkeypatch.setenv("ECHO_TEST_FLAG", "   ")
    assert config.get_bool("ECHO_TEST_FLAG", False) is False


def test_get_bool_raises_on_an_unrecognised_value(monkeypatch):
    """Treating an unrecognised value as False is how a security setting gets
    switched off without anyone noticing."""
    monkeypatch.setenv("ECHO_TEST_FLAG", "maybe")
    with pytest.raises(config.ConfigError) as excinfo:
        config.get_bool("ECHO_TEST_FLAG", False)
    assert "ECHO_TEST_FLAG" in str(excinfo.value)


def test_hsts_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SECURITY_HSTS_ENABLED", raising=False)
    assert config.hsts_enabled() is False


def test_hsts_can_be_enabled(monkeypatch):
    monkeypatch.setenv("SECURITY_HSTS_ENABLED", "true")
    assert config.hsts_enabled() is True
