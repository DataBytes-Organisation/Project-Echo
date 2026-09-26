import pytest

from app.utils.secret_loader import (
    SecretConfigurationError,
    load_secret,
)


def test_valid_secret_loads(monkeypatch):
    monkeypatch.setenv(
        "TEST_SECRET",
        "secure-test-value-123"
    )

    result = load_secret("TEST_SECRET")

    assert result == "secure-test-value-123"


def test_missing_secret_rejected(monkeypatch):
    monkeypatch.delenv(
        "TEST_SECRET",
        raising=False
    )

    with pytest.raises(SecretConfigurationError):
        load_secret("TEST_SECRET")


def test_empty_secret_rejected(monkeypatch):
    monkeypatch.setenv(
        "TEST_SECRET",
        ""
    )

    with pytest.raises(SecretConfigurationError):
        load_secret("TEST_SECRET")


def test_whitespace_secret_rejected(monkeypatch):
    monkeypatch.setenv(
        "TEST_SECRET",
        "   "
    )

    with pytest.raises(SecretConfigurationError):
        load_secret("TEST_SECRET")


@pytest.mark.parametrize(
    "value",
    [
        "changeme",
        "password",
        "secret",
        "default",
        "example",
    ],
)
def test_insecure_secret_rejected(
    monkeypatch,
    value
):
    monkeypatch.setenv(
        "TEST_SECRET",
        value
    )

    with pytest.raises(SecretConfigurationError):
        load_secret("TEST_SECRET")


def test_secret_not_exposed_in_error(monkeypatch):
    monkeypatch.delenv(
        "TEST_SECRET",
        raising=False
    )

    with pytest.raises(
        SecretConfigurationError
    ) as error:

        load_secret("TEST_SECRET")

    message = str(error.value)

    assert "TEST_SECRET" in message
    assert "secure-test-value-123" not in message