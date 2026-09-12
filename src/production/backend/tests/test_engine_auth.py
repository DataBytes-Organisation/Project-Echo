import pytest
from fastapi import HTTPException

from app.middleware.engine_auth import ENGINE_API_KEY_ENV, verify_engine_api_key


def test_no_key_configured_leaves_endpoint_open(monkeypatch):
    monkeypatch.delenv(ENGINE_API_KEY_ENV, raising=False)

    verify_engine_api_key(x_engine_api_key=None)
    verify_engine_api_key(x_engine_api_key="anything")


def test_matching_key_is_accepted(monkeypatch):
    monkeypatch.setenv(ENGINE_API_KEY_ENV, "secret-value")

    verify_engine_api_key(x_engine_api_key="secret-value")


def test_missing_key_is_rejected_when_configured(monkeypatch):
    monkeypatch.setenv(ENGINE_API_KEY_ENV, "secret-value")

    with pytest.raises(HTTPException) as exc_info:
        verify_engine_api_key(x_engine_api_key=None)

    assert exc_info.value.status_code == 401


def test_wrong_key_is_rejected_when_configured(monkeypatch):
    monkeypatch.setenv(ENGINE_API_KEY_ENV, "secret-value")

    with pytest.raises(HTTPException) as exc_info:
        verify_engine_api_key(x_engine_api_key="wrong-value")

    assert exc_info.value.status_code == 401
