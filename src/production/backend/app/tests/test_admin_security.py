import importlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.middleware.auth import signJWT

ADMIN_ROUTES=[
    ("GET", "/hmi/users"),
    ("POST", "/hmi/users/testuser/visit"),
    ("POST", "/api/admin/services/pause"),
]
SCHEMA_ROUTES=["/docs", "/redoc", "/openapi.json", "/openapi-export", "/spec/summary"]


def auth_headers(roles):
    token=signJWT({"_id": "000000000000000000000001"}, roles)
    return {"Authorization": f"Bearer {token}"}


def build_client(monkeypatch, environment):
    monkeypatch.setattr(settings, "environment", environment)
    import app.main as main_module
    importlib.reload(main_module)
    yield TestClient(main_module.app, raise_server_exceptions=False)
    monkeypatch.undo()
    importlib.reload(main_module)


@pytest.fixture
def dev_client(monkeypatch):
    yield from build_client(monkeypatch, "development")


@pytest.fixture
def prod_client(monkeypatch):
    yield from build_client(monkeypatch, "production")


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_anonymous_is_rejected(dev_client, method, path):
    assert dev_client.request(method, path).status_code in (401, 403)


@pytest.mark.parametrize("roles", [["ROLE_USER"], []])
@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_ordinary_user_is_forbidden(dev_client, method, path, roles):
    response=dev_client.request(method, path, headers=auth_headers(roles))
    assert response.status_code==403


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
def test_admin_is_allowed(dev_client, method, path):
    # the pause route gets no body, so it returns 422 and never changes any service state
    response=dev_client.request(method, path, headers=auth_headers(["ROLE_ADMIN"]))
    assert response.status_code not in (401, 403)


@pytest.mark.parametrize("path", SCHEMA_ROUTES)
def test_schema_endpoints_hidden_in_production(prod_client, path):
    assert prod_client.get(path).status_code==404


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/openapi-export", "/spec/summary"])
def test_schema_endpoints_available_in_development(dev_client, path):
    assert dev_client.get(path).status_code==200
