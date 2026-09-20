import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import signJWT

NODE_ROUTES=[
    ("GET", "/iot/nodes/node-1"),
    ("PUT", "/iot/nodes/node-1/register"),
]

client=TestClient(app, raise_server_exceptions=False)


def auth_headers(roles):
    token=signJWT({"_id": "000000000000000000000001"}, roles)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("method,path", NODE_ROUTES)
def test_missing_credentials_rejected(method, path):
    assert client.request(method, path).status_code in (401, 403)


@pytest.mark.parametrize("method,path", NODE_ROUTES)
def test_invalid_token_rejected(method, path):
    response=client.request(method, path, headers={"Authorization": "Bearer garbage"})
    assert response.status_code in (401, 403)


@pytest.mark.parametrize("method,path", NODE_ROUTES)
def test_valid_token_passes_auth(method, path):
    response=client.request(method, path, headers=auth_headers(["ROLE_USER"]))
    assert response.status_code not in (401, 403)
