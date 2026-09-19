import asyncio
import importlib
import os
import sys
import types
from unittest.mock import patch

import pytest
from fastapi import HTTPException, WebSocketDisconnect, status


# app.config creates `settings = Settings()` at import time.
# Required settings must therefore exist before importing app modules.
REQUIRED_ENV = {
    "MONGODB_URI": "mongodb://user:pass@localhost:27017/EchoNet",
    "USER_MONGODB_URI": "mongodb://user:pass@localhost:27017/UserSample",
    "JWT_SECRET": "test-secret",
}

os.environ.setdefault(
    "MONGODB_URI",
    REQUIRED_ENV["MONGODB_URI"],
)
os.environ.setdefault(
    "USER_MONGODB_URI",
    REQUIRED_ENV["USER_MONGODB_URI"],
)
os.environ.setdefault(
    "JWT_SECRET",
    REQUIRED_ENV["JWT_SECRET"],
)


from app.config import Settings, settings
from app.feature_flags import (
    ANALYTICS_EXTENSIONS_FLAG,
    REALTIME_STREAMING_FLAG,
    UnknownFeatureFlagError,
    is_feature_enabled,
    require_feature,
)
from app.routers import live


class FakeCollection:
    """
    Minimal Mongo-like collection used to import and exercise
    insights.py without connecting to the real database.
    """

    def count_documents(self, *args, **kwargs):
        return 0

    def aggregate(self, *args, **kwargs):
        return []


def load_insights_with_fake_database(monkeypatch):
    """
    Import app.routers.insights with isolated fake collections.

    insights.py imports Events, Microphones and Nodes directly from
    app.database, so replacing app.database before import prevents
    MongoDB connections/index creation during this unit test.
    """
    fake_database = types.ModuleType("app.database")

    fake_database.Events = FakeCollection()
    fake_database.Microphones = FakeCollection()
    fake_database.Nodes = FakeCollection()

    monkeypatch.setitem(
        sys.modules,
        "app.database",
        fake_database,
    )

    # Force a fresh import so the fake app.database is used.
    monkeypatch.delitem(
        sys.modules,
        "app.routers.insights",
        raising=False,
    )

    return importlib.import_module(
        "app.routers.insights"
    )


def test_feature_flags_default_to_enabled():
    with patch.dict(
        "os.environ",
        REQUIRED_ENV,
        clear=True,
    ):
        config = Settings()

    assert config.realtime_streaming_enabled is True
    assert config.analytics_extensions_enabled is True


def test_feature_flags_can_be_disabled_from_environment():
    env = {
        **REQUIRED_ENV,
        "REALTIME_STREAMING_ENABLED": "false",
        "ANALYTICS_EXTENSIONS_ENABLED": "false",
    }

    with patch.dict(
        "os.environ",
        env,
        clear=True,
    ):
        config = Settings()

    assert config.realtime_streaming_enabled is False
    assert config.analytics_extensions_enabled is False


def test_feature_flag_helper_returns_enabled_state(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "realtime_streaming_enabled",
        True,
    )

    assert (
        is_feature_enabled(
            REALTIME_STREAMING_FLAG
        )
        is True
    )


def test_feature_flag_helper_returns_disabled_state(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "realtime_streaming_enabled",
        False,
    )

    assert (
        is_feature_enabled(
            REALTIME_STREAMING_FLAG
        )
        is False
    )


def test_unknown_feature_flag_fails_loudly():
    with pytest.raises(UnknownFeatureFlagError):
        is_feature_enabled(
            "does_not_exist"
        )


def test_require_feature_allows_enabled_feature(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "analytics_extensions_enabled",
        True,
    )

    require_feature(
        ANALYTICS_EXTENSIONS_FLAG,
        "analytics extensions",
    )


def test_require_feature_returns_predictable_404_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "analytics_extensions_enabled",
        False,
    )

    with pytest.raises(HTTPException) as exc_info:
        require_feature(
            ANALYTICS_EXTENSIONS_FLAG,
            "analytics extensions",
        )

    assert exc_info.value.status_code == 404
    assert (
        exc_info.value.detail
        == "analytics extensions is disabled"
    )


def test_analytics_filter_is_blocked_when_extensions_disabled(
    monkeypatch,
):
    insights = load_insights_with_fake_database(
        monkeypatch
    )

    monkeypatch.setattr(
        settings,
        "analytics_extensions_enabled",
        False,
    )

    with pytest.raises(HTTPException) as exc_info:
        insights.insights_overview(
            species="Dingo",
        )

    assert exc_info.value.status_code == 404
    assert (
        exc_info.value.detail
        == "analytics extensions is disabled"
    )


def test_analytics_filter_is_allowed_when_extensions_enabled(
    monkeypatch,
):
    insights = load_insights_with_fake_database(
        monkeypatch
    )

    monkeypatch.setattr(
        settings,
        "analytics_extensions_enabled",
        True,
    )

    result = insights.insights_overview(
        species="Dingo",
    )

    # Fake collections are empty, but the important behaviour here is
    # that the feature gate allows the request to reach analytics logic.
    assert result["counts"]["detections"] == 0


class FakeWebSocket:
    def __init__(self):
        self.closed_code = None
        self.accepted = False
        self.query_params = {
            "token": "c12-test-token"
        }
        self.headers = {}

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed_code = code

    async def receive_text(self):
        raise WebSocketDisconnect()


def test_realtime_streaming_closes_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        live,
        "is_feature_enabled",
        lambda flag_name: False,
    )

    websocket = FakeWebSocket()

    asyncio.run(
        live.detection_stream(websocket)
    )

    assert websocket.accepted is True
    assert websocket.closed_code == status.WS_1008_POLICY_VIOLATION


def test_realtime_streaming_connects_when_enabled(
    monkeypatch,
):
    monkeypatch.setattr(
        live,
        "is_feature_enabled",
        lambda flag_name: True,
    )

    monkeypatch.setattr(
        live,
        "decodeJWT",
        lambda token: {
            "sub": "c12-test-user"
        },
    )

    state = {
        "connected": False,
        "disconnected": False,
    }

    async def fake_connect(websocket):
        state["connected"] = True

    async def fake_disconnect(websocket):
        state["disconnected"] = True

    monkeypatch.setattr(
        live.detection_stream_manager,
        "connect",
        fake_connect,
    )

    monkeypatch.setattr(
        live.detection_stream_manager,
        "disconnect",
        fake_disconnect,
    )

    websocket = FakeWebSocket()

    asyncio.run(
        live.detection_stream(websocket)
    )

    assert state["connected"] is True
    assert state["disconnected"] is True
    assert websocket.closed_code is None