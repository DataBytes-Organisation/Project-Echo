"""
Tests for app.config.Settings.

Run inside the API container / with the backend's requirements installed:
    python -m unittest test_config -v
"""
import os
import unittest
from unittest.mock import patch

REQUIRED_ENV = {
    "MONGODB_URI": "mongodb://user:pass@localhost:27017/EchoNet",
    "USER_MONGODB_URI": "mongodb://user:pass@localhost:27017/UserSample",
    "JWT_SECRET": "test-secret",
}

# app.config builds a module-level `settings = Settings()` singleton at import
# time, so the required vars must exist in the process env *before* the first
# import below — otherwise this test module would only run in a shell that
# already happens to have them exported (e.g. inside the API container).
os.environ.setdefault("MONGODB_URI", REQUIRED_ENV["MONGODB_URI"])
os.environ.setdefault("USER_MONGODB_URI", REQUIRED_ENV["USER_MONGODB_URI"])
os.environ.setdefault("JWT_SECRET", REQUIRED_ENV["JWT_SECRET"])

from pydantic import ValidationError

from app.config import Settings


class TestSettingsValidEnv(unittest.TestCase):
    def test_constructs_with_required_env_only(self):
        with patch.dict("os.environ", REQUIRED_ENV, clear=True):
            settings = Settings()
        self.assertEqual(settings.mongodb_uri, REQUIRED_ENV["MONGODB_URI"])
        self.assertEqual(settings.user_mongodb_uri, REQUIRED_ENV["USER_MONGODB_URI"])
        self.assertEqual(settings.jwt_secret, REQUIRED_ENV["JWT_SECRET"])

    def test_int_and_float_fields_are_coerced(self):
        env = {**REQUIRED_ENV, "REDIS_PORT": "6380", "REQUEST_TIMEOUT_SECONDS": "20.5"}
        with patch.dict("os.environ", env, clear=True):
            settings = Settings()
        self.assertIsInstance(settings.redis_port, int)
        self.assertEqual(settings.redis_port, 6380)
        self.assertIsInstance(settings.request_timeout_seconds, float)
        self.assertEqual(settings.request_timeout_seconds, 20.5)


class TestSettingsFailFast(unittest.TestCase):
    def test_missing_mongodb_uri_raises(self):
        env = {k: v for k, v in REQUIRED_ENV.items() if k != "MONGODB_URI"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings()
        self.assertIn("mongodb_uri", str(ctx.exception))

    def test_missing_user_mongodb_uri_raises(self):
        env = {k: v for k, v in REQUIRED_ENV.items() if k != "USER_MONGODB_URI"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings()
        self.assertIn("user_mongodb_uri", str(ctx.exception))

    def test_missing_jwt_secret_raises(self):
        env = {k: v for k, v in REQUIRED_ENV.items() if k != "JWT_SECRET"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings()
        self.assertIn("jwt_secret", str(ctx.exception))


class TestSettingsDefaults(unittest.TestCase):
    def test_optional_secrets_default_to_none(self):
        with patch.dict("os.environ", REQUIRED_ENV, clear=True):
            settings = Settings()
        self.assertIsNone(settings.twilio_account_sid)
        self.assertIsNone(settings.twilio_auth_token)
        self.assertIsNone(settings.twilio_phone_number)
        self.assertIsNone(settings.mail_username)

    def test_infra_defaults_applied_when_unset(self):
        with patch.dict("os.environ", REQUIRED_ENV, clear=True):
            settings = Settings()
        self.assertEqual(settings.mqtt_host, "ts-mqtt-server-cont")
        self.assertEqual(settings.mqtt_port, 1883)
        self.assertEqual(settings.redis_host, "echo-redis")
        self.assertEqual(settings.redis_port, 6379)
        self.assertEqual(settings.jwt_algorithm, "HS256")
        self.assertEqual(settings.mongo_db_name, "EchoNet")


if __name__ == "__main__":
    unittest.main()
