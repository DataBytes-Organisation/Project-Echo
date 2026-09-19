"""Offline tests for the explicit CI-only R2 launcher."""

import io
import os
import sys
import unittest
import wave
from unittest.mock import patch

from botocore.exceptions import StubAssertionError
from cloudflare_r2 import R2Config, R2Storage

import r2_fixture


class R2CIFixtureTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, r2_fixture.CI_ENV)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def storage(self):
        config = R2Config.from_env()
        return R2Storage(config, r2_fixture.FixtureClient(config))

    def test_bucket_listing_and_audio_are_offline_and_repeatable(self):
        with patch(
            "botocore.httpsession.URLLib3Session.send",
            side_effect=AssertionError("Unexpected network request"),
        ):
            storage = self.storage()
            for _ in range(2):
                storage.check_connection()
                self.assertEqual(storage.list_species(), [r2_fixture.SPECIES])
                data = storage.download_bytes(r2_fixture.OBJECT_KEY)
                self.assertEqual(data, r2_fixture.sample_audio())
                with wave.open(io.BytesIO(data), "rb") as audio:
                    self.assertEqual(
                        (audio.getnchannels(), audio.getframerate(), audio.getnframes()),
                        (1, 16000, 16000),
                    )

    def test_unmatched_prefix_returns_no_species(self):
        self.assertEqual(self.storage().list_species("not-the-fixture"), [])

    def test_incorrect_bucket_is_rejected(self):
        with self.assertRaises(StubAssertionError):
            self.storage()._client.head_bucket(Bucket="wrong-bucket")

    def test_incorrect_object_is_rejected(self):
        with self.assertRaises(StubAssertionError):
            self.storage().download_bytes("missing/file.wav")

    def test_fixture_requires_explicit_opt_in(self):
        del os.environ["ECHO_R2_CI_FIXTURE"]
        with self.assertRaisesRegex(RuntimeError, "explicit synthetic CI"):
            r2_fixture.validate_environment()

    def test_fixture_rejects_live_credentials(self):
        os.environ["R2_ACCESS_KEY_ID"] = "not-the-fixture-key"
        with self.assertRaisesRegex(RuntimeError, "do not use live credentials"):
            r2_fixture.FixtureClient(R2Config.from_env())

    def test_normal_configuration_still_requires_real_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Missing required Cloudflare R2"):
                R2Config.from_env()

    def test_check_mode_does_not_launch_an_application(self):
        original = R2Storage._create_client
        with patch.object(sys, "argv", ["r2_fixture.py", "check"]):
            with patch.object(r2_fixture.runpy, "run_path") as run:
                r2_fixture.main()
                run.assert_not_called()
        self.assertIs(R2Storage._create_client, original)

    def test_launches_real_entrypoints_and_restores_client(self):
        original = R2Storage._create_client
        for service, filename in (
            ("engine", "echo_engine.py"),
            ("simulator", "system_manager.py"),
        ):
            with self.subTest(service=service):
                with patch.object(sys, "argv", ["r2_fixture.py", service]):
                    with patch.object(sys, "path", list(sys.path)):
                        with patch.object(r2_fixture.runpy, "run_path") as run:
                            r2_fixture.main()
                            self.assertEqual(
                                run.call_args.args[0].replace("\\", "/"),
                                f"/app/{filename}",
                            )
                            self.assertEqual(run.call_args.kwargs, {"run_name": "__main__"})
                self.assertIs(R2Storage._create_client, original)

    def test_application_failure_is_not_hidden(self):
        with patch.object(sys, "argv", ["r2_fixture.py", "engine"]):
            with patch.object(sys, "path", list(sys.path)):
                with patch.object(
                    r2_fixture.runpy, "run_path",
                    side_effect=RuntimeError("application failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "application failed"):
                        r2_fixture.main()


if __name__ == "__main__":
    unittest.main()
