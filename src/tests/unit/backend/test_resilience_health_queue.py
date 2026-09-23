import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from bson import ObjectId

# Set required test env vars before importing app
os.environ["MONGODB_URI"] = "mongodb://localhost:27017/EchoNet"
os.environ["USER_MONGODB_URI"] = "mongodb://localhost:27017/UserSample"
os.environ["JWT_SECRET"] = "test-secret-key-32-chars-minimum-length-resilience"

# test_detection_serialization.py leaves a stub `app.database` (no __file__) in
# sys.modules; this module needs the real one for app.main, so drop the stub.
if not getattr(sys.modules.get("app.database"), "__file__", None):
    sys.modules.pop("app.database", None)

from redis import Redis
from rq import Queue, SimpleWorker
from rq.registry import FailedJobRegistry, ScheduledJobRegistry

from app.config import settings
from app.main import app
from app.queue import (
    QUEUE_NAME,
    AudioFileMissingError,
    enqueue_audio_ingest,
    process_audio_ingest_job,
)
from app.routers.health import health_dependencies, check_socket
from app.services import mqtt_client


def _last_set(mock_uploads):
    return mock_uploads.update_one.call_args_list[-1][0][1]["$set"]


class TestResilienceHealthAndQueue(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.routers.health.client")
    @patch("app.routers.health.Userclient")
    @patch("app.routers.health.redis_conn")
    @patch("app.routers.health.check_socket")
    def test_health_dependencies_all_up(self, mock_socket, mock_redis, mock_user_mongo, mock_primary_mongo):
        mock_primary_mongo.admin.command.return_value = {"ok": 1}
        mock_user_mongo.admin.command.return_value = {"ok": 1}
        mock_redis.ping.return_value = True
        mock_socket.return_value = {"status": "UP", "latency_ms": 1.5}

        response = self.client.get("/health/dependencies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")
        self.assertEqual(data["dependencies"]["mongodb_primary"]["status"], "UP")
        self.assertEqual(data["dependencies"]["mongodb_user"]["status"], "UP")
        self.assertEqual(data["dependencies"]["redis"]["status"], "UP")
        self.assertEqual(data["dependencies"]["hivemq"]["status"], "UP")

    @patch("app.routers.health.client")
    @patch("app.routers.health.Userclient")
    @patch("app.routers.health.redis_conn")
    @patch("app.routers.health.check_socket")
    def test_health_dependencies_degraded_when_redis_down(self, mock_socket, mock_redis, mock_user_mongo, mock_primary_mongo):
        mock_primary_mongo.admin.command.return_value = {"ok": 1}
        mock_user_mongo.admin.command.return_value = {"ok": 1}
        mock_redis.ping.side_effect = Exception("Connection timeout (bounded)")
        mock_socket.return_value = {"status": "UP", "latency_ms": 1.2}

        response = self.client.get("/health/dependencies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "DEGRADED")
        self.assertEqual(data["dependencies"]["redis"]["status"], "DOWN")
        self.assertIn("Connection timeout", data["dependencies"]["redis"]["error"])

    @patch("app.routers.health.client")
    @patch("app.routers.health.Userclient")
    @patch("app.routers.health.redis_conn")
    @patch("app.routers.health.check_socket")
    def test_health_dependencies_down_when_all_fail(self, mock_socket, mock_redis, mock_user_mongo, mock_primary_mongo):
        mock_primary_mongo.admin.command.side_effect = Exception("Mongo primary unavailable")
        mock_user_mongo.admin.command.side_effect = Exception("Mongo user unavailable")
        mock_redis.ping.side_effect = Exception("Redis unreachable")
        mock_socket.return_value = {"status": "DOWN", "latency_ms": None, "error": "Connection refused"}

        response = self.client.get("/health/dependencies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "DOWN")

    @patch("app.database.AudioUploads")
    def test_process_audio_ingest_job_workflow(self, mock_uploads):
        fake_id = str(ObjectId())
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"RIFF....WAVE")
            audio.flush()
            mock_uploads.find_one.return_value = {
                "_id": ObjectId(fake_id),
                "filename": "test_audio.wav",
                "path": audio.name,
            }
            mock_uploads.update_one.return_value = MagicMock(modified_count=1)

            result = process_audio_ingest_job(upload_id=fake_id, filename="test_audio.wav")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["upload_id"], fake_id)
        self.assertEqual(result["filename"], "test_audio.wav")
        self.assertIn("completed_at", result)
        # Verify MongoDB updates called for state transition
        self.assertGreaterEqual(mock_uploads.update_one.call_count, 2)
        completion = _last_set(mock_uploads)
        self.assertEqual(completion["processing_status"], "completed")
        self.assertEqual(completion["pipeline_status"], "ready_for_inference")
        self.assertEqual(completion["file_size_verified"], 12)

    @patch("app.queue.get_current_job")
    @patch("app.database.AudioUploads")
    def test_missing_file_fails_without_retry_and_is_not_ready(self, mock_uploads, mock_current_job):
        fake_job = MagicMock(retries_left=3)
        mock_current_job.return_value = fake_job
        fake_id = str(ObjectId())
        mock_uploads.find_one.return_value = {"_id": ObjectId(fake_id), "path": "uploads/gone.wav"}

        with self.assertRaises(AudioFileMissingError):
            process_audio_ingest_job(upload_id=fake_id, file_path="/nonexistent/gone.wav", filename="gone.wav")

        self.assertEqual(fake_job.retries_left, 0)
        failure = _last_set(mock_uploads)
        self.assertEqual(failure["processing_status"], "failed")
        self.assertIn("/nonexistent/gone.wav", failure["error_message"])
        for call in mock_uploads.update_one.call_args_list:
            self.assertNotEqual(call[0][1].get("$set", {}).get("pipeline_status"), "ready_for_inference")

    @patch("app.database.AudioUploads")
    def test_missing_file_fails_when_no_path_recorded(self, mock_uploads):
        mock_uploads.find_one.return_value = None

        with self.assertRaises(AudioFileMissingError):
            process_audio_ingest_job(upload_id=str(ObjectId()), filename="orphan.wav")

        self.assertEqual(_last_set(mock_uploads)["processing_status"], "failed")

    @patch("app.database.AudioUploads")
    def test_empty_file_fails(self, mock_uploads):
        mock_uploads.find_one.return_value = {"_id": ObjectId()}
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with self.assertRaises(AudioFileMissingError):
                process_audio_ingest_job(upload_id=str(ObjectId()), file_path=audio.name)

        self.assertEqual(_last_set(mock_uploads)["processing_status"], "failed")

    @patch("app.queue.get_current_job")
    @patch("app.database.AudioUploads")
    def test_transient_error_marks_retry_scheduled_then_failed_on_last_attempt(self, mock_uploads, mock_current_job):
        mock_uploads.find_one.side_effect = ConnectionError("mongo unavailable")
        fake_job = MagicMock(retries_left=2)
        mock_current_job.return_value = fake_job

        with self.assertRaises(ConnectionError):
            process_audio_ingest_job(upload_id=str(ObjectId()), file_path="/tmp/x.wav")
        self.assertEqual(_last_set(mock_uploads)["processing_status"], "retry_scheduled")
        self.assertEqual(fake_job.retries_left, 2)  # transient errors keep their retries

        fake_job.retries_left = 0
        with self.assertRaises(ConnectionError):
            process_audio_ingest_job(upload_id=str(ObjectId()), file_path="/tmp/x.wav")
        self.assertEqual(_last_set(mock_uploads)["processing_status"], "failed")

    @patch("app.queue.job_queue")
    def test_enqueue_audio_ingest_helper(self, mock_queue):
        fake_job = MagicMock()
        fake_job.id = "job-echo-12345"
        mock_queue.enqueue.return_value = fake_job

        job_id = enqueue_audio_ingest(upload_id="60c72b2f9b1e8a001c8a1234", filename="sample.wav")
        self.assertEqual(job_id, "job-echo-12345")
        mock_queue.enqueue.assert_called_once()
        retry = mock_queue.enqueue.call_args.kwargs["retry"]
        self.assertEqual(retry.max, 3)
        self.assertEqual(retry.intervals, [10, 30, 60])
        self.assertEqual(mock_queue.enqueue.call_args.kwargs["job_timeout"], settings.job_timeout_seconds)

    def test_queue_matches_agreed_setup(self):
        self.assertEqual(QUEUE_NAME, "echo-backend")
        self.assertEqual(settings.redis_url, "redis://echo-redis:6379/1")

    @patch("app.main.close_cache_client")
    @patch("app.main.stop_mqtt_client")
    @patch("app.main.start_mqtt_client")
    @patch("app.main.client")
    @patch("app.main.Userclient")
    @patch("app.main.redis_conn")
    def test_lifespan_shutdown_closes_all_owned_clients(
        self, mock_redis, mock_user_mongo, mock_primary_mongo, mock_start_mqtt, mock_stop_mqtt, mock_close_cache
    ):
        # Trigger app lifespan startup & shutdown
        with TestClient(app) as test_client:
            res = test_client.get("/")
            self.assertEqual(res.status_code, 200)
            mock_start_mqtt.assert_called_once()
            mock_stop_mqtt.assert_not_called()

        # Once context exits, all clients must have been released
        mock_stop_mqtt.assert_called_once()
        mock_primary_mongo.close.assert_called_once()
        mock_user_mongo.close.assert_called_once()
        mock_redis.close.assert_called_once()
        mock_close_cache.assert_called_once()

    @patch("app.main.close_cache_client")
    @patch("app.main.stop_mqtt_client", side_effect=RuntimeError("mqtt stuck"))
    @patch("app.main.start_mqtt_client")
    @patch("app.main.client")
    @patch("app.main.Userclient")
    @patch("app.main.redis_conn")
    def test_lifespan_cleanup_continues_after_mqtt_error(
        self, mock_redis, mock_user_mongo, mock_primary_mongo, _start, _stop, mock_close_cache
    ):
        with TestClient(app):
            pass
        mock_primary_mongo.close.assert_called_once()
        mock_user_mongo.close.assert_called_once()
        mock_redis.close.assert_called_once()
        mock_close_cache.assert_called_once()


class TestMqttClientLifecycle(unittest.TestCase):

    def tearDown(self):
        mqtt_client._client = None
        mqtt_client.connection_state = "disconnected"

    @patch("app.services.mqtt_client.mqtt.Client")
    def test_stop_disconnects_and_joins_loop(self, mock_client_cls):
        paho = mock_client_cls.return_value

        started = mqtt_client.start_mqtt_client()
        self.assertIs(started, paho)
        paho.loop_start.assert_called_once()
        # A second start reuses the owned client instead of leaking another loop thread
        self.assertIs(mqtt_client.start_mqtt_client(), paho)
        mock_client_cls.assert_called_once()

        mqtt_client.stop_mqtt_client()
        paho.disconnect.assert_called_once()
        paho.loop_stop.assert_called_once()
        self.assertEqual(mqtt_client.get_connection_state(), "disconnected")

        # Idempotent: nothing left to stop
        mqtt_client.stop_mqtt_client()
        paho.disconnect.assert_called_once()
        paho.loop_stop.assert_called_once()

    @patch("app.services.mqtt_client.mqtt.Client")
    def test_stop_joins_loop_even_if_disconnect_raises(self, mock_client_cls):
        paho = mock_client_cls.return_value
        paho.disconnect.side_effect = OSError("socket closed")

        mqtt_client.start_mqtt_client()
        mqtt_client.stop_mqtt_client()

        paho.loop_stop.assert_called_once()
        self.assertEqual(mqtt_client.get_connection_state(), "disconnected")

    def test_clean_disconnect_is_not_reported_as_reconnecting(self):
        mqtt_client.on_disconnect(None, None, 0)
        self.assertEqual(mqtt_client.get_connection_state(), "disconnected")
        mqtt_client.on_disconnect(None, None, 7)
        self.assertEqual(mqtt_client.get_connection_state(), "reconnecting")


# Runs the job through a real RQ worker against Redis to prove the retry
# bookkeeping works with RQ itself, not just with mocks. Uses a scratch queue on
# DB 15 so it never touches the live echo-backend queue on DB 1.
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://echo-redis:6379/15")


def _redis_available():
    try:
        return Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=1).ping()
    except Exception:
        return False


@unittest.skipUnless(_redis_available(), f"Redis not reachable at {TEST_REDIS_URL}")
class TestAudioIngestWithRealRqWorker(unittest.TestCase):

    def setUp(self):
        self.redis = Redis.from_url(TEST_REDIS_URL)
        self.redis.flushdb()
        self.queue = Queue("echo-backend-test", connection=self.redis)

    def tearDown(self):
        self.redis.flushdb()
        self.redis.close()

    def _run_once(self):
        SimpleWorker([self.queue], connection=self.redis).work(burst=True)

    @patch("app.database.AudioUploads")
    def test_missing_file_goes_straight_to_failed_registry(self, _uploads):
        from rq import Retry

        job = self.queue.enqueue(
            process_audio_ingest_job,
            upload_id=str(ObjectId()),
            file_path="/nonexistent/gone.wav",
            retry=Retry(max=3, interval=[10, 30, 60]),
        )
        self._run_once()

        job.refresh()
        self.assertTrue(job.is_failed)
        self.assertEqual(job.retries_left, 0)
        self.assertIn(job.id, FailedJobRegistry(queue=self.queue).get_job_ids())
        self.assertNotIn(job.id, ScheduledJobRegistry(queue=self.queue).get_job_ids())

    @patch("app.database.AudioUploads")
    def test_transient_error_is_scheduled_for_delayed_retry(self, uploads):
        from rq import Retry

        uploads.find_one.side_effect = ConnectionError("mongo unavailable")
        job = self.queue.enqueue(
            process_audio_ingest_job,
            upload_id=str(ObjectId()),
            file_path="/tmp/x.wav",
            retry=Retry(max=3, interval=[10, 30, 60]),
        )
        self._run_once()

        job.refresh()
        self.assertTrue(job.is_scheduled)
        self.assertEqual(job.retries_left, 2)
        self.assertIn(job.id, ScheduledJobRegistry(queue=self.queue).get_job_ids())
        self.assertEqual(_last_set(uploads)["processing_status"], "retry_scheduled")


if __name__ == "__main__":
    unittest.main()
