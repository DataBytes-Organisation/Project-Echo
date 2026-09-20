import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from bson import ObjectId

# Set required test env vars before importing app
os.environ["MONGODB_URI"] = "mongodb://localhost:27017/EchoNet"
os.environ["USER_MONGODB_URI"] = "mongodb://localhost:27017/UserSample"
os.environ["JWT_SECRET"] = "test-secret-key-32-chars-minimum-length-resilience"

from app.config import settings
from app.main import app
from app.queue import process_audio_ingest_job, enqueue_audio_ingest
from app.routers.health import health_dependencies, check_socket


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
        mock_uploads.find_one.return_value = {
            "_id": ObjectId(fake_id),
            "filename": "test_audio.wav",
            "path": "uploads/test_audio.wav",
        }
        mock_uploads.update_one.return_value = MagicMock(modified_count=1)

        result = process_audio_ingest_job(upload_id=fake_id, filename="test_audio.wav")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["upload_id"], fake_id)
        self.assertEqual(result["filename"], "test_audio.wav")
        self.assertIn("completed_at", result)
        # Verify MongoDB updates called for state transition
        self.assertGreaterEqual(mock_uploads.update_one.call_count, 2)

    @patch("app.queue.job_queue")
    def test_enqueue_audio_ingest_helper(self, mock_queue):
        fake_job = MagicMock()
        fake_job.id = "job-echo-12345"
        mock_queue.enqueue.return_value = fake_job

        job_id = enqueue_audio_ingest(upload_id="60c72b2f9b1e8a001c8a1234", filename="sample.wav")
        self.assertEqual(job_id, "job-echo-12345")
        mock_queue.enqueue.assert_called_once()

    @patch("app.main.client")
    @patch("app.main.Userclient")
    @patch("app.main.redis_conn")
    def test_lifespan_shutdown_closes_all_owned_clients(self, mock_redis, mock_user_mongo, mock_primary_mongo):
        # Trigger app lifespan startup & shutdown
        with TestClient(app) as test_client:
            res = test_client.get("/")
            self.assertEqual(res.status_code, 200)
        
        # Once context exits, all clients must have close() called
        mock_primary_mongo.close.assert_called_once()
        mock_user_mongo.close.assert_called_once()
        mock_redis.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
