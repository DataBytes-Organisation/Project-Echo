import sys
import types
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from bson import ObjectId
from requests import HTTPError, Response
from rq import Retry

fake_database = types.ModuleType("app.database")
fake_database.Detections = MagicMock()
fake_database.GENDER = []
fake_database.STATES_CODE = []
fake_database.AUS_STATES = []
original_database = sys.modules.get("app.database")
sys.modules["app.database"] = fake_database

fake_schemas = types.ModuleType("app.schemas")


class DummyDetection:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_schemas.Detection = DummyDetection
fake_schemas.DetectionCreate = object
original_schemas = sys.modules.get("app.schemas")
sys.modules["app.schemas"] = fake_schemas

from app.jobs import detection_webhook
from app.jobs.queue import JOB_TIMEOUT_SECONDS, enqueue_detection_webhook
from app import detections as detections_service

if original_database is None:
    sys.modules.pop("app.database", None)
else:
    sys.modules["app.database"] = original_database

if original_schemas is None:
    sys.modules.pop("app.schemas", None)
else:
    sys.modules["app.schemas"] = original_schemas


DETECTION_ID = "651f2a9f4d1f1b1c3e2a4567"


def _detection_doc():
    return {
        "_id": ObjectId(DETECTION_ID),
        "timestamp": datetime(2023, 3, 22, 13, 45, 12),
        "sensorId": "2",
        "species": "Sus Scrofa",
        "microphoneLLA": [-33.1101, 150.0567, 23.0],
        "animalEstLLA": [-33.1105, 150.0569, 23.0],
        "animalTrueLLA": [-33.1106, 150.057, 23.0],
        "animalLLAUncertainty": 10,
        "audioClip": "clip",
        "confidence": 99.4,
        "sampleRate": 48000,
    }


class DetectionWebhookJobTests(unittest.TestCase):
    @patch.object(detection_webhook.requests, "post")
    @patch.object(detection_webhook, "Detections")
    @patch.dict("os.environ", {"DETECTION_WEBHOOK_URL": ""}, clear=False)
    def test_returns_false_when_url_unset(self, detections_col, post):
        self.assertFalse(detection_webhook.process_detection_job(DETECTION_ID))
        post.assert_not_called()
        detections_col.find_one.assert_not_called()

    @patch.object(detection_webhook.requests, "post")
    @patch.object(detection_webhook, "Detections")
    @patch.dict("os.environ", {"DETECTION_WEBHOOK_URL": "http://hooks.test/detect"}, clear=False)
    def test_returns_false_when_detection_missing(self, detections_col, post):
        detections_col.find_one.return_value = None
        self.assertFalse(detection_webhook.process_detection_job(DETECTION_ID))
        post.assert_not_called()

    @patch.object(detection_webhook.requests, "post")
    @patch.object(detection_webhook, "Detections")
    @patch.dict("os.environ", {"DETECTION_WEBHOOK_URL": "http://hooks.test/detect"}, clear=False)
    def test_returns_false_when_detection_id_invalid(self, detections_col, post):
        self.assertFalse(detection_webhook.process_detection_job("not-an-object-id"))
        post.assert_not_called()
        detections_col.find_one.assert_not_called()

    @patch.object(detection_webhook.requests, "post")
    @patch.object(detection_webhook, "Detections")
    @patch.dict("os.environ", {"DETECTION_WEBHOOK_URL": "http://hooks.test/detect"}, clear=False)
    def test_returns_true_on_http_200(self, detections_col, post):
        detections_col.find_one.return_value = _detection_doc()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        post.return_value = response

        self.assertTrue(detection_webhook.process_detection_job(DETECTION_ID))
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], detection_webhook.HTTP_TIMEOUT_SECONDS)

    @patch.object(detection_webhook.requests, "post")
    @patch.object(detection_webhook, "Detections")
    @patch.dict("os.environ", {"DETECTION_WEBHOOK_URL": "http://hooks.test/detect"}, clear=False)
    def test_raises_on_http_500(self, detections_col, post):
        detections_col.find_one.return_value = _detection_doc()
        response = Response()
        response.status_code = 500
        post.return_value = response

        with self.assertRaises(HTTPError):
            detection_webhook.process_detection_job(DETECTION_ID)


class EnqueueDetectionWebhookTests(unittest.TestCase):
    @patch("app.jobs.queue.get_queue")
    def test_enqueues_with_retry_and_timeout(self, get_queue):
        queue = MagicMock()
        get_queue.return_value = queue

        enqueue_detection_webhook(DETECTION_ID)

        queue.enqueue.assert_called_once()
        args, kwargs = queue.enqueue.call_args
        self.assertEqual(args[0], detection_webhook.process_detection_job)
        self.assertEqual(args[1], DETECTION_ID)
        self.assertEqual(kwargs["job_timeout"], JOB_TIMEOUT_SECONDS)
        retry = kwargs["retry"]
        self.assertIsInstance(retry, Retry)
        self.assertEqual(retry.max, 3)
        self.assertEqual(list(retry.intervals), [10, 30, 60])


class CreateDetectionEnqueueTests(unittest.TestCase):
    @patch("app.detections.enqueue_detection_webhook")
    @patch("app.detections.Detections")
    def test_enqueues_after_mongo_insert(self, detections_col, enqueue):
        inserted_id = ObjectId(DETECTION_ID)
        order = []
        detections_col.insert_one.side_effect = lambda *_a, **_k: (
            order.append("insert") or MagicMock(inserted_id=inserted_id)
        )
        detections_col.find_one.return_value = _detection_doc()
        enqueue.side_effect = lambda *_a, **_k: order.append("enqueue")

        payload = MagicMock()
        payload.dict.return_value = {"species": "Sus Scrofa"}

        detections_service.create_detection(payload)

        self.assertEqual(order, ["insert", "enqueue"])
        enqueue.assert_called_once_with(DETECTION_ID)
        detections_col.insert_one.assert_called_once()


if __name__ == "__main__":
    unittest.main()
