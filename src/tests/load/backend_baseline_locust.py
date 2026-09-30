"""Task 3 backend load-testing baseline.

Exercises the same three workloads as backend_baseline_k6.js:
- GET /                     : liveness proxy
- POST /api/audio/upload    : audio ingestion
- POST /hmi/post_recording  : backend -> MQTT queue ingestion

The simulator should remain stopped during queue-ingestion benchmarking so the
measurement boundary ends at successful publication to the MQTT broker.
"""

from datetime import datetime, timezone
from io import BytesIO

from locust import HttpUser, between, task


class BackendBaselineUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.workload_index = self.environment.runner.user_count % 3

    @task
    def backend_workload(self):
        workload = self.workload_index % 3
        self.workload_index += 1

        if workload == 0:
            self.health_liveness()
        elif workload == 1:
            self.audio_upload()
        else:
            self.queue_ingest()

    def health_liveness(self):
        with self.client.get(
            "/",
            name="health_liveness",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Expected 200, got {response.status_code}"
                )

    def audio_upload(self):
        timestamp = datetime.now(timezone.utc).timestamp()
        filename = f"task3_loadtest_locust_{id(self)}_{timestamp}.wav"

        files = {
            "file": (
                filename,
                BytesIO(b"task3-load-test-audio"),
                "audio/wav",
            )
        }

        data = {
            "user_id": "task3-load-test",
        }

        with self.client.post(
            "/api/audio/upload",
            files=files,
            data=data,
            name="audio_upload",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Expected 200, got {response.status_code}"
                )

    def queue_ingest(self):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensorId": f"task3-loadtest-{id(self)}",
            "microphoneLLA": [-37.8136, 144.9631, 10.0],
            "animalEstLLA": [-37.8135, 144.9632, 10.0],
            "animalTrueLLA": [-37.8134, 144.9633, 10.0],
            "animalLLAUncertainty": 5.0,
            "audioClip": f"task3-loadtest-{id(self)}",
            "mode": "Recording_Mode",
            "audioFile": f"task3_loadtest_{id(self)}.wav",
        }

        with self.client.post(
            "/hmi/post_recording",
            json=payload,
            name="queue_ingest",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(
                    f"Expected 201, got {response.status_code}"
                )
