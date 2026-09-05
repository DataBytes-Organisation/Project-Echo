"""Locust load test against Project Echo's backend API.

Targets 4 real, safe, read-only, no-auth GET endpoints - verified working
by hand before being added here (see docs/team-guides/TDD_Guide.md for the
full list considered and why /insights/* was excluded: it 500s due to a
real bug - app/routers/insights.py reads a differently-named env var,
MONGO_URI instead of the rest of the app's MONGODB_URI, with no fallback
default, so it connects to Mongo with no credentials at all).

Prerequisites (see TDD_Guide.md for the full walkthrough):
    1. MongoDB + Redis running:
       docker compose -f src/deployment/docker/docker-compose.yml up echo_store echo-redis -d
    2. The backend running locally, with env vars matching docker-compose's
       echo_api service plus a host-reachable Mongo URI:
       MONGODB_URI="mongodb://root:root_password@localhost:27017/EchoNet?authSource=admin"
       MAIL_STARTTLS=true MAIL_SSL_TLS=false PYTHONIOENCODING=utf-8 \
         uvicorn app.main:app --host 127.0.0.1 --port 9000
       (PYTHONIOENCODING=utf-8 works around a real bug: app/main.py prints a
       U+2705 checkmark at import time, which crashes with UnicodeEncodeError
       under Windows' default cp1252 console encoding whenever stdout isn't a
       real interactive terminal - e.g. redirected to a file, or launched by
       a process manager.)

Run headless (example: 20 users, spawn 5/sec, 60 second burst):
    locust -f src/tests/load/locustfile.py --host http://127.0.0.1:9000 \
        --headless -u 20 -r 5 -t 60s --csv=locust_results
"""

from locust import HttpUser, task, between


class EchoApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def public_test(self):
        self.client.get("/public/public-test", name="/public/public-test")

    @task(2)
    def hmi_microphones(self):
        self.client.get("/hmi/microphones", name="/hmi/microphones")

    @task(2)
    def iot_nodes(self):
        self.client.get("/iot/nodes", name="/iot/nodes")

    @task(1)
    def engine_animal_records(self):
        self.client.get("/engine/animal_records", name="/engine/animal_records")
