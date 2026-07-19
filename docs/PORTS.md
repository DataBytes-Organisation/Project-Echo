# Port Reference

Canonical port assignments for every service in `src/components/docker-compose.yml`.

| Service | Port(s) | Notes |
|---|---|---|
| `echo_api` | 9000 | FastAPI backend (`API.Dockerfile`) |
| `model_server` | 8501 | TensorFlow Serving model server |
| `echo_hmi` | 3000 | Web dashboard (host and container both 3000) |
| `echo_mqtt` | 1883, 7001, 7070 | HiveMQ MQTT broker + control center |
| `echo_store` (Mongo) | 27017 | Primary database |
| `mongo-express` | 8888 (→ container 8081) | Mongo admin UI |
| `echo-redis` | 6379 | Cache/session store |
| IoT `management_application` | 5000 | Current sensor client/server |
| IoT `previous_implementation` | 5001 | Superseded client/server, kept for reference — moved off 5000 to avoid colliding with `management_application` |
| Device onboarding | 8883 | MQTT-over-TLS |

No RabbitMQ is used in this stack — messaging runs over MQTT (HiveMQ), not RabbitMQ.
