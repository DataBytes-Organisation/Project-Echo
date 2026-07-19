# Simulator

Owned by the simulation team. Generates synthetic sensor/microphone
network activity and animal movement events, sending them to the API
the same way real IoT sensors do.

- `src/` — simulator engine (clock, entities, factories, sensor/comms managers).
- `Simulator.Dockerfile` — container build.
- `requirements.txt` — Python dependencies for this service.
