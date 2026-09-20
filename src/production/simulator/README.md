# Simulator

Owned by the simulation team. Generates synthetic sensor/microphone
network activity and animal movement events, sending them to the API
the same way real IoT sensors do.

- `src/` — simulator engine (clock, entities, factories, sensor/comms managers).
- `Simulator.Dockerfile` — container build.
- `requirements.txt` — Python dependencies for this service.

## Cloudflare R2 audio storage

For the complete setup and test sequence, see the
[R2 deployment instructions](../../deployment/docker/README.md).

Animal Mode reads the migrated simulator dataset from the private
`project-echo-simulator-prod` bucket. Preserve the original object layout:
`<scientific-species-name>/<audio-file>`. The audio bytes, MQTT topic and message
fields are unchanged. Recording Mode continues forwarding incoming recordings.
The engine reads its species list from the same R2 configuration. Prediction
logic, backend behaviour and training tools are unchanged.

Set these values in `src/deployment/docker/.env`, keeping its other settings:

```dotenv
R2_ACCOUNT_ID=your-account-id
R2_BUCKET_NAME=project-echo-simulator-prod
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_DATASET_PREFIX=
```

Use Object Read only S3 credentials restricted to that bucket. Keep the bucket
private and never commit credentials. Leave the prefix empty for species folders
at the bucket root. For non-Docker runs, export these variables in the process
environment; the simulator does not automatically read the deployment `.env`.

From `src/deployment/docker`, validate and rebuild just the simulator:

```console
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build echo_simulator
docker compose -f docker-compose.yml run --rm --no-deps echo_simulator python -c "from comms_manager import CommsManager; m=CommsManager(); s=m.r2_load_species_list(); k=next(iter(m.audio_keys.values()))[0]; a=m.r2_storage.download_bytes(k); print('Simulator R2 read: PASS;', len(s), 'species;', len(a), 'bytes')"
```

The read check does not start the simulation or publish messages. With the rest
of the stack already running, apply the new image with
`docker compose -f docker-compose.yml up -d --no-deps echo_simulator`, then use
the existing simulator controls. Rebuild and recreate `echo_engine` too when
applying the engine migration. Other services retain their existing configuration.

For the standalone simulator Compose file, pass
`--env-file ../../deployment/docker/.env` when running from this directory.
The build uses an additional context for the shared R2 package and requires
Docker Compose 2.17 or newer.

From the repository root, run the offline regression tests after installing
the shared adapter's `requirements.txt`:

```console
python -m unittest discover -s src/tests/unit/storage -p "test_*r2.py" -v
```
