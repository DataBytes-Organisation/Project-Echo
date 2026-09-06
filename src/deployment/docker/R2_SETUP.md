# Cloudflare R2 production integration

The engine and simulator use the same private R2 bucket. Objects must retain the
existing Project Echo layout:

```text
<scientific-species-name>/<audio-file>
```

If `R2_DATASET_PREFIX` is set, the layout is nested below that prefix:

```text
<prefix>/<scientific-species-name>/<audio-file>
```

## Configure locally

1. Create an R2 API token with **Object Read only** permission for the dataset
   bucket. Do not commit or send the token through chat.
2. In `src/deployment/docker`, copy `.env_example` to `.env` if `.env` does not
   already exist.
3. Set `R2_ACCOUNT_ID`, `R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, and
   `R2_SECRET_ACCESS_KEY` in `.env`.
4. Leave `R2_DATASET_PREFIX` empty when species folders are at the bucket root.
   Otherwise set it to the parent prefix without a trailing slash.

## Run the production flow

From `src/deployment/docker`:

```powershell
docker compose up -d --build
docker compose ps
docker logs -f ts-echo-engine-cont
```

In a second terminal, initialise the simulator:

```powershell
docker exec -it ts-simulator-cont sh
cd /app
python system_manager.py
```

In a third terminal, start simulator control:

```powershell
docker exec -it ts-simulator-cont sh
cd /app
python control_sim.py
```

The engine log should show species loaded from Cloudflare R2 followed by model
predictions. To verify the final database write, inspect the API log:

```powershell
docker logs -f ts-api-cont
```

The expected end-to-end path is R2 audio to Simulator to MQTT to Engine to
TFLite inference to Backend API to MongoDB.
