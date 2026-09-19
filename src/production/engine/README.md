# Engine

## Generic Engine Pipeline

This is an entry point and sample code for how to build an end to end pipeline from loading raw audio data and generating a model

## Optimised Engine Pipeline

This is a variant of the generic engine pipeline that execute a parallel pipeline - this is ready for producing the production model.

## Docker

For step-by-step R2 setup, `.env` configuration and local acceptance testing,
see the [R2 deployment instructions](../../deployment/docker/README.md).

The engine and simulator now read species from the same private Cloudflare R2
dataset. In `src/deployment/docker/.env`, configure `R2_ACCOUNT_ID`,
`R2_BUCKET_NAME`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and
`R2_DATASET_PREFIX`. Keep the other settings, including `ENGINE_API_KEY`.
Use bucket-scoped Object Read only S3 credentials. For
`project-echo-simulator-prod/<scientific-species-name>/<audio-file>`, leave the
prefix empty. Never commit secrets or enable public bucket access.

For the full application, use the existing Compose stack. From
`src/deployment/docker`:

```console
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build echo_engine echo_simulator
```

Once the rest of the application is running, apply these two images using
`docker compose -f docker-compose.yml up -d --no-deps echo_engine echo_simulator`.
Both services receive the same R2 settings. GCP login and its credential mount
are no longer needed for either service. Inference, model class mappings,
MQTT payloads and backend delivery are unchanged.

For manual builds below, the shared R2 build context requires Docker BuildKit.
Compose builds require Docker Compose 2.17 or newer.

***NOTE Before running this, please ensure you have trained a model and placed it in models/echo_model/1/***

Run these steps to execute the engine in docker

### Setup volumes

```
docker volume create myvolume
```

### Setup network

```
docker network create --driver bridge echo-net
```

### Model Server

```
docker build --file Model.Dockerfile -t ts-echo-model .
```

```
docker run -p 8501:8501 --name ts-echo-model-cont --network echo-net -d ts-echo-model
```

### Echo Engine

```
docker build --build-context r2_storage=../infrastructure/store/cloudflare_r2 --file Engine.Dockerfile -t ts-echo-engine .
```

```
docker run --name ts-echo-engine-cont -it --rm --network echo-net -e R2_ACCOUNT_ID -e R2_BUCKET_NAME -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e R2_DATASET_PREFIX -e ENGINE_API_KEY ts-echo-engine
```

For manual `docker run`, export the listed variables in the terminal first;
unlike Compose, it does not read the deployment `.env` automatically.
