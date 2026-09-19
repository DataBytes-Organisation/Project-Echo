# Cloudflare R2 engine and simulator storage

For dashboard setup, private credentials, `.env` configuration and the full
application test, use the [R2 deployment instructions](../../../../deployment/docker/README.md).

The production engine and simulator use this package for dataset storage.
Configure both with the same bucket and prefix in the deployment `.env`; see
`../../../engine/README.md` and `../../../simulator/README.md`. A private
`project-echo-simulator-prod` bucket with an empty prefix preserves the original
`<scientific-species-name>/<audio-file>` layout. Both services need only
bucket-scoped Object Read only S3 credentials. The prototype helpers below
remain available for isolated storage checks.

- Engine: list sorted species names from the first dataset directory.
- Simulator: group objects by species, randomly select an audio object, and
  download its bytes.
- Shared storage: connect, paginate listings, upload, and download.

The integration preserves existing MQTT payloads and audio bytes; it does not
change inference, backend validation or database behaviour.

## Optional prototype demo

Create a private Standard R2 bucket, leave public access disabled, and create a
token with Object Read & Write access restricted to the prototype bucket. Set
the variables shown in `.env.example` in the local environment. Never commit
real credentials. `R2_DATASET_PREFIX` defaults to an empty string (bucket root).
Set it explicitly to `prototype` only when using the example prototype layout
below. Production engine and simulator access needs only Object Read only
credentials.

Objects must use this layout:

```text
prototype/<species name>/<audio file>
```

Install `requirements.txt`, then run from the parent `store` directory:

```powershell
python -m cloudflare_r2.demo
```

Expected final output:

```text
R2 connection: PASS
Engine species loading: PASS (... species)
Engine/simulator species agreement: PASS
Simulator random audio download: PASS (...)
Prototype integration: PASS
```

The demo alone does not verify the full application. Validate the real engine
and simulator before retiring GCP storage; other datasets and GCP services are
outside this runtime integration.
