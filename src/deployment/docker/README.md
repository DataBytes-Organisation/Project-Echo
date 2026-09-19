# Run the engine and simulator with Cloudflare R2

This guide covers the production engine/simulator storage integration, not the
old prototype demo. Commands below are for Windows **Command Prompt (CMD)**,
including a CMD terminal in VS Code. Use your own repository location.

## 1. What changes and what stays the same

- The engine reads its sorted species list from R2.
- The simulator lists the same species, downloads their audio from R2 and sends
  the existing audio messages to the engine through MQTT.
- Both services use the same bucket, prefix and shared S3-compatible client.
- Their Google storage dependencies, authentication step and credential mounts
  are replaced by R2 configuration.
- Model inference, MQTT message fields, backend validation and database logic
  are unchanged. Other GCP datasets/services are outside this integration.

Keep the migrated dataset structure exactly as it was in `project_echo_bucket_1`:

```text
project-echo-simulator-prod/
  Gymnorhina tibicen/
    recording-001.wav
  <another-scientific-species-name>/
    <original-audio-filename>
```

The names above illustrate the layout. Preserve the real species names and
filenames, including spelling and case. Do not add `prototype/` or the old GCP
bucket name as an extra directory. A folder named `sample.wav/` is not audio;
there must be an actual uploaded file. This guide does not migrate or delete data.

## 2. Prepare the private R2 bucket

Skip creation if the approved bucket and dataset already exist.

1. Sign in to the [Cloudflare dashboard](https://dash.cloudflare.com/).
2. Select the approved project account.
3. Open **Storage & databases > R2 Object Storage > Overview**.
4. If R2 is not enabled, ask the account owner to complete checkout. If payment
   details are requested, use the approved billing arrangement; do not assume
   that all usage will be free. See [Cloudflare's R2 setup guide](https://developers.cloudflare.com/r2/get-started/).
5. Select **Create bucket** and enter `project-echo-simulator-prod`.
6. Choose **Standard** storage. The current adapter uses the default R2 endpoint,
   so use the normal **Automatic** location option without a restricted
   jurisdiction. If project policy requires a restricted jurisdiction, stop and
   arrange the corresponding endpoint support before connecting this code.
7. Select **Create bucket**. See [bucket creation instructions](https://developers.cloudflare.com/r2/buckets/create-buckets/).
8. Open the bucket's **Settings**. Keep **Public Development URL** disabled and
   do not attach a public custom domain. These are separate public-access routes;
   disabling only the development URL is not sufficient if a public domain is
   attached. See [Cloudflare public-access settings](https://developers.cloudflare.com/r2/buckets/public-buckets/).
9. Confirm the **Objects** tab contains the migrated species folders and audio.

### Create the runtime credentials

1. Return to R2 **Overview > Account Details > API Tokens > Manage**.
2. Select **Create Account API token** if your account role permits it, or use
   the project-approved user token option.
3. Name it `project-echo-runtime-read`.
4. Select **Object Read only**.
5. Select **Apply to specific buckets only** and choose
   `project-echo-simulator-prod`.
6. Apply the team's expiry/IP policy, then create the token.
7. Save its **Access Key ID** and **Secret Access Key** securely. Use these S3
   credentials, not the general API token value. Copy the **Account ID** from
   the same account's R2 overview.

A separate migration token may use **Object Read & Write** for this bucket.
Do not put that token in the running project; revoke it only after migration
and verification are complete. Neither task needs **Admin Read & Write**.
See [Cloudflare R2 authentication](https://developers.cloudflare.com/r2/api/tokens/).

## 3. Open your local configuration

1. Open the repository in VS Code.
2. Select **Terminal > New Terminal** and choose **Command Prompt** from the
   terminal profile menu.
3. Replace `YOUR_PROJECT_ECHO_FOLDER` below with your actual checkout folder:

```cmd
cd /d "YOUR_PROJECT_ECHO_FOLDER\src\deployment\docker"
```

4. If `.env` does not exist, create it from the safe template:

```cmd
if not exist .env copy .env_example .env
```

5. Open it locally:

```cmd
notepad .env
```

Do not overwrite an existing `.env` with the example. Keep the existing database,
authentication, mail and other project settings. A five-line R2-only file is not
a complete configuration for the whole application.

## 4. Add or update only the R2 settings

Keep exactly one entry for each variable:

```dotenv
R2_ACCOUNT_ID=YOUR_ACCOUNT_ID
R2_BUCKET_NAME=project-echo-simulator-prod
R2_ACCESS_KEY_ID=YOUR_READ_ONLY_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY=YOUR_READ_ONLY_SECRET_ACCESS_KEY
R2_DATASET_PREFIX=
```

Replace the three credential/account placeholders privately. Do not replace
working values with placeholders. Save the file.

| Setting | What to enter |
| --- | --- |
| `R2_ACCOUNT_ID` | The account containing the bucket, not an email or zone ID. |
| `R2_BUCKET_NAME` | The approved bucket name, without a URL or slash. |
| `R2_ACCESS_KEY_ID` | Access Key ID for the bucket-scoped runtime token. |
| `R2_SECRET_ACCESS_KEY` | That same token's Secret Access Key. |
| `R2_DATASET_PREFIX` | Empty for species folders at the bucket root. |

Plain `NAME=value` works for these R2 values; quotes are not required. The client
constructs `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` and uses region `auto`.
There is no separate endpoint variable in the current implementation.

Do not paste keys into Python files, Dockerfiles, README files, commits, chat or
screenshots. Use the team's approved secret-sharing method or configure them
directly on the deployment host. The local `.env` is ignored by Git; only
`.env_example` should be shared.

If switching from a personal R2 account to the project account, replace all three
account/credential values together, verify the destination dataset, and recreate
both services. Copying configuration alone does not copy the dataset.

If you previously set R2 variables directly in CMD, clear those overrides in
this terminal so Compose reads the file:

```cmd
set R2_ACCOUNT_ID=
set R2_BUCKET_NAME=
set R2_ACCESS_KEY_ID=
set R2_SECRET_ACCESS_KEY=
set R2_DATASET_PREFIX=
```

Shell variables can override values from an env file. See
[Docker's environment-file rules](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/).

## 5. Validate and build

Open Docker Desktop and wait for its engine to run. In the same CMD terminal:

```cmd
docker info
docker compose --env-file .env -f docker-compose.yml config --quiet
docker compose --env-file .env -f docker-compose.yml build echo_engine echo_simulator
```

Stop if a command fails. `config --quiet` checks Compose syntax, not whether
credentials work or whether all application-specific settings are populated.
Do not share plain `docker compose config` output: it can contain resolved secrets.

The additional build context packages the shared R2 client into both images.
Keep it in the build; copying only engine/simulator source is not sufficient.
Use `docker-compose.yml` consistently for this guide. The test Compose file
uses a different engine base image but the same image tag; do not mix builds
from the two files while diagnosing a run.

## 6. Verify R2 without starting the simulation

Check the real simulator's listing and download path:

```cmd
docker compose --env-file .env -f docker-compose.yml run --rm --no-deps -T echo_simulator python -B -c "from comms_manager import CommsManager; m=CommsManager(); species=m.r2_load_species_list(); key=next(iter(m.audio_keys.values()))[0]; audio=m.r2_storage.download_bytes(key); print('Simulator R2 read: PASS;', len(species), 'species;', len(audio), 'bytes')"
```

Check the real engine's R2 species-loading method without starting MQTT or its
database constructor (importing the engine still loads its existing model assets):

```cmd
docker compose --env-file .env -f docker-compose.yml run --rm --no-deps -T echo_engine python -B -c "from echo_engine import EchoEngine; engine=EchoEngine.__new__(EchoEngine); species=engine.r2_load_species_list(); print('Engine R2 species: PASS;', len(species), 'species')"
```

Both commands must succeed and report the same species count. Counts depend on
the dataset; they need not equal the model's class count. These are storage
checks, not proof of predictions reaching the database.

## 7. Apply the images and check readiness

Use an approved local development database, not production. Ensure the existing
MongoDB, MQTT and backend services have their normal working configuration.
If those services are stopped:

```cmd
docker compose --env-file .env -f docker-compose.yml up -d echo_store echo_mqtt echo_api
```

Apply the updated engine and simulator:

```cmd
docker compose --env-file .env -f docker-compose.yml up -d --no-deps echo_engine echo_simulator
docker compose --env-file .env -f docker-compose.yml ps -a
docker compose --env-file .env -f docker-compose.yml logs --tail 100 echo_engine echo_simulator echo_api
```

These commands can recreate containers to apply new settings; they do not delete
database volumes. `restart` alone does not apply changed container environment
values. Check for all of the following before continuing:

- Engine: the configured TFLite model loaded, species retrieved from Cloudflare
  R2, and `Engine waiting for audio to arrive...`.
- Simulator: `Connected... waiting for start command`.
- Backend: `Application startup complete`, with no startup traceback.

The engine's separate IoT listener retains its existing broker configuration;
R2 does not replace MQTT. If startup is waiting on that listener, diagnose its
connection separately rather than changing the R2 bucket or credentials.

## 8. Run the existing 30-second local simulation

Only run this after readiness passes and the local test database is approved.
The existing simulator adds movements/detections and **replaces the microphone
collection during initialization**. Its backend movement route can also invoke
notifications. Use disposable test data and a notification-safe development
setup. Do not run it against data or recipients that must remain untouched.

```cmd
docker compose --env-file .env -f docker-compose.yml exec echo_simulator python control_sim.py
```

The controller requests `Animal_Mode`, waits about 30 seconds, then requests
`Stop`. Let it finish. If it is interrupted before Stop, send Stop explicitly:

```cmd
docker compose --env-file .env -f docker-compose.yml exec echo_simulator python -c "import paho.mqtt.publish as p; p.single('Simulator_Controls', 'Stop', hostname='echo_mqtt', port=1883)"
```

Inspect the recent local logs (redact sensitive content before sharing):

```cmd
docker compose --env-file .env -f docker-compose.yml logs --since 5m echo_simulator echo_engine echo_api
```

Confirm an audio message was sent, the engine produced a species/confidence, and
the backend accepted `POST /engine/event` with HTTP 201. In the local MongoDB
viewer, confirm a newly inserted event in the configured database's `events`
collection, with a new `_id` and `sourceType=simulator`. Do not mistake old seed
records for this run. A storage PASS alone is not full end-to-end acceptance.

Do not use `down -v`, factory reset, volume deletion or `--remove-orphans` to
solve an R2 connection problem. Do not retire GCP until the approved acceptance
checks and any separately scoped dataset migration are complete.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Docker Server section fails | Start Docker Desktop; do not reset/delete its data. |
| Missing R2 configuration | Check all five R2 entries and which env file/terminal is being used. |
| Access denied or signature error | Check account, bucket, matching S3 key pair, token validity and scope. Do not make the bucket public. |
| No species audio found | Use an empty prefix for the root layout and upload actual nonempty audio files, not folder markers. |
| `No module named cloudflare_r2` | Rebuild with the provided Compose file and its shared build context. |
| Backend missing `USER_MONGODB_URI` or `JWT_SECRET` | The existing backend configuration is incomplete. Obtain its approved settings separately; changing R2 cannot fix this. |
| Backend rejects an event | Capture the status/error and coordinate with the backend owner; do not alter unrelated schemas as part of storage migration. |
| Obsolete Compose `version` warning | It is not an R2 authentication failure. Check the actual command exit/error. |
| Boto3 Python 3.9 deprecation warning | The simulator retains its existing Python version. A runtime upgrade is a separate task. |

## Offline R2 regression tests

From the repository root, in your Python test environment:

```cmd
python -m pip install -r src/production/infrastructure/store/cloudflare_r2/requirements.txt
python -m unittest discover -s src/tests/unit/storage -p "test_*r2.py" -v
```

If `python` is not on PATH, substitute your installed Python executable. These
tests use fake credentials and mocked storage/messaging; they do not upload
data or run the simulation. Engine regression tests additionally require the
existing engine test dependencies and fixtures.
