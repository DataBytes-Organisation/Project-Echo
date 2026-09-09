# Project Echo Repository Ownership Review (v2)

> **⚠️ Predates the Krish-structure migration.** The paths in this document
> (`src/components/…`, `data/…`, `scripts/…`, flat `docs/…`) describe the
> repository layout *before* the reorganisation. See
> [`MIGRATION_KRISH_STRUCTURE.md`](MIGRATION_KRISH_STRUCTURE.md) for the current
> layout and the old→new path mapping. This file is kept as a historical record
> and should be revised or superseded.

Updated against the actual tracked repository state as of 2026-07-19. This
supersedes `Project_Echo_Repository_Ownership.pdf` at the repo root, whose
paths (`src/Components/`, `Design/`, `Tutorials/`, `src/Prototypes/`,
`src/Echo_Components_on_K8s/`) predate a folder-lowercasing and
reorganisation pass (commits `d6ccfc46`..`e8c1c158`). Kept as Markdown
instead of a PDF so it stays easy to update alongside the code.

## 0. What changed since the original review

The original review's Section 10 priority actions and Section 11 target
layout proposed nesting prototypes under `src/prototypes/` by technical
topic. The team instead took a more aggressive route: **prototype,
research-in-progress, and deployment-variant folders were moved out of the
tracked tree entirely**, into `.data/archive/`, which is git-ignored
(`.data/` in `.gitignore`). They still exist on disk for local reference but
are no longer part of the repository history going forward.

| Original recommendation | Current status |
|---|---|
| Remove `mlruns/` generated output from Git | Done — moved to `.data/mlflow_runs/mlruns`, git-ignored |
| Reorganise `src/Prototypes` by team/topic, keep visible | Superseded — moved wholesale to `.data/archive/Prototypes`, untracked |
| Merge/archive `src/Echo_Components_on_K8s` duplicate API/HMI | Done — moved to `.data/archive/Echo_Components_on_K8s`, untracked |
| Archive `src/Environments`, `src/io`, `src/loadtest` | Done — moved to `.data/archive/{Environments,io,loadtest}`, untracked |
| Move `Tutorials/` under `docs/`, organise by topic | Done — `docs/tutorials/` |
| Consolidate `Design/`, `Requirements/`, root assets | Partially done — `data/design/`, `data/assets/`; `Requirements/` folded into `docs/REQUIREMENTS.md` |
| Relocate installable `Echo/` package | Done — `src/echo/` |
| Move `src/Components/Store` out of the runtime/compose tree | **Not done** — `src/components/store/` still sits alongside the compose services; still not a compose service |
| Lowercase folders | Done | 
| Create CODEOWNERS | Done by this pass — see `.github/CODEOWNERS` |

Net effect: this review is now shorter than the original, because the
biggest source of clutter (the sprawling `Prototypes/` tree) is no longer a
repository-structure problem — it's an on-disk-only archive outside git's
purview. What's left to review is the tracked tree only.

## 1. Repository Boundary and Main Structure

The active runtime is centred on `src/components/`. Everything under
`.data/` is local-only (git-ignored) and out of scope for ownership review;
it's not part of what ships or what reviewers see in a PR.

```
Project-Echo/
├── .github/workflows/        CI/CD (Docker build)
├── .dvc/, .dvcignore         DVC tracking configuration
├── docs/                     documentation, research, tutorials, sprints
│   ├── experiments/          MLflow/DVC experiment scripts
│   ├── research/             literature review, lit review artifacts
│   ├── sprints/               sprint task tracking
│   ├── tasks/                 task briefs
│   └── tutorials/             onboarding notebooks/guides
├── data/                     non-code assets
│   ├── assets/                misc packaged assets
│   ├── design/                branding, diagrams
│   ├── samples/                sample datasets
│   ├── test_fixtures/          fixtures for automated tests
│   └── weather/                weather CSV data
├── scripts/                  standalone dev/analysis scripts
│   ├── benchmarking/
│   ├── training/
│   └── utils/
├── src/
│   ├── components/           main Docker Compose production stack
│   └── echo/                 installable Engine library (near-empty stub currently)
├── requirements.txt, setup.py
├── README.md
└── (.data/  — git-ignored local archive: old Prototypes, K8s fork,
     Environments, io, loadtest, mlruns — not reviewed here)
```

| Path | Purpose | Primary team | Status / action |
|---|---|---|---|
| `.github/workflows/` | Docker build & CI workflow | Backend | Keep |
| `.dvc/`, `.dvcignore` | Data/model versioning config | Engine | Keep; confirm remote storage is configured |
| `docs/` | Documentation, research, sprints, tutorials | Engine / Backend (mixed) | Keep; see 1.1 |
| `data/` | Non-code assets | Split by subfolder | Keep; see 1.1 |
| `scripts/` | Ad-hoc benchmarking/training/utility scripts | Engine (mostly) | Keep but clarify vs `docs/experiments/` — overlapping purpose |
| `src/components/` | Production stack | Split by service | Keep — see Sections 3–7 |
| `src/echo/` | Installable inference package | Engine | Currently just `__init__.py` — confirm whether the real package content still lives in `.data/archive` and needs porting back, or whether this is intentionally a placeholder |
| `requirements.txt`, `setup.py` | Root Python packaging | Engine | Keep |

### 1.1 `docs/` and `data/` breakdown (not covered in the original review)

These didn't exist as top-level folders in the original PDF and need explicit owners:

| Path | Purpose | Suggested owner |
|---|---|---|
| `docs/Engine_Documentation.md` | Engine docs | Engine |
| `docs/Dockerfile_Optimization_Guide.md` | Docker build guidance | Backend |
| `docs/INSTALL.md`, `docs/PORTS.md` | Setup/ops docs | Backend |
| `docs/REQUIREMENTS.md` | System requirements | Shared (Backend maintains, all teams contribute) |
| `docs/experiments/` | MLflow/DVC demo scripts | Engine |
| `docs/research/` | Literature review corpus | Engine |
| `docs/sprints/`, `docs/tasks/` | Sprint/task tracking docs | Shared |
| `docs/tutorials/` | Onboarding material | Shared |
| `data/design/` | Branding, diagrams | HMI |
| `data/assets/` | Misc packaged assets (incl. an HMI submission zip) | HMI |
| `data/samples/`, `data/weather/` | Sample/reference datasets | Engine |
| `data/test_fixtures/` | Audio fixtures for API/engine tests | Backend |

## 2. Main Production Stack — `src/components/`

```
src/components/
├── docker-compose.yml
├── docker-compose.test.yml
├── api/
├── engine/
├── hmi/
├── iot/
├── mongodb/
├── mqtt-server/
├── simulator/
├── store/
└── data_scripts/
```

| Folder | Responsibility | Owner | Runtime classification |
|---|---|---|---|
| `api/` | FastAPI services, auth, routers, DB access | Backend | Active production |
| `engine/` | Audio processing, classification, MQTT integration | Engine | Active production |
| `hmi/` | Node/Express server + browser dashboard | HMI | Active production |
| `iot/edge_inference/` | Field-device TFLite inference + MQTT publishing | IoT | Field production path |
| `iot/management_application/` | IoT management container | IoT | In compose; review integration |
| `iot/2025_t3_prototype/`, `iot/2026_t1_new_onboarding/`, `iot/previous_implementation/` | Trimester prototypes / superseded client | IoT | Not production — candidates to move under `.data/archive` like the rest of the prototype sprawl, or keep with a clear README marking status |
| `mongodb/` | DB image, init/seed data | Backend | Active production |
| `mqtt-server/` | HiveMQ broker config | Backend | Active production infra |
| `simulator/` | Animal/vocalisation simulation | Engine | Active production support |
| `store/` | Offline dataset notebooks + GCS scripts | Engine | **Not a runtime service** — still not moved out despite the original recommendation |
| `data_scripts/` | Movement prediction / vegetation density scripts | Engine | Not a runtime service; standalone analysis scripts, similar status to `store/` |
| `docker-compose*.yml` | Local runtime + CI orchestration | Backend | Authoritative local runtime |

## 3. Engine Team Ownership

```
src/components/engine/
├── echo_engine.py            current runtime (MQTT + API integration)
├── echo_engine.sh, echo_engine.json, models.config
├── Engine.Dockerfile / Engine.test.Dockerfile / Model.Dockerfile
├── models/                   echo_model, weather_model, placeholder
├── yamnet_dir/                YAMNet assets + weights (.h5 in tree — see note)
├── helpers/
├── *.ipynb                    generic/multilabel/optimised pipeline notebooks
└── test_iot_integration.py, test_iot_publisher.py
```

Plus: `src/components/simulator/`, `src/components/store/`,
`src/components/data_scripts/`, `src/echo/`, `docs/Engine_Documentation.md`,
`docs/experiments/`, `docs/research/`, `scripts/benchmarking/`,
`scripts/training/`, `.dvc/`.

| Item | Classification | Action |
|---|---|---|
| `echo_engine.py` | Production | Canonical IoT-integrated runtime |
| `models/` | Production model weights | `.gitignore` already excludes `src/components/engine/models/*` — good, confirm DVC actually tracks these instead of them being untracked-and-missing |
| `yamnet_dir/*.h5`, `.pkl` | Model weights checked into `src/components/` | **Confirmed** — `yamnet.h5` (15MB), `model_2_79.h5` (1.7MB), `model_3_82_16000.h5` (1.7MB), plus two `.pkl` files are real binary blobs committed directly to git (not LFS pointers, not covered by the `engine/models/*` gitignore rule). Move to DVC/LFS |
| `*.ipynb` in `engine/` | Development notebooks in a production folder | Move to `docs/experiments/` or a research area |
| `echo_credentials.json` (engine + simulator) | Tracked JSON with `DB_USERNAME` / `DB_PASSWORD` keys | **Confirmed non-empty values are committed in git** (both copies). Treat as a live secrets leak until a team member confirms these are throwaway local-dev credentials — if real, rotate immediately and scrub git history, then replace with `.env`-based config (there's already an `.env_example` pattern used elsewhere in this repo) |
| `src/echo/` | Installable package | Confirm content — currently only `__init__.py` |

## 4. Backend Team Ownership

```
src/components/api/
├── app/main.py, database.py, schemas.py, serializers.py, detections.py
├── app/middleware/     auth, auth_bearer, pause_guard
├── app/routers/        14 routers: auth, engine, hmi, iot, sim, sensors,
│                        detections, insights, live, projects, public,
│                        species_predictor, two_factor, weather_data, uploads/
├── app/services/       budget, projects, service_state, model_adapter
├── app/utils/
└── backend/             OpenAPI specs, K8s deployment YAML, convert_openapi.py
```

Plus: `src/components/mongodb/`, `src/components/mqtt-server/`,
`src/components/docker-compose*.yml`, `.github/workflows/`,
`docs/PORTS.md`, `docs/INSTALL.md`, `docs/Dockerfile_Optimization_Guide.md`.

| Path | Purpose | Action |
|---|---|---|
| `api/app/main.py` | FastAPI entry point | Keep — note it's git-ignored *and* force-tracked (`.gitignore` lists it as "environment-specific generated file" but `git ls-files` shows it's committed). Worth resolving that contradiction so new contributors aren't confused about whether to edit it |
| `api/app/routers/` | Per-domain API endpoints | Keep; assign secondary reviewers by route (e.g. `iot.py` also needs IoT sign-off, `hmi.py` also needs HMI sign-off) |
| `api/backend/` | OpenAPI specs + a leftover `backend-deployment.yaml`/`backend-service.yaml` (K8s manifests) | The K8s fork was archived to `.data/`, but these two manifest files still live under `api/backend/` — confirm whether they're still needed or are orphaned from the archived K8s deployment |
| `mongodb/init/` | DB init/seed JSON | Keep; review seed data for anything sensitive (e.g. `user-seed.json`, `donations-stripe-seed.json`) |
| `mqtt-server/` | HiveMQ broker image | Keep |

## 5. HMI Team Ownership

```
src/components/hmi/
├── HMI.Dockerfile
├── ai/                bio_master_*.xlsx reference spreadsheets
├── digital_assets/     branding, icons, logos (8 numbered subfolders)
├── recodings/           sample audio-event JSON recordings
└── ui/
    ├── server.js
    ├── config/, controller/, middleware/, model/, routes/
    └── public/
```

Plus: `data/design/`, `data/assets/`.

| Path | Purpose | Action |
|---|---|---|
| `hmi/ui/server.js` | Node/Express entry point | Keep |
| `hmi/ai/`, `hmi/digital_assets/`, `hmi/recodings/` | Reference spreadsheets, branding, demo data | Consider moving non-runtime assets to `data/design/` or `data/assets/` — they currently ship inside the same tree as the Dockerfile that builds the HMI image |
| `hmi/ui/robot.txt` | Likely meant to be `robots.txt` | Typo — fix if this is meant to be served |
| `hmi/ui/Server.bat` | Windows launch script | Keep; low-risk |
| `hmi/ui/node_modules/` | Tracked? | `.gitignore` excludes `node_modules/` — confirm it isn't accidentally tracked |

## 6. IoT Team Ownership

```
src/components/iot/
├── edge_inference/            iot_edge_client.py — primary field path
├── management_application/     client_pi.py, Dockerfile
├── 2025_t3_prototype/           previous trimester prototype
├── 2026_t1_new_onboarding/      onboarding + newer work
└── previous_implementation/     superseded implementation
```

| Path | Purpose | Action |
|---|---|---|
| `iot/edge_inference/iot_edge_client.py` | Field TFLite inference + MQTT | Keep as primary field path |
| `iot/management_application/client_pi.py` | Compose IoT client | Review — same hard-coded-LAN-server concern the original review flagged |
| `iot/2025_t3_prototype/`, `iot/2026_t1_new_onboarding/` | Trimester-scoped work | Keep for now; same "move to archive once superseded" pattern applied to `.data/archive/Prototypes` elsewhere |
| `iot/previous_implementation/` | Superseded implementation, including a `README.me` (typo) | Archive once confirmed unused — this is exactly the kind of folder that already got moved to `.data/archive` for other teams; inconsistent that it's still tracked here |

The Engine↔IoT dependency the original review called out
(`Prototypes/engine/torch_impl/Integrate_EfficientNetV2_Engine/` — PyTorch→ONNX/TFLite
export feeding edge inference) now lives under `.data/archive/Prototypes/engine/torch_impl/`,
i.e. **untracked**. If IoT's edge model conversion pipeline still depends on
that code, it needs to either move back into the tracked tree or be
documented as an external/local-only dependency — right now it's invisible
to anyone who clones the repo fresh.

## 7. Final Ownership Matrix

| Team | Primary tracked folders | Secondary / shared |
|---|---|---|
| Engine | `src/components/engine`, `src/components/simulator`, `src/components/store`, `src/components/data_scripts`, `src/echo`, `docs/Engine_Documentation.md`, `docs/experiments`, `docs/research`, `scripts/benchmarking`, `scripts/training`, `.dvc` | Model contracts consumed by API; IoT edge model exports (currently untracked, see §6) |
| Backend | `src/components/api`, `src/components/mongodb`, `src/components/mqtt-server`, `src/components/docker-compose*.yml`, `.github/workflows`, `docs/PORTS.md`, `docs/INSTALL.md` | Deployment support for all teams |
| HMI | `src/components/hmi`, `data/design`, `data/assets` | HMI API routes implemented by Backend; IoT device-health displays |
| IoT | `src/components/iot` | Engine TFLite/model conversion; Backend API/MQTT |
| Shared | `README.md`, `docs/REQUIREMENTS.md`, `docs/sprints`, `docs/tasks`, `docs/tutorials`, `scripts/utils`, `data/test_fixtures` | All teams |

## 8. Priority Actions (remaining, as of this pass)

- Move `src/components/store/` and `src/components/data_scripts/` out of
  the compose-adjacent tree — still open from the original review.
- Move tracked model weights (`engine/yamnet_dir/*.h5`, `*.pkl`) to DVC/Git
  LFS — the `.gitignore` rule for `engine/models/*` doesn't cover these.
- **`echo_credentials.json` (in both `engine/` and `simulator/`) has
  non-empty `DB_USERNAME`/`DB_PASSWORD` values committed to git.** Confirm
  with the team whether these are real; if so, rotate immediately and scrub
  git history — this is the highest-priority item in this review.
- Resolve the `api/app/main.py` gitignore contradiction (listed as
  "environment-specific generated" but is committed) — decide whether it's
  a template or a real file and document it.
- Confirm whether `api/backend/backend-deployment.yaml` /
  `backend-service.yaml` are orphaned leftovers from the archived K8s fork.
- Apply the same "move to `.data/archive` once superseded" treatment used
  elsewhere to `src/components/iot/previous_implementation/` and the dated
  IoT prototype folders, for consistency.
- Decide whether `.data/archive/Prototypes/engine/torch_impl` (the
  PyTorch→ONNX/TFLite export path IoT edge inference may depend on) needs
  to move back into the tracked tree, since right now a fresh clone doesn't
  have it.
- `.github/CODEOWNERS` has been generated from this document (see repo
  root) — replace the placeholder handles with real GitHub usernames or
  team slugs before it takes effect.

## 9. Note

This is an ownership and organisation review, not a migration plan. Any
file moves should go through reviewed PRs after each team confirms active
dependencies — several tracked paths above (`store/`, `data_scripts/`,
`iot/previous_implementation/`) look like straightforward moves but may
have import paths or Docker build contexts pointing at them.
