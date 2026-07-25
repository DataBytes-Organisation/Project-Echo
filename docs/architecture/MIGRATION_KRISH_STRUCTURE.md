# Repository Migration — Krish's Structure

This repository was reorganised from the previous `src/components/` layout
(Holan's structure) to **Krish's proposed structure** — a `production/` vs
`prototypes/` split under `src/`, with deployment, tests, data tooling, docs,
tutorials, and models/data separated out. Source of the target layout:
`Project_Echo_Three_Structures_Only.pdf` (Krish's structure, p. 3).

All moves used `git mv`, so file history is preserved. Nothing was deleted.

## Path mapping (old → new)

### Source components → `src/production/`
| Old | New |
| --- | --- |
| `src/components/api/` | `src/production/backend/` |
| `src/components/engine/` | `src/production/engine/` |
| `src/components/hmi/` | `src/production/hmi/` |
| `src/components/iot/` | `src/production/iot/` |
| `src/components/simulator/` | `src/production/simulator/` |
| `src/components/mongodb/` | `src/production/infrastructure/mongodb/` |
| `src/components/mqtt-server/` | `src/production/infrastructure/mqtt-server/` |
| `src/components/store/` | `src/production/infrastructure/store/` |
| `src/echo/` | `src/production/infrastructure/echo/` |

### Data tooling → `src/data_tools/`
| Old | New |
| --- | --- |
| `src/components/data_scripts/` | `src/data_tools/data_scripts/` |
| `scripts/utils/` | `src/data_tools/utils/` |
| `scripts/training/` | `src/data_tools/training/` |

### Prototypes → `src/prototypes/`
| Old | New |
| --- | --- |
| `scripts/benchmarking/` | `src/prototypes/engine/benchmarking/` |

### Deployment → `src/deployment/`
| Old | New |
| --- | --- |
| `src/components/docker-compose.yml` | `src/deployment/docker/docker-compose.yml` |
| `src/components/docker-compose.test.yml` | `src/deployment/docker/docker-compose.test.yml` |
| `src/components/.dockerignore`, `.env_example`, `dockerScript.py`, `package.json`, `package-lock.json`, `*.png` | `src/deployment/docker/` |

### Docs / tutorials / data
| Old | New |
| --- | --- |
| `docs/Engine_Documentation.md`, `docs/PORTS.md`, `docs/Dockerfile_Optimization_Guide.md`, `docs/REPOSITORY_OWNERSHIP.md` | `docs/architecture/` |
| `docs/INSTALL.md`, `docs/REQUIREMENTS.md`, `docs/sprints/`, `docs/tasks/` | `docs/team-guides/` |
| `docs/experiments/` | `docs/research/experiments/` |
| `docs/tutorials/` | `tutorials/` (top level) |
| `data/` (samples, weather, design, assets, test_fixtures) | `models-and-data/` |
| `src/components/README.md` | `src/production/README.md` |

## Net-new placeholder folders (no prior content)

Created empty with a `README.md` placeholder — see Krish's diagram:
`src/prototypes/{backend,hmi,iot,simulator,computer_vision}`,
`src/prototypes/engine/{pytorch,augmentation,event_detection,weather_noise,ensemble_overlap}`,
`src/deployment/kubernetes`, `src/tests`.

## Reference updates applied

- **`.github/CODEOWNERS`** — every path rule remapped; added broad
  `production/prototypes/data_tools/deployment/tests` tree defaults (placed
  above the specific rules so those still override; GitHub uses last-match-wins).
  Handle TODOs (`@first-last`) are unchanged and still need real GitHub usernames.
- **`src/deployment/docker/docker-compose.yml` + `.test.yml`** — build
  `context:` and volume mounts rewritten to `../../production/...`. Validated
  with `docker compose config` (contexts resolve).
- **`.github/workflows/docker-image.yml`** — path triggers `src/**/*`;
  `cd src/deployment/docker` for compose commands.
- **`setup.py`** — `package_dir` now `src/production/infrastructure` (the `echo`
  package's new home).
- **`.gitignore`** — the four `src/components/...` ignore rules remapped.
- **`README.md`** — "Full Directory Tree" rewritten to Krish's layout.
- **Markdown prose** across the repo swept for old path strings.

## Follow-ups / decisions to confirm

1. **Docker/K8s build test.** `docker compose config` passes, but a full
   `docker compose -f src/deployment/docker/docker-compose.test.yml build` was
   not run here. Validate before relying on CI.
2. **`iot/` kept whole under `production/`.** The IoT component contains both
   live and legacy/prototype subfolders (`2025_t3_prototype`,
   `previous_implementation`, etc.). Splitting these into `production/iot` vs
   `prototypes/iot` needs IoT-team input; left intact for now.
3. **`models-and-data/` is still git-tracked.** Krish's structure intends this
   for externally/DVC-tracked assets ("not normal Git"). The git-tracked
   samples/fixtures/design were moved here as-is; moving them to DVC/LFS is a
   follow-up.
4. **Embedded Kubernetes manifests** (`*-deployment.yaml`, `*-service.yaml`)
   still live beside their components. `src/deployment/kubernetes/` exists as
   their intended home; consolidating them (and rewiring any relative refs) is
   a follow-up.
5. **`docs/architecture/REPOSITORY_OWNERSHIP.md`** describes the *pre-migration*
   `src/components/` layout and its own reorg suggestions; its path references
   were intentionally left unchanged so it reads as a historical record. It
   should be revised or superseded by this document.
