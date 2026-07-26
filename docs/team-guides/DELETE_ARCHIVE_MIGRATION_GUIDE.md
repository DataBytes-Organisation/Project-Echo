# Emptying `.delete/` - Team Migration Guide

## What `.delete/` is

`.delete/` is a **frozen snapshot of the old repository layout**, captured during
the restructure into the new team-based structure (Krish's structure). It is the
former `src/Archive` tree: old `src/Components`, `Prototypes/`,
`Echo_Components_on_K8s/`, `Environments/`, `io/`, `loadtest/`, etc.

It was kept **because it may contain work that's still important to someone** - we
didn't want to delete anything blind. But it's ~3.7 GB and doesn't belong in the
repo long-term. So instead of one person guessing, **everyone moves their own
work out of `.delete/` into its correct home in the new structure.** You know your
files better than anyone - that's why this is distributed.

**Goal:** empty `.delete/archive/` by relocating each file into an *existing*
folder. When it's empty, we delete `.delete/` in a final commit.

---

## Hard rules (read before you move anything)

1. **Do NOT create new folders.** Move files only into folders that **already
   exist** in the new structure (see the map below). If nothing fits, **ask in
   the thread** - don't invent a folder.
2. **Do NOT overwrite existing files.** Some content was already migrated. If a
   filename clashes at the destination, **check which is newer** and rename yours
   (e.g. `_archive` suffix) rather than clobbering.
3. **Use `git mv`, not delete-and-re-add** - it preserves file history.
4. **Delete regenerable junk instead of moving it** (see "Do not migrate" below).
5. **Don't dump large binaries into git.** Trained models, audio, big image sets
   → coordinate with your lead for `models-and-data/` (DVC), don't `git mv` them
   into a code folder. Flag them in the thread.
6. **Claim your area in the thread first** (e.g. "taking `Prototypes/engine/
   augmentation`") so two people don't move the same folder and collide.

---

## Where things go (old archive area → EXISTING folder)

All source paths are under `.delete/archive/`.

| Archive area | Move into (already exists) | Notes |
|---|---|---|
| `Echo_Components_on_K8s/` | `src/deployment/kubernetes/` | K8s manifests, configMaps/secrets |
| `Environments/` | `src/deployment/` | env `.yaml`/`.txt`/`.bat` configs; the setup `README.md` → `docs/team-guides/` |
| `io/` | `src/data_tools/` | data export scripts; the `.csv`/`.json` data → `models-and-data/` |
| `loadtest/` | `src/tests/` | load / performance tests |
| `Prototypes/api/` | `src/prototypes/backend/` | |
| `Prototypes/Computer Vision/` | `src/prototypes/computer_vision/` | |
| `Prototypes/data/` | `src/data_tools/data_scripts/` | ⚠ `GoogleCloud_download.ipynb` already lives in `src/production/infrastructure/store/` - check before moving |
| `Prototypes/eda/` | `src/prototypes/engine/` | exploratory analysis; pure write-ups → `docs/research/` |
| `Prototypes/engine/` | `src/prototypes/engine/` | use themed subfolders - see engine table |
| `Prototypes/hmi/` | `src/prototypes/hmi/` | |
| `Prototypes/Iot/` | `src/prototypes/iot/` | |
| `Prototypes/R and D/` | `src/prototypes/hmi/` | "Project Echo Website" = frontend prototype |
| `Prototypes/sim/` | `src/prototypes/simulator/` | |

### Engine prototypes → themed subfolders (all already exist)

`src/prototypes/engine/` has pre-made themed subfolders. Match your work to one:

| Old engine subfolder(s) | Move into |
|---|---|
| `torch_impl/` | `src/prototypes/engine/pytorch/` |
| `AUGMENTATION PROTOTYPES/`, `Augmentation Tasks/`, `Audio Augmentation Comparison/`, `AudioAugmentationProbabilityTask/` | `src/prototypes/engine/augmentation/` |
| `Event Detection Tasks/`, `Completed_Event_Segmenter/`, `Event_Segmentation_YamNet/`, `Event optimizing - ND/`, `Course Detection - ND/` | `src/prototypes/engine/event_detection/` |
| `Benchmarking_and_Experimentation/`, `Testing on Real World Data/` | `src/prototypes/engine/benchmarking/` |
| `WeatherDetection/`, `Weather Detection - ND/`, `Noise_detection/`, `Removing backgroung noise tasks/` | `src/prototypes/engine/weather_noise/` |
| `Overlapping sound/`, `Working with overlapping audio/`, `Combining models pipeline/`, `Clustering/`, `Unsupervised Classification Methods/` | `src/prototypes/engine/ensemble_overlap/` |
| **Everything else engine** (Transfer Learning, Car Horn models, Visualization/CAM, yamnet, Integration_Demo, Final Trained Echo Model, …) | `src/prototypes/engine/` (root) - only drop into a themed subfolder if it *clearly* fits |

> Unsure which theme? Put it at `src/prototypes/engine/` root and note it in the
> thread - **do not create a new subfolder.**

---

## Do NOT migrate - delete these instead (regenerable)

These are pure clutter; `rm -r` them rather than moving:

- `**/node_modules/` - regenerate with `npm install`
- `**/__pycache__/`, `**/.ipynb_checkpoints/`, `*.pyc`, `*.map` - caches/build
- `**/mlflow_runs/` - generated experiment runs
- Bulk `*.npy` feature caches - regenerate from audio in preprocessing

If you're unsure whether something is regenerable, **ask before deleting.**

---

## How to move your area (step by step)

```bash
# 0. Claim your area in the thread, then pull latest.

# 1. Check the destination doesn't already have it (avoid overwrite):
ls "src/prototypes/engine/augmentation/"

# 2. Move with history preserved (quote paths with spaces):
git mv ".delete/archive/Prototypes/engine/Augmentation Tasks" \
       "src/prototypes/engine/augmentation/"

# 3. Delete junk in your area instead of moving it:
rm -r ".delete/archive/Prototypes/hmi/ui/node_modules" 2>/dev/null

# 4. Commit small and per-area with a clear message:
git commit -m "chore(archive): move engine augmentation prototypes into src/prototypes/engine/augmentation"

# 5. Push / open a PR per the team's normal flow.
```

**Name clash?** Don't overwrite - rename yours:
`git mv ".delete/archive/.../pipeline.ipynb" "src/prototypes/engine/pipeline_archive.ipynb"`

---

## Done?

When `.delete/archive/` is empty (`find .delete -type f` returns nothing), post in
the thread. A lead will remove `.delete/` in a final commit and confirm the repo
is clean.

**Questions or "where does X go?" → ask in the thread. Never create a new folder
to make something fit.**
