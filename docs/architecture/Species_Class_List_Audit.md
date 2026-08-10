# Species / Class List Inconsistency Audit

**Author**: Nolan Nguyen (Engine team) — produced alongside the Model
Provenance Audit (`Model_Provenance_Audit.md` in this folder) as a
follow-up, not a required Sprint 1 deliverable.

**Status**: Findings only, verified by direct comparison of the files
below. No production files were changed.

## Why this exists

The Model Provenance Audit found that the container actually deployed in
production defaults to the TFLite model (Candidate B), not the TF-Serving
`echo_model` (Candidate A) the docs describe. That raises an immediate
follow-up question: does the species list the rest of the system uses to
label detections actually match the model that's running? This document
answers that with a direct comparison, not an assumption.

## The four species lists found in this repo

| Source | File | Species count |
|---|---|---|
| Production static list | `src/production/engine/class_names.json` | **21** |
| TFLite model's own mapping (Candidate B) | `src/production/engine/models/efficientnetv2/class_mapping.json` | **123** |
| Real local dataset checked out on this machine | `src/prototypes/data_files/` (folder names) | **128** |
| GCS bucket documentation | `.delete/archive/Prototypes/engine/torch_impl/docs/dataset_and_setup.md` | **118** (documented, not independently re-verified in this audit) |

A fifth, *dynamic* source also exists: `echo_engine.py`'s
`gcp_load_species_list()` function pulls species names from a live GCS
bucket folder listing at runtime rather than reading a static file, so the
list the legacy engine actually uses can change without any code or config
change in this repo at all.

## Finding 1 — The production species list barely matches the model that's actually deployed

Direct set comparison (case-sensitive, exact string match):

- `class_names.json` (21) vs TFLite `class_mapping.json` (123): only **5
  species in common**.
- Allowing for capitalisation differences only (case-insensitive
  comparison): **8 species in common** — i.e. 3 more matches
  (`Capra hircus`/`Capra Hircus`, `Felis catus`/`Felis Catus`,
  `Sus scrofa`/`Sus Scrofa`) are the *same* species written with
  inconsistent capitalisation across the two files.
- Even generously, that's **8 of 21 (38%)** of the production species list
  actually represented in the model the Dockerfile defaults to serving.
  The other 13 entries in `class_names.json` correspond to no class the
  TFLite model can output, and conversely the TFLite model can output ~115
  species that `class_names.json` has no record of.

Species present in `class_names.json` with **no match at all** in the
TFLite model's class list (16 of 21):

```
Alectura lathami, Anas gracilis, Apus pacificus, Canis lupus dingo,
Caprimulgus macrurus, Centropus phasianinus, Chrysococcyx minutillus,
Eudynamys orientalis, Geopelia placida, Leipoa ocellata,
Menura novaehollandiae, Phasianus colchicus, Uperoleia laevigata
```
(`Capra hircus`, `Felis catus`, `Sus scrofa` are excluded from this list —
see the capitalisation note above.)

**Practical implication**: if any downstream code path uses
`class_names.json` to label output from the TFLite model (or vice versa),
roughly 6 in 10 labels would be either wrong or unresolvable. This wasn't
tested end-to-end as part of this audit — it's a structural risk found by
comparing the files, not a confirmed runtime bug — but it's a direct
consequence of Finding 1 in the Model Provenance Audit and should be
checked before anyone relies on `class_names.json` alongside the TFLite
model path.

## Finding 2 — The real local dataset is a near-superset of the TFLite model's classes, using a different naming convention

- All 123 TFLite classes are present in the local 128-species dataset
  (100% match).
- The 5 extra species present locally but not in the TFLite list use
  **underscore-separated** names instead of the space-separated
  `"Genus species"` format everywhere else:

  ```
  Asio_flammeus, Branta_bernicla_nigricans, Horornis_diphone,
  Meleagris_gallopavo, Spilopelia_chinensis
  ```

  This is very likely a real overlap that's only invisible to naive exact
  string matching — e.g. `Spilopelia_chinensis` (local, underscore) is
  almost certainly the same species as `Spilopelia chinensis`, which
  already appears correctly space-separated elsewhere in the TFLite list.
  Any code that joins these lists by exact string equality will silently
  miss this species.

## Finding 3 — Five entries in the TFLite class list are not species names

`class_mapping.json`'s 123-entry list includes 5 short, lowercase,
single-word entries that don't match any recognisable binomial species
name format used everywhere else in the file:

```
brant, jabwar, sheowl, spodov, wiltur
```

These look like corrupted, truncated, or placeholder labels rather than
real species (for comparison, every other one of the 123 entries follows
`"Genus species"` capitalised format). They are **not** empty/unused
classes, though: `src/prototypes/data_files/` has real, substantial audio
folders for at least two of them (`brant/` — 135 files, `sheowl/` — 128
files), so real data was genuinely collected and labelled under these
names. That makes this more likely a naming/anonymisation decision than a
processing bug, but it still isn't resolvable from the repository alone —
it would need input from whoever originally collected/labelled this data
to confirm what these five names actually refer to.

## Recommended next steps (not actioned in this audit)

1. Resolve Finding 1 before trusting `class_names.json` for any output
   that comes from the TFLite (Candidate B) model path — confirm which
   list actually corresponds to that model's output layer.
2. Standardise on one naming convention (space-separated, consistent
   capitalisation) across `class_names.json`, `class_mapping.json`, and
   the local dataset folder names, so simple exact-match joins stop
   silently losing real species like `Spilopelia_chinensis`.
3. Ask whoever trained the Candidate B TFLite model what `brant`,
   `jabwar`, `sheowl`, `spodov`, and `wiltur` are meant to represent.
4. If the team commits to a single canonical species list going forward,
   this repo's new PyTorch/Hydra pipeline
   (`src/prototypes/engine/augmentation/`) derives its class list
   automatically from whatever dataset directory is passed in
   (`dataset.py`'s `index_directory`), so pointing it at one agreed
   dataset snapshot would make the list self-documenting rather than a
   separately-maintained static file that can drift, as happened here.
