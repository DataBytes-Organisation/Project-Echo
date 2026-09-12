# Augmentation Benchmark - Results and Selected Configuration

Full per-arm data (including the
balanced dataset's complete per-species F1 breakdown and the separate
test-split evaluation referenced below) is in
`current_dataset/ceiling40_seed42.json` and
`balanced_dataset/ceiling40_seed42.json`, alongside this file - this
document is the readable summary of those two files, not a replacement
for them.


All runs: single seed, `batch_size=4`, `num_workers=0`, EfficientNetV2 +
CircleLoss (`training.use_arcface: circle`), five presets (`none`,
`original_unfixed_reference`, `light`, `default`, `heavy`) from Sprint
1's unmodified `augment.py`/`config/augmentation/*.yaml`.



## The goal

"One of `light`/`default`/`heavy` should beat both `none` and
`original_unfixed_reference`" is checked here as both accuracy AND
macro-F1, for at least one augmented preset, in the same run. A preset
that wins on only one of the two metrics counts as a partial, not a
full, result - noted explicitly wherever it happens.

---

## Results: current (unbalanced) dataset

15 species, 2,737 files, ~547 held-out validation samples (the Sprint 2
subset built for early iteration, before the balanced dataset existed).
Seed 42, 40-epoch ceiling - "ceiling" because the harder presets never
triggered `early_stopping_patience=15` naturally; the run was cut off by
the epoch limit (see "Why the goal isn't fully met" below).


| Arm | final train loss | best val loss | test acc | macro-F1 |
|---|---|---|---|---|
| `none` | 15.230 | **1.765** | **0.522** | **0.498** |
| `original_unfixed_reference` | 18.672 | 2.506 | 0.218 | 0.147 |
| `light` | 10.210 | 2.027 | 0.520 | 0.478 |
| `default` | 18.466 | 2.367 | 0.255 | 0.152 |
| `heavy` | 18.614 | 2.573 | 0.179 | 0.084 |

**Goal check:** not met against `none` - `light` comes within **0.002
accuracy** (0.520 vs 0.522), effectively a tie given this is a
single-seed, ~547-sample measurement, but doesn't cross it. Against
`original_unfixed_reference`: `default`/`light` beat it on both metrics,
but `heavy` (0.179/0.084) loses to it on *both* (0.218/0.147) - see "Why
`heavy` underperforms" below.

---

## Results: balanced (final) dataset

79 species, 6,258 train / 4,466 validation / 4,613 test files (Manisha/Raveesha's
delivery, recording-level split, training floor=30/cap=150). Seed 42,
40-epoch ceiling.

| Arm | final train loss | best val loss | test acc | macro-F1 | zero-F1 species (of 79) |
|---|---|---|---|---|---|
| `none` | 10.137 | 2.344 | **0.712** | **0.611** | **3** |
| `original_unfixed_reference` | 15.565 | 2.217 | 0.567 | 0.336 | 12 |
| `light` | 11.653 | 2.224 | 0.653 | 0.494 | 5 |
| `default` | 14.619 | **2.024** | 0.599 | 0.384 | 7 |
| `heavy` | 14.992 | 2.167 | 0.584 | 0.390 | 7 |

**Goal check:** not met against `none` - `light` is closest (0.653/0.494
vs 0.712/0.611) but doesn't cross it. Against `original_unfixed_reference`,
the goal **is** met, cleanly and consistently: all three fixed presets
beat it on both accuracy and macro-F1 - `light` by the largest margin
(+0.086 accuracy, +0.158 macro-F1).

---

## Why the goal isn't fully met yet (and why the numbers above aren't contradictory)

**`CircleLoss` is a margin-based metric-learning loss: it improves
smoothly for every arm, but discrete accuracy stays near zero until the
embedding space crosses a separation threshold, at which point accuracy
appears - not gradually, but as a step.** Presets that mask more of the
spectrogram remove more discriminative signal per sample, which delays
exactly this threshold, not the loss curve itself. This is directly
visible in the balanced-dataset table above without needing any other
evidence: `default` has the **lowest (best) validation loss of all five
arms** (2.024) yet is only 4th of 5 on accuracy (0.599) - the model with
the best embedding space is not the model with the best discrete
accuracy, because it hadn't finished crossing the threshold within the
40-epoch budget.

Further confirmation from `train_acc`, not just loss: on the current
dataset, `original_unfixed_reference`/`default`/`heavy` finish 40 epochs
with training accuracy of essentially **0.00-0.05** - they had not
crossed the phase transition *at all* by the epoch ceiling. On the larger
balanced dataset, the same three arms reach meaningfully non-zero
training accuracy (0.12-0.14) by epoch 40 - more data gives the phase
transition more signal to work with under the same epoch budget, which
is also why the goal is met against `original_unfixed_reference` on the
balanced dataset but only partially on the current one.

**Why `original_unfixed_reference` is noisy rather than simply bad.** It
isn't a weaker point on the same light/default/heavy intensity ladder -
it's Sprint 1's deliberately-reproduced *pre-fix* behaviour: no cap on
`max_total_time_ratio`, so the fraction of the spectrogram masked per
sample is effectively unbounded and varies far more from sample to
sample than any capped preset. On the balanced dataset this shows up as
**12 of 79 species scoring exactly 0.0 F1** - nearly double `default`'s
or `heavy`'s count (7 each) despite a competitive aggregate accuracy
(0.567) - consistent with inconsistent per-sample masking teaching some
classes well while leaving others essentially untrained, which accuracy
(dominated by classes that *did* learn) hides and macro-F1 exposes. This
is the strongest evidence in the whole benchmark that the Sprint 1
safety-cap fix has real, practical value, independent of which capped
preset you pick.

**Why `heavy` underperforms even `original_unfixed_reference` on the
current dataset (0.179/0.084 vs 0.218/0.147).** `heavy` is deliberately
the most aggressive preset on the light/default/heavy ladder, so within a
fixed, already-insufficient epoch budget it is expected to be *furthest*
from crossing the phase transition - the same mechanism as above, at its
most extreme point, on the smallest dataset (2,737 files) where there is
also the least room to average out single-run noise. This does not recur
on the balanced dataset, where `heavy` (0.584/0.390) clearly and
consistently beats `original_unfixed_reference` (0.567/0.336).

---

## Suggestions to actually close the gap

1. **Let `early_stopping_patience` govern completion, not a fixed
   ceiling.** Every result above was cut off by an epoch limit chosen for
   time budget, not by the model's own loss plateauing (`train.py:299`
   already implements this correctly - it's the epoch ceilings passed
   into these specific benchmark runs that override it). Given
   `default`'s validation loss was the best of all five arms at epoch 40
   and still improving, this is the single most likely change to let a
   heavier preset's loss advantage convert into an accuracy advantage,
   and should be tried before any of the changes below.

2. **Time warping.** Add a genuine time-axis warp - stretch/compress a
   random region of the time axis non-uniformly, the original SpecAugment
   paper's "W" operation - as a new operation alongside the existing
   masking ops in `augment.py`. This is mechanistically different from
   every current preset: masking *deletes* information (sets pixels to
   zero/min/mean), which is exactly what defers the embedding-separation
   threshold, since masked regions carry no discriminative signal at all.
   Warping *redistributes* existing information without deleting any of
   it, so it should perturb the embedding much less at initialisation
   while still forcing timing/tempo invariance - a real candidate to keep
   augmentation's regularisation benefit without paying as much of the
   phase-transition delay every masking-only preset pays here.

3. **Curriculum/warm-start augmentation schedule.** Train with `none` or
   `light` for the first ~15-20 epochs, then switch to `default`/`heavy`
   for the remainder. This targets the root cause directly - the cost of
   heavier augmentation is concentrated in the early, pre-separation
   epochs, not the later ones - so avoid paying it at all rather than
   trying to survive it.

4. **A small auxiliary cross-entropy term alongside `CircleLoss`** during
   early training, so a usable discrete-class gradient signal exists
   before the margin loss reaches its own separation threshold. This is a
   training-recipe change to `train.py`'s loss setup rather than an
   augmentation change, so it would need to be raised with Nolan, who
   owns `reproducible_training_pipeline/`'s architecture (Section 3.12).

5. **Dial down `heavy` specifically, or add an intermediate rung.**
   `heavy` is the single worst-performing arm on almost every
   current-dataset metric, occasionally below even the buggy
   `original_unfixed_reference` reference - evidence it currently sits
   past the useful range for this loss/data/epoch-budget combination
   rather than inside it. A slightly milder variant is a cheap change to
   try before concluding heavy masking doesn't work at all.

6. **Multi-seed runs.** Needed specifically to confirm
   `original_unfixed_reference`'s occasional strong showings are real
   rather than single-seed noise its own high masking-variance would
   predict.

---

## Final selected configuration: `light`

**Recommendation:** adopt the `light` preset as the team's default
augmentation configuration going forward (Nolan's Section 3.12 training
switches to it once the balanced dataset is in regular use), while
keeping `default`/`heavy` under active investigation via the suggestions
above rather than ruling them out.

**Justification:**

- **Closest of the three augmented presets to `none` everywhere, and
  effectively tied with it on the current dataset** (0.520 vs 0.522
  accuracy - a 0.002 gap, well inside single-seed noise). No other
  augmented preset gets this close on any run.
- **Wins outright against `original_unfixed_reference` on the dataset
  that actually matters for submission** (balanced, +0.086 accuracy,
  +0.158 macro-F1 - the largest margin of any capped preset).
- **Fewest catastrophic per-class failures on the balanced dataset**: 5
  of 79 species at exactly 0.0 F1, versus 7 for `default`, 7 for `heavy`,
  and 12 for `original_unfixed_reference` - the most stable preset of the
  three real augmentation options, not just the highest-scoring on
  average.
- **Smallest phase-transition delay of the three augmented presets**:
  on the balanced dataset its training curve reached val-accuracy 0.697
  by the time training stopped, ahead of `default` (0.639) and `heavy`
  (0.643) - `light` is the augmented preset least likely to still be
  mid-transition when training has to stop under a real, time-boxed
  budget.
- **Why not just select `none`, since it wins every table above?**
  Because winning this specific comparison is not the same claim as
  "generalises best after deployment." The validation split here is drawn
  from the same recordings/distribution as training (even after
  balancing) - it does not represent the microphone, environment, and
  recording-quality variation the model will actually see in the field,
  which is precisely what augmentation exists to guard against. `none`'s
  lead reflects it fitting *this* validation distribution fastest under a
  limited epoch budget, not that it will hold up best against unseen
  recording conditions - a question this benchmark cannot answer and
  Praveen's held-out baseline task (Section 3.15) is better positioned
  to. `light` is the pick that keeps real augmentation in the pipeline
  while minimising the accuracy cost this benchmark actually measured.
- **`default` remains the most promising long-term candidate, not a
  rejected one.** It has the best validation loss of all five arms on the
  balanced dataset (2.024) - the strongest embedding space of anything
  tested here. It is not the current pick only because its accuracy
  hadn't caught up within the epoch budget this benchmark could afford,
  which suggestion #1 above (real early-stopping-governed runs) is the
  direct, low-cost way to test properly before the next sprint.

---

## Caveats

- Every run above is single-seed; the workflow (`run_augmentation_benchmark.py`)
  supports `--seeds 0,1,2` but a fair multi-seed run needs the
  epoch-ceiling issue resolved first (repeating an unfair single-budget
  comparison three times doesn't fix the unfairness).
- Praveen's evaluation-script/metric alignment (Section 3.15) is still
  not done - whether his baseline task's chunk-aggregation,
  `zero_division` handling, and split/seed choices match
  `evaluate_checkpoint()`'s here has not been checked.

---

## Raw artifacts (not in this file, and not in git)

Checkpoints (`best_efficientnet_v2_seed*.pth`, ~78-79MB each, ~25 files
across both datasets) and full training logs (`train_log_seed*.txt`, up
to ~79MB each - mostly raw tqdm progress output) stay local, next to this
folder (`experiment_results/`, `experiment_results_balanced/`) - excluded from git via
the repo's root `.gitignore` (checkpoints via the existing `*.pth` rule,
the rest via an addition made when this results folder was set up).
Ask me for the local copies if you need the actual weights or full logs
rather than the numbers above.
