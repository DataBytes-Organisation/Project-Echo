# SpecAugment (Audio Augmentation) Usage Instructions



## 1. Required libraries

This code is designed to run inside the team's existing `projectecho` conda environment, so everyone runs augmentation the same way.

`projectecho` doesn't have PyTorch or a few other packages this code needs by default. Install them with:

```
conda activate projectecho
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 hydra-core==1.3.4 omegaconf==2.3.1 scikit-learn==1.6.1 tqdm==4.69.1 diskcache==5.6.3 lmdb==2.3.0 matplotlib==3.9.4 umap-learn==0.5.12
```

These versions are what this framework was actually built and tested against. `librosa`, `soundfile`, and `audiomentations` are also used, but `projectecho` already has those pinned (`librosa==0.11.0`, `audiomentations==0.42.0`) so don't reinstall or upgrade them, since the rest of the team's pipeline is pinned to those exact versions.

**Verify the install:**

```
python -c "import torch, torchaudio, hydra, omegaconf, sklearn, diskcache, lmdb, umap; print('ok')"
```

If that prints `ok` with no errors, everything needed is present.

## 2. What this does

During training, SpecAugment blacks out (masks) random frequency and time strips, forcing the model to learn the general shape of an animal call instead of memorising exact pixels, which makes it more robust to messy real-world recordings. It is applied to training data only (not validation data), so accuracy checks stay fair.


## 3. How it's configured

Nothing about SpecAugment is hardcoded. Its settings are separate files in `Project-Echo/src/prototypes/engine/augmentation/config/augmentation/` (preset files), and you pick one on the command line with `augmentation=<name>` when running `main.py`.

Five presets exist right now:

| Name | What it is |
|---|---|
| `none` | Augmentation off |
| `light` | Gentle masking |
| `default` | Recommended starting point - used automatically if you don't specify `augmentation=...` |
| `heavy` | Aggressive masking, for experimentation |
| `original_unfixed_reference` | Reproduces the original settings from last trimester, recommended for comparison purposes only, not real use |

The rest of this file walks through proving that mechanism actually works, first as a fast config check, then in real training.


## 4. Step-by-step: try it yourself

You have to `cd` into this directory first (`Project-Echo/src/prototypes/engine/augmentation/`).

### Step 1 - See what presets exist

```
dir config\augmentation
```

Five files, one per row of the table above. Each one is just a small YAML file. Open `config/augmentation/default.yaml` in a text editor if you want to see what one looks like before going further. The other presets are in similar structure with different values depending on their names.


### Step 2 - Prove that selecting a preset actually changes something

Before running any real (slow) training, you can quickly check that choosing a different preset genuinely changes what the model will use. Hydra has a built-in flag, `--cfg job`, that prints the full configuration, then exits immediately without training anything:

```
python main.py augmentation=none --cfg job
```

Look for the `augmentation:` section near the top of the output. For `none` it will show:
```yaml
augmentation:
  _target_: torchvision.transforms.Compose
  transforms: []
```

Now run the same command with a different preset:
```
python main.py augmentation=heavy --cfg job
```
This time the `augmentation:` section is completely different:
```yaml
augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
  - _target_: augment.SpecAugment
    p: 0.7
    n_freq_mask: 3
    n_time_mask: 3
    freq_mask_ratio: 0.2
    time_mask_ratio: 0.15
    max_total_time_ratio: 0.5
    mask_value: zero
```

The exact same command, with only the `augmentation=` value changed, produces a different configuration.

The full output for all five presets is in the [Reference](#reference) section below, if you want to see every one.

### Step 3 - Prove it actually works in a real training run

Step 2 only proves the configuration changes, so no training is involved. This step proves the same terminal-driven selection that actually controls augmentation during real training against the project's real dataset, which should be placed in this repo at `Project-Echo/models-and-data/data_files` (the downloaded `data_files` from the Onboarding Task). Thus, you should put your `data_files` folder from your local machine into the mentioned location before continuing.

One thing to know first: `config.yaml`'s checked-in default (`system.audio_data_directory: b3`) points at a folder that was never part of this repo (it's just a stale placeholder, not a sign the dataset is missing). The fix for that is one of three things bundled into the same opt-in settings file `cpu.yaml` used below.

Here is the command to run the training pipeline:
```
python main.py augmentation=default ~augmentations.audio +local=cpu training.epochs=1 training.num_workers=0 training.batch_size=4
```

- `~augmentations.audio` removes the audio-level noise augmentation (it needs a `background_noise/` folder that doesn't exist here), and this can't be folded into the bundle since it's a deletion and Hydra only supports deleting a key as a command-line override, not as config file content.
- `+local=cpu` selects the mentioned settings bundle (`config/local/cpu.yaml`) that fixes three environment gaps at once: points at the real dataset (`models-and-data/data_files`), keeps Hydra from switching working directories (which would otherwise break the path to this `augmentation` folder), and skips a pretrained-weights download that hits a Windows SSL certificate bug in this setup. It is opt-in and changes nothing for anyone who doesn't select it. See that file for the full explanation of what it does and why.
- `epochs=1` keeps this pipeline a quick proof rather than the 500-epoch default. This one stays explicit on purpose, unlike the other two, since it's a choice about how long to run, not an environment gap that needs to be addressed.
- `num_workers=0` and `batch_size=4` cut this down to the least resource-consuming settings possible. The checked-in defaults (`num_workers: 12` and `batch_size: 64` in `config.yaml`) spawn 12 parallel worker processes with each holding a 64-clip batch of decoded audio and spectrograms in memory at once, which is enough to exhaust available RAM on a laptop, i.e., `Unable to allocate ... MiB` errors and `Cache write error: I/O operation on closed file`. `num_workers=0` runs loading in the main process instead of spawning any workers and `batch_size=4` shrinks the batch itself to consume less resource, and these are necessary for this run because we only need to test if it really works, not a real training run.

At these settings this trades speed for memory safety, which runs slower per step than the checked-in defaults. You can press `Ctrl+C` to exit early without any risks. Even at 1 epoch, this runs on the full (roughly) 8,600-file dataset, so expect a full epoch to take a while rather than being instant.

Swap `augmentation=default` for `augmentation=light`, `augmentation=heavy`, `augmentation=none`, or `augmentation=original_unfixed_reference` and real training starts under that setting instead. This proves that the selection made on the command line is what actually drives training, with nothing hardcoded.

### Step 4 - Where to find the results

- Live progress prints to the console every step (loss and accuracy)
- `best_efficientnet_v2.pth` - the trained model checkpoint, saved whenever validation improves
- `events.out.tfevents.*` - a TensorBoard log (view with `tensorboard --logdir .`)

For a small-scale comparison across multiple presets at once, see `build_experiment_subset.py` and `run_validation_experiment.py` in this folder.

## Reference

### Full config for every preset

```yaml
# augmentation=none
augmentation:
  _target_: torchvision.transforms.Compose
  transforms: []

# augmentation=light
augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
  - _target_: augment.SpecAugment
    p: 0.3
    n_freq_mask: 1
    n_time_mask: 1
    freq_mask_ratio: 0.1
    time_mask_ratio: 0.05
    max_total_time_ratio: 0.2
    mask_value: zero

# augmentation=default
augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
  - _target_: augment.SpecAugment
    p: 0.5
    n_freq_mask: 2
    n_time_mask: 2
    freq_mask_ratio: 0.15
    time_mask_ratio: 0.1
    max_total_time_ratio: 0.4
    mask_value: zero

# augmentation=heavy
augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
  - _target_: augment.SpecAugment
    p: 0.7
    n_freq_mask: 3
    n_time_mask: 3
    freq_mask_ratio: 0.2
    time_mask_ratio: 0.15
    max_total_time_ratio: 0.5
    mask_value: zero

# augmentation=original_unfixed_reference
augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
  - _target_: augment.SpecAugment
    p: 0.5
    freq_mask_param: 30
    time_mask_param: 80
    n_freq_mask: 2
    n_time_mask: 2
    max_total_time_ratio: 1.0
    mask_value: zero
```

### What each parameter means

| Setting | Meaning | Typical range |
|---|---|---|
| `p` | the probability that masking does/does not happen at all on a given spectrogram | 0.0 - 1.0 |
| `n_freq_mask` | number of separate frequency-strips to cover | 1 - 3 |
| `n_time_mask` | number of separate time-strips to cover | 1 - 3 |
| `freq_mask_ratio` | max width of a frequency strip as a fraction of the total number of frequencies | 0.05 - 0.2 |
| `time_mask_ratio` | max width of a time strip as a fraction of the total number of timeframes | 0.05 - 0.2 |
| `freq_mask_param` and `time_mask_param` | max widths as a fixed pixel count instead of a ratio. If set, this overrides the ratio above. | project-dependent |
| `max_total_time_ratio` | Hard safety cap: no matter what else is configured, the combined width of all time-strips can never exceed this fraction of the clip | 0.2 - 0.5 |
| `mask_value` | What value a masked strip gets filled with: `"zero"` (the default and original value), `"min"`, or `"mean"` | - |

**Why we use ratios instead of fixed pixel counts?** The original implementation used a fixed pixel count for `time_mask_param` (80), which, at this project's audio settings (2-second audio clips with around 201 timeframes), could black out up to 80% of a clip if 2 separate time-strips is the option. A ratio-based setting (e.g. "10% of all timeframes") stays adaptive even if clip length or audio settings change later, which a fixed pixel count cannot. The `max_total_time_ratio` cap is a second safety net that applies even if someone deliberately sets fixed pixel values (this is exactly what the `original_unfixed_reference` preset demonstrates - see its config above).


### Adding a new preset

You can create a custom preset by creating a new YAML file in `config/augmentation/` and following the structure of `default.yaml` but with different values, then selecting it with `augmentation=<filename_without_.yaml>`.