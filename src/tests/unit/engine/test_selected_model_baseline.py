from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "prototypes"
    / "engine"
    / "evaluation"
    / "selected_model_baseline"
    / "baseline_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("selected_model_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_canonical_label_handles_case_underscores_and_spacing():
    assert MODULE.canonical_label("  Cervus_Unicolour ") == "cervus unicolour"
    assert MODULE.canonical_label("Spilopelia chinensis") == "spilopelia chinensis"


def test_stable_rank_is_deterministic_and_seeded():
    uri = "gs://example/species/audio.wav"
    assert MODULE.stable_rank(42, uri) == MODULE.stable_rank(42, uri)
    assert MODULE.stable_rank(42, uri) != MODULE.stable_rank(43, uri)


def test_boolean_series_parses_csv_text_safely():
    import pandas as pd

    result = MODULE.boolean_series(pd.Series(["True", "False", "1", "0", "yes", "no"]))
    assert result.tolist() == [True, False, True, False, True, False]


def test_is_augmented_matches_camel_case_and_separator_variants():
    markers = ["add_gaussian_snr", "background_noise", "pitch_shift", "time_mask"]
    assert MODULE.is_augmented("AddGaussianSNR_region_1.mp3", markers)
    assert MODULE.is_augmented("BackgroundNoise_PitchShift_region_1.mp3", markers)
    assert MODULE.is_augmented("PitchShift-TimeMask-region-1.mp3", markers)
    assert not MODULE.is_augmented("region_12.000-14.000.mp3", markers)


def test_scores_to_probabilities_preserves_probabilities():
    values = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    result, mode = MODULE.scores_to_probabilities(values)
    np.testing.assert_allclose(result, values, rtol=1e-6)
    assert mode == "probabilities"


def test_scores_to_probabilities_softmaxes_logits():
    result, mode = MODULE.scores_to_probabilities(np.array([-1.0, 0.0, 2.0]))
    assert mode == "logits_softmax"
    assert np.isclose(result.sum(), 1.0)
    assert int(np.argmax(result)) == 2


def test_adapt_input_supports_nchw_and_nhwc():
    source = np.zeros((1, 1, 128, 313), dtype=np.float32)
    nchw = MODULE.adapt_input(
        source,
        {"shape": np.array([1, 1, 128, 313]), "dtype": np.float32, "quantization": (0.0, 0)},
    )
    nhwc = MODULE.adapt_input(
        source,
        {"shape": np.array([1, 128, 313, 1]), "dtype": np.float32, "quantization": (0.0, 0)},
    )
    assert nchw.shape == (1, 1, 128, 313)
    assert nhwc.shape == (1, 128, 313, 1)
