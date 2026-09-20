"""Unit test for EfficientNetV2ArcFace.get_embedding(), added for the
similar-detection retrieval feature. Runs against the real (older,
freeze_bn) checkpoint from the first full training run, since this test
only checks the method's contract (shape, finiteness, normalisation), not
embedding quality - quality was investigated separately in
similar_detections_prototype.ipynb, which found that checkpoint's embedding
space collapsed (97% dead dimensions) and led to retraining with
norm_choice=keep_bn. This test would pass against either checkpoint; it is
not a substitute for that quality investigation.
"""

import sys
from pathlib import Path

PIPELINE_DIR = (
    Path(__file__).resolve().parents[3]  # this is already .../src
    / "prototypes"
    / "engine"
    / "reproducible_training_pipeline"
)
sys.path.insert(0, str(PIPELINE_DIR))

import pytest
import torch

hydra = pytest.importorskip("hydra", reason="Only installed in the pipeline's own .venv, not the root environment.")
omegaconf = pytest.importorskip("omegaconf", reason="Only installed in the pipeline's own .venv, not the root environment.")
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CHECKPOINT_PATH = (
    PIPELINE_DIR / "outputs" / "sprint2_first_full_training_2026-09-04" / "best_efficientnet_v2.pth"
)

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Requires the Sprint 2 full-training checkpoint, which is untracked and local-only.",
)


@pytest.fixture(scope="module")
def model():
    with initialize_config_dir(config_dir=str(PIPELINE_DIR / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=efficientnet_v2"])

    OmegaConf.set_struct(cfg, False)
    cfg.data.num_classes = 127
    OmegaConf.set_struct(cfg, True)

    from model import Model

    m = Model(cfg)
    m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    m.eval()
    return m


class TestGetEmbedding:
    def test_output_shape_matches_backbone_feature_dim(self, model):
        dummy_input = torch.randn(1, 1, 384, 200)
        with torch.no_grad():
            embedding = model.model.get_embedding(dummy_input)
        assert embedding.shape == (1, 1280)  # EfficientNetV2-S's final feature width

    def test_output_is_finite(self, model):
        dummy_input = torch.randn(1, 1, 384, 200)
        with torch.no_grad():
            embedding = model.model.get_embedding(dummy_input)
        assert torch.isfinite(embedding).all()

    def test_output_is_l2_normalised(self, model):
        dummy_input = torch.randn(1, 1, 384, 200)
        with torch.no_grad():
            embedding = model.model.get_embedding(dummy_input)
        norm = torch.linalg.norm(embedding, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_different_inputs_do_not_always_produce_identical_embeddings(self, model):
        # Not a quality check (see module docstring) - just confirms the method
        # is not hard-wired to return a constant regardless of input.
        a = torch.randn(1, 1, 384, 200)
        b = torch.randn(1, 1, 384, 200)
        with torch.no_grad():
            emb_a = model.model.get_embedding(a)
            emb_b = model.model.get_embedding(b)
        assert not torch.allclose(emb_a, emb_b)
