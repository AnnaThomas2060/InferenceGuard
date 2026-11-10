"""The served risk classifier must load the weights it was given, or say loudly that it did not.

Session 10: on transformers 5.20.1 and 5.22.2, `load_risk_classifier` could not
load AT707/InferenceGuard-Roberta-Risk (missing `post_init`), caught the error,
and silently built a 1-layer RoBERTa with random heads. Separately, the
published checkpoint was itself untrained, because the upload script loaded
weights with strict=False and published regardless.

Everything here runs offline on a tiny RoBERTa saved to a temp dir.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.risk_model.model import load_risk_classifier  # noqa: E402
from src.risk_model.roberta_model import RobertaRiskClassifier, RobertaRiskConfig  # noqa: E402

HIDDEN = 32


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A trained-looking RobertaRiskClassifier saved with save_pretrained."""
    backbone_dir = tmp_path / "backbone"
    backbone = transformers.RobertaModel(transformers.RobertaConfig(
        vocab_size=100, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=37, max_position_embeddings=64,
    ))
    backbone.save_pretrained(backbone_dir)

    config = RobertaRiskConfig(base_model_name=str(backbone_dir), hidden_size=HIDDEN)
    model = RobertaRiskClassifier(config)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.heads.parameters():
            p.add_(torch.randn_like(p))  # stand-in for training: heads differ from init
    ckpt_dir = tmp_path / "ckpt"
    model.save_pretrained(ckpt_dir)
    return ckpt_dir, model.state_dict()


def test_checkpoint_round_trips_through_the_product_loader(tiny_checkpoint):
    ckpt_dir, saved = tiny_checkpoint
    _, info = RobertaRiskClassifier.from_pretrained(str(ckpt_dir), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]

    served = load_risk_classifier(str(ckpt_dir), device="cpu")
    assert served.roberta.config.num_hidden_layers == 2
    loaded = served.state_dict()
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[k], saved[k]) for k in saved)


def test_loader_fallback_is_loud(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="src.risk_model.model"):
        load_risk_classifier(str(tmp_path / "does-not-exist"), device="cpu")
    assert any("UNTRAINED" in r.getMessage() for r in caplog.records)


# --- upload script ---------------------------------------------------------------

def test_upload_refuses_without_a_checkpoint():
    from host_roberta_on_hf import load_trained_weights

    with pytest.raises(ValueError, match="untrained"):
        load_trained_weights(object(), None)


def test_upload_refuses_an_empty_checkpoint_dir(tmp_path):
    from host_roberta_on_hf import load_trained_weights

    with pytest.raises(FileNotFoundError):
        load_trained_weights(object(), str(tmp_path))


def test_upload_refuses_weights_that_do_not_match(tiny_checkpoint, tmp_path):
    from safetensors.torch import save_file

    from host_roberta_on_hf import load_trained_weights

    ckpt_dir, _ = tiny_checkpoint
    model = RobertaRiskClassifier.from_pretrained(str(ckpt_dir))
    # e.g. a ModernBERT state dict: no key matches, which strict=False used to accept
    bad = tmp_path / "bad"
    bad.mkdir()
    save_file({"bert.embeddings.weight": torch.zeros(4, 4), "classifiers.0.weight": torch.zeros(1, 4)},
              str(bad / "model.safetensors"))
    with pytest.raises(RuntimeError, match="does not match"):
        load_trained_weights(model, str(bad))


def test_upload_accepts_matching_weights(tiny_checkpoint):
    from host_roberta_on_hf import load_trained_weights

    ckpt_dir, saved = tiny_checkpoint
    model = RobertaRiskClassifier.from_pretrained(str(ckpt_dir))
    with torch.no_grad():
        for p in model.heads.parameters():
            p.zero_()
    load_trained_weights(model, str(ckpt_dir))
    assert all(torch.equal(model.state_dict()[k], saved[k]) for k in saved if k.startswith("heads."))


# --- scoring script --------------------------------------------------------------

def test_scoring_loader_accepts_a_clean_checkpoint(tiny_checkpoint):
    from score_risk_model import load_served_model

    ckpt_dir, _ = tiny_checkpoint
    model, facts = load_served_model(str(ckpt_dir), "cpu")
    assert facts["backbone_layers"] == 2


def test_scoring_refuses_the_test_split():
    from score_risk_model import load_records

    with pytest.raises(ValueError, match="held-out"):
        load_records("test", ROOT / "data/manifests/split_manifest.json")


def test_band_shares():
    from score_risk_model import band_shares

    probs = [0.1, 0.35, 0.65, 0.9]
    truth = [0, 1, 0, 1]
    shares = band_shares(probs, truth, {"medium": 0.30, "high": 0.60})
    assert shares == {"share_low": 0.25, "share_high": 0.5, "high_precision": 0.5, "positives_missed_in_low": 0.0}
