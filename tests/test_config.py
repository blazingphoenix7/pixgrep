import json
from pathlib import Path

import pytest
from pixgrep.config import load_config, DEFAULT_MODEL_ID


def test_load_config_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({"image_root": str(tmp_path / "imgs")}))
    cfg = load_config(cfg_file)
    assert cfg.image_root == tmp_path / "imgs"
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert cfg.db_path == cfg.index_dir / "pixgrep.sqlite"
    assert cfg.emb_path == cfg.index_dir / "embeddings.npy"


def test_missing_image_root_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    with pytest.raises(ValueError):
        load_config(tmp_path / "does_not_exist.json")


def test_env_var_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXGREP_IMAGE_ROOT", str(tmp_path))
    cfg = load_config(tmp_path / "none.json")
    assert cfg.image_root == tmp_path


def test_lexical_inject_and_junk_soft_weight_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({"image_root": str(tmp_path / "imgs")}))
    cfg = load_config(cfg_file)
    assert cfg.lexical_inject_k == 50
    assert cfg.junk_soft_weight == 1.0


def test_lexical_inject_and_junk_soft_weight_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({
        "image_root": str(tmp_path / "imgs"),
        "lexical_inject_k": 10,
        "junk_soft_weight": 0.5,
    }))
    cfg = load_config(cfg_file)
    assert cfg.lexical_inject_k == 10
    assert cfg.junk_soft_weight == 0.5


def test_lexical_inject_and_junk_soft_weight_zero_disables(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({
        "image_root": str(tmp_path / "imgs"),
        "lexical_inject_k": 0,
        "junk_soft_weight": 0,
    }))
    cfg = load_config(cfg_file)
    assert cfg.lexical_inject_k == 0
    assert cfg.junk_soft_weight == 0.0


def test_feedback_db_defaults_under_index_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({"image_root": str(tmp_path / "imgs")}))
    cfg = load_config(cfg_file)
    assert cfg.feedback_db == ""
    assert cfg.feedback_db_path == cfg.index_dir / "feedback.sqlite"


def test_feedback_db_override_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    custom = str(tmp_path / "somewhere" / "fb.sqlite")
    cfg_file.write_text(json.dumps({
        "image_root": str(tmp_path / "imgs"),
        "feedback_db": custom,
    }))
    cfg = load_config(cfg_file)
    assert cfg.feedback_db == custom
    assert cfg.feedback_db_path == Path(custom)
