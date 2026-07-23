import json

import pytest

from pixgrep.config import load_config


def test_engine_defaults_to_torch(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(json.dumps({"image_root": str(tmp_path)}))
    cfg = load_config(cfg_file)
    assert cfg.engine == "torch"
    assert cfg.ov_devices == ("NPU", "CPU")


def test_openvino_engine_parsed(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "engine": "openvino",
                "ov_vision_ir": "ov_models/vision_int8.xml",
                "ov_devices": ["CPU"],
            }
        )
    )
    cfg = load_config(cfg_file)
    assert cfg.engine == "openvino"
    assert cfg.ov_devices == ("CPU",)


def test_missing_ir_raises_clean_error(tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    monkeypatch.delenv("PIXGREP_IMAGE_ROOT", raising=False)
    cfg_file = tmp_path / "config.local.json"
    cfg_file.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "engine": "openvino",
                "ov_vision_ir": str(tmp_path / "does_not_exist.xml"),
            }
        )
    )
    cfg = load_config(cfg_file)
    with pytest.raises(RuntimeError, match="convert_ov"):
        cfg.make_embedder()
