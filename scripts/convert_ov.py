"""Convert the configured vision model to an OpenVINO IR for Intel acceleration.

Produces two files next to your config's `ov_vision_ir` path:
  <name>.xml/.bin            — FP16 IR
  <name>_int8.xml/.bin       — weight-only INT8 (recommended: ~2x faster on CPU,
                               runs on Intel NPUs, retrieval-quality preserved)

Weight-only compression (INT8_ASYM) is deliberately chosen over full static
quantization: in our benchmarks full INT8 activation quantization of SigLIP2
ViTs collapsed retrieval quality, while weight-only kept recall identical to
FP16. Requires: pip install openvino nncf

Usage:
    python scripts/convert_ov.py            # uses config.local.json model_id
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pixgrep.config import load_config  # noqa: E402


class _VisionWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        out = self.m.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        return out


def main() -> int:
    import openvino as ov
    import nncf
    from transformers import AutoModel, AutoProcessor

    cfg = load_config()
    out_path = Path(cfg.ov_vision_ir) if cfg.ov_vision_ir else Path("ov_models/vision.xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_path = out_path.with_name(out_path.stem.replace("_int8", "") + ".xml")
    int8_path = fp16_path.with_name(fp16_path.stem + "_int8.xml")

    print(f"loading {cfg.model_id} ...")
    model = AutoModel.from_pretrained(
        cfg.model_id, token=False, attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(cfg.model_id, token=False)
    size = processor.image_processor.size
    side = size.get("height") or size.get("shortest_edge")

    print(f"converting vision encoder ({side}px) ...")
    t0 = time.time()
    with torch.no_grad():
        ov_model = ov.convert_model(
            _VisionWrapper(model), example_input=torch.randn(2, 3, side, side)
        )
    ov_model.reshape([-1, 3, side, side])
    ov.save_model(ov_model, fp16_path, compress_to_fp16=True)
    print(f"FP16 IR: {fp16_path} ({time.time()-t0:.0f}s)")

    print("weight-only INT8 compression ...")
    t1 = time.time()
    int8_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(int8_model, int8_path)
    print(f"INT8 IR: {int8_path} ({time.time()-t1:.0f}s)")

    print(
        "\nSet in config.local.json:\n"
        f'  "engine": "openvino",\n'
        f'  "ov_vision_ir": "{int8_path.as_posix()}",\n'
        f'  "ov_devices": ["NPU", "CPU"]   // drop NPU if your machine has none'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
