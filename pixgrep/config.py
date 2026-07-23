from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-256"
DEFAULT_STRIP_PATTERN = r"[-_ ]*(?:pro|s|l|\d+)$"


@dataclass(frozen=True)
class Config:
    image_root: Path
    index_dir: Path
    model_id: str
    group_strip_pattern: str
    batch_size: int
    engine: str = "torch"  # "torch" | "openvino"
    ov_vision_ir: str = ""  # path to converted vision IR (see scripts/convert_ov.py)
    ov_devices: tuple[str, ...] = ("NPU", "CPU")
    ov_cache_dir: str = ""

    @property
    def db_path(self) -> Path:
        return self.index_dir / "pixgrep.sqlite"

    @property
    def emb_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    def make_embedder(self):
        """Build the embedder the config asks for (torch by default)."""
        if self.engine == "openvino":
            from .embedding_ov import OVEmbedder

            return OVEmbedder(
                self.model_id,
                Path(self.ov_vision_ir),
                devices=tuple(self.ov_devices),
                batch_size=self.batch_size,
                cache_dir=Path(self.ov_cache_dir) if self.ov_cache_dir else None,
            )
        from .embedding import Embedder

        return Embedder(self.model_id)


def load_config(path="config.local.json") -> Config:
    data: dict = {}
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    image_root = os.environ.get("PIXGREP_IMAGE_ROOT") or data.get("image_root")
    if not image_root:
        raise ValueError(
            "No image_root configured. Copy config.example.json to config.local.json "
            "and set 'image_root', or set the PIXGREP_IMAGE_ROOT environment variable."
        )
    return Config(
        image_root=Path(image_root),
        index_dir=Path(data.get("index_dir", "index")),
        model_id=data.get("model_id", DEFAULT_MODEL_ID),
        group_strip_pattern=data.get("group_strip_pattern", DEFAULT_STRIP_PATTERN),
        batch_size=int(data.get("batch_size", 16)),
        engine=data.get("engine", "torch"),
        ov_vision_ir=data.get("ov_vision_ir", ""),
        ov_devices=tuple(data.get("ov_devices", ["NPU", "CPU"])),
        ov_cache_dir=data.get("ov_cache_dir", ""),
    )
