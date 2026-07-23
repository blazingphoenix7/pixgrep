from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .embedding import Embedder as _TorchEmbedder


class OVEmbedder:
    """OpenVINO-accelerated image embedder for Intel CPUs / iGPUs / NPUs.

    Loads a converted (and optionally weight-compressed) IR of the vision
    encoder and runs batches across one or more devices with simple work
    stealing. NPU devices are compiled with a static batch shape and short
    final batches are padded. Text embedding lazily falls back to the torch
    Embedder — indexing never calls it, and query-time text is one short
    string at a time.

    Create the IR with ``scripts/convert_ov.py``.
    """

    def __init__(
        self,
        model_id: str,
        vision_ir: Path,
        devices: tuple[str, ...] = ("NPU", "CPU"),
        batch_size: int = 8,
        cache_dir: Path | None = None,
        cpu_threads: int = 8,
    ):
        import openvino as ov
        from transformers import AutoProcessor

        vision_ir = Path(vision_ir)
        if not vision_ir.is_file():
            raise RuntimeError(
                f"vision IR not found: {vision_ir} — run scripts/convert_ov.py first"
            )

        self.model_id = model_id
        self.batch = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id, token=False)
        size = self.processor.image_processor.size
        self._side = size.get("height") or size.get("shortest_edge")

        core = ov.Core()
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})

        self._workers: list[tuple[str, object, bool]] = []  # (device, compiled, static)
        errors = []
        for dev in devices:
            try:
                if dev == "NPU":
                    model = core.read_model(str(vision_ir))
                    model.reshape([batch_size, 3, self._side, self._side])
                    compiled = core.compile_model(model, dev)
                    self._workers.append((dev, compiled, True))
                elif dev == "CPU":
                    compiled = core.compile_model(
                        str(vision_ir),
                        dev,
                        {
                            "PERFORMANCE_HINT": "THROUGHPUT",
                            "INFERENCE_NUM_THREADS": cpu_threads,
                            "NUM_STREAMS": "1",
                        },
                    )
                    self._workers.append((dev, compiled, False))
                else:
                    compiled = core.compile_model(str(vision_ir), dev)
                    self._workers.append((dev, compiled, False))
            except Exception as e:  # device absent or driver failure — skip it
                errors.append(f"{dev}: {e}")
        if not self._workers:
            raise RuntimeError(f"no OpenVINO device available: {'; '.join(errors)}")

        self._torch: _TorchEmbedder | None = None

    @property
    def active_devices(self) -> list[str]:
        return [d for d, _, _ in self._workers]

    def embed_images(self, images: list) -> np.ndarray:
        pil = [_TorchEmbedder._to_pil(i) for i in images]
        px = self.processor(images=pil, return_tensors="pt")["pixel_values"].numpy()
        batches = [px[i : i + self.batch] for i in range(0, len(px), self.batch)]
        results: list[np.ndarray | None] = [None] * len(batches)

        lock = threading.Lock()
        next_idx = [0]

        def run(compiled, static: bool):
            while True:
                with lock:
                    if next_idx[0] >= len(batches):
                        return
                    idx = next_idx[0]
                    next_idx[0] += 1
                chunk = batches[idx]
                n = len(chunk)
                if static and n < self.batch:
                    pad = np.repeat(chunk[-1:], self.batch - n, axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                out = compiled(chunk)[0][:n]
                results[idx] = out

        threads = [
            threading.Thread(target=run, args=(compiled, static), daemon=True)
            for _, compiled, static in self._workers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        emb = np.vstack([r for r in results if r is not None]).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._torch is None:
            self._torch = _TorchEmbedder(self.model_id)
        return self._torch.embed_texts(texts)
