# pixgrep

**grep for your images.** A local, offline **visual + semantic search engine** for large image
libraries. Point it at a folder (or network share) of images and search it two ways:

- 🔎 **Text search** — describe what you want in plain language, get ranked matches.
- 🖼️ **Reverse-image search** — drop in an image, find the visually similar ones.
- ✨ **"More like this"**, plus optional filters derived from filename patterns.

It computes an embedding for every image **once**, then answers queries instantly against a local
index. No manual tagging. No cloud services — images and data never leave your machine or network.

## Why

Image collections of 100k+ files are impossible to browse and impractical to tag by hand.
Embedding-based search makes them findable with zero tagging.

## How it works

- **Indexer** — walks the image folder, makes thumbnails, and computes embeddings
  (**SigLIP 2** for text search, **DINOv2** for visual similarity), storing them in **SQLite** + a
  local vector store. Resumable and incremental.
- **Search server** — a small **FastAPI** app for text/image queries; serves thumbnails and full images.
- **Web UI** — a lightweight browser front-end: search bar, drag-and-drop, results grid, lightbox.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Configuration

The image directory and other machine-specific settings are read from a local, **untracked** config
(`.env` / `config.local.*`). Nothing environment-specific is committed to this repo.


## Intel acceleration (OpenVINO)

On Intel machines (Core Ultra CPUs with NPUs, Arc iGPUs) pixgrep can run the
vision encoder through OpenVINO with weight-only INT8 compression — in our
benchmarks **2–4.7x faster indexing** with retrieval quality identical to
FP16 (validated on recall metrics, not just tensor parity). Full activation
quantization is deliberately avoided: it collapsed retrieval quality for
SigLIP2-style ViTs in testing.

```bash
pip install openvino nncf
python scripts/convert_ov.py   # one-time: converts + compresses the model
```

Then in `config.local.json`:

```json
{
  "engine": "openvino",
  "ov_vision_ir": "ov_models/vision_int8.xml",
  "ov_devices": ["NPU", "CPU"]
}
```

Multiple devices run in parallel with work stealing. Drop `"NPU"` if your
machine has none; the engine skips unavailable devices gracefully.

## Status

🚧 Pre-implementation / in progress.

## Stack

Python 3.11 · SigLIP 2 · DINOv2 · OpenVINO (optional acceleration) · FastAPI · SQLite · NumPy

## License

MIT
