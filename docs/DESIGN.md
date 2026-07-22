# pixgrep — Design

A local, offline **visual + semantic search engine** for large image libraries. Point it at a
directory (local or a network share) and search it by natural-language description or by example
image — with no manual tagging and no cloud services.

> All environment-specific values (the image directory, hosts, etc.) are read from a local,
> untracked config. Nothing in this repo is tied to a particular deployment.

---

## 1. Problem

Large image collections — think 100k–1M+ files accumulated over years — become impossible to
browse and impractical to tag by hand. Folder structure decays, duplicates pile up, and finding
"that one piece" or "others like this" is slow. The goal is to make such a library **findable by
meaning and by visual similarity, with zero manual tagging.**

## 2. What it does

- **Text search** — "red vintage convertible at sunset" → ranked matches.
- **Reverse-image search** — drop an image → visually similar items.
- **"More like this"** on any result.
- **Filters** derived for free from filename patterns (configurable).
- **Result → full image + its file path** for easy locating.

## 3. Non-goals

- No manual/AI tagging of the whole library (embeddings replace tags for search).
- No claims about attributes not visible in a photo (e.g. true physical size without a reference).
- Read-only: never edits, moves, or deletes source files.
- No paid cloud services; data stays local.
- No auth/HTTPS in the MVP (single-user, localhost).

## 4. Key decisions

| Decision | Rationale |
|---|---|
| **Embeddings, not tags** | One pass yields both text and image search; no vocabulary to maintain. |
| **SigLIP 2 (text) + DINOv2 (visual similarity)** | Stronger than plain CLIP on fine-grained retrieval; DINOv2 index is gated on a quality metric. |
| **Brute-force exact search (NumPy / faiss-flat)** | At ~100k–1M vectors, exact cosine is a few ms/query; no ANN index to build or maintain. |
| **Free & local; optional OpenVINO iGPU acceleration** | Runs on commodity hardware with no GPU; iGPU/OpenVINO is an optional speedup. |
| **Robustness > raw speed** | The full index build runs rarely; resume-from-checkpoint + per-file error quarantine matter most. |
| **No human labeling required** | Junk is flagged zero-shot; quality is measured with a free "filename-group recall" metric. |
| **Config-driven, service-based** | Image paths/hosts come from local config; a FastAPI service makes future multi-user trivial. |

## 5. Components

1. **Indexer** — resumable, incremental batch job: walk → decode (reduced-scale, quarantine bad files) → thumbnail → embed → zero-shot junk flag → store. Skips already-indexed files by path + mtime + size; de-duplicates by content hash.
2. **Search server** — FastAPI: text search, reverse-image search, "more like this", filters; serves thumbnails and full images.
3. **Web UI** — a lightweight static front-end: search bar, drag-and-drop, results grid, lightbox.
4. **Quality harness** — computes the free filename-group recall metric; optional manual spot-check.

## 6. Data model

**SQLite** (WAL mode; indexer is the sole writer, server reads): one row per image with path,
content hash, parsed filename metadata, dimensions, size, mtime, thumbnail path, decode status,
junk score/flag, duplicate group, and a link to its embedding row.

**Embeddings** — stored as float16 memory-mapped matrices, one per model.

## 7. Pipeline

Walk the configured root → for each image: skip if unchanged → hash → decode at reduced scale
(with truncated/CMYK/EXIF handling) → thumbnail (WebP, hash-sharded directories) → embed (batched)
→ zero-shot junk score → write row + append vector → checkpoint frequently.

**File types:** `.jpg`/`.jpeg` primarily, plus `.png`/`.tif` where present; skip source/working
files (`.psd`, `.ai`), video, and documents.

## 8. Search

- **Text:** embed the query with the text encoder → cosine vs the text-model matrix → top-k → collapse duplicate groups → apply filters.
- **Image / "more like this":** embed the image with the visual-similarity model → cosine vs its matrix → top-k.
- Query embedding happens per request (one vector, instant even on CPU), behind a small lock so concurrent clients don't collide.

## 9. Junk filtering — no labels needed

Zero-shot classification (compare each image embedding to text prompts describing "a clean product
photo" vs "a document / rendering / logo / placeholder"), nudged by folder/filename priors. Stored
as a **reversible flag**, never a deletion. Optional future upgrade: a linear probe trained on a
small labeled set.

## 10. Quality strategy

- **Primary (free, automatic):** filename-group recall — files sharing a base filename are treated as the same item; query with one and measure how many of its siblings return. Also decides whether the second (visual-similarity) index is worth adding.
- **Optional (~20 min):** a handful of real queries judged good/bad in a spreadsheet.

## 11. Speed strategy

- **First:** simplest path (PyTorch/onnxruntime CPU) to validate search *quality*.
- **Then:** OpenVINO on the iGPU (benchmark CPU-int8 vs iGPU); parallel readers to saturate the network link; reduced-scale decode; batched inference; float16.
- Estimate scales with library size and link speed; the build is a rare, resumable batch.

## 12. Multi-user readiness

Built as a service from the start: images served through the API (never raw filesystem paths in the
UI), relative URLs, single-writer SQLite (WAL), and a single portable project directory — so moving
to an always-on host later is a copy-paste. Auth/HTTPS/service-install are deferred.

## 13. Stack

Python 3.11 · SigLIP 2 · DINOv2 (via `transformers` / `optimum-intel` / `openvino`, pinned) ·
NumPy / optional `faiss-cpu` · FastAPI + uvicorn · Pillow · SQLite · static HTML/JS.

## 14. Phasing

- **P0 — Proof:** clean env; embed a sample set; compute the free recall metric; throughput probe.
- **P1 — MVP:** text + reverse-image search end-to-end on the sample; validate quality.
- **P2 — Scale:** robust, resumable full-library build; junk flagging; dedup; iGPU/OpenVINO tuning.
- **P3 — Polish:** filters, "more like this" UI, incremental refresh, multi-user hardening.
