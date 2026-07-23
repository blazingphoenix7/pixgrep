from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .search import SearchEngine

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(title="pixgrep")

    @app.get("/api/meta")
    def meta():
        return {"count": engine.count}

    @app.get("/api/search")
    def search(
        q: str = Query(...),
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
    ):
        if not q.strip():
            raise HTTPException(status_code=400, detail="empty query")
        return {"results": engine.text_search(q, k=k, min_ratio=min_ratio, min_score=min_score)}

    @app.post("/api/search/image")
    def search_image(
        file: UploadFile = File(...),
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
    ):
        data = file.file.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="could not decode image")
        return {"results": engine.image_search(img, k=k, min_ratio=min_ratio, min_score=min_score)}

    @app.get("/api/similar/{row}")
    def similar(
        row: int,
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
    ):
        try:
            return {"results": engine.similar(row, k=k, min_ratio=min_ratio, min_score=min_score)}
        except IndexError:
            raise HTTPException(status_code=404, detail="unknown row")

    @app.get("/api/image/{row}")
    def image(row: int):
        try:
            path = Path(engine.path_for(row))
        except IndexError:
            raise HTTPException(status_code=404, detail="unknown row")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file missing on disk")
        return FileResponse(path)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def root():
            return FileResponse(STATIC_DIR / "index.html")

    return app


def main() -> None:
    import argparse

    import uvicorn

    from .config import load_config

    parser = argparse.ArgumentParser(description="pixgrep search server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8177)
    args = parser.parse_args()

    cfg = load_config()
    print(f"Loading model {cfg.model_id} ...")
    engine = SearchEngine(cfg.index_dir, cfg.make_embedder())
    print(f"Index loaded: {engine.count} images. http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
