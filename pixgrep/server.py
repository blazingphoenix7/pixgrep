from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.middleware.base import BaseHTTPMiddleware

from .search import SearchEngine


class _CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        elif path == "/" or path == "/index.html" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _parse_filters(f_params: list[str]) -> dict[str, str] | None:
    """Parse repeated `f=field:value` query params into a filters dict."""
    if not f_params:
        return None
    filters: dict[str, str] = {}
    for item in f_params:
        if ":" in item:
            field, _, value = item.partition(":")
            field = field.strip()
            value = value.strip()
            if field and value:
                filters[field] = value
    return filters if filters else None


def create_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(title="pixgrep")
    app.add_middleware(_CacheControlMiddleware)

    @app.get("/api/meta")
    def meta():
        return {"count": engine.count}

    @app.get("/api/facets")
    def facets():
        if not engine._tags.has_data:
            return {}
        raw = engine._tags.facets()
        return {
            field: [
                {"value": v, "count": c}
                for v, c in sorted(val_counts.items(), key=lambda x: -x[1])
            ]
            for field, val_counts in raw.items()
        }

    @app.get("/api/search")
    def search(
        q: str = Query(...),
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
        f: list[str] = Query(default=[]),
        hw: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    ):
        if not q.strip():
            raise HTTPException(status_code=400, detail="empty query")
        filters = _parse_filters(f)
        results = engine.text_search(
            q, k=k, min_ratio=min_ratio, min_score=min_score,
            filters=filters, hybrid_weight=hw,
        )
        return {"results": results}

    @app.post("/api/search/image")
    def search_image(
        file: UploadFile = File(...),
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
        f: list[str] = Query(default=[]),
    ):
        data = file.file.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="could not decode image")
        filters = _parse_filters(f)
        results = engine.image_search(
            img, k=k, min_ratio=min_ratio, min_score=min_score, filters=filters,
        )
        return {"results": results}

    @app.get("/api/similar/{row}")
    def similar(
        row: int,
        k: int = Query(24, ge=1, le=200),
        min_ratio: float = Query(0.6, ge=0.0, le=1.0),
        min_score: float = Query(0.05, ge=0.0, le=1.0),
        f: list[str] = Query(default=[]),
    ):
        filters = _parse_filters(f)
        try:
            results = engine.similar(
                row, k=k, min_ratio=min_ratio, min_score=min_score, filters=filters,
            )
        except IndexError:
            raise HTTPException(status_code=404, detail="unknown row")
        return {"results": results}

    @app.get("/api/image/{row}")
    def image(row: int):
        try:
            path = Path(engine.path_for(row))
        except IndexError:
            raise HTTPException(status_code=404, detail="unknown row")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file missing on disk")
        return FileResponse(path)

    @app.get("/api/thumb/{row}")
    def thumb(row: int):
        thumb_path = engine.index_dir / "thumbs" / f"{row}.jpg"
        if thumb_path.is_file():
            return FileResponse(thumb_path)
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
    engine = SearchEngine(cfg.index_dir, cfg.make_embedder(), hybrid_weight=cfg.hybrid_weight, junk_threshold=cfg.junk_threshold)
    print(f"Index loaded: {engine.count} images. http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
