"""FastAPI service: DearSkin intelligent recommender + evaluation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ensure_data import ensure_data_dir
from recommender_engine import RecommendationEngine, ranking_evaluation, recommend_bundle

ENGINE: Optional[RecommendationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    data_dir = Path(__file__).resolve().parent / "data"
    ensure_data_dir(data_dir)
    ENGINE = RecommendationEngine()
    yield


app = FastAPI(title="DearSkin Intelligent Recommender", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendBody(BaseModel):
    user_id: int = Field(ge=0, description="Synthetic shopper id from demo dataset")
    approach: str = Field(
        pattern="^(collaborative|content|knowledge|all)$",
        description="Which paradigm to run",
    )
    cf_method: str = Field(
        default="user_cosine",
        pattern="^(user_cosine|user_pearson|item_cosine|mf_svd)$",
    )
    top_k: int = Field(default=8, ge=1, le=40)
    max_budget: Optional[float] = Field(default=None)
    categories: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    prefer_fragrance_free: bool = False


class RatingBody(BaseModel):
    user_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    rating: float = Field(ge=1.0, le=5.0)


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_loaded": ENGINE is not None}


@app.get("/api/meta")
def meta():
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    return {
        "n_users": ENGINE.n_users,
        "n_items": ENGINE.n_items,
        "cf_methods": ["user_cosine", "user_pearson", "item_cosine", "mf_svd"],
        "approaches": ["collaborative", "content", "knowledge", "all"],
        "categories": sorted({it.category for it in ENGINE.items.values()}),
        "brands": sorted({it.brand for it in ENGINE.items.values()}),
        "evaluation_metrics": ["Precision@K", "Recall@K", "MAP@K", "NDCG@K"],
    }


@app.get("/api/catalog")
def catalog():
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    return {"items": [vars(it) for it in sorted(ENGINE.items.values(), key=lambda x: x.item_id)]}


@app.get("/api/users/{user_id}/profile")
def user_profile(user_id: int):
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    if user_id >= ENGINE.n_users:
        raise HTTPException(404, "Unknown user")
    row = ENGINE.R[user_id]
    idx = np.where(~np.isnan(row))[0]
    history = [
        {
            "item_id": int(i),
            **vars(ENGINE.items[int(i)]),
            "rating": float(row[i]),
        }
        for i in idx
    ]
    history.sort(key=lambda x: -x["rating"])
    return {
        "user_id": user_id,
        "rated_count": len(history),
        "ratings": history,
    }


@app.post("/api/ratings")
def upsert_rating(body: RatingBody):
    """Append/update an explicit preference signal (session-local matrix)."""
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    if body.user_id >= ENGINE.n_users:
        raise HTTPException(400, "user_id out of range for demo dataset")
    if body.item_id >= ENGINE.n_items:
        raise HTTPException(400, "item_id out of range")
    ENGINE.R[body.user_id, body.item_id] = body.rating
    return {"ok": True}


@app.post("/api/recommendations")
def recommendations(body: RecommendBody):
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    if body.user_id >= ENGINE.n_users:
        raise HTTPException(404, "Unknown user")

    kb_params = {
        "max_budget": body.max_budget,
        "categories": body.categories,
        "brands": body.brands,
        "prefer_fragrance_free": body.prefer_fragrance_free,
    }

    try:
        payload = recommend_bundle(
            ENGINE,
            body.user_id,
            body.approach,
            body.cf_method,
            body.top_k,
            kb_params,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return payload


@app.get("/api/evaluation")
def evaluation(holdout_ratio: float = 0.22, max_users: Optional[int] = None):
    if ENGINE is None:
        raise HTTPException(503, "Engine not ready")
    return ranking_evaluation(ENGINE, holdout_ratio=holdout_ratio, max_users=max_users)


# Serve marketing pages + recommender UI from repo root (same origin as /api/*).
_SITE_ROOT = Path(__file__).resolve().parent.parent


@app.get("/")
def site_index():
    index = _SITE_ROOT / "index.html"
    if not index.is_file():
        raise HTTPException(404, "index.html not found")
    return FileResponse(index)


if _SITE_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=str(_SITE_ROOT), html=True), name="site")

