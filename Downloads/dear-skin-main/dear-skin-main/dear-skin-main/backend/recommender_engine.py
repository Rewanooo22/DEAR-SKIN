"""Recommendation algorithms + ranking evaluation for coursework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"


CFMethod = str  # user_cosine | user_pearson | item_cosine | mf_svd


@dataclass
class CatalogItem:
    item_id: int
    name: str
    category: str
    brand: str
    price: float
    tags: str


class RecommendationEngine:
    """Loads ratings matrix + catalog; exposes CF / CB / KB + evaluation."""

    relevance_threshold = 4.0

    def __init__(self) -> None:
        items_df = pd.read_csv(DATA_DIR / "items.csv")
        ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")

        self.n_users = int(ratings_df["user_id"].max()) + 1
        self.n_items = len(items_df)

        self.items: dict[int, CatalogItem] = {}
        for _, row in items_df.iterrows():
            self.items[int(row["item_id"])] = CatalogItem(
                item_id=int(row["item_id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                brand=str(row["brand"]),
                price=float(row["price"]),
                tags=str(row["tags"]),
            )

        self.R = np.full((self.n_users, self.n_items), np.nan, dtype=float)
        for _, row in ratings_df.iterrows():
            u, i = int(row["user_id"]), int(row["item_id"])
            self.R[u, i] = float(row["rating"])

        corpus = [
            f"{it.category} {it.brand} {it.tags}" for it in sorted(self.items.values(), key=lambda x: x.item_id)
        ]
        self._tfidf = TfidfVectorizer(max_features=120, min_df=1)
        self._item_features = self._tfidf.fit_transform(corpus)

    # --- Matrix helpers ---
    def _user_mean(self, R: np.ndarray, u: int) -> float:
        row = R[u]
        mask = ~np.isnan(row)
        return float(np.nanmean(row)) if mask.any() else 3.0

    def _masked_predict_fill(self, R_train: np.ndarray) -> np.ndarray:
        """Global mean fill for MF stability."""
        overall = np.nanmean(R_train)
        X = np.where(np.isnan(R_train), overall, R_train)
        return X, overall

    # --- Similarities ---
    def _sim_cosine_users(self, R: np.ndarray, u: int, v: int) -> float:
        mask = (~np.isnan(R[u])) & (~np.isnan(R[v]))
        if mask.sum() < 2:
            return 0.0
        a, b = R[u][mask], R[v][mask]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _sim_pearson_users(self, R: np.ndarray, u: int, v: int) -> float:
        mask = (~np.isnan(R[u])) & (~np.isnan(R[v]))
        if mask.sum() < 2:
            return 0.0
        a, b = R[u][mask].astype(float), R[v][mask].astype(float)
        a -= a.mean()
        b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _sim_cosine_items(self, R: np.ndarray, i: int, j: int) -> float:
        mask = (~np.isnan(R[:, i])) & (~np.isnan(R[:, j]))
        if mask.sum() < 2:
            return 0.0
        a, b = R[mask, i], R[mask, j]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    # --- Collaborative filtering ---
    def _predict_cf_user_based(
        self, R_train: np.ndarray, user: int, item: int, kind: str
    ) -> tuple[float | None, list[tuple[int, float]]]:
        if not np.isnan(R_train[user, item]):
            return float(R_train[user, item]), []

        neighbors: list[tuple[int, float]] = []
        sim_fn = self._sim_cosine_users if kind == "cosine" else self._sim_pearson_users

        for v in range(self.n_users):
            if v == user:
                continue
            if np.isnan(R_train[v, item]):
                continue
            s = sim_fn(R_train, user, v)
            if s > 0:
                neighbors.append((v, s))

        neighbors.sort(key=lambda x: -x[1])
        neighbors = neighbors[:35]

        if not neighbors:
            return None, []

        mu_u = self._user_mean(R_train, user)
        num = 0.0
        den = 0.0
        used_nn: list[tuple[int, float]] = []
        for v, s in neighbors:
            rv_item = R_train[v, item]
            mu_v = self._user_mean(R_train, v)
            num += s * (rv_item - mu_v)
            den += abs(s)
            used_nn.append((v, s))

        if den < 1e-9:
            return None, used_nn

        pred = mu_u + num / den
        pred = float(np.clip(pred, 1.0, 5.0))
        return pred, used_nn

    def _predict_cf_item_based(
        self, R_train: np.ndarray, user: int, item: int
    ) -> tuple[float | None, list[tuple[int, float]]]:
        if not np.isnan(R_train[user, item]):
            return float(R_train[user, item]), []

        rated_items = np.where(~np.isnan(R_train[user]))[0]
        sims: list[tuple[int, float]] = []
        for j in rated_items:
            if int(j) == item:
                continue
            sim = self._sim_cosine_items(R_train, item, int(j))
            if sim > 0:
                sims.append((int(j), sim))

        sims.sort(key=lambda x: -x[1])
        sims = sims[:35]

        if not sims:
            return None, []

        num = 0.0
        den = 0.0
        for j, s in sims:
            num += s * R_train[user, j]
            den += abs(s)

        pred = float(np.clip(num / den, 1.0, 5.0))
        return pred, sims

    def _fit_mf(self, R_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        X, overall = self._masked_predict_fill(R_train)
        Xc = X - X.mean()
        k = min(12, min(X.shape) - 1)
        if k < 1:
            return np.zeros_like(X), np.zeros_like(X), overall
        U, sigma, Vt = svds(Xc, k=k)
        Sigma = np.diag(sigma)
        recon = U @ Sigma @ Vt + X.mean()
        return recon, X, overall

    def _predict_mf(self, recon: np.ndarray, user: int, item: int) -> float:
        return float(np.clip(recon[user, item], 1.0, 5.0))

    def _score_all_cf(
        self, R_train: np.ndarray, user: int, method: CFMethod
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scores = np.full(self.n_items, np.nan)
        aux: dict[str, Any] = {"mf_recon": None}

        if method == "mf_svd":
            recon, _, _ = self._fit_mf(R_train)
            aux["mf_recon"] = recon
            for i in range(self.n_items):
                if np.isnan(R_train[user, i]):
                    scores[i] = self._predict_mf(recon, user, i)
            return scores, aux

        for i in range(self.n_items):
            if not np.isnan(R_train[user, i]):
                scores[i] = R_train[user, i]
                continue

            if method == "user_cosine":
                pred, _ = self._predict_cf_user_based(R_train, user, i, "cosine")
            elif method == "user_pearson":
                pred, _ = self._predict_cf_user_based(R_train, user, i, "pearson")
            elif method == "item_cosine":
                pred, _ = self._predict_cf_item_based(R_train, user, i)
            else:
                pred = None

            scores[i] = pred if pred is not None else np.nan

        # Fallback for cold spots: column mean popularity prior
        col_means = np.nanmean(np.where(np.isnan(R_train), np.nan, R_train), axis=0)
        overall = np.nanmean(col_means)
        for i in range(self.n_items):
            if np.isnan(scores[i]):
                scores[i] = float(col_means[i]) if not np.isnan(col_means[i]) else overall

        return scores, aux

    # --- Content-based ---
    def _user_profile_vector(self, R_train: np.ndarray, user: int):
        from scipy.sparse import diags, vstack as sp_vstack

        rated = np.where(~np.isnan(R_train[user]))[0]
        if len(rated) == 0:
            return None
        weights: list[float] = []
        rows = []
        for idx in rated:
            r = R_train[user, idx]
            w = max(float(r - 2.5), 0.15)
            weights.append(w)
            rows.append(self._item_features[idx])
        stacked = sp_vstack(rows)
        tw = sum(weights)
        if tw < 1e-9:
            return None
        prof = diags(weights, format="csr") @ stacked
        return prof.sum(axis=0) / tw

    # --- Knowledge-based ---
    def knowledge_scores(
        self,
        user: int,
        max_budget: float | None,
        categories: list[str] | None,
        brands: list[str] | None,
        prefer_fragrance_free: bool,
        R_train: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
        """Higher score = better match to constraints + predicted liking."""

        cf_scr, _ = self._score_all_cf(R_train, user, "item_cosine")
        meta: dict[int, dict[str, Any]] = {}

        out = np.zeros(self.n_items)
        cats = {c.strip().lower() for c in (categories or []) if c.strip()}
        brs = {b.strip().lower() for b in (brands or []) if b.strip()}

        for iid, it in self.items.items():
            penalty = 1.0
            reasons: list[str] = []

            if max_budget is not None and it.price > max_budget:
                penalty *= 0.15
                reasons.append(f"above budget (${it.price:.0f} > ${max_budget:.0f})")
            elif max_budget is not None:
                reasons.append(f"within budget (≤ ${max_budget:.0f})")

            if cats:
                if it.category.lower() in cats:
                    penalty *= 1.35
                    reasons.append(f"category matches ({it.category})")
                else:
                    penalty *= 0.55
                    reasons.append(f"outside preferred categories")

            if brs:
                if it.brand.lower() in brs:
                    penalty *= 1.35
                    reasons.append(f"brand matches ({it.brand})")
                else:
                    penalty *= 0.45
                    reasons.append("different brand than filters")

            if prefer_fragrance_free and "fragrance-free" in it.tags.lower():
                penalty *= 1.25
                reasons.append("fragrance-free tag matches preference")

            fit = penalty * (0.65 + 0.07 * cf_scr[iid])
            out[iid] = fit
            meta[iid] = {"reason_tags": reasons}

        return out, meta


def content_scores(engine: RecommendationEngine, R_train: np.ndarray, user: int) -> np.ndarray:
    prof = engine._user_profile_vector(R_train, user)
    if prof is None:
        return np.full(engine.n_items, 3.0)
    vec = np.squeeze(np.asarray(prof)).reshape(1, -1)
    sims = cosine_similarity(vec, engine._item_features).ravel()
    return (sims + 1.0) * 2.5  # scale roughly into rating-like [0,5]


def explain_cf(
    engine: RecommendationEngine,
    method: CFMethod,
    user: int,
    item: int,
    R_train: np.ndarray,
    aux: dict[str, Any],
) -> str:
    if method == "user_cosine":
        _, nn = engine._predict_cf_user_based(R_train, user, item, "cosine")
        if nn:
            top_v, top_s = nn[0]
            return (
                "Collaborative (user-based cosine): recommended because users whose ratings "
                f"aligned closely with yours (closest neighbor user #{top_v}, cosine similarity "
                f"{top_s:.2f}) rated this product strongly."
            )
        return (
            "Collaborative (user-based cosine): neighbors who overlap with your taste "
            "rated this item higher than their typical baseline."
        )

    if method == "user_pearson":
        _, nn = engine._predict_cf_user_based(R_train, user, item, "pearson")
        if nn:
            top_v, top_s = nn[0]
            return (
                "Collaborative (user-based Pearson): recommended because users with correlated "
                f"taste patterns (e.g. user #{top_v}, Pearson r~{top_s:.2f}) appreciated this item."
            )
        return (
            "Collaborative (user-based Pearson): prediction driven by taste correlations "
            "on items you both rated."
        )

    if method == "item_cosine":
        _, pairs = engine._predict_cf_item_based(R_train, user, item)
        if pairs:
            top_j, top_s = pairs[0]
            liked_name = engine.items[top_j].name
            return (
                "Collaborative (item-based cosine): recommended because it is similar to "
                f"items you rated ({liked_name}), cosine item-item similarity~{top_s:.2f}."
            )
        return (
            "Collaborative (item-based): neighbors in item-space resemble products "
            "you already rated positively."
        )

    recon = aux.get("mf_recon") if aux else None
    if method == "mf_svd" and recon is not None:
        return (
            "Collaborative (matrix factorization / SVD): recommended because your latent taste "
            "vector and this product's latent factor loadings combine to a high recovered score "
            "(low-rank structure learned from many user–item interactions)."
        )
    return "Collaborative filtering: score inferred from aggregated interaction patterns."


def explain_content(engine: RecommendationEngine, item_id: int, user: int, R_train: np.ndarray) -> str:
    it = engine.items[item_id]
    rated = [(j, R_train[user, j]) for j in np.where(~np.isnan(R_train[user]))[0]]
    rated.sort(key=lambda x: -x[1])
    liked_ids = [j for j, r in rated[:5] if r >= engine.relevance_threshold]
    liked_names = [engine.items[j].name for j in liked_ids[:3]]
    overlap_hint = ""
    if liked_names:
        overlap_hint = f" Content aligns with items you rated highly ({', '.join(liked_names)})."
    return (
        "Content-based: recommended because its category, brand, and ingredient tags overlap "
        f"with a TF-IDF profile built from your past ratings (here: {it.category}, "
        f"{it.brand}).{overlap_hint}"
    )


def explain_kb(engine: RecommendationEngine, item_id: int, kb_meta: dict[str, Any]) -> str:
    tags = kb_meta.get("reason_tags") or []
    detail = "; ".join(tags) if tags else "constraints applied"
    return (
        "Knowledge-based: recommended because it satisfies your stated shopping rules "
        f"({detail}), then ranked against predicted affinity from collaborative signals."
    )


def precision_at_k(rec: list[int], rel: set[int], k: int) -> float:
    rec_k = rec[:k]
    if k == 0:
        return 0.0
    hits = len(set(rec_k) & rel)
    return hits / k


def recall_at_k(rec: list[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    hits = len(set(rec[:k]) & rel)
    return hits / len(rel)


def average_precision_at_k(rec: list[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    hits = 0
    acc = 0.0
    for i, x in enumerate(rec[:k], start=1):
        if x in rel:
            hits += 1
            acc += hits / i
    return acc / len(rel)


def ndcg_at_k(rec: list[int], rel: set[int], k: int) -> float:
    def dcg(ids: list[int]) -> float:
        s = 0.0
        for idx, x in enumerate(ids, start=1):
            rel_gain = 1.0 if x in rel else 0.0
            s += rel_gain / np.log2(idx + 1)
        return s

    ideal_len = min(k, len(rel))

    def ideal_dcg() -> float:
        s = 0.0
        for idx in range(1, k + 1):
            gain = 1.0 if idx <= ideal_len else 0.0
            s += gain / np.log2(idx + 1)
        return s

    num = dcg(rec[:k])
    den = ideal_dcg()
    return float(num / den) if den > 0 else 0.0


def ranking_evaluation(
    engine: RecommendationEngine,
    holdout_ratio: float = 0.22,
    seed: int = 7,
    max_users: Optional[int] = None,
):
    """Hold out rated items per user; rank on train; measure overlap with relevant held-out."""
    rng = np.random.default_rng(seed)

    cf_methods = ["user_cosine", "user_pearson", "item_cosine", "mf_svd"]

    K = 10
    agg: dict[str, dict[str, list[float]]] = {}

    default_kb = {"max_budget": 85.0, "categories": [], "brands": [], "prefer_fragrance_free": False}

    users_list = list(range(engine.n_users))
    rng.shuffle(users_list)
    if max_users is not None:
        users_list = users_list[: max(1, min(max_users, len(users_list)))]

    for user in users_list:
        rated_idx = np.where(~np.isnan(engine.R[user]))[0]
        if len(rated_idx) < 8:
            continue

        rng.shuffle(rated_idx)
        n_hide = max(2, int(len(rated_idx) * holdout_ratio))
        hide = set(map(int, rated_idx[:n_hide]))
        train_mask = np.ones_like(engine.R[user])
        for h in hide:
            train_mask[h] = np.nan
        R_train = engine.R.copy()
        R_train[user] = np.where(np.isnan(train_mask), np.nan, engine.R[user])

        rel = {
            int(j)
            for j in hide
            if engine.R[user, j] >= engine.relevance_threshold
        }
        if not rel:
            continue

        rated_train = set(np.where(~np.isnan(R_train[user]))[0])

        def recommend_scores(scores: np.ndarray) -> list[int]:
            candidates = [(i, scores[i]) for i in range(engine.n_items) if i not in rated_train]
            candidates.sort(key=lambda x: (-x[1], x[0]))
            return [i for i, _ in candidates[: max(K * 3, 30)]]

        # CF variants
        for m in cf_methods:
            label = f"cf_{m}"
            agg.setdefault(label, {"P@K": [], "R@K": [], "MAP@K": [], "NDCG@K": []})
            scr, _ = engine._score_all_cf(R_train, user, m)
            rec = recommend_scores(scr)
            agg[label]["P@K"].append(precision_at_k(rec, rel, K))
            agg[label]["R@K"].append(recall_at_k(rec, rel, K))
            agg[label]["MAP@K"].append(average_precision_at_k(rec, rel, K))
            agg[label]["NDCG@K"].append(ndcg_at_k(rec, rel, K))

        # Content
        cs = content_scores(engine, R_train, user)
        rec_c = recommend_scores(cs)
        key_c = "content_tfidf"
        agg.setdefault(key_c, {"P@K": [], "R@K": [], "MAP@K": [], "NDCG@K": []})
        agg[key_c]["P@K"].append(precision_at_k(rec_c, rel, K))
        agg[key_c]["R@K"].append(recall_at_k(rec_c, rel, K))
        agg[key_c]["MAP@K"].append(average_precision_at_k(rec_c, rel, K))
        agg[key_c]["NDCG@K"].append(ndcg_at_k(rec_c, rel, K))

        # Knowledge (same KB profile for fairness — student can tune in UI)
        ks, _ = engine.knowledge_scores(
            user,
            default_kb["max_budget"],
            default_kb["categories"],
            default_kb["brands"],
            default_kb["prefer_fragrance_free"],
            R_train,
        )
        rec_k = recommend_scores(ks)
        key_k = "knowledge_rules"
        agg.setdefault(key_k, {"P@K": [], "R@K": [], "MAP@K": [], "NDCG@K": []})
        agg[key_k]["P@K"].append(precision_at_k(rec_k, rel, K))
        agg[key_k]["R@K"].append(recall_at_k(rec_k, rel, K))
        agg[key_k]["MAP@K"].append(average_precision_at_k(rec_k, rel, K))
        agg[key_k]["NDCG@K"].append(ndcg_at_k(rec_k, rel, K))

    def summarize(tag: str) -> dict[str, float]:
        bucket = agg.get(tag)
        if not bucket:
            return {}
        return {m: float(np.mean(vals)) if vals else 0.0 for m, vals in bucket.items()}

    cf_summaries = {m: summarize(f"cf_{m}") for m in cf_methods}
    content_summary = summarize("content_tfidf")
    kb_summary = summarize("knowledge_rules")

    analysis = build_analysis_text(cf_summaries, content_summary, kb_summary)

    return {
        "K": K,
        "users_sampled": len(users_list),
        "metrics_explained": [
            "Precision@K — fraction of top-K picks that were truly relevant (held-out liked items).",
            "Recall@K — fraction of held-out likes recovered inside top-K.",
            "MAP@K — rewards ranking relevant items nearer the top.",
            "NDCG@K — weighs gains by logarithmic rank discount ( graded relevance surrogate ).",
        ],
        "collaborative_filtering": cf_summaries,
        "content_based": content_summary,
        "knowledge_based": kb_summary,
        "analysis": analysis,
    }


def build_analysis_text(
    cf_summaries: dict[str, dict[str, float]],
    content_summary: dict[str, float],
    kb_summary: dict[str, float],
) -> dict[str, str]:
    """Pick winners + qualitative rationale."""

    def score_tuple(name: str, d: dict[str, float]) -> tuple[str, float]:
        # Composite score for headline winner (student interprets tables too)
        if not d:
            return name, -1
        composite = (
            d.get("MAP@K", 0) * 0.45 + d.get("NDCG@K", 0) * 0.35 + d.get("P@K", 0) * 0.20
        )
        return name, composite

    cf_best = max(cf_summaries.items(), key=lambda kv: score_tuple(kv[0], kv[1])[1])
    overall_candidates = [
        ("Collaborative:" + cf_best[0], cf_best[1]),
        ("Content-TFIDF", content_summary),
        ("Knowledge-rules", kb_summary),
    ]
    overall_best = max(overall_candidates, key=lambda kv: score_tuple(kv[0], kv[1])[1])

    cf_best_name, cf_best_metrics = cf_best
    reasons = []

    reasons.append(
        "Collaborative variants differ because cosine vs Pearson emphasize magnitude vs centered taste; "
        "item-based leverages product similarity instead of user neighborhoods; MF captures latent factors "
        "that pairwise similarities miss—usually strongest when data is dense."
    )

    reasons.append(
        "Content-based profiling excels when interactions are sparse but textual metadata is rich; "
        "it can diverge from CF when tags capture ingredient preferences CF cannot yet observe."
    )

    reasons.append(
        "Knowledge-based filtering injects explicit constraints (budget/brand/category); rankings shift "
        "even when affinity scores tie, explaining divergent lists versus purely behavioral CF."
    )

    conditions = (
        "Typical patterns on this synthetic storefront: MF/item-CF shine with dense ratings; "
        "Pearson user-CF helps when users rate on different scales; KB wins interpretability on tight budgets; "
        "content picks stay coherent when metadata aligns with latent tastes."
    )

    return {
        "best_cf_method": cf_best_name,
        "best_cf_metric_snapshot": cf_best_metrics,
        "best_overall_approach": overall_best[0],
        "best_overall_metric_snapshot": overall_best[1],
        "why_differences_occur": " ".join(reasons),
        "when_each_performs_better": conditions,
    }


def recommend_bundle(
    engine: RecommendationEngine,
    user: int,
    approach: str,
    cf_method: CFMethod,
    top_k: int,
    kb_params: dict[str, Any],
) -> dict[str, Any]:
    """Produce recommendations + explanations for UI."""

    R_train = engine.R.copy()

    items_out: list[dict[str, Any]] = []

    if approach == "collaborative":
        scores, aux = engine._score_all_cf(R_train, user, cf_method)
        ranked = np.argsort(-scores)
        ranked = [int(i) for i in ranked if np.isnan(R_train[user, i])][:top_k]

        for rank, iid in enumerate(ranked, start=1):
            items_out.append(
                {
                    "rank": rank,
                    "item_id": iid,
                    "name": engine.items[iid].name,
                    "category": engine.items[iid].category,
                    "brand": engine.items[iid].brand,
                    "price": engine.items[iid].price,
                    "predicted_score": float(scores[iid]),
                    "approach": "collaborative_filtering",
                    "cf_method": cf_method,
                    "explanation": explain_cf(engine, cf_method, user, iid, R_train, aux),
                }
            )

    elif approach == "content":
        scores = content_scores(engine, R_train, user)
        ranked = np.argsort(-scores)
        ranked = [int(i) for i in ranked if np.isnan(R_train[user, i])][:top_k]
        for rank, iid in enumerate(ranked, start=1):
            items_out.append(
                {
                    "rank": rank,
                    "item_id": iid,
                    "name": engine.items[iid].name,
                    "category": engine.items[iid].category,
                    "brand": engine.items[iid].brand,
                    "price": engine.items[iid].price,
                    "predicted_score": float(scores[iid]),
                    "approach": "content_based",
                    "cf_method": None,
                    "explanation": explain_content(engine, iid, user, R_train),
                }
            )

    elif approach == "knowledge":
        ks, meta = engine.knowledge_scores(
            user,
            kb_params.get("max_budget"),
            kb_params.get("categories") or [],
            kb_params.get("brands") or [],
            bool(kb_params.get("prefer_fragrance_free")),
            R_train,
        )
        ranked = np.argsort(-ks)
        ranked = [int(i) for i in ranked if np.isnan(R_train[user, i])][:top_k]
        for rank, iid in enumerate(ranked, start=1):
            items_out.append(
                {
                    "rank": rank,
                    "item_id": iid,
                    "name": engine.items[iid].name,
                    "category": engine.items[iid].category,
                    "brand": engine.items[iid].brand,
                    "price": engine.items[iid].price,
                    "predicted_score": float(ks[iid]),
                    "approach": "knowledge_based",
                    "cf_method": None,
                    "explanation": explain_kb(engine, iid, meta.get(iid, {})),
                }
            )

    elif approach == "all":
        bundles = [
            recommend_bundle(engine, user, "collaborative", cf_method, top_k, kb_params),
            recommend_bundle(engine, user, "content", cf_method, top_k, kb_params),
            recommend_bundle(engine, user, "knowledge", cf_method, top_k, kb_params),
        ]
        return {"mode": "all", "panels": bundles}

    else:
        raise ValueError("Unknown approach")

    return {"mode": "single", "user_id": user, "results": items_out}
