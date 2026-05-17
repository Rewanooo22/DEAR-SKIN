"""Generate deterministic synthetic ratings + catalog for demo / coursework."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(42)

CATEGORIES = ["Serum", "Cleanser", "Moisturizer", "Sunscreen", "Treatment"]
BRANDS = ["DearGlow", "PureLab", "SkinTheory", "AquaVeil", "BareEssence"]
TAG_POOL = (
    "hydrating clarifying anti-aging SPF ceramide niacinamide retinol vitamin-c "
    "gentle exfoliating barrier-repair oil-control sensitive acne prone "
    "brightening peptide antioxidant fragrance-free vegan soothing niacinamide-rich "
    "PHA AHA BHA moisture-lock redness calming pore-refining firming lifting "
    "pollution-defense mineral-filter dewy matte reef-safe".split()
)


def ensure_data_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.csv"
    ratings_path = out_dir / "ratings.csv"

    n_users = 60
    n_items = 72

    item_rows = []
    for i in range(n_items):
        cat = CATEGORIES[i % len(CATEGORIES)]
        brand = BRANDS[(i * 3) % len(BRANDS)]
        price = round(float(RNG.uniform(12, 95)), 2)
        tags = " ".join(sorted(RNG.choice(TAG_POOL, size=6, replace=False)))
        item_rows.append(
            {
                "item_id": i,
                "name": f"{brand} {cat} #{i + 1}",
                "category": cat,
                "brand": brand,
                "price": price,
                "tags": tags,
            }
        )

    # Latent preferences per user → correlated ratings (denser signal for CF)
    user_topics = RNG.standard_normal((n_users, 6))
    item_topics = RNG.standard_normal((n_items, 6))
    base_scores = user_topics @ item_topics.T

    ratings_rows = []
    for u in range(n_users):
        # Each user rates ~35–55 items so CF neighborhoods exist
        n_rate = int(RNG.integers(35, 56))
        chosen = RNG.choice(n_items, size=n_rate, replace=False)
        for i in chosen:
            noise = RNG.normal(0, 0.65)
            r = base_scores[u, i] + noise
            r = float(np.clip(round(r + 3.25, 2), 1.0, 5.0))
            ratings_rows.append({"user_id": u, "item_id": int(i), "rating": r})

    with items_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["item_id", "name", "category", "brand", "price", "tags"],
        )
        w.writeheader()
        w.writerows(item_rows)

    with ratings_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "rating"])
        w.writeheader()
        w.writerows(ratings_rows)


if __name__ == "__main__":
    ensure_data_dir(Path(__file__).resolve().parent / "data")
