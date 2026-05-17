# 🌸 DearSkin — *A Love Letter to Your Skin*

[![License](https://img.shields.io/badge/license-MIT-ff69b4)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-beige)]()
[![Supabase](https://img.shields.io/badge/Backend-Supabase-green)](https://supabase.com/)
[![Tech Stack](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-pink)]()

> ✨ *DearSkin* is a personalized skincare web app that recommends ingredients and products based on your skin type, concerns, and reviews from real users — all wrapped in a classy, minimalist UI.

---

## 🌟 Key Features

- **🧑‍⚕️ Personalized Skincare Profiles** — Custom onboarding for age, gender, skin type, and concerns
- **🤖 AI-Powered Ingredient Suggestions** — Get smart recommendations based on product ratings and skin concerns
- **🛍️ Marketplace-Driven Curation** — Product picks from Amazon, Sephora & more (static for MVP, dynamic in Phase 2)
- **🧪 Ingredient Compatibility Checker** — Prevents harmful combinations (e.g., AHA + Retinol)
- **📚 Skin Quiz (Phase 2)** — Determine skin type and get custom regimens
- **📸 AI Skin Diary (Future Vision)** — Use face scans to track your glow journey
- **💬 Community Stories** — Share experiences, tag products, follow others

---

## 🚀 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Supabase (Auth + PostgreSQL)
- **ML Model** : Python (FastAPI/Flask for API)
- **Libraries**: Chart.js (quiz), Anime.js (UI effects)

---

## 🧠 ML Recommendation Engine

- Dataset Sources:
  - [Skincare Products & Effects](https://www.kaggle.com/datasets/sayakmandal/skincare-products-ingredients-and-effects)
  - [INCI Ingredient List](https://www.kaggle.com/datasets/amaboh/skin-care-product-ingredients-inci-list)
  - [Sephora Product Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
- Approach:
  - Content-based filtering
  - Collaborative filtering
  - Acne/comedogenic/irritant scores

---

## ✨ What’s Next

- 🤖 AI-powered chatbot
- 🧬 Ingredient dupe finder
- ♻️ Sustainability tags (vegan, cruelty-free)
- 📅 Seasonal/hormonal skin routine updates

---

## 🧑‍💻 Getting Started (Local Setup)

1. Clone the repo and open the project folder.
2. **Marketing site (static pages):** serve the repo root with any static server, e.g.  
   `python -m http.server 8080`  
   then open `http://127.0.0.1:8080/index.html`.
3. **Intelligent recommender (coursework engine):**
   - `cd backend`
   - Create a virtual environment (recommended), then `pip install -r requirements.txt`
   - `python ensure_data.py` — writes `backend/data/*.csv` (synthetic ratings + catalog)
   - `uvicorn main:app --reload --host 127.0.0.1 --port 8000`
   - Open `http://127.0.0.1:8080/recommender.html` (or your static port) and set the API URL to `http://127.0.0.1:8000` if needed.
4. Full-dataset evaluation: `GET http://127.0.0.1:8000/api/evaluation` (omit `max_users`). The lab UI calls `?max_users=35` for a faster preview.

### What the recommender implements

- **Collaborative filtering (4 methods):** user-based cosine, user-based Pearson, item-based cosine, matrix factorization (truncated SVD).
- **Content-based:** TF-IDF profiles over category / brand / ingredient tags.
- **Knowledge-based:** budget, category, brand, and fragrance-free constraints fused with collaborative affinity.
- **Evaluation:** Precision@K, Recall@K, MAP@K, NDCG@K with per-user hold-out and narrative comparison helpers.

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more details.

---

## ❤️ Credits

Crafted with care by **Sanya Shresta**  
“A love letter to your skin.”
