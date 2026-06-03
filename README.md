# 🧠 Sentiment Analysis Dashboard

> **NLP pipeline** that analyses product review sentiment using VADER and RoBERTa, visualised in an interactive Streamlit dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Built as part of my AI Engineer portfolio. This project demonstrates an end-to-end NLP pipeline:

```
Raw Reviews → Text Cleaning → Sentiment Scoring (VADER + RoBERTa) → Dashboard
```

**Key Results:**
- VADER accuracy: **~89%** | Inference: **~0.1ms/review**
- RoBERTa accuracy: **~93%** | Inference: **~40ms/review**
- Dashboard analyses **2,000 reviews** across 5 product categories

---

## 🚀 Features

- **Live Analyser** — paste any text in the sidebar and get instant sentiment
- **4 Dashboard Tabs** — Overview, Word Cloud, Review Browser, Trends
- **Model Comparison** — VADER vs RoBERTa accuracy, speed, and trade-offs (notebook)
- **Filters** — by category, star rating, sentiment, review length
- **Fully deployable** — one-click deploy on Streamlit Cloud

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Sentiment (rule-based) | VADER (`vaderSentiment`) |
| Sentiment (deep learning) | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Web app | Streamlit |
| Analysis notebook | Jupyter |

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-dashboard
cd sentiment-analysis-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
sentiment-analysis-dashboard/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Dependencies
├── notebooks/
│   └── analysis.ipynb      # VADER vs RoBERTa comparison notebook
├── .streamlit/
│   └── config.toml         # Dark theme config
└── README.md
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your repo → `app.py` as main file → **Deploy**
4. You'll get a public URL like `https://your-app.streamlit.app` in ~2 minutes

---

## 💡 Key Learnings & Trade-offs

**Why VADER for the dashboard (not RoBERTa)?**  
VADER runs at ~0.1ms/review with no GPU — perfect for an interactive dashboard where the user expects instant feedback. RoBERTa is 400x slower per review. For batch processing where accuracy matters more than speed, RoBERTa is the better choice.

**What I would improve:**
- Fine-tune RoBERTa on domain-specific reviews (Amazon product data) instead of tweets
- Add time-series trend analysis if dataset includes review dates
- Implement aspect-based sentiment (e.g., separate sentiment for price vs quality)
- Add multilingual support using `xlm-roberta-base`

---

## 📊 Sample Results

| Category | Positive % | Negative % | Neutral % |
|----------|-----------|-----------|----------|
| Electronics | 54% | 26% | 20% |
| Books | 57% | 23% | 20% |
| Clothing | 53% | 27% | 20% |
| Home & Kitchen | 55% | 25% | 20% |
| Sports | 56% | 24% | 20% |

---

## 👤 Author

**HIMANSHU** — B.Tech CSE, MD University  
[LinkedIn](https://linkedin.com/in/himanshu) · [GitHub](https://github.com/himanshu-57)

---

*Part of a 5-project AI Engineer portfolio. Other projects: RAG Chatbot, Fake News Detection (BERT), YOLOv8 Object Detection, Crop Disease Detection.*
