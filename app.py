import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import time
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inline CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
    .positive { color: #00d4aa; }
    .negative { color: #ff4b6e; }
    .neutral  { color: #ffa500; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    div[data-testid="stSidebarContent"] { background: #161b2e; }
</style>
""", unsafe_allow_html=True)

# ── VADER (no model download needed) ─────────────────────────
@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None

# ── Load or generate dataset ──────────────────────────────────
@st.cache_data
def load_data():
    """
    In production: pd.read_csv('data/reviews.csv')
    For demo we generate a realistic synthetic dataset so the app
    runs instantly without a 500MB download.
    """
    np.random.seed(42)
    n = 2000

    categories = ["Electronics", "Books", "Clothing", "Home & Kitchen", "Sports"]
    positive_templates = [
        "Absolutely love this product, works perfectly and arrived on time.",
        "Great quality for the price, highly recommend to everyone.",
        "Exceeded my expectations, will definitely buy again.",
        "Outstanding product, customer service was also excellent.",
        "Very happy with my purchase, fast delivery and great packaging.",
        "Works exactly as described, very satisfied with the quality.",
        "Five stars without hesitation, fantastic product overall.",
        "Amazing value, much better than I expected honestly.",
    ]
    negative_templates = [
        "Terrible quality, broke after just two days of use.",
        "Very disappointed, does not match the description at all.",
        "Waste of money, completely useless product.",
        "Stopped working after a week, very poor quality control.",
        "Would not recommend, packaging was damaged and product missing parts.",
        "Horrible experience, customer service was unhelpful and rude.",
        "Total scam, nothing like the pictures shown online.",
        "Returned immediately, product was defective out of the box.",
    ]
    neutral_templates = [
        "It's okay, nothing special but does the job.",
        "Average product, meets basic expectations nothing more.",
        "Decent quality for the price, could be better though.",
        "Works fine but the instructions were a bit confusing.",
        "Product is acceptable, delivery took longer than expected.",
        "Neither great nor terrible, just an average item.",
        "It functions as advertised but feels a bit cheaply made.",
        "Not bad, not great, somewhere in the middle really.",
    ]

    weights = [0.55, 0.25, 0.20]  # positive, negative, neutral
    sentiments, texts, stars, cats = [], [], [], []

    for _ in range(n):
        s = np.random.choice(["positive", "negative", "neutral"], p=weights)
        sentiments.append(s)
        if s == "positive":
            texts.append(np.random.choice(positive_templates))
            stars.append(np.random.choice([4, 5], p=[0.3, 0.7]))
        elif s == "negative":
            texts.append(np.random.choice(negative_templates))
            stars.append(np.random.choice([1, 2], p=[0.6, 0.4]))
        else:
            texts.append(np.random.choice(neutral_templates))
            stars.append(3)
        cats.append(np.random.choice(categories))

    df = pd.DataFrame({
        "review_text": texts,
        "sentiment_label": sentiments,
        "star_rating": stars,
        "category": cats,
        "helpful_votes": np.random.randint(0, 150, n),
        "review_length": [len(t.split()) for t in texts],
    })
    return df

# ── Sentiment scorer ──────────────────────────────────────────
def score_vader(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound

def get_top_keywords(texts, n=10):
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for",
                 "of","with","it","is","was","this","that","my","i","very",
                 "product","item","buy","got","get","purchased","ordered"}
    words = []
    for t in texts:
        words += [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', t)
                  if w.lower() not in stopwords]
    return Counter(words).most_common(n)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🎛️ Filters")

df_raw = load_data()

categories_all = ["All"] + sorted(df_raw["category"].unique().tolist())
sel_category = st.sidebar.selectbox("Product Category", categories_all)

sel_stars = st.sidebar.multiselect(
    "Star Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
)

sel_sentiment = st.sidebar.multiselect(
    "Sentiment", ["positive", "negative", "neutral"],
    default=["positive", "negative", "neutral"]
)

min_len, max_len = int(df_raw.review_length.min()), int(df_raw.review_length.max())
sel_len = st.sidebar.slider("Review Length (words)", min_len, max_len, (min_len, max_len))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Live Analyser")
user_text = st.sidebar.text_area("Paste any review here:", height=100,
    placeholder="Type or paste a review to analyse it instantly...")

# ── Apply filters ─────────────────────────────────────────────
df = df_raw.copy()
if sel_category != "All":
    df = df[df["category"] == sel_category]
if sel_stars:
    df = df[df["star_rating"].isin(sel_stars)]
if sel_sentiment:
    df = df[df["sentiment_label"].isin(sel_sentiment)]
df = df[(df["review_length"] >= sel_len[0]) & (df["review_length"] <= sel_len[1])]

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# 🧠 Sentiment Analysis Dashboard")
st.markdown(
    "**NLP pipeline** — VADER lexicon scorer on Amazon-style product reviews. "
    "Filter by category, rating, or review length to explore sentiment patterns."
)
st.markdown("---")

# ── Live analyser result ──────────────────────────────────────
analyzer = load_vader()
if user_text.strip() and analyzer:
    label, score = score_vader(user_text, analyzer)
    emoji = {"positive": "✅", "negative": "❌", "neutral": "⚪"}[label]
    color = {"positive": "#00d4aa", "negative": "#ff4b6e", "neutral": "#ffa500"}[label]
    st.markdown(
        f"<div style='background:#1e2130;border-radius:10px;padding:16px 20px;"
        f"border-left:4px solid {color};margin-bottom:20px'>"
        f"<b style='color:{color}'>{emoji} Live Analysis: {label.upper()}</b>"
        f"<span style='color:#aaa;font-size:0.85rem;margin-left:16px'>Compound score: {score:.3f}</span>"
        f"<p style='margin:6px 0 0;color:#ccc;font-size:0.9rem'>{user_text[:200]}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── KPI row ───────────────────────────────────────────────────
total   = len(df)
pos_pct = (df.sentiment_label == "positive").mean() * 100
neg_pct = (df.sentiment_label == "negative").mean() * 100
neu_pct = (df.sentiment_label == "neutral").mean()  * 100
avg_star= df.star_rating.mean()

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, css in [
    (c1, f"{total:,}",    "Total Reviews",      ""),
    (c2, f"{pos_pct:.1f}%", "Positive",         "positive"),
    (c3, f"{neg_pct:.1f}%", "Negative",         "negative"),
    (c4, f"{neu_pct:.1f}%", "Neutral",          "neutral"),
    (c5, f"{avg_star:.2f}★","Avg Star Rating",  ""),
]:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value {css}'>{val}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "☁️ Word Cloud", "📋 Reviews", "📈 Trends"])

COLORS = {"positive": "#00d4aa", "negative": "#ff4b6e", "neutral": "#ffa500"}

# ── TAB 1: Overview ───────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Sentiment Distribution")
        counts = df["sentiment_label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        bars = ax.bar(counts.index,
                      counts.values,
                      color=[COLORS.get(s, "#888") for s in counts.index],
                      width=0.5, edgecolor="none")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(val), ha="center", va="bottom", color="white", fontsize=11)
        ax.set_xlabel("Sentiment", color="#aaa")
        ax.set_ylabel("Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", color="#2e2e3e", linewidth=0.5)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Sentiment by Category")
        cat_sent = df.groupby(["category", "sentiment_label"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        cols_order = [c for c in ["positive", "neutral", "negative"] if c in cat_sent.columns]
        bottom = np.zeros(len(cat_sent))
        for s in cols_order:
            ax.barh(cat_sent.index, cat_sent[s], left=bottom,
                    color=COLORS[s], label=s, height=0.6)
            bottom += cat_sent[s].values
        ax.set_xlabel("Review Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_visible(False)
        patches = [mpatches.Patch(color=COLORS[s], label=s) for s in cols_order]
        ax.legend(handles=patches, loc="lower right",
                  facecolor="#1e2130", labelcolor="white", fontsize=9)
        st.pyplot(fig)
        plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Star Rating Distribution")
        star_counts = df["star_rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.bar(star_counts.index, star_counts.values,
               color=["#ff4b6e","#ff8c42","#ffa500","#a0d4a0","#00d4aa"],
               width=0.6, edgecolor="none")
        ax.set_xlabel("Stars", color="#aaa")
        ax.set_ylabel("Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", color="#2e2e3e", linewidth=0.5)
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.markdown("#### Top 10 Keywords")
        kw = get_top_keywords(df["review_text"].tolist())
        if kw:
            words, freqs = zip(*kw)
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            colors_kw = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))
            ax.barh(words[::-1], freqs[::-1], color=colors_kw, height=0.6)
            ax.tick_params(colors="#aaa")
            ax.spines[:].set_visible(False)
            ax.grid(axis="x", color="#2e2e3e", linewidth=0.5)
            st.pyplot(fig)
            plt.close()

# ── TAB 2: Word Cloud ─────────────────────────────────────────
with tab2:
    st.markdown("#### Word Cloud by Sentiment")
    wc_sent = st.radio("Select sentiment:", ["positive", "negative", "neutral"],
                       horizontal=True)
    subset = df[df["sentiment_label"] == wc_sent]["review_text"]
    if len(subset) > 0:
        text_blob = " ".join(subset.tolist())
        stopwords_wc = {"the","a","an","and","or","but","in","on","at","to","for",
                        "of","with","it","is","was","this","that","very","just","so"}
        wc = WordCloud(
            width=900, height=400,
            background_color="#0e1117",
            colormap="cool" if wc_sent == "positive" else "autumn" if wc_sent == "negative" else "YlOrBr",
            stopwords=stopwords_wc,
            max_words=80,
            prefer_horizontal=0.8
        ).generate(text_blob)
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0e1117")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
        st.caption(f"Generated from {len(subset):,} {wc_sent} reviews")

# ── TAB 3: Reviews table ──────────────────────────────────────
with tab3:
    st.markdown("#### Sample Reviews")
    n_show = st.slider("Reviews to show", 5, 50, 15)
    sample = df.sample(min(n_show, len(df)), random_state=1)[
        ["review_text", "sentiment_label", "star_rating", "category", "helpful_votes"]
    ].rename(columns={
        "review_text": "Review",
        "sentiment_label": "Sentiment",
        "star_rating": "Stars",
        "category": "Category",
        "helpful_votes": "Helpful Votes"
    })

    def color_sentiment(val):
        c = {"positive": "#00d4aa", "negative": "#ff4b6e", "neutral": "#ffa500"}.get(val, "")
        return f"color: {c}; font-weight: 600"

    try:
        styled = sample.style.map(color_sentiment, subset=["Sentiment"])
    except AttributeError:
        styled = sample.style.applymap(color_sentiment, subset=["Sentiment"])
    st.dataframe(styled, use_container_width=True, height=420)

# ── TAB 4: Trends ─────────────────────────────────────────────
with tab4:
    st.markdown("#### Sentiment vs Star Rating (Scatter)")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    ia = SentimentIntensityAnalyzer()
    sample_trend = df.sample(min(300, len(df)), random_state=7).copy()
    sample_trend["compound"] = sample_trend["review_text"].apply(
        lambda t: ia.polarity_scores(t)["compound"]
    )

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ["positive", "negative", "neutral"]:
        sub = sample_trend[sample_trend["sentiment_label"] == s]
        ax.scatter(sub["star_rating"] + np.random.uniform(-0.15, 0.15, len(sub)),
                   sub["compound"],
                   c=COLORS[s], alpha=0.55, s=30, label=s, edgecolors="none")
    ax.axhline(0.05,  color="#555", linewidth=0.8, linestyle="--")
    ax.axhline(-0.05, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Star Rating", color="#aaa")
    ax.set_ylabel("VADER Compound Score", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_visible(False)
    ax.grid(color="#2e2e3e", linewidth=0.4)
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Review Length vs Sentiment")
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ["positive", "negative", "neutral"]:
        sub = df[df["sentiment_label"] == s]["review_length"]
        ax.hist(sub, bins=20, alpha=0.6, color=COLORS[s], label=s, edgecolor="none")
    ax.set_xlabel("Review Length (words)", color="#aaa")
    ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color="#2e2e3e", linewidth=0.4)
    ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=9)
    st.pyplot(fig)
    plt.close()

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444;font-size:0.8rem'>"
    "Built with Python · VADER · Streamlit · Matplotlib | "
    "Himanshu — AI Engineer Portfolio Project"
    "</div>",
    unsafe_allow_html=True
)
