import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0f1116; }
    h1, h2, h3 { font-family: 'Trebuchet MS', sans-serif; }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6ee7f9, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }
    .metric-card {
        background: #161a23;
        border: 1px solid #262b36;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    .verdict-real {
        background: linear-gradient(135deg, #0f2e21, #123a2a);
        border: 1px solid #1f7a52;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #3a1414, #4a1717);
        border: 1px solid #a13636;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
    }
    .footer-note {
        color: #6b7280;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# NLTK setup (downloads once, cached across reruns)
# --------------------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    return stopwords.words("english"), WordNetLemmatizer()


stop_words, lemma = setup_nltk()


def preprocess_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", "", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
    txt = word_tokenize(txt)
    txt = [i for i in txt if i not in stop_words]
    txt = [lemma.lemmatize(i) for i in txt]
    return " ".join(txt)


# --------------------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "final_model.pkl"
    vec_path = BASE_DIR / "tfidf_vectorizer.pkl"
    metrics_path = BASE_DIR / "metrics.json"

    model, vectorizer, metrics = None, None, None
    errors = []

    try:
        model = joblib.load(model_path)
    except Exception as e:
        errors.append(f"Model file could not be loaded ({e.__class__.__name__}). "
                       f"It may have been saved/transferred incorrectly (not in pure binary mode).")

    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        errors.append(f"Vectorizer file could not be loaded ({e.__class__.__name__}). "
                       f"It may have been saved/transferred incorrectly (not in pure binary mode).")

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return model, vectorizer, metrics, errors


model, vectorizer, metrics, load_errors = load_artifacts()

# Map numeric label encodings back to the original string labels, if needed
LABEL_MAP = {0: "FAKE", 1: "REAL"}


def resolve_label(raw_label):
    if isinstance(raw_label, (int, np.integer)):
        return LABEL_MAP.get(int(raw_label), str(raw_label))
    label_str = str(raw_label).upper()
    if label_str in ("0", "1"):
        return LABEL_MAP[int(label_str)]
    return label_str


def predict(text: str):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    raw_pred = model.predict(vec)[0]
    label = resolve_label(raw_pred)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        conf = float(proba[classes.index(raw_pred)])
    else:
        # SVC without probability=True has no predict_proba - use decision_function instead
        score = model.decision_function(vec)
        conf = float(1 / (1 + np.exp(-abs(score[0]))))
    return label, conf


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ TruthLens")
    st.caption("NLP-powered fake news classifier")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Classifier", "📊 Model Performance", "ℹ️ About the Project"])
    st.markdown("---")
    st.markdown("**Built by** Khush")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View%20Code-181717?logo=github)](#)"
    )

if load_errors:
    for e in load_errors:
        st.error(e)
    st.stop()

# --------------------------------------------------------------------------------------
# Page: Classifier
# --------------------------------------------------------------------------------------
if page == "🔍 Classifier":
    st.markdown('<p class="hero-title">TruthLens</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Paste a news headline or article below and let the model '
        "judge whether it looks REAL or FAKE, based on patterns learned from thousands "
        "of labeled news articles.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_main, col_side = st.columns([2, 1])

    with col_main:
        example_articles = {
            "— pick an example —": "",
            "Example: Political headline": (
                "Senate passes new infrastructure bill after months of negotiation, "
                "allocating funds for roads, bridges and broadband expansion across rural areas."
            ),
            "Example: Sensational claim": (
                "Scientists SHOCKED as secret government files reveal aliens have been "
                "living among us for decades, sources claim without evidence."
            ),
        }
        choice = st.selectbox("Try an example", list(example_articles.keys()))
        default_text = example_articles[choice]

        user_text = st.text_area(
            "Article text",
            value=default_text,
            height=220,
            placeholder="Paste a news article or headline here...",
        )

        analyze = st.button("Analyze", type="primary", use_container_width=True)

    with col_side:
        st.markdown("#### How it works")
        st.markdown(
            """
1. Text is cleaned (lowercased, URLs and punctuation removed)
2. Tokenized, stopwords removed, and lemmatized
3. Converted into **TF-IDF** features
4. Fed into a trained **Support Vector Classifier (SVC)**
5. Model outputs a label + confidence score
            """
        )
        if metrics:
            st.metric("Model test accuracy", f"{metrics['accuracy']*100:.1f}%")

    if analyze:
        if not user_text or len(user_text.strip()) < 10:
            st.warning("Please enter at least a sentence or two of article text.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence = predict(user_text)

            st.write("")
            if label == "REAL":
                st.markdown(
                    f"""
                    <div class="verdict-real">
                        <h2 style="margin:0;">✅ Predicted: REAL</h2>
                        <p style="color:#a7f3d0; margin-top:0.3rem;">
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-fake">
                        <h2 style="margin:0;">🚩 Predicted: FAKE</h2>
                        <p style="color:#fca5a5; margin-top:0.3rem;">
                            Confidence: <b>{confidence*100:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#22c55e" if label == "REAL" else "#ef4444"},
                        "steps": [
                            {"range": [0, 50], "color": "#1f2430"},
                            {"range": [50, 100], "color": "#262b36"},
                        ],
                    },
                    title={"text": "Model Confidence"},
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "⚠️ This is a statistical text-pattern classifier trained on one dataset — "
                "not a fact-checker. Always verify important claims against trusted sources."
            )

# --------------------------------------------------------------------------------------
# Page: Model performance
# --------------------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.markdown('<p class="hero-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Evaluated on a held-out 20% test split.</p>', unsafe_allow_html=True)
    st.write("")

    if metrics:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Training samples", f"{metrics['n_train']:,}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test samples", f"{metrics['n_test']:,}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        labels = metrics["labels"]
        fig = px.imshow(
            cm,
            text_auto=True,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pipeline")
        st.code(
            f"Vectorizer: {metrics.get('vectorizer', 'TF-IDF')}\n"
            f"Model: {metrics.get('model_type', 'Support Vector Classifier (SVC, C=2)')}",
            language="text",
        )
    else:
        st.info("No metrics file found. Run the training script to generate metrics.json.")

# --------------------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------------------
else:
    st.markdown('<p class="hero-title">About this project</p>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        """
**TruthLens** is a fake news detection tool built on the classic *"Fake or Real News"*
dataset (~6,300 labeled political news articles, balanced FAKE/REAL).

**Pipeline**
- Text cleaning (lowercasing, URL removal, punctuation stripping)
- Tokenization (NLTK), stopword removal, lemmatization (WordNetLemmatizer)
- TF-IDF vectorization
- Support Vector Classifier (SVC, C = 2)

**Tech stack**: Python · scikit-learn · NLTK · pandas · Streamlit · Plotly

**Limitations**
- Trained on one dataset from a specific time period — may not generalize to
  entirely new topics, satire, or non-English text.
- Judges *linguistic style/patterns*, not factual accuracy — it cannot verify
  claims against real-world facts the way a fact-checker does.
- Should be treated as a decision-support signal, not a ground-truth verdict.
        """
    )

st.markdown('<p class="footer-note">Built with Streamlit · scikit-learn · Plotly</p>', unsafe_allow_html=True)