"""
Sentiment Analyzer - Streamlit app
Model : sentiment_model.pkl  (TF-IDF + Logistic Regression pipeline trained on IMDB reviews)

Run locally :  streamlit run app.py
Files needed in the SAME folder: app.py, sentiment_model.pkl, requirements.txt
"""

import html
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

MODEL_PATH = Path(__file__).parent / "sentiment_model.pkl"

# ---------------------------------------------------------------
# THEMES  (colors for background, cards, text, accent, pos / neg)
# ---------------------------------------------------------------
THEMES = {
    "🌊 Ocean": dict(bg1="#e0f2fe", bg2="#f0f9ff", card="#ffffff", text="#0f172a",
                    muted="#475569", accent="#0284c7", accent2="#06b6d4",
                    pos="#16a34a", neg="#dc2626", border="#bae6fd"),
    "🌅 Sunset": dict(bg1="#fff1e6", bg2="#ffe4e6", card="#ffffff", text="#3b1d12",
                     muted="#7c4a3a", accent="#f97316", accent2="#e11d48",
                     pos="#15803d", neg="#b91c1c", border="#fed7aa"),
    "🌿 Forest": dict(bg1="#ecfdf5", bg2="#f0fdf4", card="#ffffff", text="#052e16",
                     muted="#3f6b50", accent="#059669", accent2="#65a30d",
                     pos="#15803d", neg="#dc2626", border="#a7f3d0"),
    "💜 Royal Purple": dict(bg1="#f5f3ff", bg2="#fdf4ff", card="#ffffff", text="#2e1065",
                           muted="#6b5b95", accent="#7c3aed", accent2="#db2777",
                           pos="#16a34a", neg="#dc2626", border="#ddd6fe"),
    "🌙 Dark Night": dict(bg1="#0b1220", bg2="#1e293b", card="#1e293b", text="#e2e8f0",
                         muted="#94a3b8", accent="#38bdf8", accent2="#818cf8",
                         pos="#4ade80", neg="#f87171", border="#334155"),
    "☀️ Clean Light": dict(bg1="#ffffff", bg2="#f8fafc", card="#ffffff", text="#111827",
                          muted="#6b7280", accent="#2563eb", accent2="#4f46e5",
                          pos="#16a34a", neg="#dc2626", border="#e5e7eb"),
}

EXAMPLES = {
    "😍 Positive": "This movie was absolutely fantastic! Great acting, a beautiful story and I loved every minute of it.",
    "😡 Negative": "Worst film ever. The plot was boring, the acting was terrible and I wasted my time.",
    "🤔 Tricky": "It was not good at all, I expected so much more from this director.",
}


# ---------------------------------------------------------------
# CORE FUNCTIONS (no UI code here)
# ---------------------------------------------------------------
def clean_text(text):
    """Same cleaning that was used while training the model."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)            # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s']", " ", text)        # keep letters only
    text = re.sub(r"\s+", " ", text).strip()      # extra spaces
    return text


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return joblib.load(path)


def predict_positive_prob(model, texts):
    """Return the probability of the POSITIVE class for every text."""
    cleaned = [clean_text(t) for t in texts]
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned)[:, pos_idx]
    scores = model.decision_function(cleaned)   # e.g. LinearSVC
    return 1.0 / (1.0 + np.exp(-scores))


def explain_text(model, text, top_n=10):
    """Words that pushed the prediction (works for linear models like Logistic Regression)."""
    try:
        vec = model.steps[0][1]
        clf = model.steps[-1][1]
        if not hasattr(clf, "coef_"):
            return []
        X = vec.transform([clean_text(text)]).tocoo()
        names = vec.get_feature_names_out()
        contrib = X.data * clf.coef_[0][X.col]
        pairs = [(names[c], float(v)) for c, v in zip(X.col, contrib)]
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_n]
    except Exception:
        return []


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_css(t):
    css = """
<style>
.stApp { background: linear-gradient(135deg, __BG1__ 0%, __BG2__ 100%); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp div[data-testid="stMarkdownContainer"] { color: __TEXT__; }
[data-testid="stCaptionContainer"] * { color: __MUTED__ !important; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: __TEXT__ !important; }
[data-testid="stSidebar"] { background: __CARD__; border-right: 1px solid __BORDER__; }
[data-testid="stSidebar"] * { color: __TEXT__; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.stTextArea textarea, .stTextInput input {
  background: __CARD__; color: __TEXT__; border: 1px solid __BORDER__; border-radius: 14px; }
.stTextArea textarea:focus { border-color: __ACCENT__; box-shadow: 0 0 0 1px __ACCENT__; }

.stButton > button, .stDownloadButton > button {
  background: linear-gradient(90deg, __ACCENT__, __ACCENT2__); border: 0; border-radius: 12px;
  padding: .55rem 1.2rem; font-weight: 600; box-shadow: 0 4px 14px __SHADOW__; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); filter: brightness(1.08); }
.stButton > button *, .stDownloadButton > button * { color: #ffffff !important; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] p { color: __TEXT__; font-weight: 600; }
.stTabs [aria-selected="true"] p { color: __ACCENT__; }
.stTabs [data-baseweb="tab-highlight"] { background-color: __ACCENT__; }

[data-testid="stFileUploaderDropzone"] { background: __CARD__; border: 2px dashed __ACCENT__; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] * { color: __TEXT__; }
[data-testid="stFileUploaderDropzone"] button { background: __ACCENT__; border: 0; }
[data-testid="stFileUploaderDropzone"] button * { color: #ffffff !important; }

[data-baseweb="select"] > div { background: __CARD__; border-color: __BORDER__; }
[data-baseweb="select"] * { color: __TEXT__; }
[data-baseweb="select"] svg { fill: __TEXT__; }
[data-baseweb="popover"] ul { background: __CARD__; }
[data-baseweb="popover"] li { background: __CARD__; color: __TEXT__; }
[data-baseweb="popover"] li:hover { background: __BORDER__; }

.hero { background: linear-gradient(90deg, __ACCENT__, __ACCENT2__); border-radius: 22px;
  padding: 2.2rem 2rem; margin-bottom: 1.2rem; box-shadow: 0 10px 30px __SHADOW__; }
.hero h1 { color: #ffffff !important; margin: 0; font-size: 2.3rem; }
.hero p { color: #ffffff !important; opacity: .92; margin: .4rem 0 0 0; font-size: 1.05rem; }

.card { background: __CARD__; border: 1px solid __BORDER__; border-radius: 18px; padding: 1.2rem 1.4rem;
  box-shadow: 0 6px 20px __SHADOW__; margin-bottom: 1rem; }
.result { display: flex; align-items: center; gap: 1.2rem; }
.result .emoji { font-size: 3.6rem; line-height: 1; }
.result .label { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.result .sub { color: __MUTED__ !important; font-size: 1rem; margin-top: .2rem; }
.note { color: __MUTED__ !important; font-size: .92rem; margin-top: .6rem; }

.stat { background: __CARD__; border: 1px solid __BORDER__; border-radius: 16px; padding: 1rem;
  text-align: center; box-shadow: 0 4px 14px __SHADOW__; }
.stat-value { font-size: 2rem; font-weight: 800; }
.stat-label { color: __MUTED__ !important; font-size: .9rem; }

.chip { display: inline-block; padding: .25rem .7rem; margin: .2rem .25rem .2rem 0; border-radius: 999px;
  font-size: .9rem; font-weight: 600; }

.table-wrap { overflow-x: auto; border-radius: 14px; border: 1px solid __BORDER__; }
.themed-table { width: 100%; border-collapse: collapse; background: __CARD__; color: __TEXT__; }
.themed-table th { background: __ACCENT__; color: #ffffff; padding: .6rem .8rem; text-align: left !important; }
.themed-table td { padding: .55rem .8rem; border-bottom: 1px solid __BORDER__; color: __TEXT__; }
</style>
"""
    shadow = "rgba(0,0,0,0.35)" if t["card"] != "#ffffff" else "rgba(0,0,0,0.08)"
    values = {
        "__BG1__": t["bg1"], "__BG2__": t["bg2"], "__CARD__": t["card"], "__TEXT__": t["text"],
        "__MUTED__": t["muted"], "__ACCENT__": t["accent"], "__ACCENT2__": t["accent2"],
        "__BORDER__": t["border"], "__SHADOW__": shadow,
    }
    for key, val in values.items():
        css = css.replace(key, val)
    return css


# ===== UI =====
# ---------------------------------------------------------------
# SIDEBAR: theme + color options
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎨 Appearance")
    theme_name = st.radio("Choose a theme", list(THEMES.keys()), key="theme")
    base_theme = THEMES[theme_name]
    accent_pick = st.color_picker("Accent color", base_theme["accent"], key=f"accent_{theme_name}")
    st.caption("Change the theme or pick your own accent color. Results update instantly.")
    st.markdown("---")
    st.markdown("## 🤖 Model")
    st.caption("TF-IDF (1-2 word n-grams) + Logistic Regression, trained on 50,000 IMDB movie reviews.")

T = dict(base_theme)
T["accent"] = accent_pick
st.markdown(build_css(T), unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("`sentiment_model.pkl` was not found. Keep it in the same folder as `app.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info("Make sure `scikit-learn==1.6.1` (the version used to train the model) is installed.")
    st.stop()

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


def set_example(text):
    st.session_state["review_text"] = text


def plot_layout(fig, height=300):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]),
    )
    return fig


def html_table(df):
    body = df.to_html(index=False, border=0, classes="themed-table", escape=True)
    return f'<div class="table-wrap">{body}</div>'


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>💬 Sentiment Analyzer</h1>'
    "<p>Paste a review and find out instantly if it is positive or negative - "
    "powered by Machine Learning.</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "📂 Batch Analysis", "📜 History", "ℹ️ About"])

# ---------------------------------------------------------------
# TAB 1: single review
# ---------------------------------------------------------------
with tab1:
    st.markdown("##### Try an example")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, key=f"ex_{name}", on_click=set_example, args=(text,), use_container_width=True)

    review = st.text_area("Enter your review", key="review_text", height=150,
                          placeholder="Type or paste a movie / product review here...")

    if st.button("✨ Analyze Sentiment", key="analyze_btn"):
        if not review.strip():
            st.warning("Please enter some text first.")
        elif not clean_text(review):
            st.warning("The text has no readable English words. Please try a different review.")
        else:
            st.session_state["last_text"] = review
            p_now = float(predict_positive_prob(model, [review])[0])
            st.session_state["history"].append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Review": (review[:80] + "...") if len(review) > 80 else review,
                "Sentiment": "Positive" if p_now >= 0.5 else "Negative",
                "Positive score (%)": round(p_now * 100, 1),
            })

    last = st.session_state["last_text"]
    if last:
        p = float(predict_positive_prob(model, [last])[0])
        is_pos = p >= 0.5
        conf = p if is_pos else 1 - p
        color = T["pos"] if is_pos else T["neg"]
        label = "Positive" if is_pos else "Negative"
        emoji = "😊" if is_pos else "😞"
        note = ""
        if conf < 0.6:
            note = '<div class="note">⚠️ Low confidence - this review may be mixed or neutral.</div>'

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(
                f'<div class="card result" style="border-left: 8px solid {color};">'
                f'<div class="emoji">{emoji}</div>'
                f'<div><div class="label" style="color:{color};">{label}</div>'
                f'<div class="sub">Confidence: {conf * 100:.1f}%</div></div></div>{note}',
                unsafe_allow_html=True,
            )

            pairs = explain_text(model, last, top_n=12)
            if pairs:
                st.markdown("**Words that influenced the result**")
                chips = ""
                for word, val in pairs:
                    c = T["pos"] if val > 0 else T["neg"]
                    chips += (f'<span class="chip" style="background:{hex_to_rgba(c, 0.15)};'
                              f'color:{c};border:1px solid {c};">{html.escape(word)}</span>')
                st.markdown(chips, unsafe_allow_html=True)
                st.caption("Green words push towards Positive, red words push towards Negative.")

        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100,
                number={"suffix": "%", "font": {"color": color}},
                title={"text": "Positive score", "font": {"color": T["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": T["muted"]},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 50], "color": hex_to_rgba(T["neg"], 0.18)},
                        {"range": [50, 100], "color": hex_to_rgba(T["pos"], 0.18)},
                    ],
                    "threshold": {"line": {"color": T["text"], "width": 3}, "thickness": 0.8, "value": 50},
                },
            ))
            st.plotly_chart(plot_layout(gauge, 290), use_container_width=True,
                            config={"displayModeBar": False})

        if pairs:
            top = pairs[:10][::-1]
            bar = go.Figure(go.Bar(
                x=[v for _, v in top], y=[w for w, _ in top], orientation="h",
                marker_color=[T["pos"] if v > 0 else T["neg"] for _, v in top],
            ))
            bar.update_layout(title="Word contribution to the prediction")
            bar.update_xaxes(gridcolor=hex_to_rgba(T["muted"], 0.25), zerolinecolor=T["muted"])
            st.plotly_chart(plot_layout(bar, 340), use_container_width=True,
                            config={"displayModeBar": False})

# ---------------------------------------------------------------
# TAB 2: batch analysis (CSV upload)
# ---------------------------------------------------------------
with tab2:
    st.markdown("### Analyze many reviews at once")
    st.caption("Upload a CSV file that has a column containing review text.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="batch_file")

    if up is not None:
        try:
            df = pd.read_csv(up, encoding_errors="replace")
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            df = None

        if df is not None:
            if df.empty or len(df.columns) == 0:
                st.warning("The file looks empty.")
            else:
                cols = list(df.columns)
                guess = next((c for c in cols if str(c).lower() in ("review", "text", "comment", "content")), cols[0])
                text_col = st.selectbox("Which column has the review text?", cols, index=cols.index(guess))

                if st.button("🚀 Run batch analysis", key="batch_btn"):
                    texts = df[text_col].fillna("").astype(str).tolist()
                    with st.spinner("Analyzing..."):
                        probs = predict_positive_prob(model, texts)
                    out = df.copy()
                    out["Sentiment"] = np.where(probs >= 0.5, "Positive", "Negative")
                    out["Positive score (%)"] = np.round(probs * 100, 1)
                    st.session_state["batch_result"] = out
                    st.session_state["batch_text_col"] = text_col

        result = st.session_state.get("batch_result")
        if result is not None:
            total = len(result)
            n_pos = int((result["Sentiment"] == "Positive").sum())
            n_neg = total - n_pos
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat"><div class="stat-value" style="color:{T["accent"]}">{total}</div>'
                        '<div class="stat-label">Total reviews</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat"><div class="stat-value" style="color:{T["pos"]}">{n_pos}</div>'
                        '<div class="stat-label">Positive</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat"><div class="stat-value" style="color:{T["neg"]}">{n_neg}</div>'
                        '<div class="stat-label">Negative</div></div>', unsafe_allow_html=True)

            pie = go.Figure(go.Pie(
                labels=["Positive", "Negative"], values=[n_pos, n_neg], hole=0.55,
                marker=dict(colors=[T["pos"], T["neg"]]), textinfo="label+percent",
            ))
            pie.update_layout(title="Sentiment split", showlegend=False)
            st.plotly_chart(plot_layout(pie, 320), use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown("**Preview (first 50 rows)**")
            st.markdown(html_table(result.head(50)), unsafe_allow_html=True)
            st.write("")
            st.download_button("⬇️ Download results as CSV", result.to_csv(index=False).encode("utf-8"),
                               file_name="sentiment_results.csv", mime="text/csv")

# ---------------------------------------------------------------
# TAB 3: history
# ---------------------------------------------------------------
with tab3:
    hist = st.session_state["history"]
    if not hist:
        st.info("No analyses yet. Go to the Analyze tab and test a review.")
    else:
        hdf = pd.DataFrame(hist)
        counts = hdf["Sentiment"].value_counts()
        h1, h2 = st.columns(2)
        with h1:
            donut = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.55,
                marker=dict(colors=[T["pos"] if s == "Positive" else T["neg"] for s in counts.index]),
            ))
            donut.update_layout(title="Your analyses")
            st.plotly_chart(plot_layout(donut, 300), use_container_width=True,
                            config={"displayModeBar": False})
        with h2:
            line = go.Figure(go.Scatter(
                x=list(range(1, len(hdf) + 1)), y=hdf["Positive score (%)"], mode="lines+markers",
                line=dict(color=T["accent"], width=3), marker=dict(size=9),
            ))
            line.update_layout(title="Positive score over time", yaxis=dict(range=[0, 100]))
            line.update_xaxes(title="Analysis #", gridcolor=hex_to_rgba(T["muted"], 0.25))
            line.update_yaxes(gridcolor=hex_to_rgba(T["muted"], 0.25))
            st.plotly_chart(plot_layout(line, 300), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown(html_table(hdf.iloc[::-1]), unsafe_allow_html=True)
        st.write("")
        if st.button("🗑️ Clear history", key="clear_hist"):
            st.session_state["history"] = []
            st.rerun()

# ---------------------------------------------------------------
# TAB 4: about
# ---------------------------------------------------------------
with tab4:
    st.markdown("### How it works")
    st.markdown(
        "1. **Clean** - the text is lowercased and HTML tags, links and punctuation are removed.\n"
        "2. **TF-IDF** - words and word pairs are turned into numbers (important words get higher weight).\n"
        "3. **Logistic Regression** - predicts the probability that the review is positive.\n"
        "4. **Explain** - each word's weight shows why the model chose Positive or Negative."
    )
    try:
        vocab = len(model.steps[0][1].vocabulary_)
        st.markdown(f'<div class="card"><b>Vocabulary size:</b> {vocab:,} words / word pairs<br>'
                    "<b>Training data:</b> IMDB Dataset of 50K Movie Reviews<br>"
                    "<b>Expected accuracy:</b> about 89-90% on movie reviews</div>",
                    unsafe_allow_html=True)
    except Exception:
        pass
    st.caption("Note: the model was trained on movie reviews, so it works best on review-style text. "
               "It only understands English and gives Positive / Negative (no neutral class).")
