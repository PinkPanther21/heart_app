import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="CardioScan AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CUSTOM CSS — Medical Luxury Dark Theme
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ---- Root Variables ---- */
:root {
    --bg-deep:       #060a12;
    --bg-card:       rgba(13, 20, 35, 0.85);
    --bg-input:      #0d1423;
    --accent-red:    #ff3e5e;
    --accent-teal:   #00d4aa;
    --accent-blue:   #3b82f6;
    --text-primary:  #eef2ff;
    --text-muted:    #7e8fa6;
    --border:        rgba(255,255,255,0.07);
    --glow-red:      rgba(255,62,94,0.25);
    --glow-teal:     rgba(0,212,170,0.20);
}

/* ---- Background ---- */
.stApp {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}

/* Animated mesh gradient behind everything */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(255,62,94,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 90% 80%, rgba(0,212,170,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(59,130,246,0.05) 0%, transparent 70%);
    pointer-events: none;
}

/* ---- Block container ---- */
.block-container {
    padding: 2.5rem 3rem 4rem 3rem !important;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

/* ---- Hero Header ---- */
.hero-wrap {
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-teal);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 100px;
    padding: 5px 16px;
    margin-bottom: 1.1rem;
    margin-top: 1.1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem;
}
.hero-title span {
    color: var(--accent-red);
    font-style: italic;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
}

/* Heartbeat line decoration */
.heartbeat-line {
    width: 160px;
    height: 36px;
    margin: 1.4rem auto 0;
    opacity: 0.6;
}

/* ---- Cards ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 28px 32px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset,
                0 20px 60px rgba(0,0,0,0.4) !important;
}

/* ---- Section heading ---- */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--text-primary);
    margin: 0 0 0.3rem;
    font-weight: 400;
}
.section-sub {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-bottom: 1.8rem;
}

/* ---- Input labels ---- */
label, .stNumberInput label, div[data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 4px !important;
}

/* ---- Number inputs ---- */
.stNumberInput input {
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 11px 14px !important;
    font-size: 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus {
    border: 1px solid var(--accent-teal) !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.12) !important;
    outline: none !important;
}
.stNumberInput [data-testid="stNumberInputStepDown"],
.stNumberInput [data-testid="stNumberInputStepUp"] {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: var(--text-muted) !important;
}

/* ---- Predict button ---- */
.predict-btn-wrap .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff3e5e 0%, #ff6a38 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 24px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    box-shadow: 0 6px 30px var(--glow-red) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.predict-btn-wrap .stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(255,62,94,0.45) !important;
}

/* ---- Risk result cards ---- */
.result-low {
    background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,212,170,0.04));
    border: 1px solid rgba(0,212,170,0.35);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 1.2rem;
}
.result-high {
    background: linear-gradient(135deg, rgba(255,62,94,0.14), rgba(255,62,94,0.04));
    border: 1px solid rgba(255,62,94,0.4);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 1.2rem;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    margin: 0.3rem 0 0.5rem;
    font-weight: 400;
}
.result-prob {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
    font-family: 'DM Serif Display', serif;
}
.result-desc {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.8rem;
    max-width: 400px;
    line-height: 1.6;
}

/* ---- Progress bar ---- */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4aa, #3b82f6, #ff3e5e) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
    height: 8px !important;
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: var(--accent-teal) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ---- Expander ---- */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 8px 20px !important;
}
summary {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 14px 0 !important;
}

/* ---- Divider ---- */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.6rem 0 !important;
}

/* ---- Footer ---- */
footer { visibility: hidden; }
.footer-custom {
    text-align: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 3rem;
    letter-spacing: 0.04em;
}
.footer-dot { color: var(--accent-red); margin: 0 6px; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

/* ---- Column spacing ---- */
[data-testid="stHorizontalBlock"] { gap: 1.2rem !important; }

</style>
""", unsafe_allow_html=True)


# =========================================
# LOAD MODEL & DATA
# =========================================
@st.cache_resource
def load_artifacts():
    model  = pickle.load(open("heart_model.pkl",  "rb"))
    scaler = pickle.load(open("scaler.pkl",        "rb"))
    df     = pd.read_csv("health_data.csv")
    return model, scaler, df

model, scaler, df = load_artifacts()
X_columns = df.drop("cardio", axis=1).columns


# =========================================
# HERO HEADER
# =========================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">🫀 Cardiovascular Intelligence</div>
    <h1 class="hero-title">CardioScan <span>AI</span></h1>
    <p class="hero-subtitle">
        Enter patient vitals below for an instant, machine-learning powered
        cardiovascular risk assessment.
    </p>
    <!-- simple SVG heartbeat line -->
    <svg class="heartbeat-line" viewBox="0 0 160 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polyline points="0,18 28,18 36,4 44,32 52,18 68,18 76,10 84,26 90,18 160,18"
                stroke="#ff3e5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
</div>
""", unsafe_allow_html=True)


# =========================================
# INPUT FORM
# =========================================
with st.container(border=True):
    st.markdown('<p class="section-heading">Patient Information</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">All fields are pre-filled with dataset medians. Adjust values as needed.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cols   = [col1, col2, col3]
    inputs = {}

    for i, col_name in enumerate(X_columns):
        default_val = float(df[col_name].median())
        label       = col_name.replace("_", " ").title()
        step        = 1.0 if df[col_name].nunique() < 20 else 0.1
        inputs[col_name] = cols[i % 3].number_input(
            label=label,
            value=default_val,
            step=step,
            format="%.1f"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Run Risk Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================
# PREDICTION RESULT
# =========================================
if predict_clicked:
    user_input  = [inputs[c] for c in X_columns]
    input_df    = pd.DataFrame([user_input], columns=X_columns)
    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="section-heading">Risk Assessment Result</p>', unsafe_allow_html=True)
        st.markdown("---")

        r_col1, r_col2 = st.columns([3, 2])

        with r_col1:
            if prediction == 0:
                st.markdown(f"""
                <div class="result-low">
                    <div style="font-size:2rem">✅</div>
                    <div class="result-label" style="color:#00d4aa;">Low Risk Detected</div>
                    <div class="result-prob" style="color:#00d4aa;">{probability:.1f}<span style="font-size:1.5rem">%</span></div>
                    <p class="result-desc">
                        The model estimates a low probability of cardiovascular disease based
                        on the provided parameters. Regular check-ups are still recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-high">
                    <div style="font-size:2rem">⚠️</div>
                    <div class="result-label" style="color:#ff3e5e;">High Risk Detected</div>
                    <div class="result-prob" style="color:#ff3e5e;">{probability:.1f}<span style="font-size:1.5rem">%</span></div>
                    <p class="result-desc">
                        The model flags an elevated cardiovascular risk. Please consult a
                        qualified cardiologist for a thorough clinical evaluation.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with r_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Risk Probability Score**")
            st.progress(min(int(probability), 100))
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-muted); margin-top:6px;">
                <span>0% — No Risk</span><span>100% — High Risk</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            confidence = max(probability, 100 - probability)
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.metric("Prediction", "High Risk 🔴" if prediction == 1 else "Low Risk 🟢")

        st.markdown("""
        <p style="font-size:0.78rem; color:#475569; margin-top:1rem;">
        ⚠️  This tool is for educational purposes only and does not constitute medical advice.
        Always consult a licensed healthcare professional for clinical decisions.
        </p>
        """, unsafe_allow_html=True)


# =========================================
# MODEL STATS EXPANDER
# =========================================
with st.expander("📊  Model & Dataset Statistics"):
    X       = df.drop("cardio", axis=1)
    y       = df["cardio"]
    X_sc    = scaler.transform(X)
    y_pred  = model.predict(X_sc)
    acc     = np.mean(y_pred == y)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy",  f"{acc * 100:.2f}%")
    m2.metric("Dataset Rows",      f"{len(df):,}")
    m3.metric("Features Used",     str(len(X_columns)))
    m4.metric("Positive Cases",    f"{int(y.sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    class_balance = y.value_counts(normalize=True) * 100
    st.write("**Class Distribution**")
    cb_col1, cb_col2 = st.columns(2)
    cb_col1.metric("Healthy (0)", f"{class_balance.get(0, 0):.1f}%")
    cb_col2.metric("At Risk  (1)", f"{class_balance.get(1, 0):.1f}%")


# =========================================
# FOOTER
# =========================================
st.markdown("""
<div class="footer-custom">
    CardioScan AI
    <span class="footer-dot">•</span>
    Streamlit + Scikit-learn
    <span class="footer-dot">•</span>
    Logistic Regression
    <span class="footer-dot">•</span>
    For Educational Use Only
</div>
""", unsafe_allow_html=True)