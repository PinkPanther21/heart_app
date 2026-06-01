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

.stApp {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}

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

.block-container {
    padding: 2.5rem 3rem 4rem 3rem !important;
    max-width: 1400px;
    position: relative;
    z-index: 1;
}

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
.hero-title span { color: var(--accent-red); font-style: italic; }
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
}
.heartbeat-line { width: 160px; height: 36px; margin: 1.4rem auto 0; opacity: 0.6; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 28px 32px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 20px 60px rgba(0,0,0,0.4) !important;
}

.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--text-primary);
    margin: 0 0 0.3rem;
    font-weight: 400;
}
.section-sub { font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1.8rem; }

label, .stNumberInput label, div[data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin-bottom: 4px !important;
}

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

.id-lookup-wrap {
    background: rgba(0,212,170,0.06);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 1.6rem;
}
.id-lookup-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-teal);
    margin-bottom: 4px;
}
.id-lookup-desc { font-size: 0.82rem; color: var(--text-muted); }

.stAlert { border-radius: 12px !important; }

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

.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4aa, #3b82f6, #ff3e5e) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 999px !important;
    height: 8px !important;
}

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

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.6rem 0 !important; }

footer { visibility: hidden; }
.footer-custom {
    text-align: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 3rem;
    letter-spacing: 0.04em;
}
.footer-dot { color: var(--accent-red); margin: 0 6px; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
[data-testid="stHorizontalBlock"] { gap: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)


# =========================================
# FIELD METADATA
# Human-readable labels, units, and ranges
# for every possible column in this dataset.
# =========================================
FIELD_META = {
    "age":         {"label": "Age (years)",         "unit": "yrs",   "fmt": "%.1f", "step": 0.1,  "is_int": False},
    "gender":      {"label": "Gender (1=F, 2=M)",   "unit": "",      "fmt": "%d",   "step": 1,    "is_int": True},
    "height":      {"label": "Height (cm)",          "unit": "cm",    "fmt": "%d",   "step": 1,    "is_int": True},
    "weight":      {"label": "Weight (kg)",          "unit": "kg",    "fmt": "%.1f", "step": 0.1,  "is_int": False},
    "ap_hi":       {"label": "Systolic BP (mmHg)",   "unit": "mmHg",  "fmt": "%d",   "step": 1,    "is_int": True},
    "ap_lo":       {"label": "Diastolic BP (mmHg)",  "unit": "mmHg",  "fmt": "%d",   "step": 1,    "is_int": True},
    "cholesterol": {"label": "Cholesterol (1-3)",    "unit": "",      "fmt": "%d",   "step": 1,    "is_int": True},
    "gluc":        {"label": "Glucose (1-3)",        "unit": "",      "fmt": "%d",   "step": 1,    "is_int": True},
    "smoke":       {"label": "Smoker (0/1)",         "unit": "",      "fmt": "%d",   "step": 1,    "is_int": True},
    "alco":        {"label": "Alcohol (0/1)",        "unit": "",      "fmt": "%d",   "step": 1,    "is_int": True},
    "active":      {"label": "Physically Active (0/1)", "unit": "",   "fmt": "%d",   "step": 1,    "is_int": True},
}


# =========================================
# LOAD MODEL & DATA
# =========================================
@st.cache_resource
def load_artifacts():
    model  = pickle.load(open("heart_model.pkl",  "rb"))
    scaler = pickle.load(open("scaler.pkl",        "rb"))

    # Prefer the cleaned CSV produced by train_model.py
    import os
    csv_path = "health_data_clean.csv" if os.path.exists("health_data_clean.csv") else "health_data.csv"
    df = pd.read_csv(csv_path)

    # Safety net: if age is still in days, convert
    if "age" in df.columns and df["age"].max() > 200:
        df["age"] = (df["age"] / 365).round(1)

    # Drop id if present (was excluded during training)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return model, scaler, df

model, scaler, df = load_artifacts()

# Columns the scaler actually knows about (ground truth)
SCALER_COLS = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else [
    c for c in df.columns if c != "cardio"
]

target_col = "cardio"
X_columns  = SCALER_COLS   # exactly what the model expects — no id, no cardio


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
    st.markdown('<p class="section-sub">Look up an existing patient by ID, or enter new patient vitals manually.</p>',
                unsafe_allow_html=True)

    # ---- ID LOOKUP ----
    # We look up by row index since id column was dropped
    st.markdown("""
    <div class="id-lookup-wrap">
        <div class="id-lookup-title">🔍 Patient Record Lookup</div>
        <div class="id-lookup-desc">
            Enter a row number to auto-fill all fields from the dataset.
            You can edit any value after loading.
        </div>
    </div>
    """, unsafe_allow_html=True)

    id_col_ui, btn_col, status_col = st.columns([2, 1, 3])

    max_row = len(df) - 1

    with id_col_ui:
        lookup_row = st.number_input(
            "Row Number",
            min_value=0,
            max_value=max_row,
            value=0,
            step=1,
            format="%d"
        )

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        load_clicked = st.button("Load Record", use_container_width=True)

    with status_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if load_clicked:
            st.success(f"✅ Record #{int(lookup_row)} loaded successfully.")

    st.markdown("---")

    # ---- PREFILL VALUES ----
    if load_clicked:
        row = df.iloc[int(lookup_row)]
        prefill = {col: float(row[col]) for col in X_columns}
    else:
        prefill = {col: float(df[col].median()) for col in X_columns}

    # ---- INPUT FIELDS ----
    col1, col2, col3 = st.columns(3)
    cols   = [col1, col2, col3]
    inputs = {}

    for i, col_name in enumerate(X_columns):
        meta = FIELD_META.get(col_name, None)

        # Fallback for any unexpected column
        if meta is None:
            is_int = df[col_name].nunique() < 20
            meta = {
                "label": col_name.replace("_", " ").title(),
                "fmt":   "%d" if is_int else "%.1f",
                "step":  1    if is_int else 0.1,
                "is_int": is_int,
            }

        val = prefill[col_name]

        if meta["is_int"]:
            inputs[col_name] = cols[i % 3].number_input(
                label=meta["label"],
                value=int(round(val)),
                step=int(meta["step"]),
                format=meta["fmt"],
                key=f"input_{col_name}"
            )
        else:
            inputs[col_name] = cols[i % 3].number_input(
                label=meta["label"],
                value=round(val, 1),
                step=meta["step"],
                format=meta["fmt"],
                key=f"input_{col_name}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick reference guide
    with st.expander("ℹ️  Field Reference Guide"):
        st.markdown("""
        | Field | Values | Meaning |
        |---|---|---|
        | **Age** | years (e.g. 45.5) | Patient age in years |
        | **Gender** | 1 or 2 | 1 = Female, 2 = Male |
        | **Height** | cm (e.g. 168) | Height in centimetres |
        | **Weight** | kg (e.g. 72.5) | Weight in kilograms |
        | **ap_hi** | mmHg (e.g. 120) | Systolic blood pressure |
        | **ap_lo** | mmHg (e.g. 80) | Diastolic blood pressure |
        | **Cholesterol** | 1, 2, or 3 | 1=Normal, 2=Above normal, 3=Well above normal |
        | **Gluc** | 1, 2, or 3 | 1=Normal, 2=Above normal, 3=Well above normal |
        | **Smoke** | 0 or 1 | 0=Non-smoker, 1=Smoker |
        | **Alco** | 0 or 1 | 0=No alcohol, 1=Alcohol use |
        | **Active** | 0 or 1 | 0=Inactive, 1=Physically active |
        """)

    st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Run Risk Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================
# VALIDATION HELPERS
# =========================================
def validate_inputs(inp):
    """Return list of warning strings for out-of-range values."""
    warnings = []
    if "age" in inp and not (1 <= inp["age"] <= 120):
        warnings.append(f"Age {inp['age']} is outside normal range (1–120 years).")
    if "ap_hi" in inp and not (70 <= inp["ap_hi"] <= 250):
        warnings.append(f"Systolic BP {inp['ap_hi']} is outside normal range (70–250 mmHg).")
    if "ap_lo" in inp and not (40 <= inp["ap_lo"] <= 150):
        warnings.append(f"Diastolic BP {inp['ap_lo']} is outside normal range (40–150 mmHg).")
    if "ap_hi" in inp and "ap_lo" in inp and inp["ap_lo"] >= inp["ap_hi"]:
        warnings.append("Diastolic BP must be less than Systolic BP.")
    if "height" in inp and not (100 <= inp["height"] <= 220):
        warnings.append(f"Height {inp['height']} cm is outside normal range (100–220 cm).")
    if "weight" in inp and not (30 <= inp["weight"] <= 200):
        warnings.append(f"Weight {inp['weight']} kg is outside normal range (30–200 kg).")
    return warnings


# =========================================
# PREDICTION RESULT
# =========================================
if predict_clicked:

    # Validate first
    warnings = validate_inputs(inputs)
    if warnings:
        st.markdown("<br>", unsafe_allow_html=True)
        for w in warnings:
            st.warning(f"⚠️ {w}")

    # Build input in exact column order the scaler expects
    user_input   = [inputs[c] for c in X_columns]
    input_df     = pd.DataFrame([user_input], columns=X_columns)   # columns match SCALER_COLS
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
            st.markdown("**Risk Probability Score**")
            st.progress(min(int(probability), 100))
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                        color:var(--text-muted); margin-top:6px;">
                <span>0% — No Risk</span><span>100% — High Risk</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            confidence = max(probability, 100 - probability)
            st.metric("Model Confidence", f"{confidence:.1f}%")
            st.metric("Prediction", "High Risk 🔴" if prediction == 1 else "Low Risk 🟢")

        # Input summary
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋  Input Summary"):
            summary_data = {}
            for col_name, val in inputs.items():
                meta  = FIELD_META.get(col_name, {})
                label = meta.get("label", col_name.replace("_", " ").title())
                unit  = meta.get("unit", "")
                summary_data[label] = f"{val} {unit}".strip()
            st.table(pd.DataFrame(summary_data.items(), columns=["Parameter", "Value"]))

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
    y     = df[target_col]
    X_all = df[X_columns]
    X_sc  = scaler.transform(X_all)
    y_pred = model.predict(X_sc)
    acc   = np.mean(y_pred == y)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Dataset Rows",     f"{len(df):,}")
    m3.metric("Features Used",    str(len(X_columns)))
    m4.metric("Positive Cases",   f"{int(y.sum()):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    class_balance = y.value_counts(normalize=True) * 100
    st.write("**Class Distribution**")
    cb_col1, cb_col2 = st.columns(2)
    cb_col1.metric("Healthy (0)",  f"{class_balance.get(0, 0):.1f}%")
    cb_col2.metric("At Risk  (1)", f"{class_balance.get(1, 0):.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Feature Statistics (cleaned data)**")
    display_cols = [c for c in ["age", "height", "weight", "ap_hi", "ap_lo"] if c in df.columns]
    if display_cols:
        st.dataframe(df[display_cols].describe().round(1), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Model Coefficients**")
    coef_df = pd.DataFrame({
        "Feature":     X_columns,
        "Coefficient": model.coef_[0].round(4)
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True)


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
