## CardioScan AI — Cardiovascular Risk Prediction (Streamlit)

**CardioScan AI** is an educational machine-learning web application that predicts cardiovascular risk using a pre-trained **Logistic Regression** model. Users enter patient vitals (numeric fields), and the app returns a **Low Risk / High Risk** result with a probability score.

> ⚠️ Disclaimer: This project is for educational purposes only and does **not** provide medical advice. Always consult a licensed healthcare professional for real clinical decisions.

---

### Features
- Interactive **Streamlit** dashboard (dark “medical luxury” theme)
- Input fields auto-filled with **dataset medians** (adjust values as needed)
- Uses saved artifacts:
  - `heart_model.pkl` (trained model)
  - `scaler.pkl` (feature scaler)
  - `health_data_clean.csv` (cleaned dataset used to provide correct defaults & units)
- Outputs:
  - Risk category (Low/High)
  - Risk probability score
  - “Model Confidence” metric
- Model & dataset statistics section:
  - Overall accuracy (on the cleaned dataset shown in the app)
  - Dataset rows
  - Features used
  - Positive cases
  - Class distribution

---

### Tech Stack
- **Python**
- **Streamlit**
- **scikit-learn**
- **pandas**, **numpy**
- **pickle** (loading model/scaler)

---

### Project Structure
```text
heart_app/
├─ app.py
├─ train_model.py
├─ requirement.txt
├─ heart_model.pkl
├─ scaler.pkl
├─ health_data.csv
└─ health_data_clean.csv
```

---

### Setup & Run Locally

#### 1) Install dependencies
```bash
pip install -r requirement.txt
```

#### 2) Start the Streamlit app
```bash
streamlit run app.py
```

#### 3) Open the app
Streamlit will print a local URL (typically `http://localhost:8501`). Open it in your browser.

---

## How It Works (High Level)
1. On startup, the app loads:
   - `heart_model.pkl`
   - `scaler.pkl`
   - `health_data_clean.csv` (preferred) or falls back to `health_data.csv` if the cleaned file is missing
2. It determines the feature columns **from the scaler**:
   - uses `scaler.feature_names_in_` as the ground-truth feature order
   - prevents feature-name / feature-order mismatch bugs
3. User inputs are assembled into a single-row `DataFrame` using the exact scaler feature order.
4. The app scales inputs using `scaler.pkl`.
5. The model predicts:
   - `0` → Low Risk
   - `1` → High Risk
6. It displays prediction + probability.

---

## Training Fixes Applied (Important)
During modifications, two common data issues were fixed. These are the reasons the app works reliably now.

### Issue 1: `id` column must NOT be included in the scaler
**Root cause**
- If `id` is included during training, the scaler is fitted with an extra feature (`id`).
- But the app does not use `id` as a medical feature, causing a feature mismatch at inference.

**Fix**
- `train_model.py` drops `id` before fitting `StandardScaler` and training the model.
- `app.py` also drops `id` if it exists in the loaded CSV.

### Issue 2: `age` stored in days (convert to years)
**Root cause**
- The dataset stores age in **days** (e.g., ~21914 days ≈ 60 years).
- If the model is trained using converted age (years) but the app feeds raw days, predictions become inconsistent.

**Fix**
- `train_model.py` converts `age` from days → years when `age.max() > 200`.
- `app.py` includes a safety net:
  - if loaded `age` values look like days (`max > 200`), it converts them to years.

---

### Notes / Limitations
- This tool is **not clinically validated**.
- The prediction is only as good as the underlying dataset and trained model.
- Always treat outputs as **educational/demo** guidance, not medical truth.

---

### Educational Use
Use this project to:
- Demonstrate how ML inference can be wrapped in a UI
- Show preprocessing (cleaning + scaling) + prediction pipeline
- Build a real-time interactive dashboard for model outputs


