## CardioScan AI — Cardiovascular Risk Prediction (Streamlit)

**CardioScan AI** is an educational machine-learning web application that predicts cardiovascular risk using a pre-trained **Logistic Regression** model. Users enter patient vitals (numeric fields), and the app returns a **Low Risk / High Risk** result with a probability score.

> ⚠️ **Disclaimer:** This project is for educational purposes only and does **not** provide medical advice. Always consult a licensed healthcare professional for real clinical decisions.

---

### Features
- Interactive **Streamlit** dashboard (dark “medical luxury” theme)
- Input fields auto-filled with **dataset medians** (adjust values as needed)
- Uses saved artifacts:
  - `heart_model.pkl` (trained model)
  - `scaler.pkl` (feature scaler)
  - `health_data.csv` (dataset used to define feature list + default values)
- Outputs:
  - Risk category (Low/High)
  - Risk probability score
  - “Model Confidence” metric
- Model & dataset statistics section:
  - Overall accuracy (on the dataset)
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
├─ requirement.txt
├─ heart_model.pkl
├─ scaler.pkl
└─ health_data.csv
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

### How It Works (High Level)
1. On startup, the app loads:
   - `heart_model.pkl`
   - `scaler.pkl`
   - `health_data.csv`
2. It determines feature columns from the CSV: all columns except `cardio`.
3. User inputs are assembled into a DataFrame using the same feature order.
4. The app scales inputs using `scaler.pkl`.
5. The model predicts:
   - `0` → Low Risk
   - `1` → High Risk
6. It displays prediction + probability.

---

### Notes / Limitations
- This tool is **not clinically validated**.
- The prediction is only as good as the underlying dataset and trained model.
- Always treat outputs as **educational/demo** guidance, not medical truth.

---

### Educational Use
Use this project to:
- Demonstrate how ML inference can be wrapped in a UI
- Show preprocessing (scaling) + prediction pipeline
- Build a real-time interactive dashboard for model outputs

