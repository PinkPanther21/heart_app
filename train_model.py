import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# =========================================
# LOAD DATA
# =========================================
heart_data = pd.read_csv('health_data.csv')
print("Raw data shape:", heart_data.shape)
print("\nFirst few rows:")
print(heart_data.head())
print("\nColumn info:")
heart_data.info()
print("\nNull values:", heart_data.isnull().sum().to_dict())
print("Duplicates:", heart_data.duplicated().sum())
print("\nTarget distribution:\n", heart_data['cardio'].value_counts())

# =========================================
# CLEAN & FIX UNITS
# =========================================

# Drop id column — not a medical feature, causes scaler mismatch in app
if 'id' in heart_data.columns:
    heart_data = heart_data.drop(columns=['id'])
    print("\n[✓] Dropped 'id' column")

# Age is stored in DAYS in this dataset — convert to years
if heart_data['age'].max() > 200:
    heart_data['age'] = (heart_data['age'] / 365).round(1)
    print("[✓] Converted age from days → years")

# Remove physiologically impossible values
before = len(heart_data)

# Height: keep 100–220 cm
heart_data = heart_data[(heart_data['height'] >= 100) & (heart_data['height'] <= 220)]

# Weight: keep 30–200 kg
heart_data = heart_data[(heart_data['weight'] >= 30) & (heart_data['weight'] <= 200)]

# Systolic BP (ap_hi): keep 70–250
heart_data = heart_data[(heart_data['ap_hi'] >= 70) & (heart_data['ap_hi'] <= 250)]

# Diastolic BP (ap_lo): keep 40–150
heart_data = heart_data[(heart_data['ap_lo'] >= 40) & (heart_data['ap_lo'] <= 150)]

# Diastolic must be less than systolic
heart_data = heart_data[heart_data['ap_lo'] < heart_data['ap_hi']]

# Age: keep 1–120 years
heart_data = heart_data[(heart_data['age'] >= 1) & (heart_data['age'] <= 120)]

after = len(heart_data)
print(f"[✓] Removed {before - after} outlier rows — {after} rows remain")

print("\nCleaned data stats:")
print(heart_data[['age', 'height', 'weight', 'ap_hi', 'ap_lo']].describe().round(1))

# =========================================
# FEATURES & TARGET
# =========================================
X = heart_data.drop(columns='cardio')
Y = heart_data['cardio']

print("\nFeatures used:", X.columns.tolist())
print("Feature count:", len(X.columns))
print("Target distribution:\n", Y.value_counts())

# =========================================
# TRAIN / TEST SPLIT
# =========================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    stratify=Y,
    random_state=2
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# =========================================
# SCALE
# =========================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("[✓] Scaler fitted on features:", scaler.feature_names_in_.tolist())

# =========================================
# TRAIN MODEL
# =========================================
model = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    C=0.1,              # slight regularisation for generalisation
    random_state=2
)
model.fit(X_train_sc, Y_train)

# =========================================
# EVALUATE
# =========================================
train_pred = model.predict(X_train_sc)
test_pred  = model.predict(X_test_sc)

train_acc = accuracy_score(Y_train, train_pred)
test_acc  = accuracy_score(Y_test,  test_pred)

print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
print(f"Testing  Accuracy : {test_acc  * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, test_pred))
print("\nClassification Report:\n", classification_report(Y_test, test_pred))

# =========================================
# SANITY-CHECK PREDICTION
# (values now in human-readable units)
# =========================================
sample_data = {
    col: float(X.median()[col]) for col in X.columns
}
# Override a few to make an obvious high-risk profile
sample_data['age']    = 60.0
sample_data['ap_hi']  = 160
sample_data['ap_lo']  = 100
sample_data['cholesterol'] = 3

sample_df = pd.DataFrame([sample_data])
sample_sc = scaler.transform(sample_df)
sample_pred = model.predict(sample_sc)[0]
sample_prob = model.predict_proba(sample_sc)[0][1] * 100

print(f"\nSanity check (60yr, BP 160/100, high cholesterol):")
print(f"  Prediction : {'Heart Disease' if sample_pred == 1 else 'Healthy'}")
print(f"  Probability: {sample_prob:.1f}%")

# =========================================
# SAVE ARTIFACTS
# =========================================
pickle.dump(model,  open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl",      "wb"))

# Also save the cleaned dataset so the app reads correct values
heart_data.to_csv("health_data_clean.csv", index=False)

print("\n[✓] Saved: heart_model.pkl")
print("[✓] Saved: scaler.pkl")
print("[✓] Saved: health_data_clean.csv  ← use this in app.py")
