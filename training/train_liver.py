# ============================================================
# SmartHealthAI - Liver Disease AutoML Training (Ensemble: RF + XGBoost)
# Parameters: ALT, AST, ALP, Albumin, Total Protein, Bilirubin
# ============================================================

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb  # Install via: pip install xgboost

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "liver_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "liver_automl.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("[INFO] Loading liver dataset...")
print("[DEBUG] DATA_PATH:", DATA_PATH)
print("[DEBUG] File exists:", os.path.exists(DATA_PATH))

try:
    df = pd.read_csv(DATA_PATH, engine="python", encoding='latin1')
except UnicodeDecodeError:
    print("[WARNING] latin1 encoding failed. Trying cp1252...")
    df = pd.read_csv(DATA_PATH, engine="python", encoding='cp1252')

print("[INFO] Raw columns:", df.columns.tolist())

# ============================================================
# STANDARDIZE COLUMN NAMES (AUTO)
# ============================================================

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
)

COLUMN_MAP = {
    # Existing mappings (keep for flexibility)
    "alt": "alt",
    "sgpt": "alt",
    "alanine_aminotransferase": "alt",

    "ast": "ast",
    "sgot": "ast",
    "aspartate_aminotransferase": "ast",

    "alp": "alp",
    "alkaline_phosphatase": "alp",

    "albumin": "albumin",

    "total_protein": "total_protein",
    "tp": "total_protein",

    "total_bilirubin": "total_bilirubin",
    "bilirubin_total": "total_bilirubin",

    "direct_bilirubin": "direct_bilirubin",

    "age": "age",

    "dataset": "label",
    "class": "label",
    "target": "label",
    "liver_disease": "label",

    # New mappings for normalized column names from your file
    "age_of_the_patient": "age",
    "gender_of_the_patient": "gender",
    "alkphos_alkaline_phosphotase": "alp",  # Maps to 'alp'
    "sgpt_alamine_aminotransferase": "alt",  # Maps to 'alt'
    "sgot_aspartate_aminotransferase": "ast",  # Maps to 'ast'
    "total_protiens": "total_protein",  # Maps to 'total_protein'
    "alb_albumin": "albumin",  # Maps to 'albumin'
    "a_g_ratio_albumin_and_globulin_ratio": "albumin_globulin_ratio",  # Optional, if needed
    "result": "label"  # Maps to 'label'
}

df.rename(columns=COLUMN_MAP, inplace=True)

print("[INFO] Normalized columns:", df.columns.tolist())

# ============================================================
# REQUIRED FEATURES (ONLY MEDICAL ONES)
# ============================================================

FEATURES = [
    "alt",
    "ast",
    "alp",
    "albumin",
    "total_protein",
    "total_bilirubin"
]

# Optional feature
if "direct_bilirubin" in df.columns:
    FEATURES.append("direct_bilirubin")

if "age" in df.columns:
    FEATURES.append("age")

TARGET = "label"

# ============================================================
# CLEAN DATA
# ============================================================

df = df[FEATURES + [TARGET]]
df.dropna(inplace=True)

df[TARGET] = df[TARGET].apply(lambda x: 1 if x in [1, "yes", "positive"] else 0)

print("[INFO] Final dataset shape:", df.shape)
print("[INFO] Features used:", FEATURES)

# ============================================================
# SPLIT DATA
# ============================================================

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# ENSEMBLE MODEL (Random Forest + XGBoost)
# ============================================================

print("\n[INFO] Building ensemble model (Random Forest + XGBoost)...")

# Define individual models
rf = RandomForestClassifier(n_estimators=400, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=400, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Create voting ensemble (soft voting for probabilities)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model)],
    voting='soft'  # Averages probabilities for better recall
)

# Train the ensemble
ensemble.fit(X_train_scaled, y_train)

# Predict and evaluate
preds = ensemble.predict(X_test_scaled)
probs = ensemble.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "model": "Ensemble (RF + XGBoost)",
    "accuracy": accuracy_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "f1": f1_score(y_test, preds),
    "roc_auc": roc_auc_score(y_test, probs)
}

print("[INFO] Ensemble metrics:", metrics)

# ============================================================
# SAVE MODEL
# ============================================================

final_model = {
    "model": ensemble,
    "scaler": scaler,
    "features": FEATURES
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(final_model, f)

print("\n[SUCCESS] Liver ensemble model saved at:")
print(MODEL_PATH)