# ============================================================
# SmartHealthAI
# AutoML Training Script for Diabetes Prediction
# Dataset: BRFSS Diabetes Health Indicators
# ============================================================

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "diabetes.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "automl_models", "diabetes_automl.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "training_diabetes.log")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ------------------------------------------------------------
# LOGGING FUNCTION
# ------------------------------------------------------------

def log(message):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("[INFO] Loading dataset...")
log("Loading dataset")

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset loaded")
log("Dataset loaded successfully")

# ------------------------------------------------------------
# DATA INSPECTION
# ------------------------------------------------------------

print("\n[INFO] Initial dataset shape:", df.shape)
log(f"Initial shape: {df.shape}")

# ------------------------------------------------------------
# TARGET TRANSFORMATION
# ------------------------------------------------------------

# Diabetes_012:
# 0 = No Diabetes
# 1 = Prediabetes
# 2 = Diabetes

print("[INFO] Converting target labels...")
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x >= 1 else 0)
log("Converted Diabetes_012 to binary target")

# ------------------------------------------------------------
# FEATURE SELECTION (CLINICAL RELEVANCE)
# ------------------------------------------------------------

selected_features = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]

print("[INFO] Selecting clinically relevant features...")
log("Selecting clinically relevant features")

X = df[selected_features]
y = df["Diabetes_binary"]

# ------------------------------------------------------------
# MISSING VALUE HANDLING
# ------------------------------------------------------------

print("[INFO] Handling missing values...")
log("Handling missing values")

X = X.fillna(X.median())

# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("[INFO] Splitting data into train/test...")
log("Splitting dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

print("[INFO] Scaling features...")
log("Scaling features")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# AUTO MODEL DEFINITIONS (AutoML SEARCH SPACE)
# ------------------------------------------------------------

print("[INFO] Initializing model candidates...")
log("Initializing model candidates")

models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

# ------------------------------------------------------------
# MODEL TRAINING + EVALUATION
# ------------------------------------------------------------

results = {}

print("\n[INFO] Starting AutoML training process...\n")
log("Starting AutoML training")

for name, model in models.items():
    print(f"[TRAINING] {name}")
    log(f"Training model: {name}")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }

    print(f"  Accuracy : {acc}")
    print(f"  Precision: {prec}")
    print(f"  Recall   : {rec}")
    print(f"  F1-score : {f1}")
    print(f"  ROC-AUC  : {auc}\n")

    log(f"{name} | Acc:{acc} Prec:{prec} Rec:{rec} F1:{f1} AUC:{auc}")

# ------------------------------------------------------------
# GOVERNANCE-BASED MODEL SELECTION
# ------------------------------------------------------------

print("\n[INFO] Applying governance-based model selection...")
log("Applying governance-based model selection")

# Governance rule:
# Diabetes → prioritize RECALL (minimize false negatives)

best_model_name = None
best_recall = -1

for name, metrics in results.items():
    if metrics["recall"] > best_recall:
        best_recall = metrics["recall"]
        best_model_name = name

best_model = results[best_model_name]["model"]

print(f"\n[SELECTED MODEL] {best_model_name}")
log(f"Selected model: {best_model_name}")

# ------------------------------------------------------------
# FINAL EVALUATION REPORT
# ------------------------------------------------------------

print("\n================ FINAL MODEL REPORT ================\n")

final_pred = best_model.predict(X_test_scaled)
final_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print("Model Name:", best_model_name)
print("Accuracy :", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall   :", recall_score(y_test, final_pred))
print("F1 Score :", f1_score(y_test, final_pred))
print("ROC-AUC  :", roc_auc_score(y_test, final_prob))

print("\nClassification Report:\n")
print(classification_report(y_test, final_pred))

log("Final model evaluation completed")

# ------------------------------------------------------------
# MODEL SERIALIZATION
# ------------------------------------------------------------

print("\n[INFO] Saving trained model...")
log("Saving trained model")

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "features": selected_features,
        "metrics": results[best_model_name],
        "trained_on": str(datetime.now())
    }, f)

print("[SUCCESS] Model saved to:", MODEL_SAVE_PATH)
log("Model saved successfully")

# ------------------------------------------------------------
# TRAINING COMPLETED
# ------------------------------------------------------------

print("\n================ TRAINING COMPLETED SUCCESSFULLY ================\n")
log("Training completed successfully")
