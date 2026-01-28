# ============================================================
# SmartHealthAI - Model Controller
# AutoML + Medical Governance Logic
# ============================================================

import os
import pickle
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# Base Directory Setup
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
AUTOML_MODEL_DIR = os.path.join(MODEL_DIR, "automl_models")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "system.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ------------------------------------------------------------
# Logging Utility
# ------------------------------------------------------------

def log_event(message):
    with open(LOG_FILE, "a") as log:
        log.write(f"[{datetime.now()}] {message}\n")

# ------------------------------------------------------------
# Governance Rules (Medical Constraints)
# ------------------------------------------------------------

GOVERNANCE_RULES = {
    "diabetes": {
        "required_features": [
            "pregnancies", "glucose", "blood_pressure",
            "skin_thickness", "insulin", "bmi",
            "diabetes_pedigree", "age"
        ],
        "priority_metric": "recall",
        "risk_threshold": 0.40
    },

    "liver": {
        "required_features": [
            "age", "total_bilirubin", "direct_bilirubin",
            "alkaline_phosphotase", "alamine_aminotransferase",
            "aspartate_aminotransferase", "total_proteins",
            "albumin", "albumin_globulin_ratio"
        ],
        "priority_metric": "f1_score",
        "risk_threshold": 0.45
    },

    "breast_cancer": {
        "required_features": "ALL",
        "priority_metric": "recall_auc",
        "risk_threshold": 0.30
    }
}

# ------------------------------------------------------------
# Load AutoML Models
# ------------------------------------------------------------

def load_automl_model(disease):
    model_path = os.path.join(AUTOML_MODEL_DIR, f"{disease}_automl.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AutoML model not found for {disease}")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    log_event(f"Loaded AutoML model for {disease}")
    return model

AUTOML_MODELS = {
    "diabetes": load_automl_model("diabetes"),
    "liver": load_automl_model("liver"),
    "cancer": load_automl_model("cancer")
}

# ------------------------------------------------------------
# Feature Validation Layer
# ------------------------------------------------------------

def validate_features(disease, input_data):
    rules = GOVERNANCE_RULES[disease]

    if rules["required_features"] == "ALL":
        return True

    missing = []
    for feature in rules["required_features"]:
        if feature not in input_data:
            missing.append(feature)

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return True

# ------------------------------------------------------------
# Input Vector Builder
# ------------------------------------------------------------

def build_input_vector(disease, input_data):
    rules = GOVERNANCE_RULES[disease]

    if rules["required_features"] == "ALL":
        return np.array([list(input_data.values())])

    vector = []
    for feature in rules["required_features"]:
        vector.append(input_data[feature])

    return np.array([vector])

# ------------------------------------------------------------
# Risk Governance Logic
# ------------------------------------------------------------

def apply_risk_governance(disease, probability):
    threshold = GOVERNANCE_RULES[disease]["risk_threshold"]

    if probability >= threshold:
        decision = "HIGH RISK"
    else:
        decision = "LOW RISK"

    return decision

# ------------------------------------------------------------
# Explainability Layer (Basic but Real)
# ------------------------------------------------------------

def extract_feature_importance(model):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist()

    if hasattr(model, "coef_"):
        return model.coef_.flatten().tolist()

    return []

# ------------------------------------------------------------
# Main Prediction Controller
# ------------------------------------------------------------

def predict_disease(disease, input_data):
    """
    Central prediction method governed by medical rules
    """

    log_event(f"Prediction request received for {disease}")

    # Step 1: Validate disease
    if disease not in GOVERNANCE_RULES:
        raise ValueError("Unsupported disease type")

    # Step 2: Validate input features
    validate_features(disease, input_data)

    # Step 3: Build input vector
    input_vector = build_input_vector(disease, input_data)

    # Step 4: Select AutoML model
    model_key = disease if disease != "breast_cancer" else "cancer"
    model = AUTOML_MODELS[model_key]

    # Step 5: Predict
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]

    # Step 6: Apply governance threshold
    risk_level = apply_risk_governance(disease, probability)

    # Step 7: Explainability
    importance = extract_feature_importance(model)

    # Step 8: Format result
    result = {
        "disease": disease,
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "explanation": importance
    }

    log_event(f"Prediction completed for {disease} | Risk: {risk_level}")

    return result

# ------------------------------------------------------------
# Governance Audit Report Generator
# ------------------------------------------------------------

def generate_governance_report():
    report = {
        "system": "SmartHealthAI",
        "models_loaded": list(AUTOML_MODELS.keys()),
        "rules": GOVERNANCE_RULES,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    log_event("Governance audit report generated")
    return report

# ------------------------------------------------------------
# System Health Check
# ------------------------------------------------------------

def system_health_check():
    status = {
        "automl_models_loaded": True,
        "log_system": os.path.exists(LOG_FILE),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return status

# ------------------------------------------------------------
# Manual Test Execution (Optional)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example test (Diabetes)
    sample_input = {
        "pregnancies": 2,
        "glucose": 145,
        "blood_pressure": 80,
        "skin_thickness": 25,
        "insulin": 120,
        "bmi": 32.5,
        "diabetes_pedigree": 0.6,
        "age": 45
    }

    output = predict_disease("diabetes", sample_input)
    print(output)
