# ============================================================
# SmartHealthAI - Governance Rules Engine
# Clinical & Ethical Constraint Layer
# ============================================================

import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# SYSTEM METADATA
# ------------------------------------------------------------

SYSTEM_NAME = "SmartHealthAI"
VERSION = "1.0"
CREATED_BY = "AutoML + Clinical Governance Framework"
LAST_UPDATED = "2026-01-20"

# ------------------------------------------------------------
# GLOBAL CLINICAL PRINCIPLES
# ------------------------------------------------------------

CLINICAL_PRINCIPLES = {
    "patient_safety_first": True,
    "minimize_false_negatives": True,
    "model_explainability_required": True,
    "no_black_box_decisions": True,
    "assistive_not_diagnostic": True
}

# ------------------------------------------------------------
# DISEASE-SPECIFIC GOVERNANCE DEFINITIONS
# ------------------------------------------------------------

DISEASE_GOVERNANCE = {

    # =========================
    # DIABETES GOVERNANCE
    # =========================
    "diabetes": {
        "disease_name": "Diabetes Mellitus",
        "input_type": "static",
        "criticality": "high",

        "mandatory_features": [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age"
        ],

        "feature_ranges": {
            "pregnancies": (0, 20),
            "glucose": (50, 300),
            "blood_pressure": (40, 200),
            "skin_thickness": (0, 100),
            "insulin": (0, 900),
            "bmi": (10.0, 70.0),
            "diabetes_pedigree": (0.0, 3.0),
            "age": (1, 120)
        },

        "priority_metric": "recall",
        "min_recall_required": 0.85,

        "risk_thresholds": {
            "low": 0.30,
            "moderate": 0.50,
            "high": 0.70
        },

        "recommendations": {
            "low": "Maintain healthy lifestyle.",
            "moderate": "Consult physician for further tests.",
            "high": "Immediate medical consultation advised."
        }
    },

    # =========================
    # LIVER DISEASE GOVERNANCE
    # =========================
    "liver": {
        "disease_name": "Liver Disease",
        "input_type": "static",
        "criticality": "high",

        "mandatory_features": [
            "age",
            "total_bilirubin",
            "direct_bilirubin",
            "alkaline_phosphotase",
            "alamine_aminotransferase",
            "aspartate_aminotransferase",
            "total_proteins",
            "albumin",
            "albumin_globulin_ratio"
        ],

        "feature_ranges": {
            "age": (1, 120),
            "total_bilirubin": (0.1, 75.0),
            "direct_bilirubin": (0.0, 50.0),
            "alkaline_phosphotase": (20, 2000),
            "alamine_aminotransferase": (5, 2000),
            "aspartate_aminotransferase": (5, 2000),
            "total_proteins": (2.0, 10.0),
            "albumin": (1.0, 6.0),
            "albumin_globulin_ratio": (0.1, 3.0)
        },

        "priority_metric": "f1_score",
        "min_f1_required": 0.80,

        "risk_thresholds": {
            "low": 0.35,
            "moderate": 0.55,
            "high": 0.75
        },

        "recommendations": {
            "low": "Routine monitoring recommended.",
            "moderate": "Liver function tests suggested.",
            "high": "Urgent hepatologist consultation required."
        }
    },

    # =========================
    # BREAST CANCER GOVERNANCE
    # =========================
    "breast_cancer": {
        "disease_name": "Breast Cancer",
        "input_type": "static",
        "criticality": "very_high",

        "mandatory_features": "ALL",

        "priority_metric": "recall_auc",
        "min_recall_required": 0.90,
        "min_auc_required": 0.90,

        "risk_thresholds": {
            "low": 0.20,
            "moderate": 0.40,
            "high": 0.60
        },

        "recommendations": {
            "low": "No malignancy detected.",
            "moderate": "Further imaging tests recommended.",
            "high": "Immediate oncological evaluation required."
        }
    }
}

# ------------------------------------------------------------
# VALIDATION FUNCTIONS
# ------------------------------------------------------------

def validate_feature_presence(disease, input_data):
    rules = DISEASE_GOVERNANCE[disease]

    if rules["mandatory_features"] == "ALL":
        if len(input_data) == 0:
            raise ValueError("No input features provided")
        return True

    missing = []
    for feature in rules["mandatory_features"]:
        if feature not in input_data:
            missing.append(feature)

    if missing:
        raise ValueError(f"Missing mandatory features: {missing}")

    return True


def validate_feature_ranges(disease, input_data):
    ranges = DISEASE_GOVERNANCE[disease].get("feature_ranges", {})

    for feature, value in input_data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if value < min_val or value > max_val:
                raise ValueError(
                    f"Feature '{feature}' value {value} out of range ({min_val}-{max_val})"
                )

    return True

# ------------------------------------------------------------
# RISK ASSESSMENT LOGIC
# ------------------------------------------------------------

def assess_risk_level(disease, probability):
    thresholds = DISEASE_GOVERNANCE[disease]["risk_thresholds"]

    if probability >= thresholds["high"]:
        return "HIGH"
    elif probability >= thresholds["moderate"]:
        return "MODERATE"
    else:
        return "LOW"

# ------------------------------------------------------------
# CLINICAL DECISION CONTROL
# ------------------------------------------------------------

def clinical_decision_gate(disease, probability, model_metrics=None):
    rules = DISEASE_GOVERNANCE[disease]
    risk = assess_risk_level(disease, probability)

    decision = {
        "risk_level": risk,
        "allowed": True,
        "message": rules["recommendations"][risk.lower()]
    }

    # Enforce metric governance
    if model_metrics:
        if "recall" in rules and model_metrics.get("recall", 1.0) < rules.get("min_recall_required", 0):
            decision["allowed"] = False
            decision["message"] = "Model recall below clinical safety threshold."

        if "min_f1_required" in rules and model_metrics.get("f1", 1.0) < rules["min_f1_required"]:
            decision["allowed"] = False
            decision["message"] = "Model F1-score below acceptable clinical level."

    return decision

# ------------------------------------------------------------
# EXPLAINABILITY GOVERNANCE
# ------------------------------------------------------------

def validate_explainability(feature_importance):
    if feature_importance is None or len(feature_importance) == 0:
        raise ValueError("Explainability information missing.")

    if not isinstance(feature_importance, (list, np.ndarray)):
        raise ValueError("Invalid explainability format.")

    return True

# ------------------------------------------------------------
# FULL GOVERNANCE PIPELINE
# ------------------------------------------------------------

def apply_full_governance(disease, input_data, probability, feature_importance=None, model_metrics=None):
    validate_feature_presence(disease, input_data)
    validate_feature_ranges(disease, input_data)
    validate_explainability(feature_importance)

    decision = clinical_decision_gate(disease, probability, model_metrics)

    audit_record = {
        "system": SYSTEM_NAME,
        "disease": disease,
        "probability": round(float(probability), 4),
        "decision": decision,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return audit_record

# ------------------------------------------------------------
# GOVERNANCE SUMMARY
# ------------------------------------------------------------

def governance_summary():
    return {
        "system": SYSTEM_NAME,
        "version": VERSION,
        "principles": CLINICAL_PRINCIPLES,
        "diseases_supported": list(DISEASE_GOVERNANCE.keys()),
        "last_updated": LAST_UPDATED
    }

# ------------------------------------------------------------
# MANUAL TEST BLOCK
# ------------------------------------------------------------

if __name__ == "__main__":

    test_input = {
        "pregnancies": 3,
        "glucose": 165,
        "blood_pressure": 85,
        "skin_thickness": 32,
        "insulin": 130,
        "bmi": 34.2,
        "diabetes_pedigree": 0.8,
        "age": 50
    }

    test_probability = 0.78
    test_importance = [0.12, 0.30, 0.05, 0.04, 0.06, 0.20, 0.10, 0.13]

    output = apply_full_governance(
        disease="diabetes",
        input_data=test_input,
        probability=test_probability,
        feature_importance=test_importance,
        model_metrics={"recall": 0.88}
    )

    print(output)
