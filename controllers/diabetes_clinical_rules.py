# ============================================================
# SmartHealthAI - Diabetes Clinical Rules Engine v2
# Medical Diagnostic Thresholds (ADA/WHO Guidelines)
# ============================================================

import numpy as np


def score_glucose(glucose):
    if glucose <= 0: return 0.0
    if glucose < 70: return 0.05
    elif glucose <= 90: return 0.02 + (glucose - 70) * 0.003
    elif glucose <= 99: return 0.08 + (glucose - 90) * 0.013
    elif glucose <= 110: return 0.40 + (glucose - 100) * 0.015
    elif glucose <= 125: return 0.55 + (glucose - 110) * 0.02
    elif glucose <= 150: return 0.80 + (glucose - 126) * 0.005
    elif glucose <= 200: return 0.92 + (glucose - 150) * 0.0006
    else: return min(0.95 + (glucose - 200) * 0.0003, 0.99)


def score_bp(bp):
    if bp <= 0: return 0.0
    if bp < 90: return 0.05
    elif bp <= 120: return 0.02 + (bp - 90) * 0.003
    elif bp <= 129: return 0.15 + (bp - 120) * 0.012
    elif bp <= 139: return 0.30 + (bp - 130) * 0.015
    elif bp <= 180: return 0.50 + (bp - 140) * 0.005
    else: return min(0.70 + (bp - 180) * 0.003, 0.85)


def score_bmi(bmi):
    if bmi <= 0: return 0.0
    if bmi < 18.5: return 0.05
    elif bmi <= 22.0: return 0.02 + (bmi - 18.5) * 0.008
    elif bmi <= 24.9: return 0.05 + (bmi - 22.0) * 0.017
    elif bmi <= 29.9: return 0.25 + (bmi - 25.0) * 0.051
    elif bmi <= 34.9: return 0.50 + (bmi - 30.0) * 0.04
    elif bmi <= 39.9: return 0.70 + (bmi - 35.0) * 0.02
    else: return min(0.80 + (bmi - 40.0) * 0.01, 0.95)


def score_age(age):
    if age <= 0: return 0.0
    if age < 25: return 0.02 + (age / 25) * 0.06
    elif age < 35: return 0.08 + ((age - 25) / 10) * 0.10
    elif age < 45: return 0.18 + ((age - 35) / 10) * 0.12
    elif age < 55: return 0.30 + ((age - 45) / 10) * 0.10
    elif age < 65: return 0.40 + ((age - 55) / 10) * 0.10
    else: return min(0.50 + ((age - 65) / 20) * 0.10, 0.60)


def score_hba1c(hba1c):
    """HbA1c: <5.7 Normal, 5.7-6.4 Prediabetes, >=6.5 Diabetes"""
    if hba1c <= 0: return 0.0
    if hba1c < 5.0: return 0.02
    elif hba1c < 5.7: return 0.05 + (hba1c - 5.0) * 0.07
    elif hba1c < 6.0: return 0.45 + (hba1c - 5.7) * 0.1
    elif hba1c < 6.5: return 0.55 + (hba1c - 6.0) * 0.18
    elif hba1c < 8.0: return 0.85 + (hba1c - 6.5) * 0.03
    else: return min(0.93 + (hba1c - 8.0) * 0.01, 0.99)


def score_insulin(insulin):
    """Fasting insulin: 2-25 uIU/mL normal"""
    if insulin <= 0: return 0.0
    if insulin < 2: return 0.15
    elif insulin <= 15: return 0.02 + (insulin - 2) * 0.005
    elif insulin <= 25: return 0.08 + (insulin - 15) * 0.015
    elif insulin <= 100: return 0.25 + (insulin - 25) * 0.005
    elif insulin <= 300: return 0.60 + (insulin - 100) * 0.001
    else: return min(0.80, 0.80)


def score_skin_thickness(st):
    """Triceps skin fold: 10-50mm normal range"""
    if st <= 0: return 0.0
    if st <= 20: return 0.02
    elif st <= 30: return 0.05 + (st - 20) * 0.01
    elif st <= 40: return 0.15 + (st - 30) * 0.02
    else: return min(0.35 + (st - 40) * 0.015, 0.65)


def score_family_history(has_history):
    """Family history of diabetes is a strong risk factor"""
    return 0.55 if has_history else 0.02


def score_physical_activity(is_active):
    """Physical inactivity increases diabetes risk"""
    return 0.05 if is_active else 0.35


# --- Classification helpers ---

def classify_glucose(g):
    if g < 70: return "Hypoglycemia"
    elif g <= 99: return "Normal"
    elif g <= 125: return "Prediabetes"
    elif g <= 200: return "Diabetic"
    else: return "Critical High"

def classify_hba1c(h):
    if h <= 0: return "Not Provided"
    elif h < 5.7: return "Normal"
    elif h < 6.5: return "Prediabetes"
    else: return "Diabetic"

def classify_bp(bp):
    if bp < 90: return "Low"
    elif bp <= 120: return "Normal"
    elif bp <= 129: return "Elevated"
    elif bp <= 139: return "Stage 1 Hypertension"
    else: return "Stage 2 Hypertension"

def classify_bmi(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi <= 24.9: return "Normal"
    elif bmi <= 29.9: return "Overweight"
    elif bmi <= 34.9: return "Obese Class I"
    elif bmi <= 39.9: return "Obese Class II"
    else: return "Obese Class III"

def classify_insulin(ins):
    if ins <= 0: return "Not Provided"
    elif ins < 2: return "Low"
    elif ins <= 25: return "Normal"
    elif ins <= 100: return "Elevated"
    else: return "High"


# ============================================================
# MAIN HYBRID PREDICTION
# ============================================================

def hybrid_diabetes_predict(glucose, blood_pressure, bmi, age,
                            ml_model, ml_scaler=None,
                            hba1c=0, insulin=0, skin_thickness=0,
                            family_history=False, physical_activity=True):
    
    # --- Collect all scores ---
    g_score = score_glucose(glucose)
    bp_sc = score_bp(blood_pressure)
    bmi_sc = score_bmi(bmi)
    age_sc = score_age(age)
    hba1c_sc = score_hba1c(hba1c) if hba1c > 0 else None
    ins_sc = score_insulin(insulin) if insulin > 0 else None
    st_sc = score_skin_thickness(skin_thickness) if skin_thickness > 0 else None
    fam_sc = score_family_history(family_history)
    phy_sc = score_physical_activity(physical_activity)

    # --- Dynamic weighting based on what's provided ---
    weights = {}
    scores = {}
    
    # Always-present parameters
    weights["glucose"] = 0.30
    scores["glucose"] = g_score
    weights["bmi"] = 0.15
    scores["bmi"] = bmi_sc
    weights["bp"] = 0.08
    scores["bp"] = bp_sc
    weights["age"] = 0.07
    scores["age"] = age_sc
    weights["family"] = 0.06
    scores["family"] = fam_sc
    weights["activity"] = 0.04
    scores["activity"] = phy_sc
    
    # Optional high-value parameters
    if hba1c_sc is not None:
        weights["hba1c"] = 0.22
        scores["hba1c"] = hba1c_sc
    if ins_sc is not None:
        weights["insulin"] = 0.05
        scores["insulin"] = ins_sc
    if st_sc is not None:
        weights["skin"] = 0.03
        scores["skin"] = st_sc
    
    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w
    
    # Weighted clinical score
    clinical_score = sum(scores[k] * weights[k] for k in scores)
    
    # --- ML Model Score ---
    features = np.array([[glucose, blood_pressure, bmi, age]])
    if ml_scaler:
        features = ml_scaler.transform(features)
    ml_prob = float(ml_model.predict_proba(features)[0][1])
    
    # --- Blend: 55% clinical, 45% ML ---
    hybrid = (clinical_score * 0.55) + (ml_prob * 0.45)

    # --- Medical rule adjustments (NOT hard floors) ---
    # Count how many diagnostic criteria are met
    diag_flags = 0
    diag_total = 1  # glucose always checked
    
    if glucose >= 126: diag_flags += 1
    
    if hba1c > 0:
        diag_total += 1
        if hba1c >= 6.5: diag_flags += 1
    
    # Both glucose AND HbA1c confirm diabetes
    if diag_flags >= 2:
        hybrid = max(hybrid, 0.88 + min(diag_flags * 0.03, 0.10))
    elif diag_flags == 1:
        # Only one marker above threshold — adjust based on actual values
        if glucose >= 200:
            hybrid = max(hybrid, 0.90)
        elif glucose >= 126:
            # Scale from 0.75 to 0.92 based on how far above 126
            extra = min((glucose - 126) / 74, 1.0)
            hybrid = max(hybrid, 0.75 + extra * 0.17)
        elif hba1c >= 6.5:
            extra = min((hba1c - 6.5) / 3.5, 1.0)
            hybrid = max(hybrid, 0.75 + extra * 0.15)

    # Prediabetes boosters
    prediab_count = 0
    if 100 <= glucose <= 125: prediab_count += 1
    if 0 < hba1c and 5.7 <= hba1c < 6.5: prediab_count += 1
    if bmi >= 25: prediab_count += 1
    
    if prediab_count >= 2:
        hybrid = max(hybrid, 0.45 + prediab_count * 0.08)
    
    # All normal values — cap low
    all_normal = (glucose < 100 and bmi < 25 and blood_pressure <= 120 
                  and (hba1c <= 0 or hba1c < 5.7))
    if all_normal and not family_history:
        hybrid = min(hybrid, 0.15)
    
    # Clamp
    hybrid = min(max(hybrid, 0.01), 0.99)
    
    # --- Classification ---
    # Use actual medical criteria, not just the score
    if diag_flags >= 1 or hybrid >= 0.65:
        result = "Diabetic"
    elif (100 <= glucose <= 125) or (0 < hba1c and 5.7 <= hba1c < 6.5) or (0.35 <= hybrid < 0.65):
        result = "Prediabetic"
    else:
        result = "Non-Diabetic"

    # Risk level
    if hybrid >= 0.75: risk_level = "VERY HIGH"
    elif hybrid >= 0.50: risk_level = "HIGH"
    elif hybrid >= 0.30: risk_level = "MODERATE"
    else: risk_level = "LOW"

    # Confidence
    if result == "Non-Diabetic":
        confidence = round((1 - hybrid) * 100, 1)
    else:
        confidence = round(hybrid * 100, 1)

    # --- Parameter breakdown ---
    breakdown = {
        "glucose": {
            "value": glucose, "unit": "mg/dL",
            "status": classify_glucose(glucose),
            "risk_score": round(g_score * 100, 1),
            "normal_range": "70 – 99 mg/dL",
            "prediabetes_range": "100 – 125 mg/dL",
            "diabetic_threshold": "≥ 126 mg/dL"
        },
        "bmi": {
            "value": round(bmi, 1), "unit": "kg/m²",
            "status": classify_bmi(bmi),
            "risk_score": round(bmi_sc * 100, 1),
            "normal_range": "18.5 – 24.9",
            "risk_threshold": "≥ 25 (Overweight)"
        },
        "blood_pressure": {
            "value": blood_pressure, "unit": "mm Hg",
            "status": classify_bp(blood_pressure),
            "risk_score": round(bp_sc * 100, 1),
            "normal_range": "90 – 120 mm Hg"
        },
        "age": {
            "value": int(age), "unit": "years",
            "risk_score": round(age_sc * 100, 1),
            "note": "ADA recommends screening at 35+" if age >= 35 else "Below screening age"
        }
    }
    
    if hba1c > 0:
        breakdown["hba1c"] = {
            "value": hba1c, "unit": "%",
            "status": classify_hba1c(hba1c),
            "risk_score": round(hba1c_sc * 100, 1),
            "normal_range": "Below 5.7%",
            "prediabetes_range": "5.7% – 6.4%",
            "diabetic_threshold": "≥ 6.5%"
        }
    if insulin > 0:
        breakdown["insulin"] = {
            "value": insulin, "unit": "µIU/mL",
            "status": classify_insulin(insulin),
            "risk_score": round(ins_sc * 100, 1),
            "normal_range": "2 – 25 µIU/mL"
        }
    if skin_thickness > 0:
        breakdown["skin_thickness"] = {
            "value": skin_thickness, "unit": "mm",
            "risk_score": round(st_sc * 100, 1),
            "normal_range": "10 – 50 mm"
        }
    
    breakdown["family_history"] = {
        "value": "Yes" if family_history else "No",
        "risk_score": round(fam_sc * 100, 1)
    }
    breakdown["physical_activity"] = {
        "value": "Active" if physical_activity else "Inactive",
        "risk_score": round(phy_sc * 100, 1)
    }

    # --- Medical explanation (simple language) ---
    explanation = []
    
    # Glucose explanation
    if glucose >= 200:
        explanation.append({
            "icon": "🔴", "title": "Glucose is Critically High",
            "text": f"Your fasting glucose is {glucose} mg/dL. Anything above 200 mg/dL is a strong sign of diabetes. You should see a doctor immediately."
        })
    elif glucose >= 126:
        explanation.append({
            "icon": "🟠", "title": "Glucose is Above Diabetic Threshold",
            "text": f"Your fasting glucose is {glucose} mg/dL. The medical limit for diabetes is 126 mg/dL or above. A second confirmatory test is recommended."
        })
    elif glucose >= 100:
        explanation.append({
            "icon": "🟡", "title": "Glucose is in Prediabetes Range",
            "text": f"Your fasting glucose is {glucose} mg/dL. Normal is below 100. Between 100-125 is considered prediabetes — your body is having trouble managing sugar."
        })
    else:
        explanation.append({
            "icon": "🟢", "title": "Glucose is Normal",
            "text": f"Your fasting glucose is {glucose} mg/dL, which is within the healthy range (70-99 mg/dL). Good job!"
        })
    
    # HbA1c explanation
    if hba1c > 0:
        if hba1c >= 6.5:
            explanation.append({
                "icon": "🔴", "title": "HbA1c Indicates Diabetes",
                "text": f"Your HbA1c is {hba1c}%. This shows your average blood sugar over 3 months has been high. 6.5% or above = diabetes per ADA guidelines."
            })
        elif hba1c >= 5.7:
            explanation.append({
                "icon": "🟡", "title": "HbA1c Shows Prediabetes",
                "text": f"Your HbA1c is {hba1c}%. Between 5.7-6.4% means prediabetes — your blood sugar has been slightly elevated over time."
            })
        else:
            explanation.append({
                "icon": "🟢", "title": "HbA1c is Normal",
                "text": f"Your HbA1c is {hba1c}%, which is healthy (below 5.7%). Your average blood sugar over 3 months is well controlled."
            })
    
    # BMI explanation
    if bmi >= 30:
        explanation.append({
            "icon": "🟠", "title": "BMI Shows Obesity",
            "text": f"Your BMI is {round(bmi,1)}. Above 30 is obesity, which is a major risk factor for Type 2 diabetes. Even losing 5-7% body weight can significantly reduce risk."
        })
    elif bmi >= 25:
        explanation.append({
            "icon": "🟡", "title": "BMI Shows Overweight",
            "text": f"Your BMI is {round(bmi,1)}. Between 25-29.9 is overweight. This increases your diabetes risk, especially combined with other factors."
        })
    else:
        explanation.append({
            "icon": "🟢", "title": "BMI is Healthy",
            "text": f"Your BMI is {round(bmi,1)}, which is in the normal range (18.5-24.9). This is good for diabetes prevention."
        })
    
    # BP explanation
    if blood_pressure >= 140:
        explanation.append({
            "icon": "🟠", "title": "Blood Pressure is High",
            "text": f"Your BP is {blood_pressure} mm Hg. High blood pressure often goes hand-in-hand with diabetes and increases heart disease risk."
        })
    elif blood_pressure > 120:
        explanation.append({
            "icon": "🟡", "title": "Blood Pressure is Slightly Elevated",
            "text": f"Your BP is {blood_pressure} mm Hg. Normal is below 120. Slightly elevated BP can be an early warning sign."
        })
    else:
        explanation.append({
            "icon": "🟢", "title": "Blood Pressure is Normal",
            "text": f"Your BP is {blood_pressure} mm Hg — within the healthy range."
        })
    
    # Family history
    if family_history:
        explanation.append({
            "icon": "🟡", "title": "Family History Increases Risk",
            "text": "Having a parent or sibling with diabetes significantly increases your risk. Regular screening is strongly recommended."
        })
    
    # Physical activity
    if not physical_activity:
        explanation.append({
            "icon": "🟡", "title": "Physical Inactivity Adds Risk",
            "text": "Not being physically active is a known risk factor. Just 30 minutes of moderate exercise 5 days a week can reduce your risk by up to 58%."
        })
    
    # Overall verdict
    if result == "Diabetic":
        explanation.append({
            "icon": "⚠️", "title": "What This Means",
            "text": "Based on your numbers, there are strong indicators of diabetes. However, a proper medical diagnosis requires TWO separate abnormal test results. Please visit a doctor for confirmation with lab tests."
        })
    elif result == "Prediabetic":
        explanation.append({
            "icon": "⚡", "title": "What This Means",
            "text": "Your numbers suggest prediabetes — your body is starting to have difficulty processing sugar. The good news is that with diet changes and exercise, prediabetes can often be reversed."
        })
    else:
        explanation.append({
            "icon": "✅", "title": "What This Means",
            "text": "Your numbers look healthy! Continue maintaining a balanced diet and regular physical activity. Get routine checkups as recommended for your age."
        })

    return {
        "result": result,
        "confidence": confidence,
        "hybrid_score": round(hybrid, 4),
        "clinical_score": round(clinical_score, 4),
        "ml_score": round(ml_prob, 4),
        "risk_level": risk_level,
        "parameter_breakdown": breakdown,
        "explanation": explanation
    }
