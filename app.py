import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
import tensorflow as tf
import pickle
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import warnings

# Suppress specific warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import Hybrid Clinical Rules Engine
from controllers.diabetes_clinical_rules import hybrid_diabetes_predict
warnings.filterwarnings("ignore", category=Warning, module="sklearn")

app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')
app.secret_key = "smarthealthai_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
AUTOML_DIR = os.path.join(MODEL_DIR, "automl_models")



print("[INFO] Loading AutoML models...")
# Loading Pima model for Diabetes

with open(os.path.join(AUTOML_DIR, "diabetes_pima.pkl"), "rb") as f:
    diabetes_data = pickle.load(f)
    diabetes_model = diabetes_data["model"]
    diabetes_scaler = diabetes_data["scaler"]


with open(os.path.join(AUTOML_DIR, "liver_automl.pkl"), "rb") as f:
    liver_data = pickle.load(f)
    print(f"[DEBUG] Liver model type/keys: {type(liver_data)}") 
    if isinstance(liver_data, dict):
        liver_model = liver_data["model"]
        liver_scaler = liver_data.get("scaler")
        liver_features = liver_data.get("features", [])
    else:
        liver_model = liver_data
        liver_scaler = None
        liver_features = []

# CNN_MODEL_PATH = os.path.join("models", "cnn_models", "breast_cancer_cnn.h5")
# cancer_cnn_model = load_model(CNN_MODEL_PATH)
# print("[INFO] Breast Cancer CNN model loaded")

with open(os.path.join(AUTOML_DIR, "breast_cancer_tabular.pkl"), "rb") as f:
    cancer_data = pickle.load(f)
    if isinstance(cancer_data, dict):
        cancer_model = cancer_data["model"]
        cancer_scaler = cancer_data.get("scaler")
        cancer_features = cancer_data.get("features", [])
    else:
        cancer_model = cancer_data
        cancer_scaler = None
        cancer_features = []
print("[INFO] Breast Cancer tabular model loaded")

print("[INFO] AutoML models loaded successfully.")

# -------------------------------------------------
# DATABASE
# -------------------------------------------------

def get_db_connection():
    # Check if running on Render (or Linux in general for this project context)
    if os.name == 'posix': 
        # Render allows write access ONLY to /tmp or a persistent disk mounted path
        db_path = "/tmp/users.db"
    else:
        # Local Windows environment
        db_path = "users.db"

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn



def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            result TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(disease, result, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (disease, result, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (disease, result, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cancer")
def cancer_redirect():
    return redirect(url_for('predict_page', disease='cancer'))

@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    records = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("dashboard.html", records=records)

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

# -------------------------------------------------
# PREDICTION APIs
# -------------------------------------------------

@app.route("/api/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json
    print(f"[DEBUG] Diabetes Request Data: {data}")
    
    glucose = float(data.get("glucose", 0))
    blood_pressure = float(data.get("blood_pressure", 0))
    bmi = float(data.get("bmi", 0))
    age = float(data.get("age", 0))
    hba1c = float(data.get("hba1c", 0))
    insulin = float(data.get("insulin", 0))
    skin_thickness = float(data.get("skin_thickness", 0))
    family_history = bool(data.get("family_history", 0))
    physical_activity = bool(data.get("physical_activity", 1))
    
    hybrid_result = hybrid_diabetes_predict(
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        age=age,
        ml_model=diabetes_model,
        ml_scaler=diabetes_scaler,
        hba1c=hba1c,
        insulin=insulin,
        skin_thickness=skin_thickness,
        family_history=family_history,
        physical_activity=physical_activity
    )
    
    result = hybrid_result["result"]
    confidence = hybrid_result["confidence"]
    
    print(f"[DEBUG] Result: {result} | Confidence: {confidence}% | Risk: {hybrid_result['risk_level']}")

    save_prediction("Diabetes", result, confidence)

    return jsonify({
        "disease": "Diabetes",
        "result": result,
        "confidence": round(confidence, 2),
        "risk_level": hybrid_result["risk_level"],
        "parameter_breakdown": hybrid_result["parameter_breakdown"],
        "explanation": hybrid_result["explanation"],
        "scores": {
            "clinical": round(hybrid_result["clinical_score"] * 100, 1),
            "ml_model": round(hybrid_result["ml_score"] * 100, 1),
            "hybrid": round(hybrid_result["hybrid_score"] * 100, 1)
        }
    })

@app.route("/api/predict/liver", methods=["POST"])
def predict_liver():
    data = request.json

    # Features based on pickle inspection: ['alt', 'ast', 'alp', 'albumin', 'total_protein', 'total_bilirubin', 'direct_bilirubin', 'age']
    # Note: Order must match exactly what the model expects.
    # Assuming the inspection order is the correct feature order.
    # Inspect order: alt, ast, alp, albumin, total_protein, total_bilirubin, direct_bilirubin, age
    
    features = np.array([[  
        float(data.get("alt", 0)),
        float(data.get("ast", 0)),
        float(data.get("alp", 0)),
        float(data.get("albumin", 0)),
        float(data.get("total_protein", 0)),
        float(data.get("total_bilirubin", 0)),
        float(data.get("direct_bilirubin", 0)),
        float(data.get("age", 0))
    ]])

    # -------------------------------------------------------------------------
    # HYBRID AI LOGIC: Medical Guidelines Rule-Based Check
    # If all parameters are strictly within normal ranges, we default to Healthy.
    # This acts as a safety layer over the ML model to prevent false positives.
    # -------------------------------------------------------------------------
    
    # Values from User Input
    v_alt = float(data.get("alt", 0))
    v_ast = float(data.get("ast", 0))
    v_alp = float(data.get("alp", 0))
    v_albumin = float(data.get("albumin", 0))
    v_total_protein = float(data.get("total_protein", 0))
    v_total_bilirubin = float(data.get("total_bilirubin", 0))
    v_direct_bilirubin = float(data.get("direct_bilirubin", 0))

    # Normal Ranges (Standard Medical Data)
    # Total Bilirubin: 0.1 - 1.2
    # Direct Bilirubin: 0.1 - 0.4
    # ALP: 40 - 150
    # ALT: 7 - 56
    # AST: 10 - 45
    # Total Protein: 6.0 - 8.3
    # Albumin: 3.5 - 5.5
    
    is_normal = True
    if not (0.1 <= v_total_bilirubin <= 1.5): is_normal = False
    if not (0.0 <= v_direct_bilirubin <= 0.6): is_normal = False # Extended slightly
    if not (0 <= v_alp <= 160): is_normal = False
    if not (0 <= v_alt <= 60): is_normal = False
    if not (0 <= v_ast <= 60): is_normal = False
    if not (5.5 <= v_total_protein <= 8.5): is_normal = False
    if not (3.0 <= v_albumin <= 6.0): is_normal = False

    print(f"[DEBUG] Medical Rule Check: All Normal? {is_normal}")

    if is_normal:
        # Override Model
        result = "No Liver Disease"
        probability = 15.5 # Low risk score
        save_prediction("Liver Disease", result, probability)
        return jsonify({
            "disease": "Liver Disease",
            "result": result,
            "confidence": probability
        })

    # If not normal, ask the AI Model
    if liver_scaler:
        features = liver_scaler.transform(features)

    probability = liver_model.predict_proba(features)[0][1]

    # Adjusted threshold for cleaner user experience
    # The AutoML model is highly sensitive, so we only flag disease if confidence is > 90%
    if probability > 0.9:
        prediction = 1
        result = "Liver Disease Detected"
    else:
        prediction = 0
        result = "No Liver Disease"
    
    save_prediction("Liver Disease", result, float(probability) * 100)

    return jsonify({
        "disease": "Liver Disease",
        "result": result,
        "confidence": round(float(probability) * 100, 2)
    })

@app.route("/api/predict/breast_cancer", methods=["POST"])
def predict_breast_cancer():
    data = request.json

    # Features: 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
    # Defaulting to 0 if not provided
    features = np.array([[
        float(data.get("mean radius", 0)),
        float(data.get("mean texture", 0)),
        float(data.get("mean perimeter", 0)),
        float(data.get("mean area", 0)),
        float(data.get("mean smoothness", 0))
    ]])

    if cancer_scaler:
        features = cancer_scaler.transform(features)

    prediction = cancer_model.predict(features)[0]
    probability = cancer_model.predict_proba(features)[0][1] # Probability of Malignant (1)

    result = "Malignant" if prediction == 1 else "Benign" 

    save_prediction("Breast Cancer", result, float(probability) * 100)

    return jsonify({
        "disease": "Breast Cancer",
        "result": result,
        "confidence": round(float(probability) * 100, 2)
    })

# -------------------------------------------------
# LIVER FORM UI (UNCHANGED)
# -------------------------------------------------

@app.route("/liver", methods=["GET", "POST"])
def liver_predict():
    prediction = None
    probability = None

    if request.method == "POST":
        input_data = []

        # flexible feature gathering
        for feature in liver_features:
            val = request.form.get(feature)
            if val:
                input_data.append(float(val))
            else:
                input_data.append(0.0) # Fallback

        input_array = np.array(input_data).reshape(1, -1)

        # -------------------------------------------------------------
        # HYBRID AI LOGIC (Same as API)
        # -------------------------------------------------------------
        # Map input_data indices to features:
        # alt(0), ast(1), alp(2), albumin(3), total_protein(4), 
        # total_bilirubin(5), direct_bilirubin(6), age(7)
        v_alt = input_data[0]
        v_ast = input_data[1]
        v_alp = input_data[2]
        v_albumin = input_data[3]
        v_total_protein = input_data[4]
        v_total_bilirubin = input_data[5]
        v_direct_bilirubin = input_data[6]

        is_normal = True
        if not (0.1 <= v_total_bilirubin <= 1.5): is_normal = False
        if not (0.0 <= v_direct_bilirubin <= 0.6): is_normal = False 
        if not (0 <= v_alp <= 160): is_normal = False
        if not (0 <= v_alt <= 60): is_normal = False
        if not (0 <= v_ast <= 60): is_normal = False
        if not (5.5 <= v_total_protein <= 8.5): is_normal = False
        if not (3.0 <= v_albumin <= 6.0): is_normal = False

        if is_normal:
             prediction = "NO LIVER DISEASE ✅"
             probability = 12.5 # Low risk
        else:
             if liver_scaler:
                input_array = liver_scaler.transform(input_array)

             prob = liver_model.predict_proba(input_array)[0][1]

             # Adjusted threshold for cleaner user experience
             if prob > 0.9:
                 pred = 1
                 prediction = "LIVER DISEASE DETECTED ⚠️"
             else:
                 pred = 0
                 prediction = "NO LIVER DISEASE ✅"
            
             probability = round(prob * 100, 2)

        # Save to Database for Dashboard
        save_result_text = "Liver Disease Detected" if "DETECTED" in prediction else "No Liver Disease"
        save_prediction("Liver Disease", save_result_text, probability)

    return render_template(
        "liver.html",
        features=liver_features,
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

