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
warnings.filterwarnings("ignore", category=Warning, module="sklearn")

app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')
app.secret_key = "smarthealthai_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
AUTOML_DIR = os.path.join(MODEL_DIR, "automl_models")

DATABASE_PATH = os.path.join(BASE_DIR, "database", "users.db")

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
    conn = sqlite3.connect(DATABASE_PATH)
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

    # Features: glucose, blood_pressure, bmi, age
    features = np.array([[  
        float(data.get("glucose", 0)),
        float(data.get("blood_pressure", 0)),
        float(data.get("bmi", 0)),
        float(data.get("age", 0))
    ]])

    if diabetes_scaler:
        features = diabetes_scaler.transform(features)

    prediction = diabetes_model.predict(features)[0]
    probability = diabetes_model.predict_proba(features)[0][1]

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    save_prediction("Diabetes", result, float(probability) * 100)

    return jsonify({
        "disease": "Diabetes",
        "result": result,
        "confidence": round(float(probability) * 100, 2)
    })

@app.route("/api/predict/liver", methods=["POST"])
def predict_liver():
    data = request.json

    features = np.array([[  
        data["age"],
        data["total_bilirubin"],
        data["direct_bilirubin"],
        data["alkaline_phosphotase"],
        data["alamine_aminotransferase"],
        data["aspartate_aminotransferase"],
        data["total_proteins"],
        data["albumin"],
        data["albumin_globulin_ratio"]
    ]])

    if liver_scaler:
        features = liver_scaler.transform(features)

    prediction = liver_model.predict(features)[0]
    probability = liver_model.predict_proba(features)[0][1]

    result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
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

        if liver_scaler:
            input_array = liver_scaler.transform(input_array)

        pred = liver_model.predict(input_array)[0]
        prob = liver_model.predict_proba(input_array)[0][1]

        prediction = "LIVER DISEASE DETECTED ⚠️" if pred == 1 else "NO LIVER DISEASE ✅"
        probability = round(prob * 100, 2)

    return render_template(
        "liver.html",
        features=liver_features,
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    # app.run(debug=True)
