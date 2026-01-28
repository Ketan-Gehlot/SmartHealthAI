import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "pima_diabetes.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "automl_models", "diabetes_pima.pkl")

# Headers for Pima dataset
HEADERS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]

def train():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, names=HEADERS)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # User wants to use only: Glucose, BloodPressure, BMI, Age
    selected_features = ["glucose", "blood_pressure", "bmi", "age"]
    target = "outcome"

    print(f"Selected features: {selected_features}")

    X = df[selected_features]
    y = df[target]

    # Handle zeros in Glucose, BP, BMI (0 is invalid for these)
    # Replace 0 with NaN then fill with mean
    for col in ["glucose", "blood_pressure", "bmi"]:
        X[col] = X[col].replace(0, np.nan)
        X[col] = X[col].fillna(X[col].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    # Save dictionary
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": selected_features,
        "accuracy": acc
    }

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
