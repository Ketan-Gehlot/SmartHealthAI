import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "automl_models", "breast_cancer_tabular.pkl")

def train():
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = data.target

    # User wants simplified features.
    # Selecting 5 intuitive features: 
    # 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
    selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    
    print(f"Selected features: {selected_features}")

    X = df[selected_features]
    y = target

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
        "accuracy": acc,
        "description": "Simplified tabular Breast Cancer model"
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
