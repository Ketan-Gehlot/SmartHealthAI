
import pickle
import os
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check the OLD model
MODEL_PATH = os.path.join(BASE_DIR, "models", "automl_models", "diabetes_automl.pkl")

def test_old_model():
    print(f"Loading OLD model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            
        model = data if not isinstance(data, dict) else data["model"]
        scaler = data.get("scaler") if isinstance(data, dict) else None
        
        print(f"Model type: {type(model)}")
        
        # This model might expect 21 features (BRFSS) or 8 (Pima)?
        # The logs said "Initial shape: (253680, 22)". So 21 features + target.
        # Construct 21 zeros
        features_zeros = np.zeros((1, 21))
        
        # Try to predict
        try:
            prob = model.predict_proba(features_zeros)[0][1]
            print(f"Prediction for Zeros (21 features): {prob*100:.2f}%")
        except:
            print("Failed with 21 features.")
            
        # Try with 4 features (if it was Pima but misnamed?)
        features_4 = np.zeros((1, 4))
        try:
            prob = model.predict_proba(features_4)[0][1]
            print(f"Prediction for Zeros (4 features): {prob*100:.2f}%")
        except:
             print("Failed with 4 features.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_old_model()
