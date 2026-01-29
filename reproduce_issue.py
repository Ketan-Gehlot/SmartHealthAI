
import pickle
import os
import numpy as np
import warnings

# Suppress sklearn warnings like in app.py
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=Warning, module="sklearn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "automl_models", "diabetes_pima.pkl")

def test_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            scaler = data["scaler"]
            print("Model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test Case 1: All Zeros (what app.py might see if keys missing)
    features_zeros = np.array([[0.0, 0.0, 0.0, 0.0]])
    
    # Test Case 2: Typical Non-Diabetic values
    # Glucose=85, BP=70, BMI=20, Age=25
    features_healthy = np.array([[85.0, 70.0, 20.0, 25.0]])

    # Test Case 3: Typical Diabetic values
    # Glucose=180, BP=90, BMI=35, Age=50
    features_diabetic = np.array([[180.0, 90.0, 35.0, 50.0]])

    test_cases = [
        ("Zeros", features_zeros),
        ("Healthy", features_healthy),
        ("Diabetic", features_diabetic)
    ]

    for name, feats in test_cases:
        print(f"\n--- Testing {name} ---")
        print(f"Input: {feats}")
        
        try:
            # Transform
            if scaler:
                feats_scaled = scaler.transform(feats)
                print(f"Scaled input: {feats_scaled}")
            else:
                feats_scaled = feats
                print("No scaler found.")

            # Predict
            prediction = model.predict(feats_scaled)[0]
            probability = model.predict_proba(feats_scaled)[0]
            
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {probability}")
            # app.py uses probability[1] for "confidence" of Diabetes
            conf = probability[1] * 100
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
            print(f"Result: {result} ({conf:.2f}%)")
            
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_model()
