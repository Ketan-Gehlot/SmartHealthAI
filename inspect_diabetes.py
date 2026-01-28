import pickle
import os
import numpy as np

path = r"c:\Users\DELL\Desktop\FINAL YEAR PROJECT 2026\models\automl_models\diabetes_automl.pkl"

try:
    with open(path, "rb") as f:
        data = pickle.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
            if "model" in data:
                 print(f"Model object: {type(data['model'])}")
        else:
            print(f"Model object: {type(data)}")
        
        # specific check for previous failure
        # try predicting with dummy data
        try:
             model = data["model"] if isinstance(data, dict) else data
             # Try Pima shape (1 sample, 8 features)
             dummy = np.zeros((1, 8))
             print("Attempting prediction with 8 features...")
             model.predict(dummy)
             print("Prediction success with 8 features.")
        except Exception as e:
             print(f"Prediction failed with 8 features: {e}")
             
        try:
             model = data["model"] if isinstance(data, dict) else data
             # Try BRFSS shape (1 sample, 21 features based on training script)
             dummy = np.zeros((1, 21))
             print("Attempting prediction with 21 features...")
             model.predict(dummy)
             print("Prediction success with 21 features.")
        except Exception as e:
             print(f"Prediction failed with 21 features: {e}")

except Exception as e:
    print(f"Error loading pickle: {e}")
