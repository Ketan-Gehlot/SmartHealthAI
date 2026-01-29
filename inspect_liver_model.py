import pickle
import os
import xgboost
import sklearn

base_path = r"c:\Users\DELL\Desktop\FINAL YEAR PROJECT 2026\models\automl_models"
file_path = os.path.join(base_path, "liver_automl.pkl")

try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"Loaded type: {type(data)}")
    
    if isinstance(data, dict):
        model = data.get("model")
        print(f"Model key found. Model type: {type(model)}")
        print(f"Model details: {model}")
        
        # Check if it's an ensemble
        if hasattr(model, 'estimators_'):
            print("Ensemble detected!")
            for i, est in enumerate(model.estimators_):
                print(f"Estimator {i}: {type(est)}")
    else:
        print(f"Model data object: {data}")

except Exception as e:
    print(f"Error: {e}")
