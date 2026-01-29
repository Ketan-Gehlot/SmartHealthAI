import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOML_DIR = os.path.join(BASE_DIR, "models", "automl_models")

try:
    with open(os.path.join(AUTOML_DIR, "liver_automl.pkl"), "rb") as f:
        liver_data = pickle.load(f)
        print(f"Type: {type(liver_data)}")
        if isinstance(liver_data, dict):
            print(f"Keys: {liver_data.keys()}")
            if "features" in liver_data:
                print(f"Features: {liver_data['features']}")
            else:
                print("No 'features' key found.")
        else:
            print("Data is not a dict.")
except Exception as e:
    print(f"Error: {e}")
