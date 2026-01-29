import pickle
import os
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

base_path = r"c:\Users\DELL\Desktop\FINAL YEAR PROJECT 2026"
model_path = os.path.join(base_path, "models", "automl_models", "liver_automl.pkl")

print(f"Loading model from: {model_path}")

with open(model_path, "rb") as f:
    data = pickle.load(f)

model = None
scaler = None

if isinstance(data, dict):
    model = data.get("model")
    scaler = data.get("scaler")
    feature_names = data.get("features")
    print(f"Loaded Dictionary. Keys: {data.keys()}")
    print(f"EXPECTED FEATURES FROM PICKLE: {feature_names}")
else:
    model = data
    print("Loaded direct model object")

print(f"Model Type: {type(model)}")

# Try to find expected features
expected_features = []
if hasattr(model, "feature_names_in_"):
    expected_features = model.feature_names_in_
elif hasattr(model, "estimators_"):
    # Check first estimator
    est = model.estimators_[0]
    if hasattr(est, "feature_names_in_"):
        expected_features = est.feature_names_in_

print(f"Model expects features: {expected_features}")

# Create a "Normal" input sample
# Ref Normal Ranges:
# Age: 30
# Total_Bilirubin: 0.8 (0.1-1.2)
# Direct_Bilirubin: 0.2 (0.1-0.3)
# Alkaline_Phosphotase: 85 (44-147)
# Alamine_Aminotransferase: 20 (7-56)
# Aspartate_Aminotransferase: 20 (10-40)
# Total_Protiens: 7.0 (6.0-8.3)
# Albumin: 4.0 (3.5-5.5)
# Albumin_and_Globulin_Ratio: ~1.0 

# Current app.py order:
# alt, ast, alp, albumin, total_protein, total_bilirubin, direct_bilirubin, age
# User Provided Values (Reported 92.7% Risk)
# Age: 22
# Total Bilirubin: 0.7
# Direct Bilirubin: 0.2
# ALP: 66
# ALT: 44
# AST: 33
# Total Proteins: 7
# Albumin: 4

# App Order: alt, ast, alp, albumin, total_protein, total_bilirubin, direct_bilirubin, age
normal_values = [44.0, 33.0, 66.0, 4.0, 7.0, 0.7, 0.2, 22.0]

# Abnormal Values (High Bilirubin, High Enzymes, low Protein/Albumin)
abnormal_values = [
    200.0, # ALT (High)
    150.0, # AST (High)
    500.0, # ALP (High)
    2.0,   # Albumin (Low)
    5.0,   # Total Protein (Low)
    5.5,   # Total Bilirubin (High)
    2.5,   # Direct Bilirubin (High)
    55.0   # Age
]

for values, name in [(normal_values, "NORMAL"), (abnormal_values, "ABNORMAL")]:
    print(f"\nTesting with '{name}' Values: {values}")
    features = np.array([values])

    if scaler:
        features = scaler.transform(features)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    print(f"Prediction: {pred}")
    print(f"Probabilities: {prob}")
    
    # Current App Logic Assumption: 1 = Disease
    app_result = "Disease Detected" if pred == 1 else "No Disease"
    print(f"Current App Logic says: {app_result}")


# ==========================================
# TEST ON REAL DATA
# ==========================================
print("\n--- Testing on REAL Data Samples ---")
try:
    df = pd.read_csv(os.path.join(base_path, "data", "raw", "liver_train.csv"))

    # Normalize columns same as training
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_"))
    col_map = {
        "alkphos_alkaline_phosphotase": "alp",
        "alkaline_phosphotase": "alp",
        "sgpt_alamine_aminotransferase": "alt",
        "sgot_aspartate_aminotransferase": "ast",
        "total_protiens": "total_protein",
        "alb_albumin": "albumin",
        "total_bilirubin": "total_bilirubin",
        "direct_bilirubin": "direct_bilirubin",
        "age": "age",
        "gender": "gender",
        "result": "label",
        "dataset": "label"
    }
    df.rename(columns=col_map, inplace=True)
    
    # Handle Label
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'positive'] else 0)
        
        # Find a Healthy Sample (Label 0)
        healthy_df = df[df['label'] == 0]
        if not healthy_df.empty:
            # Pick a few samples
            for i in range(min(3, len(healthy_df))):
                sample = healthy_df.iloc[[i]]
                print(f"\nFound Healthy Sample #{i} (Label 0):")
                # Ensure we only pick valid columns that exist in the inputs
                valid_feats = [f for f in feature_names if f in sample.columns]
                print(sample[valid_feats].to_string(index=False))
                
                # Extract features in correct order
                real_features = sample[feature_names].values
                
                if scaler:
                    real_features = scaler.transform(real_features)
                    
                pred = model.predict(real_features)[0]
                prob = model.predict_proba(real_features)[0]
                print(f"Prediction: {pred} (Should be 0)")
                print(f"Probability: {prob}")
        else:
            print("No Class 0 samples found in CSV?!")
            print("Head:", df.head())
    else:
        print("Label column not found after rename.")
        print("Columns:", df.columns.tolist())

except Exception as e:
    print(f"Error testing real data: {e}")
