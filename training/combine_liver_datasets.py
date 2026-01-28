# ============================================================
# Combine Indian Liver Patient Datasets (AUTO-DETECT FORMAT)
# ============================================================

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILE_1 = os.path.join(BASE_DIR, "data", "raw", "indian_liver_patient.csv")
FILE_2 = os.path.join(BASE_DIR, "data", "raw", "liver_train")  # NO EXTENSION

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "liver_combined.csv")

print("[INFO] Loading datasets with auto-detection...")

# ---------------- LOAD FILE 1 ----------------
df1 = pd.read_csv(FILE_1)
print("[INFO] Dataset 1 loaded:", df1.shape)

# ---------------- LOAD FILE 2 (AUTO-DETECT) ----------------
df2 = None

for ext in [".csv", ".xls", ".xlsx"]:
    path = FILE_2 + ext
    if os.path.exists(path):
        print(f"[INFO] Found second dataset: {path}")
        try:
            if ext == ".csv":
                df2 = pd.read_csv(path, encoding="latin1", engine="python")
            else:
                df2 = pd.read_excel(path)
            break
        except Exception as e:
            print("[ERROR] Failed loading:", path)
            print(e)

if df2 is None:
    raise FileNotFoundError("Second liver dataset not found or unreadable")

print("[INFO] Dataset 2 loaded:", df2.shape)

# ---------------- STANDARDIZE COLUMN NAMES ----------------
# ------------------------------------------------------------
# AUTO ALIGN COLUMNS BASED ON AVAILABILITY
# ------------------------------------------------------------

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df1 = normalize_columns(df1)
df2 = normalize_columns(df2)

print("[DEBUG] df1 columns:", df1.columns.tolist())
print("[DEBUG] df2 columns:", df2.columns.tolist())

# ---- Define preferred liver features ----
preferred_features = [
    "age",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alt",
    "ast",
    "total_protein",
    "albumin",
    "albumin_globulin_ratio"
]

# ---- Find common features ----
common_features = [f for f in preferred_features if f in df1.columns and f in df2.columns]

print("[INFO] Common features used:", common_features)

# ---- Label handling ----
label_candidates = ["label", "dataset", "class", "target", "diagnosis"]

def find_label(df):
    for col in label_candidates:
        if col in df.columns:
            return col
    return None

label1 = find_label(df1)
label2 = find_label(df2)

if label1 is None or label2 is None:
    raise ValueError("Label column not found in one of the datasets")

df1["label"] = df1[label1].apply(lambda x: 1 if x in [1, "yes", "positive", "liver"] else 0)
df2["label"] = df2[label2].apply(lambda x: 1 if x in [1, "yes", "positive", "liver"] else 0)

# ---- Final selection ----
df1 = df1[common_features + ["label"]]
df2 = df2[common_features + ["label"]]



# ---------------- COMBINE ----------------
df = pd.concat([df1, df2], ignore_index=True)

# ---------------- CLEAN ----------------
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["label"] = df["label"].apply(lambda x: 1 if x == 1 else 0)

print("[INFO] Final shape:", df.shape)
print("[INFO] Final columns:", df.columns.tolist())

# ---------------- SAVE ----------------
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("[SUCCESS] liver_combined.csv created successfully")
print(OUTPUT_FILE)
