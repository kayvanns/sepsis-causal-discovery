import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import os

INPUT_PATH  = "data/processed/analysis.csv"
OUTPUT_DIR  = "data/chtc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORICAL_COLS = ["gender", "race"]
KEEP_COLS = [
    "anchor_age", "gender", "race", "ckd_baseline", "BMI",
    "Heart Rate (max)", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
    "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)",
    "Temperature (Max)", "Hemoglobin (min)", "Fluid Balance (mL)", "WBC (max)", "Creatinine (max)", "sofa_score",
    "Antibiotics", "Vasopressors", "Diuretics",
    "aki_24h_onset", "mechvent_24h_onset",
    "aki_post24h", "mechvent_post24h",
    "hospital_expire_flag"
]

# variables to create indicators for (all with any missingness)
INDICATOR_COLS = [
    "BMI", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
    "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)",
    "Temperature (Max)", "Hemoglobin (min)", "Fluid Balance (mL)", "WBC (max)", "Creatinine (max)"
]

def encode_categoricals(df):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    return df

def add_indicators(df):
    df = df.copy()
    for col in INDICATOR_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(float)
            print(f"  indicator: {col}_missing  ({df[col].isnull().sum()} missing)")
    return df

def impute_simple(df):
    imp = SimpleImputer(strategy="mean")
    arr = imp.fit_transform(df)
    return pd.DataFrame(arr, columns=df.columns)

def impute_knn(df, n_neighbors=5):
    scaler       = StandardScaler()
    complete     = df.dropna()
    scaler.fit(complete)
    scaled       = scaler.transform(df)
    imputed      = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    arr          = scaler.inverse_transform(imputed)
    return pd.DataFrame(arr, columns=df.columns)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INPUT_PATH)[KEEP_COLS]
df = encode_categoricals(df)
print(f"  {df.shape[0]} rows, {df.shape[1]} cols")

# ── Dataset 1: raw (no imputation, no indicators) ─────────────────────────────
print("\nBuilding data_raw.csv...")
df.to_csv(f"{OUTPUT_DIR}/data_raw.csv", index=False)
print(f"  Saved. Shape: {df.shape}")

# ── Dataset 2: simple imputation, no indicators ───────────────────────────────
print("\nBuilding data_simple.csv...")
df_simple = impute_simple(df)
df_simple.to_csv(f"{OUTPUT_DIR}/data_simple.csv", index=False)
print(f"  Saved. Shape: {df_simple.shape}")

# ── Dataset 3: simple imputation + indicators ─────────────────────────────────
print("\nBuilding data_simple_indicator.csv...")
df_with_ind = add_indicators(df)
df_simple_ind = impute_simple(df_with_ind)
df_simple_ind.to_csv(f"{OUTPUT_DIR}/data_simple_indicator.csv", index=False)
print(f"  Saved. Shape: {df_simple_ind.shape}")

# ── Dataset 4: KNN imputation + indicators ────────────────────────────────────
print("\nBuilding data_knn_indicator.csv...")
df_knn_ind = impute_knn(df_with_ind)
df_knn_ind.to_csv(f"{OUTPUT_DIR}/data_knn_indicator.csv", index=False)
print(f"  Saved. Shape: {df_knn_ind.shape}")

print("\nAll datasets saved to", OUTPUT_DIR)