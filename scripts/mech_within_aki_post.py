import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

def load_and_preprocess_data(file_path, outcome:str, treatment:str, covariates:list, continuous:list):
    
    df = pd.read_csv(file_path)
    df["blood_pressure_min"] = df["blood_pressure_min"].where(df["blood_pressure_min"] > 0, np.nan)
    df["lactate_max"] = df["lactate_max"].where(df["lactate_max"] <= 30, np.nan)
    print(df[["blood_pressure_min", "lactate_max", "sofa_score_x", "anchor_age"]].describe())
    df = df[[outcome, treatment] + covariates]  # Select relevant columns
    print(f"Initial data shape: {df.shape}")
    df = df.dropna(subset=[outcome, treatment])  # Drop rows with missing values in key columns
    print(f"Data shape after dropping missing values in outcome and treatment: {df.shape}")
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    # Standardize continuous variables
    X = df_imputed[[treatment] + covariates]
    y = df_imputed[outcome]
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[continuous] = scaler.fit_transform(X[continuous])  
    return X_scaled, y

def run_regression(X, y):
    X_with_const = sm.add_constant(X)
    model = sm.Logit(y, X_with_const).fit()
    return model

def main():
    file_path = "/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv"
    outcome    = "aki_post24h"
    treatment  = "mechvent_24h_onset"
    covariates = ["anchor_age",
        "blood_pressure_min",
        "lactate_max",
        "sofa_score_x","aki_24h_onset_y"
    ]
    continuous = ["blood_pressure_min",
        "lactate_max","sofa_score_x"  ]
    X, y = load_and_preprocess_data(file_path, outcome, treatment, covariates, continuous)
    model = run_regression(X, y)
    print(model.summary())


if __name__ == "__main__":
    main()