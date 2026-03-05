"""
KCI Missingness Indicator Script

Tests PC and FCI with KCI with missingness indicator columns.



Outputs:
    graphs/kci_<n_indicators>_indicators.png  (if successful)
    kci_results.txt                            (summary of pass/fail per run)
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci, fastkci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# column groups for background knowledge and analysis

DEMO_COLS   = ["anchor_age", "gender", "race"]
PHYS_COLS   = [
    "heart_rate_max",
    "blood_pressure_min",
    "spO2_min",
    "FiO2_max",
    "lactate_max",
    "bilirubin_max",
    "platelet_max",
    "inr_max",
    "temp_max_F",
]
TREAT_COLS  = ["antibiotics_given", "vaso_given"]
OUT24_COLS  = ["aki_24h_onset_stage_y", "mechvent_24h_onset"]
POST24_COLS = ["aki_post24h_stage", "mechvent_post24h"]
MORT_COLS   = ["hospital_expire_flag"]

CORE_COLS = DEMO_COLS + PHYS_COLS + TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS

TIER_MAP = {
    **{c: 0 for c in DEMO_COLS},
    **{c: 1 for c in PHYS_COLS + TREAT_COLS + OUT24_COLS},
    **{c: 2 for c in POST24_COLS},
    **{c: 3 for c in MORT_COLS},
}

CATEGORICAL_COLS = ["gender", "race"]
BINARY_COLS = [
    "hospital_expire_flag", "antibiotics_given", "vaso_given",
    "aki_24h_onset_stage_y", "mechvent_24h_onset",
    "aki_post24h_stage", "mechvent_post24h",
]

# Columns to generate missingness indicators for, in the order they'll be added
INDICATOR_COLS = [
    "heart_rate_max",
    "blood_pressure_min",
    "spO2_min",
    "FiO2_max",
    "lactate_max",
    "bilirubin_max",
    "platelet_max",
    "inr_max",
    "temp_max_F",
]

SAMPLE_SIZE = 800
ALPHA       = 0.1
SEED        = 42

# Data loadng and preprocessing

def load_data(path: str):
    df = pd.read_csv(path)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    available = [c for c in CORE_COLS if c in df.columns]
    missing   = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        print(f"Columns not found and skipped: {missing}")
    return df[available].copy(), available


# missingness indicator construction

def make_indicator(df_raw: pd.DataFrame, col: str) -> pd.Series:
    """
    Binary column: 1 if original value was missing, 0 if observed.
    Returns None if the column has no missing values (pointless as indicator)
    or if variance is too low (would cause KCI to crash).
    """
    if col not in df_raw.columns:
        return None, f"{col} not in dataframe"

    indicator = df_raw[col].isna().astype(float)
    missing_rate = indicator.mean()

    if missing_rate == 0.0:
        return None, f"{col} has no missing values — indicator is constant"
    if missing_rate < 0.01 or missing_rate > 0.99:
        return None, (
            f"{col} missing rate is {missing_rate:.2%} — indicator is near-constant "
            f"and will cause KCI variance error"
        )

    print(f"  [indicator] {col}_missing: {missing_rate:.2%} missing — valid for KCI")
    return indicator.rename(f"{col}_missing"), None


# Background knowledge construction

def build_background_knowledge(nodes: list, col_names: list) -> BackgroundKnowledge:
    col_to_node = {col: nodes[i] for i, col in enumerate(col_names)}
    bk = BackgroundKnowledge()
    for col, tier in TIER_MAP.items():
        if col in col_to_node:
            bk.add_node_to_tier(col_to_node[col], tier)
    for i, c1 in enumerate(DEMO_COLS):
        for c2 in DEMO_COLS[i+1:]:
            if c1 in col_to_node and c2 in col_to_node:
                bk.add_forbidden_by_node(col_to_node[c1], col_to_node[c2])
                bk.add_forbidden_by_node(col_to_node[c2], col_to_node[c1])
    for col in col_names:
        indicator_name = f"{col}_missing"
        if col in col_to_node and indicator_name in col_to_node:
            bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[col])
    return bk
 


# Main execution loop

def main():
    os.makedirs("graphs", exist_ok=True)

    print("Loading data ...")
    df_raw, col_names = load_data("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")
    print(f"Using {len(col_names)} columns, {len(df_raw)} rows.\n")

    # Simple impute for the base data
    raw_arr    = df_raw.to_numpy().astype(float)
    imputed_arr = SimpleImputer(strategy="mean").fit_transform(raw_arr)
    df_imputed  = pd.DataFrame(imputed_arr, columns=col_names)

    # Subsample for faster KCI runs 
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(df_imputed), size=SAMPLE_SIZE, replace=False)

    # Pre-build all valid indicators (check variance before any runs)
    print("Checking missingness indicators ...")
    valid_indicators   = []   


    for col in INDICATOR_COLS:
        indicator, reason = make_indicator(df_raw, col)
        if indicator is  None:
            print(f"  skipping {col}_missing: {reason}")
        else:
            valid_indicators.append((col, indicator))

    df_run = df_imputed.copy()

    for col, series in valid_indicators:
        df_run[f"{col}_missing"] = series.values
        
    indicator_names = [f"{col}_missing" for col, _ in valid_indicators]

    print("PC with all indicators")

    all_cols   = col_names + indicator_names
    data_sample  = (df_run[all_cols].to_numpy().astype(float))[sample_idx]

    results = []
    try:
            # First pass — get nodes for BK
            cg = pc(data_sample, alpha=ALPHA, indep_test=kci)
            bk = build_background_knowledge(cg.G.get_nodes(), all_cols)

            # Second pass — with BK
            cg = pc(data_sample, alpha=ALPHA, indep_test=kci,background_knowledge=bk)

            out_path = f"graphs/pc_all_indicators_kci.png"
            pyd = GraphUtils.to_pydot(cg.G, labels=all_cols)
            pyd.write_png(out_path)

            print(f"SUCCESS → {out_path}")
            results.append(("pc_all_indicators", indicator_names, "SUCCESS", ""))

    except Exception as e:
        print(f"FAILED: {e}")
        results.append(("pc_all_indicators", indicator_names, "FAILED", str(e)))
    
    try:
    # get node objects for BK
        g0, _ = fci(data_sample, independence_test_method=kci, alpha=ALPHA)
        bk = build_background_knowledge(g0.get_nodes(), all_cols)
        # apply BK
        graph, _ = fci(data_sample, independence_test_method=kci,alpha=ALPHA, background_knowledge=bk)
        out_path = f"graphs/fci_all_indicators_kci.png"
        pyd = GraphUtils.to_pydot(graph, labels=all_cols)
        pyd.write_png(out_path)
        print(f"SUCCESS → {out_path}")
        results.append(("fci_all_indicators", indicator_names, "SUCCESS", ""))

    except Exception as e:
        print(f"FAILED: {e}")
        results.append(("fci_all_indicators", indicator_names, "FAILED", str(e)))
    
    return results

if __name__ == "__main__":
    main()