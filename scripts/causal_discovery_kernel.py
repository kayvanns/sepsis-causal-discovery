import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import traceback
import pickle
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

def impute_knn(arr: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    KNN imputation with standard scaling.
    Scale → impute → inverse scale so that the returned array
    is back in the original feature units.
    """
    scaler = StandardScaler()
    complete_rows = arr[~np.isnan(arr).any(axis=1)]
    scaler.fit(complete_rows)

    arr_scaled         = scaler.transform(arr)          
    arr_imputed_scaled = KNNImputer(n_neighbors=n_neighbors).fit_transform(arr_scaled)
    arr_imputed        = scaler.inverse_transform(arr_imputed_scaled)
    return arr_imputed
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
    """
      Tier 0  demographics   — nothing can cause these
      Tier 1  physiology + treatments + 24-h outcomes
      Tier 3  post-24-h outcomes
      Tier 4  mortality      — cannot cause anything
    """

    col_to_node = {col: nodes[i] for i, col in enumerate(col_names)}
    bk = BackgroundKnowledge()
    for col, tier in TIER_MAP.items():
        if col in col_to_node:
            bk.add_node_to_tier(col_to_node[col], tier)
    for col in col_names:
        indicator_name = f"{col}_missing"
        if indicator_name in col_to_node:
            bk.add_node_to_tier(col_to_node[indicator_name], 1)

    for i, c1 in enumerate(DEMO_COLS):
        for c2 in DEMO_COLS[i+1:]:
            if c1 in col_to_node and c2 in col_to_node:
                bk.add_forbidden_by_node(col_to_node[c1], col_to_node[c2])
                bk.add_forbidden_by_node(col_to_node[c2], col_to_node[c1])

    for col in col_names:
            indicator_name = f"{col}_missing"
            if col in col_to_node and indicator_name in col_to_node:
                bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[col])
                for demo in DEMO_COLS:
                    if demo in col_to_node:
                        bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[demo])
    return bk
 

def run_and_save(algo, data_sample, all_cols, run_name, alpha=ALPHA):
    try:
        if algo == "PC":
            cg = pc(data_sample, alpha=alpha, indep_test=kci)
            bk = build_background_knowledge(cg.G.get_nodes(), all_cols)
            cg = pc(data_sample, alpha=alpha, indep_test=kci, background_knowledge=bk)
            graph = cg.G
        elif algo == "FCI":
            g0, _ = fci(data_sample, independence_test_method=kci, alpha=alpha)
            bk = build_background_knowledge(g0.get_nodes(), all_cols)
            graph, _ = fci(data_sample, independence_test_method=kci, alpha=alpha, background_knowledge=bk)

        out_path   = f"graphs/{run_name}.png"
        graph_path = f"graphs/{run_name}.pkl"

        pyd = GraphUtils.to_pydot(graph, labels=all_cols)
        pyd.write_png(out_path)

        with open(graph_path, "wb") as f:
            pickle.dump((graph, all_cols), f)

        print(f"  SUCCESS → {out_path}")
        print(f"  Graph object saved → {graph_path}")
        return "SUCCESS"

    except Exception as e:
        print(f"  FAILED: {traceback.format_exc()}")
        return "FAILED"


def main():
    os.makedirs("graphs", exist_ok=True)

    print("Loading data ...")
    df_raw, col_names = load_data("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")
    print(f"Using {len(col_names)} columns, {len(df_raw)} rows.\n")
    raw_arr = df_raw.to_numpy().astype(float)

    print("Imputing ...")
    simple_arr = SimpleImputer(strategy="mean").fit_transform(raw_arr)
    knn_arr    = impute_knn(raw_arr)
    print("Done.\n")
    
    print("Checking missingness indicators ...")
    valid_indicators   = []   


    for col in INDICATOR_COLS:
        indicator, reason = make_indicator(df_raw, col)
        if indicator is  None:
            print(f"  skipping {col}_missing: {reason}")
        else:
            valid_indicators.append((col, indicator))


    indicator_names = [f"{col}_missing" for col, _ in valid_indicators]
    df_simple = pd.DataFrame(simple_arr, columns=col_names)
    df_knn    = pd.DataFrame(knn_arr,    columns=col_names)
    for col, series in valid_indicators:
        df_simple[f"{col}_missing"] = series.values
        df_knn[f"{col}_missing"]    = series.values
    all_cols = col_names + indicator_names
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(df_simple), size=SAMPLE_SIZE, replace=False)
    data_simple = df_simple[all_cols].to_numpy().astype(float)[sample_idx]
    data_knn    = df_knn[all_cols].to_numpy().astype(float)[sample_idx]
    data_simple_no_ind = df_simple[col_names].to_numpy().astype(float)[sample_idx]
    RUNS = [
    ("PC",  data_simple,        all_cols,  "PC_kci_simple_indicator"),
    ("FCI", data_simple,        all_cols,  "FCI_kci_simple_indicator"),
    ("PC",  data_knn,           all_cols,  "PC_kci_knn_indicator"),
    ("FCI", data_knn,           all_cols,  "FCI_kci_knn_indicator"),
    ("PC",  data_simple_no_ind, col_names, "PC_kci_simple"),
    ("FCI", data_simple_no_ind, col_names, "FCI_kci_simple")
]

    for algo, data,run_cols, run_name in RUNS:
        print(f"\n--- {run_name} ---")
        run_and_save(algo, data, run_cols, run_name)

    print("\nDone.")


if __name__ == "__main__":
    main()