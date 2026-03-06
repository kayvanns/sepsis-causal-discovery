"""
Causal Discovery Comparison Script

Algorithms  : PC, FCI
Indep tests : Fisher-Z, MV-Fisher-Z
Miss strats : Simple mean imputation, KNN imputation, Raw NaNs (MV-Fisher-Z only)

Background knowledge enforced on every run.

Outputs: graphs/<ALGO>_<TEST>_<IMPUTE>.png
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import traceback

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
    **{c: 1 for c in PHYS_COLS + TREAT_COLS+OUT24_COLS},
    **{c: 2 for c in POST24_COLS},
    **{c: 3 for c in MORT_COLS},
}

CATEGORICAL_COLS = ["gender", "race"]
BINARY_COLS = [
    "hospital_expire_flag", "antibiotics_given", "vaso_given",
    "aki_24h_onset_stage_y", "mechvent_24h_onset",
    "aki_post24h_stage", "mechvent_post24h",
]

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
        print(f"[WARN] Columns not found and skipped: {missing}")
    return df[available].copy(), available


def impute_simple(arr: np.ndarray) -> np.ndarray:
    # Mean imputation
    return SimpleImputer(strategy="mean").fit_transform(arr)


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
 

#  Algorithms

ALPHA = 0.05

# (algorithm, indep_test_fn, test_label, impute_strategy, use_mvpc)

RUNS = [
    ("PC",  fisherz,    "fisherz",    "simple", False),
     ("PC",  fisherz,    "fisherz",    "simple_indicator", False),
    ("PC",  fisherz,    "fisherz",    "knn_indicator",    False),
    ("PC",  mv_fisherz, "mv_fisherz", "raw",    True),
    ("FCI", fisherz,    "fisherz",    "simple", False),
    ("FCI", fisherz,    "fisherz",    "simple_indicator", False),
    ("FCI", fisherz,    "fisherz",    "knn_indicator",    False),
    ("FCI", mv_fisherz, "mv_fisherz", "raw",    False),
]


# Main execution loop

def main():
    os.makedirs("graphs", exist_ok=True)

    print("Loading data ...")
    df, col_names = load_data("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")
    print(f"Using {len(col_names)} columns, {len(df)} rows.\n")

    raw_arr = df.to_numpy().astype(float)
    
    print("Pre-computing imputations ...")
    simple_arr = impute_simple(raw_arr)
    knn_arr    = impute_knn(raw_arr)
    print("Done.\n")

    # build indicator columns from raw df before imputation
    df_simple = pd.DataFrame(simple_arr, columns=col_names)
    df_knn    = pd.DataFrame(knn_arr,    columns=col_names)

    indicator_names = []
    for col in PHYS_COLS:
        if col in df.columns and df[col].isna().mean() > 0.01:
            ind = df[col].isna().astype(float)
            df_simple[f"{col}_missing"] = ind.values
            df_knn[f"{col}_missing"]    = ind.values
            indicator_names.append(f"{col}_missing")
            print(f"  [indicator] {col}: {df[col].isna().mean():.2%} missing")

    simple_indicator_cols = col_names + indicator_names
    knn_indicator_cols    = col_names + indicator_names
    np.random.seed(42)

    sample_idx = np.random.choice(len(df_simple), size=5000, replace=False)

    data_by_strat = {
        "raw":              (raw_arr, col_names),
        "simple":           (simple_arr, col_names),
        "simple_indicator": (df_simple[simple_indicator_cols].to_numpy()[sample_idx], simple_indicator_cols),
        "knn_indicator":    (df_knn[knn_indicator_cols].to_numpy()[sample_idx], knn_indicator_cols)}

    for algo, test_fn, test_label, impute_strat, use_mvpc in RUNS:
        run_name = f"{algo}_{test_label}_{impute_strat}"
        print(f"Running {run_name} ...")
        data,run_cols = data_by_strat[impute_strat]

        try:
            if algo == "PC":
                # get node objects for BK
                cg = pc(data, alpha=ALPHA, indep_test=test_fn,
                        mvpc=use_mvpc)
                bk = build_background_knowledge(cg.G.get_nodes(), run_cols)
                # apply BK
                cg = pc(data, alpha=ALPHA, indep_test=test_fn,
                        mvpc=use_mvpc, background_knowledge=bk)
                graph = cg.G

            elif algo == "FCI":
                # get node objects for BK
                g0, _ = fci(data, independence_test_method=test_fn,
                            alpha=ALPHA)
                bk = build_background_knowledge(g0.get_nodes(), run_cols)
                # apply BK
                graph, _ = fci(data, independence_test_method=test_fn,
                               alpha=ALPHA, background_knowledge=bk)

            out_path = f"graphs/{run_name}.png"
            pyd = GraphUtils.to_pydot(graph, labels=run_cols)
            pyd.write_png(out_path)
            print(f"  Saved -> {out_path}")

        except Exception as e:
            print(f"  FAILED: {traceback.format_exc()}")

    print("\nDone. All graphs saved to ./graphs/")


if __name__ == "__main__":
    main()
