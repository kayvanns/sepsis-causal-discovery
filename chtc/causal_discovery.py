import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import pickle
import traceback
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, mv_fisherz, kci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

# ── Column definitions ─────────────────────────────────────────────────────────
DEMO_COLS   = ["anchor_age", "gender", "race", "ckd_baseline", "BMI"]
PHYS_COLS   = ["Heart Rate (max)", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
               "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)",
               "Temperature (Max)", "Hemoglobin (min)", "Fluid Balance (mL)", "WBC (max)", "sofa_score", "Creatinine (max)"]
TREAT_COLS  = ["Antibiotics", "Vasopressors", "Diuretics"]
OUT24_COLS  = ["aki_24h_onset", "mechvent_24h_onset"]
POST24_COLS = ["aki_post24h", "mechvent_post24h"]
MORT_COLS   = ["hospital_expire_flag"]
CORE_COLS   = DEMO_COLS + PHYS_COLS + TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS

TIER_MAP = {
    **{c: 0 for c in DEMO_COLS},
    **{c: 1 for c in PHYS_COLS + ['aki_24h_onset']},
    **{c:2 for c in TREAT_COLS + [ "mechvent_24h_onset"]},
    **{c: 3 for c in POST24_COLS},
    **{c: 4 for c in MORT_COLS},
}




RUNS = [
    ("PC",  fisherz,    False),  # used with simple, simple_indicator, knn_indicator
    ("PC",  mv_fisherz, True),   # used with raw
    ("FCI", fisherz,    False),  # used with simple, simple_indicator, knn_indicator
    ("FCI", mv_fisherz, False),  # used with raw
    ("PC",kci,False),  # used with simple, simple_indicator, knn_indicator
    ("FCI",kci,False),  # used with simple, simple_indicator, knn_indicator
]

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    col_names = df.columns.tolist()
    print(f"  {len(df)} rows, {len(col_names)} columns")
    return df, col_names

# ── Background knowledge — temporal tiers only ────────────────────────────────
def build_background_knowledge(col_names):
    nodes = [GraphNode(col) for col in col_names]
    col_to_node = {col: nodes[i] for i, col in enumerate(col_names)}
    bk = BackgroundKnowledge()

    for col, tier in TIER_MAP.items():
        if col in col_to_node:
            bk.add_node_to_tier(col_to_node[col], tier)

    for col in col_names:
        if col.endswith("_missing") and col in col_to_node:
            bk.add_node_to_tier(col_to_node[col], 1)

    return bk

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_index", type=int, required=True)
    parser.add_argument("--run_name",  type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--alpha",           type=float, default=0.05)
    parser.add_argument("--sample_size",     type=int,   default=-1)
    parser.add_argument("--depth", type=int, default=-1)
    args = parser.parse_args()
    
    start = time.time()

    algo, test_fn, use_mvpc = RUNS[args.run_index]
    run_name = args.run_name
    print(f"Run: {run_name} (index {args.run_index})")

    print("Loading data...")
    df, col_names = load_data(args.data_path)


    data = df.to_numpy().astype(float)
    if args.sample_size > 0 and args.sample_size < len(df):
        np.random.seed(42)
        idx  = np.random.choice(len(df), size=args.sample_size, replace=False)
        data = data[idx]
        print(f"  Subsampled to {args.sample_size} rows")

    try:
        bk = build_background_knowledge(col_names)

        if algo == "PC":
            cg = pc(data, alpha=args.alpha, indep_test=test_fn, mvpc=use_mvpc,node_names=col_names,background_knowledge=bk)
            graph = cg.G

        elif algo == "FCI":
            graph, _ = fci(data, independence_test_method=test_fn, alpha=args.alpha,node_names=col_names,background_knowledge=bk,depth=args.depth) 

        pkl_path = f"{run_name}.pkl"
        png_path = f"{run_name}.png"
        with open(pkl_path, "wb") as f:
            pickle.dump((graph, col_names), f)
        pyd = GraphUtils.to_pydot(graph, labels=col_names)
        pyd.write_png(png_path)
        print(f"  Saved: {pkl_path}")
        print(f"  Saved: {png_path}")

    except Exception:
        print(f"  FAILED:\n{traceback.format_exc()}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s ({elapsed/3600:.2f}hrs)")

if __name__ == "__main__":
    main()