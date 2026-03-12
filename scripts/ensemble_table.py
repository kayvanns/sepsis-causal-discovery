import pickle
import pandas as pd
import os
def build_ensemble_table(graph_dir: str):
    run_edges = []
    for file in os.listdir(graph_dir):
        if not file.endswith(".pkl"):
            continue
        with open(os.path.join(graph_dir, file), "rb") as f:
            graph, cols = pickle.load(f)
        run_name = file.replace(".pkl", "")
        matrix = graph.graph
        n = len(cols)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == -1 and matrix[j, i] == 1:
                    run_edges.append({"run":run_name, "cause": cols[i], "effect": cols[j]})
        df_run_edges = pd.DataFrame(run_edges)
        df_run_edges["edge"] = df_run_edges["cause"] + "-->" + df_run_edges["effect"]
        df_run_edges["present"]= 1
    df_edges = df_run_edges.pivot_table(index ="edge", columns="run", values="present", aggfunc="max", fill_value=0).reset_index()
    run_cols = [c for c in df_edges.columns if c != "edge"]
    df_edges["agreement_score"] = df_edges[run_cols].sum(axis=1)
    table = df_edges.sort_values("agreement_score", ascending=False)
    return table
    
table = build_ensemble_table("graphs")
table.to_csv("ensemble_table_v2.csv", index=False)
