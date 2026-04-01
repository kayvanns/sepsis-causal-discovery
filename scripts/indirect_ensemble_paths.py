import pickle
import pandas as pd
import os
from collections import defaultdict

def get_direct_edges(matrix, cols):
    n = len(cols)
    edges = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                edges[i].append(j)
    return edges

def find_all_paths(edges, start, end):
    all_paths = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        for neighbor in edges[node]:
            if neighbor == end:
                all_paths.append(path + [neighbor])
            elif neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    return all_paths

def build_ensemble_table(graph_dir):
    run_edges = []
    for file in os.listdir(graph_dir):
        if not file.endswith(".pkl"):
            continue
        run_name = file.replace(".pkl", "")
        with open(os.path.join(graph_dir, file), "rb") as f:
            graph, cols = pickle.load(f)
        matrix = graph.graph
        n = len(cols)
        edges = get_direct_edges(matrix, cols)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                paths = find_all_paths(edges, i, j)
                indirect_paths = [p for p in paths if len(p) > 2]
                if indirect_paths:
                    for path in indirect_paths:
                        run_edges.append({
                            "run":           run_name,
                            "cause":         cols[i],
                            "effect":        cols[j],
                            "path":          " --> ".join([cols[k] for k in path]),
                            "intermediaries": " --> ".join([cols[k] for k in path[1:-1]]),
                            "path_length":   len(path) - 1,
                        })

    df = pd.DataFrame(run_edges)

    df["edge"] = df["cause"] + " --> " + df["effect"]
    df["present"] = 1

    table = df.pivot_table(
        index=["edge","cause","effect", "path", "intermediaries", "path_length"],
        columns="run",
        values="present",
        aggfunc="max",
        fill_value=0
    ).reset_index()

    run_cols = [c for c in table.columns if c not in ["edge","cause","effect", "path", "intermediaries", "path_length"]]
    table["agreement_score"] = table[run_cols].sum(axis=1)
    table = table.sort_values(["edge", "agreement_score"], ascending=[True, False]).reset_index(drop=True)
    return table


table = build_ensemble_table("graphs")
table.to_csv("indirect_ensemble_paths.csv", index=False)
