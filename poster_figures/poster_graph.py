
import matplotlib.pyplot as plt
import pickle
import networkx as nx

with open("graphs/FCI_mv_fisherz_raw_v2.pkl", "rb") as f:
    graph, cols = pickle.load(f)
mat = graph.graph
n = len(cols)
G = nx.DiGraph()
for i in range(n):
    for j in range(n):
        if mat[i,j] == -1 and mat[j,i] == 1:
            G.add_edge(cols[i], cols[j])

plt.figure(figsize=(20,14))
pos = {
    "anchor_age":            (0, 4),
    "gender":                (0, 2),
    "race":                  (0, 0),
    "heart_rate_max":        (2, 5),
    "blood_pressure_min":    (2, 3),
    "spO2_min":              (2, 1),
    "FiO2_max":              (2, -1),
    "lactate_max":           (4, 5),
    "bilirubin_max":         (4, 3),
    "platelet_max":          (4, 1),
    "inr_max":               (4, -1),
    "temp_max_F":            (4, -3),
    "antibiotics_given":     (6, 4),
    "vaso_given":            (6, 2),
    "aki_24h_onset_stage_y": (6, 0),
    "mechvent_24h_onset":    (6, -2),
    "aki_post24h_stage":     (8, 2),
    "mechvent_post24h":      (8, 0),
    "hospital_expire_flag":  (10, 1),
}
nx.draw_networkx(G, pos,
    node_color="#DE050C",
    node_size=3000,
    font_color="#DE050C",
    font_size=10,
    edge_color="#1A1A1A",
    arrows=True,
    arrowsize=20,
    width=1.5,
    bbox=dict(boxstyle="round", fc="#F6EFF0")
)
plt.margins(0.2)
plt.axis("off")
plt.tight_layout()
plt.savefig("poster_graph.png", dpi=300, bbox_inches="tight")
