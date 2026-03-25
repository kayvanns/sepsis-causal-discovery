import pandas as pd

table = pd.read_csv("ensemble_table_v2.csv")

# get all edges where vaso_given is the effect 
causes_aki = table[
    (table["edge"].str.endswith("-->aki_24h_onset_stage_y")) &
    (table["agreement_score"] >= 2)
][["edge", "agreement_score"]].sort_values("agreement_score", ascending=False)

# get all edges where vaso_given is the cause
aki_causes = table[
    (table["edge"].str.startswith("aki_24h_onset_stage_y-->")) &
    (table["agreement_score"] >= 2)
][["edge", "agreement_score"]].sort_values("agreement_score", ascending=False)

print("=== What causes aki_24h_onset_stage_y (adjustment set candidates) ===")
print(causes_aki.to_string(index=False))

print("=== What aki_24h_onset_stage_y causes ===")
print(aki_causes.to_string(index=False))