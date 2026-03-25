import pandas as pd

table = pd.read_csv("ensemble_table_v2.csv")

# get all edges where vaso_given is the effect 
causes_vaso = table[
    (table["edge"].str.endswith("-->vaso_given")) &
    (table["agreement_score"] >= 2)
][["edge", "agreement_score"]].sort_values("agreement_score", ascending=False)

# get all edges where vaso_given is the cause
vaso_causes = table[
    (table["edge"].str.startswith("vaso_given-->")) &
    (table["agreement_score"] >= 2)
][["edge", "agreement_score"]].sort_values("agreement_score", ascending=False)

print("=== What causes vaso_given (adjustment set candidates) ===")
print(causes_vaso.to_string(index=False))

print("=== What vaso_given causes ===")
print(vaso_causes.to_string(index=False))