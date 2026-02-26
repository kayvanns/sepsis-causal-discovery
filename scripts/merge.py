import pandas as pd

cohort = pd.read_csv("../data/raw/sepsis_cohort.csv")
engineered = pd.read_csv("../data/raw/sepsis_engineered.csv")


df = cohort.merge(engineered, on="stay_id", how="inner")

print("Merged shape:", df.shape)

df.to_csv("../data/processed/analysis.csv", index=False)
