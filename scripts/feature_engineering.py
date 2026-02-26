import pandas as pd
from datetime import timedelta
import datetime as dt
import numpy as np

columns = ['hadm_id','subject_id','stay_id',
    'anchor_age',
    'gender',
    'race',
    'admission_type',
    'admission_location',
    'admittime',
    'dischtime',
    'hospital_expire_flag',
    'intime',
    'outtime',
    'ICU_length',
    'Hospital_length']

vitals = {"heart_rate_max":{'itemid':220045, 'agg':'max'}, "blood_pressure_min":{'itemid':220181,"agg":'min'},"spO2_min":{'itemid':220277,'agg':'min'},"FiO2_max":{'itemid':223835, 'agg':'max'},"temperature_max_C":{'itemid':223762, 'agg':'max'},"temperature_max_F":{'itemid':223761,'agg':'max'},"gsc_motor_min":{'itemid':223901,'agg':'min'},"gsc_verbal_min":{'itemid':223900,'agg':'min'},"gsc_eye_min":{'itemid':220739,'agg':'min'}}

labevents = {"sodium_max":{'itemid':[50983,52623],'agg':'max'}, "sodium_min":{'itemid':[50983,52623],'agg':'min'},"potassium_max":{'itemid':[52610,50971],'agg':'max'},"bun_max":{'itemid':[51006,52647], 'agg':'max'},"creatinine_max":{'itemid':[50912,52546],'agg':'max'},"glucose_min":{'itemid':[50931,52569],'agg':'min'},"pH_min":{'itemid':[50820],'agg':'min'},"lactate_max":{'itemid':[50813, 52442, 53154],'agg':'max'}, "platelet_max":{'itemid':[51704,51265],'agg':'max'},"wbc_max":{'itemid':[51301, 51755, 51756],'agg':'max'},"hemoglobin_min":{'itemid':[50811, 51222, 51640],'agg':'min'},"ast_max":{'itemid':[53088,50878],'agg':'max'},"alt_max":{'itemid':[50861],'agg':'max'},"bilirubin_max":{'itemid':[50885,53089],'agg':'max'},"inr_max":{'itemid':[51675,51237],'agg':'max'}}

antibiotics = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ciprofloxacin', 'Ciprofloxacin HCl', 'Meropenem', 'CefePIME', 'CeftriaXONE', 'MetRONIDAZOLE (FLagyl)', 'CefTRIAXone', 'Acyclovir', 'CefazoLIN', 'Sulfameth/Trimethoprim DS', 'Tobramycin', 'Azithromycin', 'Levofloxacin', 'Ampicillin', 'Erythromycin', 'Clindamycin', 'Aztreonam', 'CeFAZolin', 'moxifloxacin', 'Linezolid', 'Micafungin', 'Sulfamethoxazole-Trimethoprim', 'Doxycycline Hyclate', 'CefTAZidime', 'MetroNIDAZOLE', 'Sulfameth/Trimethoprim SS']
antibiotic_patterns = [
    'adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicill', 'amphotericin',
    'anidulafungin', 'ancef', 'clavulanate', 'ampicillin', 'augmentin', 'avelox',
    'avidoxy', 'azactam', 'azithromycin', 'aztreonam', 'axetil', 'bactocill', 'bactrim',
    'bactroban', 'bethkis', 'biaxin', 'bicillin l-a', 'cayston', 'cefazolin', 'cedax',
    'cefoxitin', 'ceftazidime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
    'cefepime', 'cefotan', 'cefotetan', 'cefotaxime', 'ceftaroline', 'cefpodoxime',
    'cefpirome', 'cefprozil', 'ceftibuten', 'ceftin', 'ceftriaxone', 'cefuroxime',
    'cephalexin', 'cephalothin', 'cephapririn', 'chloramphenicol', 'cipro', 'ciprofloxacin',
    'claforan', 'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dicloxacillin',
    'dirithromycin', 'doryx', 'doxycy', 'duricef', 'dynacin', 'ery-tab', 'eryped', 'eryc',
    'erythrocin', 'erythromycin', 'factive', 'flagyl', 'fortaz', 'furadantin', 'garamycin',
    'gentamicin', 'kanamycin', 'keflex', 'kefzol', 'ketek', 'levaquin', 'levofloxacin',
    'lincocin', 'linezolid', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin',
    'metronidazole', 'meropenem', 'methicillin', 'minocin', 'minocycline', 'monodox',
    'monurol', 'morgidox', 'moxatag', 'moxifloxacin', 'mupirocin', 'myrac', 'nafcillin',
    'neomycin', 'nicazel doxy 30', 'nitrofurantoin', 'norfloxacin', 'noroxin', 'ocudox',
    'ofloxacin', 'omnicef', 'oracea', 'oraxyl', 'oxacillin', 'pc pen vk', 'pce dispertab',
    'panixine', 'pediazole', 'penicillin', 'periostat', 'pfizerpen', 'piperacillin',
    'tazobactam', 'primsol', 'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin',
    'smz-tmp', 'septra', 'septra ds', 'solodyn', 'spectracef', 'streptomycin',
    'sulfadiazine', 'sulfamethoxazole', 'trimethoprim', 'sulfatrim', 'sulfisoxazole',
    'suprax', 'synercid', 'tazicef', 'tetracycline', 'timentin', 'tobramycin', 'unasyn',
    'vancocin', 'vancomycin', 'vantin', 'vibativ', 'vibra-tabs', 'vibramycin', 'zinacef',
    'zithromax', 'zosyn', 'zyvox'
]

def is_antibiotic(medication):
    med_lower = str(medication).lower()
    return any(pattern in med_lower for pattern in antibiotic_patterns)

vasoactive_agents = ['Norepinephrine', 'Epinephrine', 'Vasopressin', 'Phenylephrine','Dopamine','Dobutamine','Milrinone']

icd_codes_septic_shock = ["R6521","78552"]
icd_codes_sepsis = ["R6520","99592","99591","A41"]
icd_codes_kidney = ["N17","584"]

procedure_keywords = ["ventilation", "endotracheal", "intubation", "mechanical ventilation"]


    
def get_vitals(df, before, after,chartevents):
    df = df.copy()
    df["end_window"] = (df["sepsis_onset_time"] + timedelta(hours=after))
    df["start_window"] = (df["sepsis_onset_time"] - timedelta(hours=before))
    c = chartevents.copy()
    c["charttime"] =pd.to_datetime(c["charttime"])
    merged  = c.merge(df[["stay_id","intime","end_window","start_window"]],on="stay_id", how="right")
    mask =  (merged['start_window'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for vital, info in vitals.items():
        test = merged[merged["itemid"]==info["itemid"]].groupby("stay_id")["valuenum"].agg(info["agg"]).reset_index(name=vital)
        df = df.merge(test, on="stay_id",how="left")
    return df

def get_labs(df,labs):
    df = df.copy()
    l = labs.copy()
    l["charttime"] = pd.to_datetime(l["charttime"])
    merged = df[["hadm_id","start_window","end_window"]].merge(l, on="hadm_id", how="left")
    mask =  (merged['start_window'] <= merged['charttime']) & (merged['charttime']<=merged["end_window"])
    merged = merged[mask]
    for event, info in labevents.items():
        test = merged[merged["itemid"].isin(info["itemid"])].groupby("hadm_id")["valuenum"].agg(info["agg"]).reset_index(name=event)
        df = df.merge(test, on="hadm_id",how="left")
    return df
        
def get_medications(df,pharmacy):
    df = df.copy()
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"],errors="coerce")
    merged = p.merge(df[["hadm_id","start_window","end_window"]], on="hadm_id", how="inner")
    mask = (merged["starttime"] >= merged["start_window"]) & (merged["starttime"] <= merged["end_window"])
    merged = merged[mask]

    merged['is_antibiotic'] = merged['medication'].apply(is_antibiotic)
    ab = merged[merged['is_antibiotic']]
    vaso = merged[merged["medication"].isin(vasoactive_agents)]

    ab_flag = ab.groupby("hadm_id").size().rename("antibiotics_given") > 0
    vaso_flag = vaso.groupby("hadm_id").size().rename("vaso_given") > 0

    df = df.merge(ab_flag, on="hadm_id", how="left")
    df = df.merge(vaso_flag, on="hadm_id", how="left")
    df["antibiotics_given"] = df["antibiotics_given"].notna().astype(int)
    df["vaso_given"] = df["vaso_given"].notna().astype(int)
    return df
    
def get_max_creatinine_bun(df,labs):
    creatinine = labs[ (labs["itemid"].isin([50912,52546])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_cre = creatinine.groupby("hadm_id")["valuenum"].max().reset_index(name="creatinine_admission_max")
    bun = labs[(labs["itemid"].isin([51006,52647])) & (labs["hadm_id"].isin(df["hadm_id"]))]
    max_bun = bun.groupby("hadm_id")["valuenum"].max().reset_index(name="bun_admission_max")
    df = df.merge(max_cre, on="hadm_id", how="left")
    df = df.merge(max_bun, on="hadm_id", how="left")
    return df

def get_time_to_first_antibiotic(df,pharmacy):
    df = df.copy()
    p = pharmacy.copy()
    p["starttime"] = pd.to_datetime(p["starttime"],errors="coerce")
    merged = p.merge(df[["hadm_id","admittime"]],on="hadm_id", how="right")
    antibiotics_df = merged[merged["medication"].isin(antibiotics)]
    mask = antibiotics_df["starttime"] >= antibiotics_df["admittime"]
    antibiotics_df = antibiotics_df[mask]
    first = antibiotics_df.groupby("hadm_id")["starttime"].min().reset_index(name="first_antibiotic_time")
    df = df.merge(first, on="hadm_id", how="left")
    df["time_to_first_antibiotic_hrs"] = (df["first_antibiotic_time"] - df["admittime"]).dt.total_seconds() / 3600
    return df

def get_procedures(df):
    procedures_diagnoses = procedures[procedures["hadm_id"].isin(df["hadm_id"])]
    procedures_diagnoses = procedures_diagnoses.merge(d_procedures,on=["icd_code","icd_version"], how="left")
    procedure_mask = procedures_diagnoses['long_title'].str.contains('|'.join(procedure_keywords), case=False, na=False)
    procedure_procs = procedures_diagnoses[procedure_mask]
    procedure_procs_hadm = procedure_procs["hadm_id"]
    df['vent_or_intubation'] = df['hadm_id'].isin(procedure_procs_hadm).astype(int)
    return df
    
def get_bmi(df):
    o = omr.copy()
    o["chartdate"] = pd.to_datetime(o["chartdate"])
    o = o[o["result_name"].isin(["Height (Inches)", "Weight (Lbs)"])]
    merged = o.merge(df[["subject_id", "hadm_id", "admittime"]],on="subject_id", how="right")
    merged = merged[merged["chartdate"] >= merged["admittime"]]
    pivoted = merged.pivot_table(index=["subject_id", "hadm_id", "chartdate"], columns="result_name",values="result_value", aggfunc="first").reset_index()
    pivoted = pivoted.sort_values(["hadm_id", "chartdate"]).groupby("hadm_id").first().reset_index()
    pivoted["Height (Inches)"] = pd.to_numeric(pivoted["Height (Inches)"], errors="coerce")
    pivoted["Weight (Lbs)"] = pd.to_numeric(pivoted["Weight (Lbs)"], errors="coerce")
    pivoted["BMI"] = (pivoted["Weight (Lbs)"] / (pivoted["Height (Inches)"] ** 2))*703
    pivoted = pivoted[["hadm_id", "BMI"]]
    df = df.merge(pivoted, on="hadm_id", how="left")
    return df

def get_diagnosis_flags(df,diagnoses):
    dx = diagnoses[["hadm_id","icd_code"]].copy()
    dx = dx[dx["hadm_id"].isin(df["hadm_id"])]
    dx["icd_code"] = dx["icd_code"].astype(str)
    dx["septic_shock"] = dx["icd_code"].str.startswith(tuple(icd_codes_septic_shock)).astype(int)
    dx["sepsis"] = dx["icd_code"].str.startswith(tuple(icd_codes_sepsis)).astype(int)
    dx["arf"] = dx["icd_code"].str.startswith(tuple(icd_codes_kidney)).astype(int)
    dx = dx.groupby("hadm_id")[["septic_shock","sepsis","arf"]].max().reset_index()
    return df.merge(dx,on="hadm_id",how="left")

def get_max_temperature(row):
    temp_f = row['temperature_max_F']
    temp_c = row['temperature_max_C']
    if pd.isna(temp_f) and pd.isna(temp_c):
        return np.nan

    elif pd.isna(temp_c):
        return temp_f

    elif pd.isna(temp_f):
        return temp_c * 9/5 + 32

    else:
        temp_c_as_f = temp_c * 9/5 + 32
        return max(temp_f, temp_c_as_f)
    