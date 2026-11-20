"""
Step 1: Data Preparation & BI Verification
Group 13 | UMaT BSc Mining Engineering
Explainable ML for Reduced-Input Estimation of Empirical Rock Blastability Index
"""

import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Data\BI-GROUP 13.xlsx"
OUTPUT_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATA PREPARATION & BI VERIFICATION")
print("=" * 60)

df = pd.read_excel(DATA_PATH, sheet_name="data")
dd = pd.read_excel(DATA_PATH, sheet_name="data_dictionary")

print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")

# ── 2. Missing values ─────────────────────────────────────────────────────────
print("\n[2] Missing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
mv = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
mv = mv[mv["Missing Count"] > 0]
if mv.empty:
    print("    No missing values found!")
else:
    print(mv.to_string())

# ── 3. Data types ─────────────────────────────────────────────────────────────
print("\n[3] Column data types:")
print(f"    Numeric columns : {list(df.select_dtypes(include='number').columns)}")
print(f"    Text columns    : {list(df.select_dtypes(include='object').columns)}")

# ── 4. Verify Lilly's BI formula ──────────────────────────────────────────────
print("\n[4] Verifying Lilly's BI formula: BI = 0.5 * (RMD + JPS + JPO + SGI + H)")

# Check the rating columns exist
lilly_cols = ["rmd_rating", "jps_rating", "jpo_rating", "sgi", "hardness_H"]
present = [c for c in lilly_cols if c in df.columns]
missing_lilly = [c for c in lilly_cols if c not in df.columns]

print(f"    Rating columns present : {present}")
if missing_lilly:
    print(f"    Rating columns MISSING : {missing_lilly}")

if len(present) == 5:
    df["BI_calculated"] = 0.5 * (
        df["rmd_rating"] + df["jps_rating"] + df["jpo_rating"] + df["sgi"] + df["hardness_H"]
    )
    df["BI_diff"] = (df["BI"] - df["BI_calculated"]).abs()
    max_diff = df["BI_diff"].max()
    mean_diff = df["BI_diff"].mean()
    exact_match = (df["BI_diff"] < 0.001).sum()
    print(f"    Max difference (dataset BI vs calculated): {max_diff:.4f}")
    print(f"    Mean difference                          : {mean_diff:.4f}")
    print(f"    Exact matches (diff < 0.001)             : {exact_match} / {len(df)}")
    if max_diff < 0.01:
        print("    ✅ BI column verified — matches Lilly's formula perfectly.")
    else:
        print("    ⚠️  Discrepancies found — using dataset BI column as reference.")
else:
    print("    ⚠️  Cannot verify — some rating columns missing from dataset.")

# ── 5. BI distribution ────────────────────────────────────────────────────────
print("\n[5] BI (target variable) summary:")
print(f"    Min    : {df['BI'].min():.2f}")
print(f"    Max    : {df['BI'].max():.2f}")
print(f"    Mean   : {df['BI'].mean():.2f}")
print(f"    Median : {df['BI'].median():.2f}")
print(f"    Std Dev: {df['BI'].std():.2f}")

# BI category boundaries (from Lilly 1986 — to be confirmed with supervisor)
# Commonly: Easy < 20, Medium 20–40, Hard > 40
print("\n[6] BI Category distribution (Lilly thresholds: Easy<20, Medium 20-40, Hard>40):")
def bi_category(bi):
    if bi < 20: return "Easy"
    elif bi <= 40: return "Medium"
    else: return "Hard"

df["BI_category"] = df["BI"].apply(bi_category)
print(df["BI_category"].value_counts().to_string())

# ── 6. Encode categorical columns ─────────────────────────────────────────────
print("\n[7] Encoding categorical (text) columns...")
text_cols = df.select_dtypes(include="object").columns.tolist()
# Exclude ID/metadata columns
exclude = ["record_id", "panel_id", "set_id", "domain_label", "BI_category"]
encode_cols = [c for c in text_cols if c not in exclude]
print(f"    Columns to encode: {encode_cols}")

df_encoded = df.copy()
encoding_maps = {}
for col in encode_cols:
    unique_vals = df[col].dropna().unique()
    mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
    df_encoded[col] = df[col].map(mapping)
    encoding_maps[col] = mapping
    print(f"    {col}: {mapping}")

# ── 7. Define predictor columns ───────────────────────────────────────────────
print("\n[8] Defining eligible predictor columns...")

# Exclude: target (BI), BI components (rmd/jps/jpo/sgi/H ratings — they'd give perfect score),
#          ID/metadata cols, and derived cols
excluded_as_predictors = [
    "record_id", "panel_id", "set_id", "domain_label",
    "rmd_class", "rmd_rating", "jps_rating", "jpo_class", "jpo_rating", "sgi", "hardness_H",
    "BI", "BI_calculated", "BI_diff", "BI_category"
]
predictor_cols = [c for c in df_encoded.columns if c not in excluded_as_predictors]
print(f"    Eligible predictors ({len(predictor_cols)}): {predictor_cols}")

# ── 8. Save cleaned dataset ───────────────────────────────────────────────────
df_encoded.to_csv(rf"{OUTPUT_PATH}\data_cleaned.csv", index=False)
pd.DataFrame({"predictor": predictor_cols}).to_csv(rf"{OUTPUT_PATH}\predictor_list.csv", index=False)
print(f"\n[9] Cleaned dataset saved to: data_cleaned.csv")
print(f"    Predictor list saved to  : predictor_list.csv")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
