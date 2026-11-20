"""
Step 2: Train/Test Split + Baseline Models + Full Random Forest
Group 13 | UMaT BSc Mining Engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"
DATA_PATH  = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Data\BI-GROUP 13.xlsx"

print("=" * 60)
print("STEP 2: BASELINES + FULL RANDOM FOREST MODEL")
print("=" * 60)

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, sheet_name="data")

# Encode categoricals
cat_cols = ["roughness", "lithology", "alteration", "rmd_class", "jpo_class"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Drop infill (54.9% missing) and Lilly rating components (trivial predictors)
excluded = [
    "record_id", "panel_id", "set_id", "domain_label", "infill",
    "rmd_class", "rmd_rating", "jps_rating", "jpo_class", "jpo_rating",
    "sgi", "hardness_H", "BI"
]
predictor_cols = [c for c in df.columns if c not in excluded]
print(f"\n[1] Using {len(predictor_cols)} predictors: {predictor_cols}")

# Drop rows with any remaining NaNs
df_clean = df.dropna(subset=predictor_cols + ["BI"]).reset_index(drop=True)
print(f"[2] Records after dropping NaNs: {len(df_clean)}")

X = df_clean[predictor_cols].values
y = df_clean["BI"].values

# Groups for grouped splitting: use bench_rl as proxy group
groups = df_clean["bench_rl"].values

# ── Grouped train/test split (80/20) ─────────────────────────────────────────
print("\n[3] Grouped train/test split (80/20 by bench_rl)...")
unique_benches = np.unique(groups)
np.random.seed(42)
np.random.shuffle(unique_benches)
split_idx = int(len(unique_benches) * 0.8)
train_benches = set(unique_benches[:split_idx])
test_benches  = set(unique_benches[split_idx:])

train_mask = np.array([g in train_benches for g in groups])
test_mask  = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
groups_train = groups[train_mask]

print(f"    Train set : {X_train.shape[0]} records ({len(train_benches)} benches)")
print(f"    Test set  : {X_test.shape[0]} records ({len(test_benches)} benches)")

# Save split indices for reuse in later steps
np.save(rf"{CODES_PATH}\train_mask.npy", train_mask)
np.save(rf"{CODES_PATH}\test_mask.npy",  test_mask)
np.save(rf"{CODES_PATH}\X.npy",          X)
np.save(rf"{CODES_PATH}\y.npy",          y)
np.save(rf"{CODES_PATH}\groups.npy",     groups)
pd.Series(predictor_cols).to_csv(rf"{CODES_PATH}\predictor_cols.csv", index=False)

# ── Helper: evaluation metrics ────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # BI category agreement
    def cat(v): return "Hard" if v > 40 else "Medium" if v >= 20 else "Easy"
    cat_true = [cat(v) for v in y_true]
    cat_pred = [cat(v) for v in y_pred]
    cat_acc  = np.mean([a == b for a, b in zip(cat_true, cat_pred)])
    print(f"\n  {'─'*45}")
    print(f"  Model     : {name}")
    print(f"  R²        : {r2:.4f}")
    print(f"  RMSE      : {rmse:.4f} BI units")
    print(f"  MAE       : {mae:.4f} BI units")
    print(f"  Category  : {cat_acc*100:.1f}% agreement")
    return {"Model": name, "R2": round(r2,4), "RMSE": round(rmse,4),
            "MAE": round(mae,4), "Cat_Acc": round(cat_acc,4)}

results = []

# ── Baseline 1: Mean predictor ────────────────────────────────────────────────
print("\n[4] BASELINE 1 — Training-set mean predictor")
mean_pred = np.full(len(y_test), y_train.mean())
print(f"    Predicts constant BI = {y_train.mean():.4f} for every observation")
results.append(evaluate("Mean Baseline", y_test, mean_pred))

# ── Baseline 2: Linear Regression ────────────────────────────────────────────
print("\n[5] BASELINE 2 — Linear Regression (all predictors)")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append(evaluate("Linear Regression (Full)", y_test, lr_pred))

# ── Full RF Model ─────────────────────────────────────────────────────────────
print("\n[6] FULL RANDOM FOREST — Tuning with Grouped 5-Fold CV...")

# Simple grid search over key hyperparameters
from sklearn.model_selection import ParameterGrid

gkf = GroupKFold(n_splits=5)
param_grid = {
    "n_estimators": [200, 500],
    "max_depth":    [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.5]
}

best_score, best_params = -np.inf, None
grid = list(ParameterGrid(param_grid))
print(f"    Testing {len(grid)} parameter combinations...")

for i, params in enumerate(grid):
    rf = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train,
                                cv=gkf, groups=groups_train,
                                scoring="r2")
    mean_cv = cv_scores.mean()
    if mean_cv > best_score:
        best_score = mean_cv
        best_params = params

print(f"    Best CV R²    : {best_score:.4f}")
print(f"    Best params   : {best_params}")

# Train final RF on full training set with best params
rf_full = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
rf_full.fit(X_train, y_train)
rf_pred = rf_full.predict(X_test)
results.append(evaluate("RF Full (all predictors)", y_test, rf_pred))

# ── Save results & model ──────────────────────────────────────────────────────
import pickle
with open(rf"{CODES_PATH}\rf_full_model.pkl", "wb") as f:
    pickle.dump(rf_full, f)

results_df = pd.DataFrame(results)
results_df.to_csv(rf"{CODES_PATH}\model_results.csv", index=False)

print("\n\n[7] RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

print(f"\n[8] Full RF model saved to: rf_full_model.pkl")
print(f"    Results saved to       : model_results.csv")

# Feature importance from full RF
fi = pd.DataFrame({
    "Feature": predictor_cols,
    "Importance": rf_full.feature_importances_
}).sort_values("Importance", ascending=False)
fi.to_csv(rf"{CODES_PATH}\feature_importance_full_rf.csv", index=False)
print(f"\n[9] Top 10 features by RF importance:")
print(fi.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
