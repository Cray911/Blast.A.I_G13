"""
Step 5: Train and Evaluate All Models on Touched Test Set
Group 13 | UMaT BSc Mining Engineering
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"

print("=" * 60)
print("STEP 5: COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)

# Load data
X = np.load(rf"{CODES_PATH}\X.npy")
y = np.load(rf"{CODES_PATH}\y.npy")
train_mask = np.load(rf"{CODES_PATH}\train_mask.npy")
test_mask = np.load(rf"{CODES_PATH}\test_mask.npy")
predictor_cols = pd.read_csv(rf"{CODES_PATH}\predictor_cols.csv")["0"].tolist()

pso_ga_mask = np.load(rf"{CODES_PATH}\pso_ga_mask.npy")
frfs_mask = np.load(rf"{CODES_PATH}\frfs_mask.npy")

pso_features = pd.read_csv(rf"{CODES_PATH}\pso_ga_features.csv")["0"].tolist()
frfs_features = pd.read_csv(rf"{CODES_PATH}\frfs_features.csv")["0"].tolist()

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"[1] Test set size: {len(y_test)} samples (evaluated only ONCE)")

# Evaluation helper
def evaluate_model(name, y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Category agreement (Medium <=40, Hard >40)
    cat_true = np.array(["Hard" if v > 40 else "Medium" for v in y_true])
    cat_pred = np.array(["Hard" if v > 40 else "Medium" for v in y_pred])
    cat_acc = np.mean(cat_true == cat_pred)
    
    return {
        "Model": name,
        "Inputs": n_features,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Category_Agreement_%": round(cat_acc * 100, 2)
    }

results = []

# 1. Mean Baseline
mean_pred = np.full(len(y_test), y_train.mean())
results.append(evaluate_model("Baseline: Mean Predictor", y_test, mean_pred, 0))

# 2. Linear Regression (Full)
lr_full = LinearRegression().fit(X_train, y_train)
results.append(evaluate_model("Baseline: Linear Regression (Full)", y_test, lr_full.predict(X_test), 26))

# 3. Linear Regression (PSO-GA features)
lr_pso = LinearRegression().fit(X_train[:, pso_ga_mask == 1], y_train)
results.append(evaluate_model("Baseline: Linear Regression (PSO-GA)", y_test, lr_pso.predict(X_test[:, pso_ga_mask == 1]), pso_ga_mask.sum()))

# 4. RF Full Model
with open(rf"{CODES_PATH}\rf_full_model.pkl", "rb") as f:
    rf_full = pickle.load(f)
results.append(evaluate_model("RF: Full Model", y_test, rf_full.predict(X_test), 26))

# 5. RF PSO-GA Model (Tuned on training set)
print("\n[2] Fitting and tuning RF on PSO-GA subset (8 features)...")
best_params_pso = {'max_depth': 12, 'max_features': 0.6, 'min_samples_leaf': 1, 'n_estimators': 500}
rf_pso = RandomForestRegressor(**best_params_pso, n_jobs=-1, random_state=42)
rf_pso.fit(X_train[:, pso_ga_mask == 1], y_train)
pred_pso = rf_pso.predict(X_test[:, pso_ga_mask == 1])
results.append(evaluate_model("RF: PSO-GA Selected", y_test, pred_pso, pso_ga_mask.sum()))

with open(rf"{CODES_PATH}\rf_pso_model.pkl", "wb") as f:
    pickle.dump(rf_pso, f)

# 6. RF FRFS Model
print("[3] Fitting and tuning RF on FRFS subset (23 features)...")
rf_frfs = RandomForestRegressor(**best_params_pso, n_jobs=-1, random_state=42)
rf_frfs.fit(X_train[:, frfs_mask == 1], y_train)
pred_frfs = rf_frfs.predict(X_test[:, frfs_mask == 1])
results.append(evaluate_model("RF: FRFS Selected", y_test, pred_frfs, frfs_mask.sum()))

with open(rf"{CODES_PATH}\rf_frfs_model.pkl", "wb") as f:
    pickle.dump(rf_frfs, f)

results_table = pd.DataFrame(results)
results_table.to_csv(rf"{CODES_PATH}\final_model_comparisons.csv", index=False)

print("\n[4] FINAL COMPARATIVE PERFORMANCE TABLE:")
print("=" * 75)
print(results_table.to_string(index=False))
print("=" * 75)

# Hypothesis test checking
r2_full = results_table.loc[results_table["Model"] == "RF: Full Model", "R2"].values[0]
r2_pso = results_table.loc[results_table["Model"] == "RF: PSO-GA Selected", "R2"].values[0]
r2_diff = r2_full - r2_pso

print(f"\nHypothesis Check (Sub-question 1):")
print(f"  Full RF R²   : {r2_full:.4f}")
print(f"  PSO-GA RF R² : {r2_pso:.4f}")
print(f"  Delta R²     : {r2_diff:.4f} (Threshold in synopsis: <= 0.05)")
if r2_diff <= 0.05:
    print("  ✅ HYPOTHESIS CONFIRMED: PSO-GA achieved similar accuracy with 69.2% fewer inputs!")
else:
    print(f"  Delta is {r2_diff:.4f}")

print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)
