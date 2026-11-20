"""
Step 7: Bootstrap Stability Assessment (30 Resamples)
Group 13 | UMaT BSc Mining Engineering
Quantifying subset stability (Jaccard similarity) and explanation stability (Spearman rho)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"

print("=" * 60)
print("STEP 7: BOOTSTRAP STABILITY ASSESSMENT (30 RESAMPLES)")
print("=" * 60)

# Load data
X = np.load(rf"{CODES_PATH}\X.npy")
y = np.load(rf"{CODES_PATH}\y.npy")
train_mask = np.load(rf"{CODES_PATH}\train_mask.npy")
predictor_cols = pd.read_csv(rf"{CODES_PATH}\predictor_cols.csv")["0"].tolist()

X_train, y_train = X[train_mask], y[train_mask]
n_samples, n_features = X_train.shape
N_BOOTSTRAP = 30

print(f"[1] Loaded training data: {n_samples} samples, {n_features} features.")
print(f"[2] Running {N_BOOTSTRAP} bootstrap resamples to assess engineering stability...")

selected_subsets = []
shap_rankings = []

np.random.seed(42)

for b in range(1, N_BOOTSTRAP + 1):
    # 1. Bootstrap sample with replacement
    boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_b, y_b = X_train[boot_indices], y_train[boot_indices]
    
    # 2. Fast Feature Selection on Bootstrap Sample
    # Hybrid search with 12 particles, 8 iterations for speed
    pop_size = 12
    max_iter = 8
    positions = np.random.choice([0, 1], size=(pop_size, n_features), p=[0.6, 0.4])
    for p in positions:
        if p.sum() == 0:
            p[np.random.randint(0, n_features)] = 1
            
    best_fit = 999.0
    best_mask = positions[0].copy()
    
    for it in range(max_iter):
        for i in range(pop_size):
            sel_idx = np.where(positions[i] == 1)[0]
            if len(sel_idx) == 0:
                continue
            rf_quick = RandomForestRegressor(n_estimators=30, max_depth=8, max_features=0.5, n_jobs=-1, random_state=b)
            # OOB or train RMSE approximation for bootstrap speed
            rf_quick.fit(X_b[:, sel_idx], y_b)
            preds = rf_quick.predict(X_b[:, sel_idx])
            rmse = np.sqrt(mean_squared_error(y_b, preds))
            fit = 0.85 * rmse + 0.15 * (len(sel_idx) / n_features * 5.0)
            if fit < best_fit:
                best_fit = fit
                best_mask = positions[i].copy()
                
        # Genetic mutation/crossover step
        cross_pt = np.random.randint(1, n_features)
        for i in range(pop_size):
            if np.random.rand() < 0.3:
                positions[i] = np.where(np.random.rand(n_features) < 0.5, best_mask, positions[i])
            if positions[i].sum() == 0:
                positions[i][np.random.randint(0, n_features)] = 1
                
    selected_subsets.append(best_mask)
    
    # 3. Model Fit & SHAP on Bootstrap Sample
    rf_boot = RandomForestRegressor(n_estimators=60, max_depth=10, max_features=0.5, n_jobs=-1, random_state=b)
    rf_boot.fit(X_b, y_b)
    
    # Fast SHAP on all 26 features for rank correlation
    explainer = shap.TreeExplainer(rf_boot)
    # Subsample 80 instances for fast SHAP computation
    shap_vals = explainer.shap_values(X_b[:80])
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_rankings.append(mean_abs_shap)
    
    if b % 5 == 0 or b == N_BOOTSTRAP:
        print(f"    Completed Bootstrap {b:02d}/{N_BOOTSTRAP} -> Selected {best_mask.sum()} features")

# ── 4. Calculate Stability Metrics ────────────────────────────────────────────
print("\n[3] Computing Pairwise Stability Metrics across 30 Resamples...")

# Pairwise Jaccard Similarity: J(A, B) = |A and B| / |A or B|
jaccard_scores = []
for i in range(N_BOOTSTRAP):
    for j in range(i + 1, N_BOOTSTRAP):
        s_i = set(np.where(selected_subsets[i] == 1)[0])
        s_j = set(np.where(selected_subsets[j] == 1)[0])
        inter = len(s_i.intersection(s_j))
        union = len(s_i.union(s_j))
        jaccard = inter / union if union > 0 else 1.0
        jaccard_scores.append(jaccard)

# Pairwise Spearman Rank Correlation of SHAP values
spearman_corrs = []
for i in range(N_BOOTSTRAP):
    for j in range(i + 1, N_BOOTSTRAP):
        corr, _ = spearmanr(shap_rankings[i], shap_rankings[j])
        spearman_corrs.append(corr)

jaccard_arr = np.array(jaccard_scores)
spearman_arr = np.array(spearman_corrs)

stability_summary = pd.DataFrame([
    {
        "Metric": "Jaccard Similarity (Subset Selection)",
        "Mean": round(np.mean(jaccard_arr), 4),
        "Std": round(np.std(jaccard_arr), 4),
        "Median": round(np.median(jaccard_arr), 4),
        "Min": round(np.min(jaccard_arr), 4),
        "Max": round(np.max(jaccard_arr), 4)
    },
    {
        "Metric": "Spearman Correlation (SHAP Attribution)",
        "Mean": round(np.mean(spearman_arr), 4),
        "Std": round(np.std(spearman_arr), 4),
        "Median": round(np.median(spearman_arr), 4),
        "Min": round(np.min(spearman_arr), 4),
        "Max": round(np.max(spearman_arr), 4)
    }
])

stability_summary.to_csv(rf"{CODES_PATH}\stability_metrics_summary.csv", index=False)
pd.DataFrame({"Pair": range(len(jaccard_scores)), "Jaccard": jaccard_scores, "Spearman_rho": spearman_corrs}).to_csv(rf"{CODES_PATH}\stability_pairwise_distributions.csv", index=False)

print("\n" + "=" * 70)
print("BOOTSTRAP STABILITY RESULTS (30 RESAMPLES):")
print("=" * 70)
print(stability_summary.to_string(index=False))
print("=" * 70)

print("\n[4] Frequency of Feature Selection Across 30 Resamples:")
feat_freq = np.mean(selected_subsets, axis=0) * 100
freq_df = pd.DataFrame({
    "Feature": predictor_cols,
    "Selection_Frequency_%": feat_freq
}).sort_values("Selection_Frequency_%", ascending=False)
freq_df.to_csv(rf"{CODES_PATH}\feature_selection_frequencies.csv", index=False)
print(freq_df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 7 COMPLETE: All Stability Analyses Finalized!")
print("=" * 60)
