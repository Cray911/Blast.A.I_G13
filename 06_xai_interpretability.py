"""
Step 6: Explainable AI (XAI) - SHAP, LIME, and Counterfactual Sensitivity Analysis
Group 13 | UMaT BSc Mining Engineering
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"
FIG_PATH = os.path.join(CODES_PATH, "figures")
os.makedirs(FIG_PATH, exist_ok=True)

print("=" * 60)
print("STEP 6: EXPLAINABLE AI (SHAP, LIME, COUNTERFACTUALS)")
print("=" * 60)

# Load data and trained PSO-GA model
X = np.load(rf"{CODES_PATH}\X.npy")
y = np.load(rf"{CODES_PATH}\y.npy")
train_mask = np.load(rf"{CODES_PATH}\train_mask.npy")
test_mask = np.load(rf"{CODES_PATH}\test_mask.npy")
pso_ga_mask = np.load(rf"{CODES_PATH}\pso_ga_mask.npy")
pso_features = pd.read_csv(rf"{CODES_PATH}\pso_ga_features.csv")["0"].tolist()

X_train_pso = X[train_mask][:, pso_ga_mask == 1]
y_train = y[train_mask]
X_test_pso = X[test_mask][:, pso_ga_mask == 1]
y_test = y[test_mask]

with open(rf"{CODES_PATH}\rf_pso_model.pkl", "rb") as f:
    rf_pso = pickle.load(f)

print(f"[1] Loaded PSO-GA model with {len(pso_features)} features: {pso_features}")

# ── 1. SHAP Analysis ───────────────────────────────────────────────────────────
print("\n[2] Computing SHAP values using TreeExplainer...")
explainer = shap.TreeExplainer(rf_pso)
shap_values_train = explainer.shap_values(X_train_pso)
shap_values_test = explainer.shap_values(X_test_pso)

# Global Importance (Mean Absolute SHAP)
mean_abs_shap = np.mean(np.abs(shap_values_train), axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": pso_features,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values("Mean_Abs_SHAP", ascending=False)
shap_importance_df.to_csv(rf"{CODES_PATH}\shap_feature_importance.csv", index=False)

print("\n  Global SHAP Feature Importance:")
print(shap_importance_df.to_string(index=False))

# Plot 1: SHAP Beeswarm Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_test, X_test_pso, feature_names=pso_features, show=False)
plt.title("SHAP Global Feature Attribution (Test Set) - Group 13", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "shap_summary_beeswarm.png"), dpi=300)
plt.close()

# Plot 2: SHAP Bar Plot
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values_test, X_test_pso, feature_names=pso_features, plot_type="bar", show=False)
plt.title("Mean |SHAP| Feature Impact on Blastability Index (BI)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "shap_importance_bar.png"), dpi=300)
plt.close()

# Plot 3: SHAP Dependence Plot for the top driver (dip_direction_deg)
idx_top = pso_features.index("dip_direction_deg")
plt.figure(figsize=(8, 5))
shap.dependence_plot(idx_top, shap_values_train, X_train_pso, feature_names=pso_features, show=False)
plt.title("SHAP Dependence Plot: Discontinuity Dip Direction vs BI Impact", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "shap_dependence_dip_direction.png"), dpi=300)
plt.close()

print("  Saved SHAP plots to figures/ directory.")

# ── 2. LIME Local Explanations ────────────────────────────────────────────────
print("\n[3] Generating LIME Local Surrogate Explanations...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_pso,
    feature_names=pso_features,
    mode="regression",
    random_state=42
)

# Explain an instance of "Hard" rock (high BI) and "Medium" rock
pred_test = rf_pso.predict(X_test_pso)
hard_idx = np.where(pred_test > 42)[0][0]
medium_idx = np.where(pred_test < 36)[0][0]

lime_results = []
for name, idx in [("Hard_Rock_Domain", hard_idx), ("Medium_Rock_Domain", medium_idx)]:
    exp = lime_explainer.explain_instance(X_test_pso[idx], rf_pso.predict, num_features=5)
    exp.save_to_file(os.path.join(FIG_PATH, f"lime_explanation_{name}.html"))
    lime_results.append({
        "Domain": name,
        "Actual_BI": round(y_test[idx], 2),
        "Predicted_BI": round(pred_test[idx], 2),
        "Top_Drivers": [f"{feat}: {weight:+.2f}" for feat, weight in exp.as_list()]
    })

pd.DataFrame(lime_results).to_csv(rf"{CODES_PATH}\lime_local_explanations.csv", index=False)
print("  Generated LIME explanations for representative Hard and Medium rock domains.")

# ── 3. Feasibility-Constrained Counterfactual Explanations ─────────────────────
print("\n[4] Generating Feasibility-Constrained Counterfactual Explanations...")
# Counterfactual question: For a Hard rock mass (e.g. BI = 44.5), what minimal parameter
# adjustments within observed physical bounds would reduce the estimated BI below 40 (Medium)?
target_hard_x = X_test_pso[hard_idx].copy()
original_bi = rf_pso.predict([target_hard_x])[0]

# Observed physical bounds from mine data
mins = np.min(X_train_pso, axis=0)
maxs = np.max(X_train_pso, axis=0)

# Perturbation search along key geotechnical variables (spacing, dip direction, persistence, density)
cf_candidates = []
for delta_spacing in np.linspace(0, 500, 20):
    for delta_dip_dir in np.linspace(-60, 60, 20):
        cand = target_hard_x.copy()
        # Joint spacing index
        sp_idx = pso_features.index("spacing_mode_mm")
        dd_idx = pso_features.index("dip_direction_deg")
        
        cand[sp_idx] = np.clip(cand[sp_idx] - delta_spacing, mins[sp_idx], maxs[sp_idx])
        cand[dd_idx] = np.clip(cand[dd_idx] + delta_dip_dir, mins[dd_idx], maxs[dd_idx])
        
        bi_cand = rf_pso.predict([cand])[0]
        if bi_cand <= 40.0:  # Successfully shifted category to Medium
            dist = np.linalg.norm((cand - target_hard_x) / (maxs - mins))
            cf_candidates.append({
                "New_BI": round(bi_cand, 2),
                "Category": "Medium",
                "Spacing_mm": round(cand[sp_idx], 1),
                "Dip_Dir_deg": round(cand[dd_idx], 1),
                "Distance": dist
            })

if cf_candidates:
    cf_df = pd.DataFrame(cf_candidates).sort_values("Distance").head(5)
    cf_df.to_csv(rf"{CODES_PATH}\counterfactual_sensitivity.csv", index=False)
    print("  Counterfactual Sensitivity Results (Top Solution):")
    print(f"    Original: Predicted BI = {original_bi:.2f} (Hard)")
    print(f"    Counterfactual: New BI = {cf_df.iloc[0]['New_BI']} ({cf_df.iloc[0]['Category']})")
    print(f"    Sensitivity shifts: Spacing -> {cf_df.iloc[0]['Spacing_mm']} mm, Dip Dir -> {cf_df.iloc[0]['Dip_Dir_deg']} deg")

print("\n" + "=" * 60)
print("STEP 6 COMPLETE: All XAI models, explanations, and figures saved.")
print("=" * 60)
