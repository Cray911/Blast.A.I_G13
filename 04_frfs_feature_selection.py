"""
Step 4: Dependency-based Fuzzy-Rough Feature Selection (FRFS)
Group 13 | UMaT BSc Mining Engineering
Reference: Shen and Jensen (2004)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"

print("=" * 60)
print("STEP 4: FUZZY-ROUGH FEATURE SELECTION (FRFS)")
print("=" * 60)

# Load data
X = np.load(rf"{CODES_PATH}\X.npy")
y = np.load(rf"{CODES_PATH}\y.npy")
train_mask = np.load(rf"{CODES_PATH}\train_mask.npy")
predictor_cols = pd.read_csv(rf"{CODES_PATH}\predictor_cols.csv")["0"].tolist()

X_train, y_train = X[train_mask], y[train_mask]
n_samples, n_features = X_train.shape

print(f"[1] Loaded training data: {n_samples} samples, {n_features} features.")

# Normalize continuous features to [0, 1] for fuzzy similarity
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_train)

# Decision classes: BI categories (Medium vs Hard)
# In Lilly (1986): Medium (20-40), Hard (>40)
decision_classes = np.array([1 if val > 40 else 0 for val in y_train])
unique_classes = np.unique(decision_classes)
print(f"[2] Decision classes defined: Class 0 (Medium: <=40): {(decision_classes == 0).sum()} samples | Class 1 (Hard: >40): {(decision_classes == 1).sum()} samples")

# Fuzzy similarity relation for attribute a between sample x and y:
# R_a(x, y) = max(0, 1 - |x_a - y_a| / (4 * std_a)) or standard Gaussian similarity:
# R_a(x, y) = exp( - (x_a - y_a)^2 / (2 * sigma_a^2) )
# For computational efficiency and numerical stability, use variance-scaled Euclidean fuzzy similarity:
stds = np.std(X_norm, axis=0)
stds[stds == 0] = 1.0  # Avoid div by zero

# Precompute pairwise feature distance matrices for fast greedy evaluation
print("[3] Precomputing fuzzy similarity kernels...")
# Subsample 150 points for tractability if needed, or compute exact on all 385 points (385x385 = 148k pairs, fits in memory)
N = n_samples
diff_matrices = []
for a in range(n_features):
    diff = np.abs(X_norm[:, a, None] - X_norm[:, a])
    # Triangular / Lukasiewicz fuzzy similarity: R_a(x,y) = max(0, 1 - |x - y| / (3*std))
    sim = np.maximum(0, 1.0 - diff / (3.0 * stds[a]))
    diff_matrices.append(sim)

# Crisp Equivalence classes on Decision Attribute D:
# D_eq[x, y] = 1 if decision_classes[x] == decision_classes[y] else 0
D_eq = (decision_classes[:, None] == decision_classes).astype(float)

# Function to compute fuzzy dependency degree gamma_B(D)
# gamma_B(D) = (1/N) * sum_x (POS_B(D)(x))
# where POS_B(D)(x) = max_C [ mu_B_lower(C)(x) ]
# For crisp decision classes C: mu_B_lower(C)(x) = inf_y [ I(R_B(x,y), C(y)) ]
# Using Lukasiewicz / standard fuzzy implicator: I(a, b) = min(1, 1 - a + b)
def compute_fuzzy_dependency(subset_indices):
    if len(subset_indices) == 0:
        return 0.0
    
    # Combined fuzzy relation R_B = T-norm (min) over selected attributes
    R_B = diff_matrices[subset_indices[0]].copy()
    for idx in subset_indices[1:]:
        R_B = np.minimum(R_B, diff_matrices[idx])
        
    # For each class c, compute lower approximation:
    # lower_c(x) = min_y [ 1 - R_B(x, y) + (1 if class(y)==c else 0) ]
    pos_x = np.zeros(N)
    for c in unique_classes:
        C_y = (decision_classes == c).astype(float) # shape (N,)
        # Implicator matrix: I(x, y) = 1 - R_B(x, y) + C_y[y]
        impl = np.clip(1.0 - R_B + C_y[None, :], 0.0, 1.0)
        lower_c = np.min(impl, axis=1) # inf over y
        pos_x = np.maximum(pos_x, lower_c)
        
    gamma = np.mean(pos_x)
    return gamma

# Forward Greedy Reduct Search (QuickReduct)
print("\n[4] Running Fuzzy-Rough QuickReduct forward selection...")
selected = []
gamma_current = 0.0
gamma_history = []

for step in range(n_features):
    best_candidate = -1
    best_gamma = gamma_current
    
    for candidate in range(n_features):
        if candidate not in selected:
            candidate_subset = selected + [candidate]
            gamma = compute_fuzzy_dependency(candidate_subset)
            if gamma > best_gamma:
                best_gamma = gamma
                best_candidate = candidate
                
    # Check stopping criterion: if improvement is negligible (< 0.005)
    if best_candidate == -1 or (best_gamma - gamma_current) < 0.005:
        print(f"    Stopping: dependency degree reached plateau at step {step} (gamma = {gamma_current:.4f})")
        break
        
    selected.append(best_candidate)
    gamma_current = best_gamma
    print(f"    Step {step+1}: Added '{predictor_cols[best_candidate]}' -> Fuzzy Dependency gamma = {gamma_current:.4f}")
    gamma_history.append({"step": step+1, "feature": predictor_cols[best_candidate], "gamma": gamma_current})

selected_features_frfs = [predictor_cols[i] for i in selected]
frfs_mask = np.zeros(n_features, dtype=int)
frfs_mask[selected] = 1

print("\n[5] FRFS Complete!")
print(f"    Final Features Selected ({len(selected_features_frfs)}): {selected_features_frfs}")
print(f"    Reduction: {n_features} -> {len(selected_features_frfs)} features ({(1 - len(selected_features_frfs)/n_features)*100:.1f}% reduction)")

# Save FRFS features
np.save(rf"{CODES_PATH}\frfs_mask.npy", frfs_mask)
pd.Series(selected_features_frfs).to_csv(rf"{CODES_PATH}\frfs_features.csv", index=False)
pd.DataFrame(gamma_history).to_csv(rf"{CODES_PATH}\frfs_history.csv", index=False)

print("\n" + "=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
