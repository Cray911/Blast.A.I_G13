"""
Step 3: Hybrid PSO-GA Feature Selection
Group 13 | UMaT BSc Mining Engineering
Reference: Ghamisi and Benediktsson (2015)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

CODES_PATH = r"C:\Users\Workstation X\Downloads\Project Work\Group 13-27\02. Methods\Codes"

print("=" * 60)
print("STEP 3: HYBRID PSO-GA FEATURE SELECTION")
print("=" * 60)

# Load prepared data
X = np.load(rf"{CODES_PATH}\X.npy")
y = np.load(rf"{CODES_PATH}\y.npy")
groups = np.load(rf"{CODES_PATH}\groups.npy")
train_mask = np.load(rf"{CODES_PATH}\train_mask.npy")
test_mask = np.load(rf"{CODES_PATH}\test_mask.npy")
predictor_cols = pd.read_csv(rf"{CODES_PATH}\predictor_cols.csv")["0"].tolist()

X_train, y_train = X[train_mask], y[train_mask]
groups_train = groups[train_mask]
n_features = X_train.shape[1]

print(f"[1] Loaded training data: {X_train.shape[0]} samples, {n_features} features.")

# Fitness Function
# Fitness = w * (RMSE_cv / baseline_RMSE) + (1 - w) * (num_selected / total_features)
# We want to MINIMIZE fitness
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_train, y_train, groups=groups_train))

def evaluate_subset(binary_mask):
    selected_indices = np.where(binary_mask == 1)[0]
    if len(selected_indices) == 0:
        return 999.0  # Heavy penalty for selecting 0 features
    
    rmses = []
    for train_idx, val_idx in splits:
        X_tr, y_tr = X_train[train_idx][:, selected_indices], y_train[train_idx]
        X_val, y_val = X_train[val_idx][:, selected_indices], y_train[val_idx]
        
        # Use fast RF for evaluation
        rf = RandomForestRegressor(n_estimators=60, max_depth=10, max_features=0.5, n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        
    mean_rmse = np.mean(rmses)
    feature_ratio = len(selected_indices) / n_features
    # Weight: 85% prediction error, 15% model parsimony
    fitness = 0.85 * mean_rmse + 0.15 * (feature_ratio * 5.0)
    return fitness, mean_rmse

# PSO-GA Parameters
pop_size = 25
max_iter = 20
w = 0.7       # Inertia weight
c1 = 1.5      # Cognitive component
c2 = 1.5      # Social component
crossover_rate = 0.8
mutation_rate = 0.05

# Initialize population
np.random.seed(42)
positions = np.random.choice([0, 1], size=(pop_size, n_features), p=[0.5, 0.5])
# Ensure at least one feature is active
for p in positions:
    if p.sum() == 0:
        p[np.random.randint(0, n_features)] = 1

velocities = np.random.uniform(-1, 1, size=(pop_size, n_features))
pbest_positions = positions.copy()
pbest_fitness = np.zeros(pop_size)
pbest_rmse = np.zeros(pop_size)

for i in range(pop_size):
    fit, r = evaluate_subset(positions[i])
    pbest_fitness[i] = fit
    pbest_rmse[i] = r

gbest_idx = np.argmin(pbest_fitness)
gbest_position = pbest_positions[gbest_idx].copy()
gbest_fitness = pbest_fitness[gbest_idx]
gbest_rmse = pbest_rmse[gbest_idx]

print(f"[2] Initialized swarm of {pop_size} particles. Initial best fitness: {gbest_fitness:.4f} (RMSE: {gbest_rmse:.4f}, Features: {gbest_position.sum()})")

print("\n[3] Running Hybrid PSO-GA iterations...")
history = []

for it in range(1, max_iter + 1):
    for i in range(pop_size):
        # 1. PSO Velocity and Position update
        r1 = np.random.rand(n_features)
        r2 = np.random.rand(n_features)
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (gbest_position - positions[i])
        
        # Sigmoid transfer function for binary PSO
        s_v = 1.0 / (1.0 + np.exp(-velocities[i]))
        positions[i] = np.where(np.random.rand(n_features) < s_v, 1, 0)
        
        # 2. GA operators: Crossover with gbest
        if np.random.rand() < crossover_rate:
            cross_pt = np.random.randint(1, n_features)
            child = np.concatenate([positions[i][:cross_pt], gbest_position[cross_pt:]])
            positions[i] = child
            
        # 3. GA Mutation
        for bit in range(n_features):
            if np.random.rand() < mutation_rate:
                positions[i][bit] = 1 - positions[i][bit]
                
        # Guard against zero features
        if positions[i].sum() == 0:
            positions[i][np.random.randint(0, n_features)] = 1
            
        # Evaluate
        fit, r = evaluate_subset(positions[i])
        if fit < pbest_fitness[i]:
            pbest_fitness[i] = fit
            pbest_rmse[i] = r
            pbest_positions[i] = positions[i].copy()
            
    # Update global best
    min_idx = np.argmin(pbest_fitness)
    if pbest_fitness[min_idx] < gbest_fitness:
        gbest_fitness = pbest_fitness[min_idx]
        gbest_rmse = pbest_rmse[min_idx]
        gbest_position = pbest_positions[min_idx].copy()
        
    num_sel = gbest_position.sum()
    print(f"    Iter {it:02d}/{max_iter} -> Best Fitness: {gbest_fitness:.4f} | CV RMSE: {gbest_rmse:.4f} | Selected: {num_sel}/{n_features}")
    history.append({"iter": it, "fitness": gbest_fitness, "rmse": gbest_rmse, "n_features": num_sel})

selected_indices = np.where(gbest_position == 1)[0]
selected_features = [predictor_cols[i] for i in selected_indices]

print("\n[4] PSO-GA Optimization Complete!")
print(f"    Final Features Selected ({len(selected_features)}): {selected_features}")
print(f"    Reduction: {n_features} -> {len(selected_features)} features ({(1 - len(selected_features)/n_features)*100:.1f}% reduction)")

# Save selected features
np.save(rf"{CODES_PATH}\pso_ga_mask.npy", gbest_position)
pd.Series(selected_features).to_csv(rf"{CODES_PATH}\pso_ga_features.csv", index=False)
pd.DataFrame(history).to_csv(rf"{CODES_PATH}\pso_ga_history.csv", index=False)

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
