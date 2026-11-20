"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── App Configuration ──
st.set_page_config(
    page_title="Rock Blastability AI | Group 13 UMaT",
    page_icon="⛏️",
    layout="wide"
)

# ── Paths & Loading ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_pso_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "pso_ga_features.csv")
DATA_PATH = os.path.join(BASE_DIR, "..", "Data", "BI-GROUP 13.xlsx")

@st.cache_resource
def load_model_and_explainer():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_data
def load_feature_list():
    df_feat = pd.read_csv(FEATURES_PATH)
    return df_feat["0"].tolist()

model, explainer = load_model_and_explainer()
feature_names = load_feature_list()

# ── Styling ──
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="main-title">⛏️ Explainable AI Blastability Index (BI) Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">University of Mines and Technology (UMaT) | Department of Mining Engineering — Group 13<br><i>Supervisor: Dr. Festus Kunkyin-Saadaari</i></div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction & XAI", "📂 Batch Sheet Predictor", "📊 Research Metrics & Stability"])

# ── TAB 1: Single Prediction ──
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("1. Geotechnical Field Inputs")
        st.write("Adjust the 8 optimal parameters selected by **Hybrid PSO-GA**:")
        
        # Preset buttons
        pcol1, pcol2 = st.columns(2)
        preset_hard = pcol1.button("📌 Load Hard Rock Domain Preset")
        preset_med = pcol2.button("📌 Load Medium Rock Domain Preset")
        
        # Defaults
        val_dd = 180.0 if not preset_hard and not preset_med else (240.0 if preset_hard else 90.0)
        val_sp = 300.0 if not preset_hard and not preset_med else (650.0 if preset_hard else 120.0)
        val_dip = 55.0 if not preset_hard and not preset_med else (75.0 if preset_hard else 30.0)
        val_dens = 2.75 if not preset_hard and not preset_med else (2.95 if preset_hard else 2.60)
        val_pers = 3.5 if not preset_hard and not preset_med else (6.0 if preset_hard else 1.5)
        val_az = 95.0 if not preset_hard and not preset_med else (140.0 if preset_hard else 60.0)
        val_rot = 85.0 if not preset_hard and not preset_med else (120.0 if preset_hard else 60.0)
        val_north = 590000.0
        
        # Sliders
        c1, c2 = st.columns(2)
        with c1:
            dip_dir = st.slider("Discontinuity Dip Direction (°)", 0.0, 360.0, float(val_dd), 1.0, help="Azimuth of maximum dip")
            spacing = st.slider("Joint Spacing Mode (mm)", 10.0, 1500.0, float(val_sp), 10.0, help="Dominant joint spacing")
            dip = st.slider("Discontinuity Dip (°)", 0.0, 90.0, float(val_dip), 1.0, help="Inclination of joint planes")
            density = st.slider("Rock Density (t/m³)", 2.0, 3.5, float(val_dens), 0.05, help="Specific gravity measure")
            
        with c2:
            persistence = st.slider("Joint Persistence (m)", 0.1, 15.0, float(val_pers), 0.1, help="Trace length of joint planes")
            face_az = st.slider("Face Azimuth (°)", 0.0, 360.0, float(val_az), 1.0, help="Bench strike orientation")
            rot_press = st.slider("Rotation Pressure (bar)", 20.0, 200.0, float(val_rot), 5.0, help="MWD drilling response")
            northing = st.number_input("Northing Coordinate (m)", 500000.0, 700000.0, float(val_north), 1000.0)

    input_dict = {
        'northing': northing,
        'face_azimuth_deg': face_az,
        'dip_deg': dip,
        'dip_direction_deg': dip_dir,
        'spacing_mode_mm': spacing,
        'persistence_m': persistence,
        'density_t_m3': density,
        'rotation_pressure_bar': rot_press
    }
    input_vector = np.array([[input_dict[f] for f in feature_names]])

    with col_output:
        st.subheader("2. AI Blastability Estimation")
        
        # Predict
        pred_bi = float(model.predict(input_vector)[0])
        category = "Hard Rock Mass" if pred_bi > 40.0 else ("Medium Rock Mass" if pred_bi >= 20.0 else "Easy Rock Mass")
        color = "#DC2626" if pred_bi > 40.0 else ("#059669" if pred_bi >= 20.0 else "#2563EB")
        
        mcol1, mcol2 = st.columns([1, 1.2])
        with mcol1:
            st.metric(label="Predicted Lilly BI", value=f"{pred_bi:.2f}")
        with mcol2:
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 12px 18px; border-radius: 8px; text-align: center; font-size: 1.1rem; font-weight: 600; margin-top: 5px;">
                {category}
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        # Lilly Scale Progress Meter
        progress_val = min(max((pred_bi - 15.0) / 45.0, 0.0), 1.0)
        st.progress(progress_val)
        st.caption(f"Scale: Easy (<20) | Medium (20 - 40) | Hard (>40) — Model Output: {pred_bi:.2f}")
        
        # Blasting operational note
        if pred_bi > 40.0:
            st.warning("⚠️ **Blasting Guidance:** High blastability index indicates substantial blasting resistance. Increased powder factor ($PF$) or tighter pattern spacing recommended to minimize oversize and secondary breaking.")
        else:
            st.success("✅ **Blasting Guidance:** Favorable blastability. Standard powder factor recommended to prevent over-pulverization, excessive ground vibration, and backbreak.")

        st.write("---")
        st.subheader("3. Live SHAP Attribution (Why did the AI predict this?)")
        
        # Live SHAP values
        shap_vals = explainer.shap_values(input_vector)[0]
        base_val = float(np.ravel(explainer.expected_value)[0])
        
        # Plot SHAP waterfall bar
        fig, ax = plt.subplots(figsize=(7, 3.5))
        y_pos = np.arange(len(feature_names))
        # Sort by absolute SHAP impact
        sort_idx = np.argsort(np.abs(shap_vals))
        
        sorted_feats = [feature_names[i] for i in sort_idx]
        sorted_shaps = [shap_vals[i] for i in sort_idx]
        colors = ['#DC2626' if s > 0 else '#2563EB' for s in sorted_shaps]
        
        ax.barh(y_pos, sorted_shaps, color=colors, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_feats, fontsize=9)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Blastability Index (BI)", fontsize=9)
        ax.set_title(f"Base Value: {base_val:.2f} ➔ Final BI: {pred_bi:.2f}", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 2: Batch Predictor ──
with tab2:
    st.subheader("📂 Batch Prediction from Excel Sheet")
    st.write("Upload a routine bench drill/mapping Excel file to estimate Blastability Index across all panels simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                bdf = pd.read_csv(uploaded_file)
            else:
                bdf = pd.read_excel(uploaded_file)
                
            st.write(f"Loaded {len(bdf)} rows.")
            missing_cols = [c for c in feature_names if c not in bdf.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in sheet: {missing_cols}")
            else:
                batch_X = bdf[feature_names].values
                bdf["Predicted_BI"] = np.round(model.predict(batch_X), 2)
                bdf["BI_Category"] = bdf["Predicted_BI"].apply(lambda v: "Hard" if v > 40 else ("Medium" if v >= 20 else "Easy"))
                
                st.success("✅ Prediction complete!")
                st.dataframe(bdf[["Predicted_BI", "BI_Category"] + feature_names].head(15))
                
                # Download button
                csv_bytes = bdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_bytes,
                    file_name="Estimated_Blastability_Index_Group13.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ── TAB 3: Research Metrics & Stability ──
with tab3:
    st.subheader("📊 Thesis Experimental Benchmark")
    st.write("Summary of held-out test evaluations (163 untouched bench records) and 30-resample stability analysis:")
    
    comp_df = pd.DataFrame([
        {"Model": "Mean Baseline", "Inputs": 0, "R²": -0.0102, "RMSE": 4.5383, "MAE": 3.5889, "Category Agreement": "74.85%"},
        {"Model": "Linear Regression (Full)", "Inputs": 26, "R²": 0.5546, "RMSE": 3.0136, "MAE": 2.4477, "Category Agreement": "72.39%"},
        {"Model": "Random Forest (Full)", "Inputs": 26, "R²": 0.8692, "RMSE": 1.6331, "MAE": 1.0623, "Category Agreement": "93.87%"},
        {"Model": "Random Forest (PSO-GA)", "Inputs": 8, "R²": 0.8862, "RMSE": 1.5231, "MAE": 0.9489, "Category Agreement": "95.09%"},
        {"Model": "Random Forest (FRFS)", "Inputs": 23, "R²": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Agreement": "93.25%"}
    ])
    st.table(comp_df)
    
    st.subheader("🛡️ 30-Resample Engineering Stability (Sub-question 2)")
    scol1, scol2 = st.columns(2)
    with scol1:
        st.metric("SHAP Ranking Stability (Spearman ρ)", "0.8280 ± 0.055", help="Correlations across 30 bootstrap resamples")
        st.caption("Values above 0.80 indicate that the model's explanations are dependable and resilient to sampling variability.")
    with scol2:
        st.metric("Feature Subset Stability (Jaccard J)", "0.3049", help="Subset overlap across resamples")
        st.caption("Core structural features (Dip Direction: 100%, Spacing: 96.7%, Dip: 73.3%) were selected nearly unanimously.")
