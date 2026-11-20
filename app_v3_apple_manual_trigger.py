"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
Restyled with Apple Product Design Language & Explicit "Predict BI" Trigger
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
    page_title="Blastability AI | Group 13 UMaT",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom Apple-Inspired CSS ──
st.markdown("""
<style>
    /* Hide Streamlit default chrome & menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {display: none;}

    /* Global typography and background */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #1D1D1F !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Page container padding */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }

    /* Hero Typography */
    .apple-hero {
        margin-bottom: 1.8rem;
    }
    .apple-title {
        font-size: 2.4rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #1D1D1F;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
    }
    .apple-subtitle {
        font-size: 1.02rem;
        font-weight: 400;
        color: #6E6E73;
        letter-spacing: -0.01em;
        line-height: 1.4;
        margin: 0;
    }

    /* Apple Cards */
    .apple-card {
        background-color: #F5F8FC;
        border: 1px solid #E5E5EA;
        border-radius: 18px;
        padding: 1.6rem 1.75rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    .apple-card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
    }
    .apple-card-subtitle {
        font-size: 0.88rem;
        color: #6E6E73;
        margin: 0 0 1.2rem 0;
    }

    /* iOS Segmented Control Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F5F8FC !important;
        border: 1px solid #E5E5EA !important;
        border-radius: 9999px !important;
        padding: 4px !important;
        gap: 4px !important;
        width: fit-content !important;
        margin-bottom: 1.8rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px !important;
        border-radius: 9999px !important;
        padding: 0 20px !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #6E6E73 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Primary & Secondary Pill Buttons */
    .stButton > button {
        border-radius: 9999px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:not([kind="primary"]) {
        border: 1px solid #E5E5EA !important;
        background-color: #FFFFFF !important;
        color: #007AFF !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border-color: #007AFF !important;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
    }
    
    /* Primary Action Button (Predict BI) */
    .stButton > button[kind="primary"] {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(0, 122, 255, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0062CC !important;
        box-shadow: 0 6px 18px rgba(0, 122, 255, 0.4) !important;
    }

    /* Sliders styling */
    .stSlider [data-baseweb="slider"] {
        color: #007AFF !important;
    }

    /* Metric Display */
    .metric-value {
        font-size: 3.2rem;
        font-weight: 700;
        color: #1D1D1F;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #6E6E73;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    /* Class Badge (Shades of Blue Only) */
    .apple-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        text-align: center;
    }
    .badge-easy {
        background-color: #EBF4FE;
        color: #007AFF;
        border: 1px solid #C3DEFD;
    }
    .badge-medium {
        background-color: #E0EFFF;
        color: #0062CC;
        border: 1px solid #B0D6FD;
    }
    .badge-hard {
        background-color: #D2E7FF;
        color: #0040A8;
        border: 1px solid #99C5FA;
    }

    /* Thin Apple Arc/Bar Gauge */
    .gauge-track {
        width: 100%;
        height: 8px;
        background-color: #E5E5EA;
        border-radius: 9999px;
        overflow: hidden;
        margin: 14px 0 6px 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #70B1F5 0%, #007AFF 50%, #0040A8 100%);
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.76rem;
        color: #6E6E73;
    }

    /* Operational guidance box */
    .guidance-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: #1D1D1F;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──
def render_html(html_str: str):
    cleaned = "\n".join(line.strip() for line in html_str.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

# ── Paths & Data Loading ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_pso_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "pso_ga_features.csv")

@st.cache_resource
def load_model_and_explainer():
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    exp = shap.TreeExplainer(m)
    return m, exp

@st.cache_data
def load_feature_list():
    df_feat = pd.read_csv(FEATURES_PATH)
    return df_feat["0"].tolist()

@st.cache_data
def get_training_bounds():
    X = np.load(os.path.join(BASE_DIR, "X.npy"))
    train_mask = np.load(os.path.join(BASE_DIR, "train_mask.npy"))
    pso_mask = np.load(os.path.join(BASE_DIR, "pso_ga_mask.npy"))
    X_train_pso = X[train_mask][:, pso_mask == 1]
    mins = np.min(X_train_pso, axis=0)
    maxs = np.max(X_train_pso, axis=0)
    feats = load_feature_list()
    return {feats[i]: {"min": round(float(mins[i]), 2), "max": round(float(maxs[i]), 2)} for i in range(len(feats))}

model, explainer = load_model_and_explainer()
feature_names = load_feature_list()
train_bounds = get_training_bounds()

# ── Hero Section ──
render_html("""
<div class="apple-hero">
    <div class="apple-title">Rock Blastability AI</div>
    <div class="apple-subtitle">Explainable reduced-input Blastability Index (BI) estimation • Group 13, UMaT Mining Engineering</div>
</div>
""")

# ── iOS Segmented Tabs ──
tab1, tab2, tab3 = st.tabs(["Prediction & XAI", "Batch Upload", "Performance & Stability"])

# ── TAB 1: Single Prediction & Live XAI ──
with tab1:
    col_input, col_output = st.columns([1, 1.15], gap="large")
    
    with col_input:
        render_html("""
        <div class="apple-card">
            <div class="apple-card-title">Geotechnical Parameters</div>
            <div class="apple-card-subtitle">8 optimal inputs selected by Hybrid PSO-GA</div>
        </div>
        """)
        
        # Presets
        pcol1, pcol2 = st.columns(2)
        preset_hard = pcol1.button("Hard Domain Preset")
        preset_med = pcol2.button("Medium Domain Preset")
        
        # Preset handling via session state
        if preset_hard:
            st.session_state["val_dd"] = 240.0
            st.session_state["val_sp"] = 220.0
            st.session_state["val_dip"] = 75.0
            st.session_state["val_dens"] = 2.56
            st.session_state["val_pers"] = 6.0
            st.session_state["val_az"] = 140.0
            st.session_state["val_rot"] = 55.0
            st.session_state["val_north"] = 1800.0
        elif preset_med:
            st.session_state["val_dd"] = 90.0
            st.session_state["val_sp"] = 80.0
            st.session_state["val_dip"] = 35.0
            st.session_state["val_dens"] = 2.46
            st.session_state["val_pers"] = 1.5
            st.session_state["val_az"] = 60.0
            st.session_state["val_rot"] = 35.0
            st.session_state["val_north"] = 800.0
            
        # Defaults within observed ranges
        val_dd = st.session_state.get("val_dd", 180.0)
        val_sp = st.session_state.get("val_sp", 120.0)
        val_dip = st.session_state.get("val_dip", 55.0)
        val_dens = st.session_state.get("val_dens", 2.50)
        val_pers = st.session_state.get("val_pers", 3.5)
        val_az = st.session_state.get("val_az", 95.0)
        val_rot = st.session_state.get("val_rot", 45.0)
        val_north = st.session_state.get("val_north", 1200.0)
        
        c1, c2 = st.columns(2)
        with c1:
            dip_dir = st.slider("Discontinuity Dip Direction (°)", 0.0, 360.0, float(val_dd), 1.0, key="in_dip_dir")
            spacing = st.slider("Joint Spacing Mode (mm)", 5.0, 500.0, float(val_sp), 5.0, key="in_spacing")
            dip = st.slider("Discontinuity Dip (°)", 0.0, 90.0, float(val_dip), 1.0, key="in_dip")
            density = st.slider("Rock Density (t/m³)", 2.0, 3.2, float(val_dens), 0.02, key="in_density")
            
        with c2:
            persistence = st.slider("Joint Persistence (m)", 0.1, 20.0, float(val_pers), 0.1, key="in_persistence")
            face_az = st.slider("Face Azimuth (°)", 0.0, 360.0, float(val_az), 1.0, key="in_face_az")
            rot_press = st.slider("Rotation Pressure (bar)", 15.0, 100.0, float(val_rot), 1.0, key="in_rot_press")
            northing = st.number_input("Northing Coordinate (m)", -500.0, 5000.0, float(val_north), 50.0, key="in_northing")

        # Current input dictionary
        current_inputs = {
            'northing': northing,
            'face_azimuth_deg': face_az,
            'dip_deg': dip,
            'dip_direction_deg': dip_dir,
            'spacing_mode_mm': spacing,
            'persistence_m': persistence,
            'density_t_m3': density,
            'rotation_pressure_bar': rot_press
        }
        
        st.write("")
        # Primary Action Button
        predict_clicked = st.button("Predict BI", type="primary", use_container_width=True)

        if predict_clicked:
            # Check for range violations against training data bounds
            range_warnings = []
            for f, val in current_inputs.items():
                f_min = train_bounds[f]["min"]
                f_max = train_bounds[f]["max"]
                if val < f_min or val > f_max:
                    range_warnings.append(
                        f"<strong>{f}</strong> ({val}) is outside training data range [{f_min}, {f_max}]."
                    )
            
            input_vector = np.array([[current_inputs[f] for f in feature_names]])
            
            with st.spinner("Computing prediction & SHAP attribution..."):
                pred_bi = float(model.predict(input_vector)[0])
                category = "Hard Rock Mass" if pred_bi > 40.0 else ("Medium Rock Mass" if pred_bi >= 20.0 else "Easy Rock Mass")
                shap_vals = explainer.shap_values(input_vector)[0]
                base_val = float(np.ravel(explainer.expected_value)[0])
                
                # Store in session_state
                st.session_state["prediction_result"] = {
                    "pred_bi": pred_bi,
                    "category": category,
                    "shap_vals": shap_vals,
                    "base_val": base_val,
                    "input_snapshot": current_inputs.copy(),
                    "warnings": range_warnings
                }

    with col_output:
        has_result = "prediction_result" in st.session_state
        
        if not has_result:
            # Empty state before first click
            render_html("""
            <div class="apple-card" style="text-align: center; padding: 4.5rem 2rem;">
                <div style="font-size: 2.4rem; margin-bottom: 0.8rem; color: #007AFF;">⛏️</div>
                <div class="apple-card-title">Enter your measurements and press Predict BI</div>
                <div class="apple-card-subtitle" style="margin-bottom: 0; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Adjust the 8 geotechnical field parameters on the left and click the <strong>Predict BI</strong> button to calculate blastability resistance and generate live SHAP explanations.
                </div>
            </div>
            """)
        else:
            res = st.session_state["prediction_result"]
            pred_bi = res["pred_bi"]
            category = res["category"]
            shap_vals = res["shap_vals"]
            base_val = res["base_val"]
            snapshot = res["input_snapshot"]
            warnings_list = res["warnings"]
            
            # Check if inputs changed since last prediction
            inputs_changed = any(
                not np.isclose(current_inputs[k], snapshot[k], atol=1e-3)
                for k in current_inputs
            )
            
            if inputs_changed:
                render_html("""
                <div style="background-color: #FFF9E6; border: 1px solid #FFE58F; border-radius: 12px; padding: 10px 16px; margin-bottom: 14px; font-size: 0.86rem; color: #D46B08; display: flex; align-items: center; gap: 8px;">
                    <span>⚠️</span>
                    <span>Inputs changed since this prediction. Press <strong>Predict BI</strong> to update.</span>
                </div>
                """)
                
            if warnings_list:
                for w_txt in warnings_list:
                    render_html(f"""
                    <div style="background-color: #FFF2F0; border: 1px solid #FFCCC7; border-radius: 12px; padding: 10px 16px; margin-bottom: 10px; font-size: 0.84rem; color: #CF1322;">
                        ⚠️ Domain Warning: {w_txt} Prediction may extrapolate.
                    </div>
                    """)

            # Prediction & Gauge Card
            badge_cls = "badge-hard" if pred_bi > 40.0 else ("badge-medium" if pred_bi >= 20.0 else "badge-easy")
            pct = min(max((pred_bi - 15.0) / 45.0 * 100.0, 0.0), 100.0)
            
            guidance = (
                "<strong>High Blast Resistance:</strong> Tighter pattern or increased powder factor recommended to limit oversize."
                if pred_bi > 40.0 else
                "<strong>Favorable Blastability:</strong> Standard powder factor recommended to prevent backbreak and excessive fines."
            )

            render_html(f"""
            <div class="apple-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="metric-label">Estimated Lilly BI</div>
                        <div class="metric-value">{pred_bi:.2f}</div>
                    </div>
                    <div>
                        <span class="apple-badge {badge_cls}">{category}</span>
                    </div>
                </div>
                
                <div class="gauge-track">
                    <div class="gauge-fill" style="width: {pct:.1f}%;"></div>
                </div>
                <div class="gauge-labels">
                    <span>Easy (&lt;20)</span>
                    <span>Medium (20 – 40)</span>
                    <span>Hard (&gt;40)</span>
                </div>
                
                <div class="guidance-box">
                    {guidance}
                </div>
            </div>
            """)

            # Separate Card for SHAP Explanation
            render_html("""
            <div class="apple-card">
                <div class="apple-card-title">Attribution &amp; Explanation</div>
                <div class="apple-card-subtitle">SHAP feature contributions pushing BI higher or lower</div>
            </div>
            """)

            fig, ax = plt.subplots(figsize=(6.8, 3.2), facecolor="#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            
            sort_idx = np.argsort(np.abs(shap_vals))
            sorted_feats = [feature_names[i] for i in sort_idx]
            sorted_shaps = [shap_vals[i] for i in sort_idx]
            colors = ['#007AFF' if s > 0 else '#B8D3F2' for s in sorted_shaps]
            
            y_pos = np.arange(len(sorted_feats))
            ax.barh(y_pos, sorted_shaps, color=colors, height=0.55, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_feats, fontsize=8.5, color='#1D1D1F')
            ax.axvline(0, color='#E5E5EA', linestyle='-', linewidth=1.0)
            
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#E5E5EA')
            ax.spines['bottom'].set_linewidth(0.8)
            
            ax.tick_params(axis='x', colors='#6E6E73', labelsize=8)
            ax.tick_params(axis='y', length=0)
            ax.set_title(f"Baseline Reference: {base_val:.2f}  ➔  Predicted Output: {pred_bi:.2f}", fontsize=9, color='#6E6E73', pad=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── TAB 2: Batch Predictor ──
with tab2:
    render_html("""
    <div class="apple-card">
        <div class="apple-card-title">Batch Sheet Estimation</div>
        <div class="apple-card-subtitle">Upload routine drill/blast mapping sheets (.xlsx or .csv) for bulk prediction</div>
    </div>
    """)
    
    uploaded_file = st.file_uploader("", type=["xlsx", "csv"], label_visibility="collapsed")
    if uploaded_file:
        try:
            bdf = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            missing_cols = [c for c in feature_names if c not in bdf.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                batch_X = bdf[feature_names].values
                bdf["Predicted_BI"] = np.round(model.predict(batch_X), 2)
                bdf["BI_Category"] = bdf["Predicted_BI"].apply(lambda v: "Hard" if v > 40 else ("Medium" if v >= 20 else "Easy"))
                
                st.dataframe(bdf[["Predicted_BI", "BI_Category"] + feature_names].head(12), use_container_width=True)
                
                csv_bytes = bdf.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions (.csv)",
                    data=csv_bytes,
                    file_name="Estimated_Blastability_Index.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Processing error: {e}")

# ── TAB 3: Research Metrics & Stability ──
with tab3:
    col_t1, col_t2 = st.columns([1.2, 1], gap="large")
    
    with col_t1:
        render_html("""
        <div class="apple-card">
            <div class="apple-card-title">Model Performance Comparison</div>
            <div class="apple-card-subtitle">Evaluated on 163 held-out bench records (unseen benches)</div>
        </div>
        """)
        
        comp_df = pd.DataFrame([
            {"Model": "Mean Baseline", "Inputs": 0, "R²": -0.0102, "RMSE": 4.5383, "MAE": 3.5889, "Category Match": "74.85%"},
            {"Model": "Linear Regression (Full)", "Inputs": 26, "R²": 0.5546, "RMSE": 3.0136, "MAE": 2.4477, "Category Match": "72.39%"},
            {"Model": "Random Forest (Full)", "Inputs": 26, "R²": 0.8692, "RMSE": 1.6331, "MAE": 1.0623, "Category Match": "93.87%"},
            {"Model": "Random Forest (PSO-GA)", "Inputs": 8, "R²": 0.8862, "RMSE": 1.5231, "MAE": 0.9489, "Category Match": "95.09%"},
            {"Model": "Random Forest (FRFS)", "Inputs": 23, "R²": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Match": "93.25%"}
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with col_t2:
        render_html("""
        <div class="apple-card">
            <div class="apple-card-title">Engineering Stability</div>
            <div class="apple-card-subtitle">30 bootstrap resamples with replacement</div>
            
            <div style="margin-bottom: 1.2rem;">
                <div class="metric-label">SHAP Ranking Stability (Spearman ρ)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #007AFF;">0.8280 ± 0.055</div>
                <div style="font-size: 0.82rem; color: #6E6E73; margin-top: 2px;">
                    Confirms that model explanations remain robust across field sampling variations.
                </div>
            </div>
            
            <div>
                <div class="metric-label">Core Subset Selection Overlap</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #007AFF;">100% / 96.7%</div>
                <div style="font-size: 0.82rem; color: #6E6E73; margin-top: 2px;">
                    Dip Direction selected in 30/30 resamples; Spacing selected in 29/30 resamples.
                </div>
            </div>
        </div>
        """)
