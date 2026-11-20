"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
Restyled with Apple Product Design Language (Clean, White & Blue, iOS Segmented Controls)
"""

import os
import pickle
import textwrap
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

    /* Pill Buttons */
    .stButton > button {
        border-radius: 9999px !important;
        border: 1px solid #E5E5EA !important;
        background-color: #FFFFFF !important;
        color: #007AFF !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    .stButton > button:hover {
        background-color: #007AFF !important;
        color: #FFFFFF !important;
        border-color: #007AFF !important;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
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

model, explainer = load_model_and_explainer()
feature_names = load_feature_list()

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
        
        val_dd = 180.0 if not preset_hard and not preset_med else (240.0 if preset_hard else 90.0)
        val_sp = 300.0 if not preset_hard and not preset_med else (650.0 if preset_hard else 120.0)
        val_dip = 55.0 if not preset_hard and not preset_med else (75.0 if preset_hard else 30.0)
        val_dens = 2.75 if not preset_hard and not preset_med else (2.95 if preset_hard else 2.60)
        val_pers = 3.5 if not preset_hard and not preset_med else (6.0 if preset_hard else 1.5)
        val_az = 95.0 if not preset_hard and not preset_med else (140.0 if preset_hard else 60.0)
        val_rot = 85.0 if not preset_hard and not preset_med else (120.0 if preset_hard else 60.0)
        val_north = 590000.0
        
        c1, c2 = st.columns(2)
        with c1:
            dip_dir = st.slider("Discontinuity Dip Direction (°)", 0.0, 360.0, float(val_dd), 1.0)
            spacing = st.slider("Joint Spacing Mode (mm)", 10.0, 1500.0, float(val_sp), 10.0)
            dip = st.slider("Discontinuity Dip (°)", 0.0, 90.0, float(val_dip), 1.0)
            density = st.slider("Rock Density (t/m³)", 2.0, 3.5, float(val_dens), 0.05)
            
        with c2:
            persistence = st.slider("Joint Persistence (m)", 0.1, 15.0, float(val_pers), 0.1)
            face_az = st.slider("Face Azimuth (°)", 0.0, 360.0, float(val_az), 1.0)
            rot_press = st.slider("Rotation Pressure (bar)", 20.0, 200.0, float(val_rot), 5.0)
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
        # Prediction & Gauge Card
        pred_bi = float(model.predict(input_vector)[0])
        category = "Hard Rock Mass" if pred_bi > 40.0 else ("Medium Rock Mass" if pred_bi >= 20.0 else "Easy Rock Mass")
        badge_cls = "badge-hard" if pred_bi > 40.0 else ("badge-medium" if pred_bi >= 20.0 else "badge-easy")
        
        # Calculate percentage for thin linear arc gauge
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
            <div class="apple-card-subtitle">Live SHAP feature contributions pushing BI higher or lower</div>
        </div>
        """)

        shap_vals = explainer.shap_values(input_vector)[0]
        base_val = float(np.ravel(explainer.expected_value)[0])

        # Styled clean Matplotlib figure matching Apple design
        fig, ax = plt.subplots(figsize=(6.8, 3.2), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        
        sort_idx = np.argsort(np.abs(shap_vals))
        sorted_feats = [feature_names[i] for i in sort_idx]
        sorted_shaps = [shap_vals[i] for i in sort_idx]
        
        # Apple blue for positive push, subtle grey-blue for negative push
        colors = ['#007AFF' if s > 0 else '#B8D3F2' for s in sorted_shaps]
        
        y_pos = np.arange(len(sorted_feats))
        ax.barh(y_pos, sorted_shaps, color=colors, height=0.55, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_feats, fontsize=8.5, color='#1D1D1F')
        ax.axvline(0, color='#E5E5EA', linestyle='-', linewidth=1.0)
        
        # Remove spines for clean borderless look
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
