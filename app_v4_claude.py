"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
Restyled with Apple Product Design Language & Explicit "Predict BI" Trigger
"""

import os
import json
import math
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

# ── Class cut-offs and gauge scale (cite the source of these cut-offs in your thesis) ──
EASY_MAX, HARD_MIN = 20.0, 40.0
GAUGE_MIN, GAUGE_MAX = 15.0, 60.0
BADGE = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}

def classify(v: float) -> str:
    return "Hard" if v > HARD_MIN else ("Medium" if v >= EASY_MAX else "Easy")

# ── Paths & Data Loading ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_pso_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "pso_ga_features.csv")
BOUNDS_PATH = os.path.join(BASE_DIR, "training_bounds.json")

@st.cache_resource
def load_model_and_explainer():
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    return m, shap.TreeExplainer(m)

@st.cache_data
def load_feature_list():
    return pd.read_csv(FEATURES_PATH)["0"].tolist()

@st.cache_data
def get_training_bounds():
    """Min / max / median of each input over the TRAINING rows only.
    Read from training_bounds.json when it exists. Otherwise computed once from
    X.npy + masks and saved to training_bounds.json, so X.npy (your full dataset)
    does not have to be shipped if you deploy the app online."""
    if os.path.exists(BOUNDS_PATH):
        with open(BOUNDS_PATH) as f:
            return json.load(f)
    X = np.load(os.path.join(BASE_DIR, "X.npy"))
    train_mask = np.load(os.path.join(BASE_DIR, "train_mask.npy"))
    pso_mask = np.load(os.path.join(BASE_DIR, "pso_ga_mask.npy"))
    Xtr = X[train_mask][:, pso_mask == 1]
    feats = load_feature_list()
    bounds = {feats[i]: {"min": float(Xtr[:, i].min()), "max": float(Xtr[:, i].max()),
                         "median": float(np.median(Xtr[:, i]))} for i in range(len(feats))}
    try:
        with open(BOUNDS_PATH, "w") as f:
            json.dump(bounds, f, indent=2)
    except OSError:
        pass
    return bounds

model, explainer = load_model_and_explainer()
feature_names = load_feature_list()
train_bounds = get_training_bounds()

absent = [f for f in feature_names if f not in train_bounds]
if absent:
    st.error(f"training_bounds.json has no entry for: {absent}. Delete it and rerun so it is rebuilt.")
    st.stop()

# ── Input definitions: label and step for each feature ──
INPUT_SPEC = {
    "dip_direction_deg": ("Discontinuity Dip Direction (°)", 1.0),
    "spacing_mode_mm": ("Joint Spacing Mode (mm)", 5.0),
    "dip_deg": ("Discontinuity Dip (°)", 1.0),
    "density_t_m3": ("Rock Density (t/m³)", 0.01),
    "persistence_m": ("Joint Persistence (m)", 0.1),
    "face_azimuth_deg": ("Face Azimuth (°)", 1.0),
    "rotation_pressure_bar": ("Rotation Pressure (bar)", 1.0),
    "northing": ("Northing Coordinate (m)", 1.0),
}
ordered_feats = [f for f in INPUT_SPEC if f in feature_names] + [f for f in feature_names if f not in INPUT_SPEC]

def widget_range(f):
    """Range allowed in the widgets: the training range, rounded INWARD to 2 decimals."""
    b = train_bounds[f]
    lo = math.ceil(round(b["min"] * 100, 6)) / 100
    hi = math.floor(round(b["max"] * 100, 6)) / 100
    return (lo, hi) if hi > lo else (round(b["min"], 2), round(b["min"], 2) + 0.01)

RANGES = {f: widget_range(f) for f in feature_names}

def clamp(f, v):
    lo, hi = RANGES[f]
    return round(float(min(max(v, lo), hi)), 2)

# ── Linked number box + slider (session state is the single source of truth) ──
for f in feature_names:
    b = train_bounds[f]
    start = clamp(f, b.get("median", (b["min"] + b["max"]) / 2))
    st.session_state.setdefault(f"num_{f}", start)
    st.session_state.setdefault(f"sld_{f}", start)

def sync_from_num(f):
    st.session_state[f"sld_{f}"] = st.session_state[f"num_{f}"]

def sync_from_sld(f):
    st.session_state[f"num_{f}"] = st.session_state[f"sld_{f}"]

PRESET_HARD = {"dip_direction_deg": 240.0, "spacing_mode_mm": 220.0, "dip_deg": 75.0, "density_t_m3": 2.56,
               "persistence_m": 6.0, "face_azimuth_deg": 140.0, "rotation_pressure_bar": 55.0, "northing": 1800.0}
PRESET_MEDIUM = {"dip_direction_deg": 90.0, "spacing_mode_mm": 80.0, "dip_deg": 35.0, "density_t_m3": 2.46,
                 "persistence_m": 1.5, "face_azimuth_deg": 60.0, "rotation_pressure_bar": 35.0, "northing": 800.0}

def apply_preset(values):
    for f, v in values.items():
        if f in RANGES:
            v = clamp(f, v)  # presets are kept inside the training range too
            st.session_state[f"num_{f}"] = v
            st.session_state[f"sld_{f}"] = v

# ── Batch prediction ──
def predict_batch(bdf: pd.DataFrame) -> pd.DataFrame:
    X = bdf[feature_names].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().sum()
    if bad.any():
        raise ValueError("Missing or non-numeric values in: " + ", ".join(f"{c} ({n} rows)" for c, n in bad[bad > 0].items()))
    out = bdf.copy()
    out["Predicted_BI"] = np.round(model.predict(X.values), 2)
    out["BI_Category"] = out["Predicted_BI"].apply(classify)
    mins = pd.Series({f: train_bounds[f]["min"] for f in feature_names})
    maxs = pd.Series({f: train_bounds[f]["max"] for f in feature_names})
    outside = (X < mins) | (X > maxs)
    out["Out_of_Range_Inputs"] = outside.apply(lambda r: ", ".join(r.index[r]), axis=1)
    return out

# ── Hero Section ──
render_html("""
<div class="apple-hero">
    <div class="apple-title">Rock Blastability AI</div>
    <div class="apple-subtitle">Explainable reduced-input Blastability Index (BI) estimation • Group 13, UMaT Mining Engineering</div>
</div>
""")

tab1, tab2, tab3 = st.tabs(["Prediction & XAI", "Batch Upload", "Performance & Stability"])

# ── TAB 1: Single Prediction & XAI ──
with tab1:
    col_input, col_output = st.columns([1, 1.15], gap="large")

    with col_input:
        render_html("""
        <div class="apple-card">
            <div class="apple-card-title">Geotechnical Parameters</div>
            <div class="apple-card-subtitle">8 optimal inputs selected by Hybrid PSO-GA. Type a value or drag the slider: they stay in sync, and both are limited to the training data range.</div>
        </div>
        """)

        pcol1, pcol2 = st.columns(2)
        pcol1.button("Hard Domain Preset", on_click=apply_preset, args=(PRESET_HARD,), use_container_width=True)
        pcol2.button("Medium Domain Preset", on_click=apply_preset, args=(PRESET_MEDIUM,), use_container_width=True)

        cols = st.columns(2)
        half = (len(ordered_feats) + 1) // 2
        for i, f in enumerate(ordered_feats):
            label, step = INPUT_SPEC.get(f, (f, round((RANGES[f][1] - RANGES[f][0]) / 100, 4)))
            lo, hi = RANGES[f]
            with cols[0 if i < half else 1]:
                st.number_input(label, min_value=lo, max_value=hi, step=step, format="%.2f",
                                key=f"num_{f}", on_change=sync_from_num, args=(f,),
                                help=f"Training data range: {lo:g} to {hi:g}")
                st.slider(label, min_value=lo, max_value=hi, step=step, format="%.2f",
                          key=f"sld_{f}", on_change=sync_from_sld, args=(f,), label_visibility="collapsed")

        st.caption("Northing is a site coordinate, not a rock property: use this model only at the mine it was trained on.")

        current_inputs = {f: float(st.session_state[f"num_{f}"]) for f in feature_names}

        st.write("")
        predict_clicked = st.button("Predict BI", type="primary", use_container_width=True)

        if predict_clicked:
            input_vector = np.array([[current_inputs[f] for f in feature_names]])
            with st.spinner("Computing prediction & SHAP attribution..."):
                pred_bi = float(model.predict(input_vector)[0])
                tree_preds = None
                if hasattr(model, "estimators_"):
                    x32 = input_vector.astype(np.float32)
                    tree_preds = np.array([t.predict(x32)[0] for t in model.estimators_])
                shap_vals = explainer.shap_values(input_vector)[0]
                base_val = float(np.ravel(explainer.expected_value)[0])
                st.session_state["prediction_result"] = {
                    "pred_bi": pred_bi,
                    "category": classify(pred_bi),
                    "tree_preds": tree_preds,
                    "shap_vals": shap_vals,
                    "base_val": base_val,
                    "input_snapshot": current_inputs.copy(),
                }

    with col_output:
        if "prediction_result" not in st.session_state:
            render_html("""
            <div class="apple-card" style="text-align: center; padding: 4.5rem 2rem;">
                <div style="font-size: 2.4rem; margin-bottom: 0.8rem; color: #007AFF;">⛏️</div>
                <div class="apple-card-title">Enter your measurements and press Predict BI</div>
                <div class="apple-card-subtitle" style="margin-bottom: 0; max-width: 440px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Set the 8 geotechnical field parameters on the left, then click <strong>Predict BI</strong> to get the estimate, its tree spread and a SHAP explanation.
                </div>
            </div>
            """)
        else:
            res = st.session_state["prediction_result"]
            pred_bi, category = res["pred_bi"], res["category"]
            shap_vals, base_val, snapshot = res["shap_vals"], res["base_val"], res["input_snapshot"]
            tree_preds = res["tree_preds"]

            if any(not np.isclose(current_inputs[k], snapshot[k], atol=1e-3) for k in current_inputs):
                render_html("""
                <div style="background-color: #FFF9E6; border: 1px solid #FFE58F; border-radius: 12px; padding: 10px 16px; margin-bottom: 14px; font-size: 0.86rem; color: #D46B08;">
                    ⚠️ Inputs changed since this prediction. Press <strong>Predict BI</strong> to update.
                </div>
                """)

            spread_html = ""
            if tree_preds is not None:
                sd = float(tree_preds.std())
                p5, p95 = (float(v) for v in np.percentile(tree_preds, [5, 95]))
                spread_html = (
                    f'<div style="font-size:0.86rem;color:#6E6E73;margin-top:10px;">'
                    f'Tree spread ±{sd:.2f} &nbsp;•&nbsp; 5–95% of the {len(tree_preds)} trees: {p5:.1f} – {p95:.1f}</div>'
                )
                if classify(p5) != classify(p95):
                    spread_html += (
                        '<div style="font-size:0.84rem;color:#D46B08;margin-top:6px;">'
                        'The spread crosses a class boundary, so treat the class as uncertain.</div>'
                    )

            pct = min(max((pred_bi - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN) * 100.0, 0.0), 100.0)
            guidance = (
                "<strong>High Blast Resistance:</strong> Tighter pattern or increased powder factor may be needed to limit oversize."
                if pred_bi > HARD_MIN else
                "<strong>Favorable Blastability:</strong> A standard powder factor is likely suitable, watching for backbreak and excessive fines."
            )

            render_html(f"""
            <div class="apple-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="metric-label">Estimated Lilly BI</div>
                        <div class="metric-value">{pred_bi:.2f}</div>
                    </div>
                    <div>
                        <span class="apple-badge {BADGE[category]}">{category} Rock Mass</span>
                    </div>
                </div>
                {spread_html}
                <div class="gauge-track">
                    <div class="gauge-fill" style="width: {pct:.1f}%;"></div>
                </div>
                <div class="gauge-labels">
                    <span>Easy (&lt;{EASY_MAX:g})</span>
                    <span>Medium ({EASY_MAX:g} – {HARD_MIN:g})</span>
                    <span>Hard (&gt;{HARD_MIN:g})</span>
                </div>
                <div class="guidance-box">
                    {guidance}
                    <div style="font-size:0.78rem;color:#6E6E73;margin-top:6px;">Indicative only. Confirm against site blast trials.</div>
                </div>
            </div>
            """)
            st.caption("Tree spread shows how much the individual trees disagree. It is not a formal confidence interval.")

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
        <div class="apple-card-subtitle">Upload a sheet (.xlsx or .csv) with the 8 input columns. Rows with inputs outside the training range are flagged.</div>
    </div>
    """)

    template = pd.DataFrame([{f: round(train_bounds[f].get("median", 0.0), 2) for f in feature_names}])
    st.download_button("Download blank template (.csv)", template.to_csv(index=False).encode("utf-8"),
                       file_name="BI_input_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload sheet", type=["xlsx", "csv"], label_visibility="collapsed")
    if uploaded_file:
        try:
            bdf = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file)
            missing_cols = [c for c in feature_names if c not in bdf.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                result = predict_batch(bdf)
                n_out = int((result["Out_of_Range_Inputs"] != "").sum())
                if n_out:
                    st.warning(f"{n_out} of {len(result)} rows have inputs outside the training range. "
                               "Their predictions are extrapolations and should not be trusted.")
                else:
                    st.success(f"All {len(result)} rows are inside the training range.")
                show_cols = ["Predicted_BI", "BI_Category", "Out_of_Range_Inputs"] + feature_names
                st.dataframe(result[show_cols], use_container_width=True, height=420)
                st.download_button("Download predictions (.csv)", result.to_csv(index=False).encode("utf-8"),
                                   file_name="Estimated_Blastability_Index.csv", mime="text/csv")
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
            {"Model": "Random Forest (FRFS)", "Inputs": 23, "R²": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Match": "93.25%"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption("Category match is the share of the 163 test records placed in the correct Easy/Medium/Hard class. "
                   "The mean baseline already scores 74.85%, and the three random forests differ by only a few records, "
                   "so read them as comparable accuracy with fewer inputs for the PSO-GA model.")

    with col_t2:
        render_html("""
        <div class="apple-card">
            <div class="apple-card-title">Engineering Stability</div>
            <div class="apple-card-subtitle">30 bootstrap resamples with replacement</div>

            <div style="margin-bottom: 1.2rem;">
                <div class="metric-label">SHAP Ranking Stability (Spearman ρ)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #007AFF;">0.8280 ± 0.055</div>
                <div style="font-size: 0.82rem; color: #6E6E73; margin-top: 2px;">
                    Suggests the feature rankings stay broadly similar across resamples (ρ = 1 would mean identical rankings).
                </div>
            </div>

            <div>
                <div class="metric-label">Selection Frequency (Dip Direction / Spacing)</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #007AFF;">100% / 96.7%</div>
                <div style="font-size: 0.82rem; color: #6E6E73; margin-top: 2px;">
                    Dip Direction was selected in 30/30 resamples; Spacing in 29/30.
                </div>
            </div>
        </div>
        """)
