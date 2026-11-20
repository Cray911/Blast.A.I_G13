"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
Scrollable Bento Dashboard Layout
Apple-inspired styling, dark mode toggle, no emojis, borderless rounded cards,
full-screen 1.5s loading overlay.
"""
import os
import json
import math
import pickle
import time
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Dark Mode State ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
DARK = st.session_state.dark_mode

# ── Theme Palette ──
THEME_LIGHT = {
    "bg": "#FFFFFF", "text": "#1D1D1F", "text_secondary": "#6E6E73",
    "card_bg": "#F5F8FC", "card_bg_elev": "#FFFFFF",
    "accent": "#007AFF", "accent_hover": "#0062CC",
    "track": "#E5E5EA",
    "badge_easy_bg": "#EBF4FE", "badge_easy_fg": "#007AFF",
    "badge_med_bg": "#E0EFFF",  "badge_med_fg": "#0062CC",
    "badge_hard_bg": "#D2E7FF", "badge_hard_fg": "#0040A8",
    "info_bg": "#EEF4FF",    "info_fg": "#1D1D1F",
    "success_bg": "#E9F7EE", "success_fg": "#1E6B3A",
    "warning_bg": "#FFF9E6", "warning_fg": "#8A5A00",
    "error_bg": "#FFECEC",   "error_fg": "#8A1F1F",
    "chart_bg": "#FFFFFF",   "chart_grid": "#E5E5EA",
    "shap_pos": "#007AFF",   "shap_neg": "#B8D3F2",
    "overlay_bg": "rgba(255, 255, 255, 0.7)",
}
THEME_DARK = {
    "bg": "#000000", "text": "#F5F5F7", "text_secondary": "#98989D",
    "card_bg": "#1C1C1E", "card_bg_elev": "#2C2C2E",
    "accent": "#0A84FF", "accent_hover": "#409CFF",
    "track": "#38383A",
    "badge_easy_bg": "#0A2A4A", "badge_easy_fg": "#64B5F6",
    "badge_med_bg": "#0D3454",  "badge_med_fg": "#4A9EFF",
    "badge_hard_bg": "#123E66", "badge_hard_fg": "#90C8FF",
    "info_bg": "#0F1E33",    "info_fg": "#D6E4F5",
    "success_bg": "#0E2A1B", "success_fg": "#9BE3B6",
    "warning_bg": "#332400", "warning_fg": "#FFCE6B",
    "error_bg": "#3A1414",   "error_fg": "#FF9E9E",
    "chart_bg": "#1C1C1E",   "chart_grid": "#38383A",
    "shap_pos": "#0A84FF",   "shap_neg": "#4A5A6A",
    "overlay_bg": "rgba(0, 0, 0, 0.7)",
}
C = THEME_DARK if DARK else THEME_LIGHT

# ── CSS: theme variables + main stylesheet ──
theme_vars = f""":root {{
--bg: {C['bg']};
--text: {C['text']};
--text-secondary: {C['text_secondary']};
--card-bg: {C['card_bg']};
--card-bg-elev: {C['card_bg_elev']};
--accent: {C['accent']};
--accent-hover: {C['accent_hover']};
--track: {C['track']};
--badge-easy-bg: {C['badge_easy_bg']};
--badge-easy-fg: {C['badge_easy_fg']};
--badge-med-bg: {C['badge_med_bg']};
--badge-med-fg: {C['badge_med_fg']};
--badge-hard-bg: {C['badge_hard_bg']};
--badge-hard-fg: {C['badge_hard_fg']};
--overlay-bg: {C['overlay_bg']};
}}"""

MAIN_CSS = """
/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden; display: none; }

/* Global typography + background */
html, body, [class*="css"], .stApp {
    font-family: -apple-system, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
}

/* Hero */
.apple-hero { margin-bottom: 1.6rem; }
.apple-title {
    font-size: 2.4rem; font-weight: 600; letter-spacing: -0.03em;
    color: var(--text); line-height: 1.15; margin: 0 0 0.4rem 0;
}
.apple-subtitle {
    font-size: 1.02rem; font-weight: 400; color: var(--text-secondary);
    letter-spacing: -0.01em; line-height: 1.4; margin: 0;
}

/* Cards */
.apple-card {
    background-color: var(--card-bg);
    border: none;
    border-radius: 18px;
    padding: 1.6rem 1.75rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}
.apple-card-title {
    font-size: 1.15rem; font-weight: 600; color: var(--text);
    letter-spacing: -0.02em; margin: 0 0 0.25rem 0;
}
.apple-card-subtitle {
    font-size: 0.88rem; color: var(--text-secondary);
    margin: 0 0 1.2rem 0;
}

/* ── Bento Micro-Cards for Inputs ── */
.micro-card {
    background-color: var(--card-bg);
    border: none;
    border-radius: 16px;
    padding: 1rem 1.1rem 0.6rem 1.1rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
    height: 100%;
}
.micro-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.micro-unit {
    font-size: 0.72rem;
    color: var(--text-secondary);
    opacity: 0.7;
    margin-bottom: 0.5rem;
}

/* Section headers */
.section-header {
    margin: 2.4rem 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--track);
}
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.section-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin: 0;
}

/* Buttons */
.stButton > button {
    border-radius: 9999px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
    border: none !important;
}
.stButton > button:not([kind="primary"]) {
    background-color: var(--card-bg-elev) !important;
    color: var(--accent) !important;
}
.stButton > button:not([kind="primary"]):hover {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
}
.stButton > button[kind="primary"] {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    padding: 0.85rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--accent-hover) !important;
    box-shadow: 0 6px 18px rgba(0, 122, 255, 0.4) !important;
}

/* Download button */
.stDownloadButton > button {
    border-radius: 9999px !important;
    border: none !important;
    background-color: var(--card-bg-elev) !important;
    color: var(--accent) !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.2rem !important;
}
.stDownloadButton > button:hover {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
}

/* Inputs */
.stNumberInput > div, .stNumberInput > div > div,
.stTextInput > div > div, .stSelectbox > div > div,
.stTextArea > div > div {
    border-radius: 12px !important;
    border: none !important;
    background-color: var(--card-bg-elev) !important;
}
.stNumberInput input, .stTextInput input {
    color: var(--text) !important;
}

/* File uploader */
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
    border-radius: 14px !important;
    border: none !important;
    background-color: var(--card-bg-elev) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 9999px !important;
    border: none !important;
}

/* Dataframe */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] > div,
[data-testid="stDataFrameResizable"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: none !important;
    outline: none !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: none !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    border: none !important;
}

/* Metric display */
.metric-value {
    font-size: 4.2rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.04em; line-height: 1;
}
.metric-label {
    font-size: 0.82rem; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;
}

/* Badges */
.apple-badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    text-align: center;
    border: none;
}
.badge-easy   { background-color: var(--badge-easy-bg); color: var(--badge-easy-fg); }
.badge-medium { background-color: var(--badge-med-bg);  color: var(--badge-med-fg); }
.badge-hard   { background-color: var(--badge-hard-bg); color: var(--badge-hard-fg); }

/* Gauge */
.gauge-track {
    width: 100%; height: 10px; background-color: var(--track);
    border-radius: 9999px; overflow: hidden; margin: 18px 0 8px 0;
}
.gauge-fill {
    height: 100%; border-radius: 9999px;
    background: linear-gradient(90deg, #70B1F5 0%, #007AFF 50%, #0040A8 100%);
}
.gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: var(--text-secondary);
}

/* Guidance box */
.guidance-box {
    background-color: var(--card-bg-elev);
    border: none;
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 18px;
    font-size: 0.92rem;
    color: var(--text);
    line-height: 1.5;
}

/* ── FULL SCREEN LOADING OVERLAY ── */
.loading-overlay {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: var(--overlay-bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    z-index: 9999999;
    display: flex;
    justify-content: center;
    align-items: center;
}
.loading-box {
    background-color: var(--card-bg);
    padding: 2.5rem 3rem;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    border: none;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.loading-dots { display: flex; justify-content: center; gap: 8px; margin-bottom: 1rem; }
.loading-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background-color: var(--accent);
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%           { transform: scale(1.1); opacity: 1; }
}
.loading-text { color: var(--text); font-size: 1rem; font-weight: 500; }
"""
st.markdown(f"<style>{theme_vars}{MAIN_CSS}</style>", unsafe_allow_html=True)

# ── Helpers ──
def render_html(html_str: str):
    cleaned = "\n".join(line.strip() for line in html_str.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

def banner(kind: str, text: str):
    bg = C[f"{kind}_bg"]; fg = C[f"{kind}_fg"]
    st.markdown(
        f'<div style="background-color:{bg};color:{fg};padding:12px 16px;'
        f'border-radius:14px;font-size:0.9rem;margin:8px 0;border:none;">{text}</div>',
        unsafe_allow_html=True,
    )

def show_loading_overlay():
    st.markdown(
        f'<div class="loading-overlay">'
        f'  <div class="loading-box">'
        f'    <div class="loading-dots">'
        f'      <div class="loading-dot"></div>'
        f'      <div class="loading-dot"></div>'
        f'      <div class="loading-dot"></div>'
        f'    </div>'
        f'    <div class="loading-text">Computing prediction and SHAP attribution...</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Class cut-offs ──
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

# ── Input definitions ──
INPUT_SPEC = {
    "dip_direction_deg":     ("Dip Direction",      "deg", 1.0),
    "spacing_mode_mm":       ("Joint Spacing",      "mm",  5.0),
    "dip_deg":               ("Dip Angle",          "deg", 1.0),
    "density_t_m3":          ("Rock Density",       "t/m³", 0.01),
    "persistence_m":         ("Persistence",        "m",   0.1),
    "face_azimuth_deg":      ("Face Azimuth",       "deg", 1.0),
    "rotation_pressure_bar": ("Rotation Pressure",  "bar", 1.0),
    "northing":              ("Northing",           "m",   1.0),
}

ordered_feats = [f for f in INPUT_SPEC if f in feature_names] + [f for f in feature_names if f not in INPUT_SPEC]

def widget_range(f):
    b = train_bounds[f]
    lo = math.ceil(round(b["min"] * 100, 6)) / 100
    hi = math.floor(round(b["max"] * 100, 6)) / 100
    return (lo, hi) if hi > lo else (round(b["min"], 2), round(b["min"], 2) + 0.01)

RANGES = {f: widget_range(f) for f in feature_names}

def clamp(f, v):
    lo, hi = RANGES[f]
    return round(float(min(max(v, lo), hi)), 2)

# ── Session state for input widgets ──
for f in feature_names:
    b = train_bounds[f]
    start = clamp(f, b.get("median", (b["min"] + b["max"]) / 2))
    st.session_state.setdefault(f"num_{f}", start)
    st.session_state.setdefault(f"sld_{f}", start)

def sync_from_num(f): st.session_state[f"sld_{f}"] = st.session_state[f"num_{f}"]
def sync_from_sld(f): st.session_state[f"num_{f}"] = st.session_state[f"sld_{f}"]

PRESET_HARD = { "dip_direction_deg": 240.0, "spacing_mode_mm": 220.0, "dip_deg": 75.0, "density_t_m3": 2.56,
                "persistence_m": 6.0, "face_azimuth_deg": 140.0, "rotation_pressure_bar": 55.0, "northing": 1800.0}
PRESET_MEDIUM = { "dip_direction_deg": 90.0, "spacing_mode_mm": 80.0, "dip_deg": 35.0, "density_t_m3": 2.46,
                  "persistence_m": 1.5, "face_azimuth_deg": 60.0, "rotation_pressure_bar": 35.0, "northing": 800.0}

def apply_preset(values):
    for f, v in values.items():
        if f in RANGES:
            v = clamp(f, v)
            st.session_state[f"num_{f}"] = v
            st.session_state[f"sld_{f}"] = v

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

# ═══════════════════════════════════════════════════════════════
#                    BENTO DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════

# ── Hero Row: Title + Dark Mode Toggle ──
hero_col, toggle_col = st.columns([5, 1])
with hero_col:
    render_html("""
    <div class="apple-hero">
        <div class="apple-title">Rock Blastability AI</div>
        <div class="apple-subtitle">Explainable reduced-input Blastability Index (BI) estimation &bull; Group 13, UMaT Mining Engineering</div>
    </div>
    """)
with toggle_col:
    st.write("")
    st.toggle("Dark Mode", key="dark_mode", help="Switch between light and dark theme")

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — INPUTS (Bento Grid)
# ═══════════════════════════════════════════════════════════════
render_html("""
<div class="section-header">
    <div class="section-title">Geotechnical Parameters</div>
    <div class="section-subtitle">Eight essential inputs, selected by Hybrid PSO-GA. Adjust any value &mdash; sliders and number boxes stay perfectly in sync.</div>
</div>
""")

# Preset buttons row
pc1, pc2, pc3 = st.columns([1, 1, 2.5])
pc1.button("Hard Domain Preset", on_click=apply_preset, args=(PRESET_HARD,), use_container_width=True)
pc2.button("Medium Domain Preset", on_click=apply_preset, args=(PRESET_MEDIUM,), use_container_width=True)

# 4-column bento grid, 2 rows = 8 micro-cards
row1 = st.columns(4, gap="medium")
row2 = st.columns(4, gap="medium")

for i, f in enumerate(ordered_feats):
    title, unit, step = INPUT_SPEC.get(f, (f, "", round((RANGES[f][1] - RANGES[f][0]) / 100, 4)))
    lo, hi = RANGES[f]
    col = row1[i] if i < 4 else row2[i - 4]

    with col:
        render_html(f"""
        <div class="micro-card">
            <div class="micro-title">{title}</div>
            <div class="micro-unit">{unit} &nbsp;&bull;&nbsp; Range: {lo:g} &ndash; {hi:g}</div>
        </div>
        """)
        st.number_input(
            title, min_value=lo, max_value=hi, step=step, format="%.2f",
            key=f"num_{f}", on_change=sync_from_num, args=(f,),
            label_visibility="collapsed"
        )
        st.slider(
            title, min_value=lo, max_value=hi, step=step, format="%.2f",
            key=f"sld_{f}", on_change=sync_from_sld, args=(f,),
            label_visibility="collapsed"
        )

st.caption("Northing is a site coordinate, not a rock property: use this model only at the mine it was trained on.")

current_inputs = {f: float(st.session_state[f"num_{f}"]) for f in feature_names}

# Full-width Predict button
predict_clicked = st.button("Predict BI", type="primary", use_container_width=True)

loading_slot = st.empty()
if predict_clicked:
    with loading_slot:
        show_loading_overlay()
    time.sleep(1.5)
    input_vector = np.array([[current_inputs[f] for f in feature_names]])
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
    loading_slot.empty()

# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — RESULT HERO (Full Width)
# ═══════════════════════════════════════════════════════════════
if "prediction_result" in st.session_state:
    res = st.session_state["prediction_result"]
    pred_bi, category = res["pred_bi"], res["category"]
    shap_vals, base_val, snapshot = res["shap_vals"], res["base_val"], res["input_snapshot"]
    tree_preds = res["tree_preds"]

    render_html("""
    <div class="section-header">
        <div class="section-title">Prediction Result</div>
        <div class="section-subtitle">Estimated Blastability Index and operational guidance</div>
    </div>
    """)

    if any(not np.isclose(current_inputs[k], snapshot[k], atol=1e-3) for k in current_inputs):
        banner("warning", "Inputs changed since this prediction. Press <strong>Predict BI</strong> to update.")

    spread_html = ""
    if tree_preds is not None:
        sd = float(tree_preds.std())
        p5, p95 = (float(v) for v in np.percentile(tree_preds, [5, 95]))
        spread_html = (
            f'<div style="font-size:0.92rem;color:{C["text_secondary"]};margin-top:14px;">'
            f'Tree spread &plusmn;{sd:.2f} &nbsp;&bull;&nbsp; 5&ndash;95% of the {len(tree_preds)} trees: {p5:.1f} &ndash; {p95:.1f}</div>'
        )
        if classify(p5) != classify(p95):
            spread_html += (
                f'<div style="font-size:0.88rem;color:{C["warning_fg"]};margin-top:8px;">'
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
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.5rem;">
            <div style="flex: 1; min-width: 280px;">
                <div class="metric-label">Estimated Lilly BI</div>
                <div class="metric-value">{pred_bi:.2f}</div>
                {spread_html}
            </div>
            <div style="text-align: right;">
                <span class="apple-badge {BADGE[category]}">{category} Rock Mass</span>
            </div>
        </div>
        <div class="gauge-track">
            <div class="gauge-fill" style="width: {pct:.1f}%;"></div>
        </div>
        <div class="gauge-labels">
            <span>Easy (&lt;{EASY_MAX:g})</span>
            <span>Medium ({EASY_MAX:g} &ndash; {HARD_MIN:g})</span>
            <span>Hard (&gt;{HARD_MIN:g})</span>
        </div>
        <div class="guidance-box">
            {guidance}
            <div style="font-size:0.82rem;color:{C['text_secondary']};margin-top:8px;">Indicative only. Confirm against site blast trials.</div>
        </div>
    </div>
    """)
    st.caption("Tree spread shows how much the individual trees disagree. It is not a formal confidence interval.")

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 3 — SHAP EXPLANATION (Full Width)
    # ═══════════════════════════════════════════════════════════════
    render_html("""
    <div class="section-header">
        <div class="section-title">Attribution &amp; Explanation</div>
        <div class="section-subtitle">SHAP feature contributions pushing the BI higher or lower</div>
    </div>
    """)

    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=C["chart_bg"])
    ax.set_facecolor(C["chart_bg"])
    sort_idx = np.argsort(np.abs(shap_vals))
    sorted_feats = [feature_names[i] for i in sort_idx]
    sorted_shaps = [shap_vals[i] for i in sort_idx]
    colors = [C["shap_pos"] if s > 0 else C["shap_neg"] for s in sorted_shaps]
    y_pos = np.arange(len(sorted_feats))
    ax.barh(y_pos, sorted_shaps, color=colors, height=0.55, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats, fontsize=9.5, color=C["text"])
    ax.axvline(0, color=C["chart_grid"], linestyle='-', linewidth=1.0)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(C["chart_grid"])
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', colors=C["text_secondary"], labelsize=8.5)
    ax.tick_params(axis='y', length=0)
    ax.set_title(f"Baseline Reference: {base_val:.2f}  \u2192  Predicted Output: {pred_bi:.2f}",
                 fontsize=10, color=C["text_secondary"], pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — ADVANCED TOOLS (Side by Side)
# ═══════════════════════════════════════════════════════════════
render_html("""
<div class="section-header">
    <div class="section-title">Advanced Tools</div>
    <div class="section-subtitle">Batch processing and model research metrics</div>
</div>
""")

tool_col1, tool_col2 = st.columns([1.15, 1], gap="large")

# ── Left: Batch Upload ──
with tool_col1:
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
                banner("error", f"Missing required columns: {missing_cols}")
            else:
                result = predict_batch(bdf)
                n_out = int((result["Out_of_Range_Inputs"] != "").sum())
                if n_out:
                    banner("warning", f"{n_out} of {len(result)} rows have inputs outside the training range. "
                                      "Their predictions are extrapolations and should not be trusted.")
                else:
                    banner("success", f"All {len(result)} rows are inside the training range.")
                show_cols = ["Predicted_BI", "BI_Category", "Out_of_Range_Inputs"] + feature_names
                st.dataframe(result[show_cols], use_container_width=True, height=420)
                st.download_button("Download predictions (.csv)", result.to_csv(index=False).encode("utf-8"),
                                   file_name="Estimated_Blastability_Index.csv", mime="text/csv")
        except Exception as e:
            banner("error", f"Processing error: {e}")

# ── Right: Performance & Stability ──
with tool_col2:
    render_html("""
    <div class="apple-card">
        <div class="apple-card-title">Model Performance Comparison</div>
        <div class="apple-card-subtitle">Evaluated on 163 held-out bench records (unseen benches)</div>
    </div>
    """)
    comp_df = pd.DataFrame([
        {"Model": "Mean Baseline", "Inputs": 0, "R\u00b2": -0.0102, "RMSE": 4.5383, "MAE": 3.5889, "Category Match": "74.85%"},
        {"Model": "Linear Regression (Full)", "Inputs": 26, "R\u00b2": 0.5546, "RMSE": 3.0136, "MAE": 2.4477, "Category Match": "72.39%"},
        {"Model": "Random Forest (Full)", "Inputs": 26, "R\u00b2": 0.8692, "RMSE": 1.6331, "MAE": 1.0623, "Category Match": "93.87%"},
        {"Model": "Random Forest (PSO-GA)", "Inputs": 8, "R\u00b2": 0.8862, "RMSE": 1.5231, "MAE": 0.9489, "Category Match": "95.09%"},
        {"Model": "Random Forest (FRFS)", "Inputs": 23, "R\u00b2": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Match": "93.25%"},
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.caption("Category match is the share of the 163 test records placed in the correct Easy/Medium/Hard class. "
               "The mean baseline already scores 74.85%, and the three random forests differ by only a few records, "
               "so read them as comparable accuracy with fewer inputs for the PSO-GA model.")

    render_html(f"""
    <div class="apple-card">
        <div class="apple-card-title">Engineering Stability</div>
        <div class="apple-card-subtitle">30 bootstrap resamples with replacement</div>
        <div style="margin-bottom: 1.2rem;">
            <div class="metric-label">SHAP Ranking Stability (Spearman \u03c1)</div>
            <div style="font-size: 2rem; font-weight: 700; color: {C['accent']};">0.8280 &plusmn; 0.055</div>
            <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">
                Suggests the feature rankings stay broadly similar across resamples (\u03c1 = 1 would mean identical rankings).
            </div>
        </div>
        <div>
            <div class="metric-label">Selection Frequency (Dip Direction / Spacing)</div>
            <div style="font-size: 2rem; font-weight: 700; color: {C['accent']};">100% / 96.7%</div>
            <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">
                Dip Direction was selected in 30/30 resamples; Spacing in 29/30.
            </div>
        </div>
    </div>
    """)