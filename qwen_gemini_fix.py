"""
Streamlit Web Application: Explainable AI Blastability Index (BI) Estimator
Group 13 | UMaT BSc Mining Engineering
Dual Layout: Classic Studio View & Scrollable Bento Canvas View
Deep Green Theme, Sliding Pill Navigation & Toggles, full-screen 1.5s loading overlay.
v7: Integrated theme_v7.py styling with sliding pill tabs and toggles.
"""
import os
import json
import math
import pickle
import io
import time
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shap

# ── App Configuration ──
st.set_page_config(
    page_title="Blastability AI | Group 13 UMaT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── State Management (synced cleanly before theme & layout are resolved) ──
if "theme_radio" not in st.session_state:
    st.session_state.theme_radio = "Light"
if "layout_radio" not in st.session_state:
    st.session_state.layout_radio = "Classic"

st.session_state.dark_mode = (st.session_state.theme_radio == "Dark")
st.session_state.canvas_mode = (st.session_state.layout_radio == "Canvas")

DARK = st.session_state.dark_mode
CANVAS = st.session_state.canvas_mode

# ── Theme Palette (deep green) ──
THEME_LIGHT = {
    "bg": "#F2F5F2", "text": "#0E1A14", "text_secondary": "#5A6B61",
    "card_bg": "#FFFFFF", "card_bg_elev": "#F3F7F4",
    "accent": "#14573A", "accent_hover": "#0D4129",
    "track": "#DCE6DF",
    "bar": "#0F1C15", "bar_text": "#C9D8CE", "bar_dim": "#8FA597",
    "pill": "#FFFFFF", "pill_text": "#0F1C15",
    "badge_easy_bg": "#E4F2E9", "badge_easy_fg": "#1E7B4E",
    "badge_med_bg": "#C8E4D2", "badge_med_fg": "#14583A",
    "badge_hard_bg": "#14412D", "badge_hard_fg": "#D9F2E4",
    "class_easy": "#6DBE8E", "class_med": "#2E8B5A", "class_hard": "#0F4A30",
    "band_easy": "#EAF5EE", "band_med": "#DCEEE3", "band_hard": "#CBE5D6",
    "band_easy_fg": "#2A7F52", "band_med_fg": "#1B6843", "band_hard_fg": "#0F4A30",
    "gauge_a": "#8ED1A9", "gauge_b": "#2E8B5A", "gauge_c": "#0F4A30",
    "info_bg": "#E6F1EA", "info_fg": "#0E1A14",
    "success_bg": "#E1F3E8", "success_fg": "#1B6B3F",
    "warning_bg": "#FFF6DD", "warning_fg": "#8A5A00",
    "error_bg": "#FFECEC", "error_fg": "#8A1F1F",
    "chart_bg": "#FFFFFF", "chart_grid": "#E1E9E3",
    "shap_pos": "#1B7F52", "shap_neg": "#A8CDB7",
    "overlay_bg": "rgba(242, 245, 242, 0.72)",
    "shadow": "0 1px 2px rgba(15, 28, 21, 0.04), 0 10px 30px rgba(15, 28, 21, 0.06)",
    "bar_shadow": "0 14px 34px rgba(10, 24, 17, 0.28)",
}

THEME_DARK = {
    "bg": "#060B08", "text": "#E9F2EC", "text_secondary": "#8EA196",
    "card_bg": "#0F1813", "card_bg_elev": "#17231C",
    "accent": "#3BB57C", "accent_hover": "#55CB92",
    "track": "#25342B",
    "bar": "#13211A", "bar_text": "#B7C9BD", "bar_dim": "#7E9487",
    "pill": "#E9F3EC", "pill_text": "#0B1510",
    "badge_easy_bg": "#143323", "badge_easy_fg": "#7FD5A3",
    "badge_med_bg": "#1A4630", "badge_med_fg": "#62CB92",
    "badge_hard_bg": "#CDEFDC", "badge_hard_fg": "#0F3B27",
    "class_easy": "#6CCB96", "class_med": "#35A56C", "class_hard": "#1D7049",
    "band_easy": "#12231A", "band_med": "#163022", "band_hard": "#1B3C2A",
    "band_easy_fg": "#7FD5A3", "band_med_fg": "#62CB92", "band_hard_fg": "#8BDDAB",
    "gauge_a": "#4FA67A", "gauge_b": "#2FB574", "gauge_c": "#9BE8BD",
    "info_bg": "#10211A", "info_fg": "#D6E8DC",
    "success_bg": "#0E2A1B", "success_fg": "#9BE3B6",
    "warning_bg": "#332400", "warning_fg": "#FFCE6B",
    "error_bg": "#3A1414", "error_fg": "#FF9E9E",
    "chart_bg": "#0F1813", "chart_grid": "#25342B",
    "shap_pos": "#3BB57C", "shap_neg": "#45604F",
    "overlay_bg": "rgba(4, 9, 6, 0.74)",
    "shadow": "0 1px 2px rgba(0, 0, 0, 0.4), 0 10px 30px rgba(0, 0, 0, 0.35)",
    "bar_shadow": "0 14px 34px rgba(0, 0, 0, 0.55)",
}

C = THEME_DARK if DARK else THEME_LIGHT

# ── CSS: theme variables + main stylesheet ──
theme_vars = ":root {" + "".join(f"--{k.replace('_', '-')}: {v};" for k, v in C.items()) + "}"

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
    padding-top: 1.4rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

/* ═══════ TOP BAR (dark pill, inspired by a floating nav) ═══════ */
.st-key-topbar {
    background: var(--bar);
    border-radius: 9999px;
    padding: 9px 10px 9px 12px;
    box-shadow: var(--bar-shadow);
    margin-bottom: 2rem;
}
.st-key-topbar [data-testid="stHorizontalBlock"] { align-items: center !important; gap: 0.6rem !important; }
.brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.brand-logo {
    width: 42px; height: 42px; flex: 0 0 42px; border-radius: 50%;
    background: var(--pill); display: flex; align-items: center; justify-content: center;
}
.brand-name { font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.15; }
.brand-sub  { font-size: 0.74rem; line-height: 1.2; margin-top: 2px; }
.stApp .brand-name { color: #FFFFFF !important; }
.stApp .brand-sub  { color: var(--bar-dim) !important; }

/* ═══════ SLIDING PILL (tabs + toggles share one idea) ═══════ */
.stTabs [role="tablist"] {
    display: grid !important;
    grid-auto-flow: column !important;
    grid-auto-columns: 1fr !important;
    position: relative !important;
    width: fit-content !important;
    max-width: 100% !important;
    margin: 0 auto 2rem auto !important;
    padding: 6px !important;
    gap: 0 !important;
    background: var(--bar) !important;
    border: none !important;
    border-radius: 9999px !important;
    box-shadow: var(--bar-shadow) !important;
    overflow: visible !important;
}
.stTabs [role="tablist"]::before {
    content: "";
    position: absolute; z-index: 0; pointer-events: none;
    top: 6px; bottom: 6px; left: 6px;
    width: calc((100% - 12px) / 4);
    border-radius: 9999px;
    background: var(--pill);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.28);
    transition: transform 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}
.stTabs [role="tablist"]:has([role="tab"]:nth-child(2)[aria-selected="true"])::before { transform: translateX(100%); }
.stTabs [role="tablist"]:has([role="tab"]:nth-child(3)[aria-selected="true"])::before { transform: translateX(200%); }
.stTabs [role="tablist"]:has([role="tab"]:nth-child(4)[aria-selected="true"])::before { transform: translateX(300%); }
.stTabs [role="tab"] {
    position: relative; z-index: 1;
    height: 44px !important;
    padding: 0 26px !important;
    display: flex !important; align-items: center; justify-content: center;
    background: transparent !important;
    border: none !important;
    border-radius: 9999px !important;
    box-shadow: none !important;
    white-space: nowrap;
    cursor: pointer;
}
.stApp .stTabs [role="tab"] p, .stApp .stTabs [role="tab"] span, .stApp .stTabs [role="tab"] div {
    color: var(--bar-text) !important;
    font-size: 0.92rem !important; font-weight: 500 !important; margin: 0 !important;
    transition: color 0.3s ease;
}
.stApp .stTabs [role="tab"]:hover p { color: #FFFFFF !important; }
.stApp .stTabs [role="tab"][aria-selected="true"] p,
.stApp .stTabs [role="tab"][aria-selected="true"] span { color: var(--pill-text) !important; font-weight: 600 !important; }
.stTabs [role="tab"][aria-selected="true"] { background: transparent !important; box-shadow: none !important; }
.stTabs [role="tablist"]::after { display: none !important; }
.react-aria-SelectionIndicator,
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-testid="stTabPanel"] { padding-top: 0.2rem; }

/* Radio groups inside the top bar become sliding two-way switches */
.st-key-topbar [role="radiogroup"] {
    display: grid !important;
    grid-auto-flow: column !important;
    grid-auto-columns: 1fr !important;
    position: relative !important;
    gap: 0 !important;
    padding: 4px !important;
    width: 100% !important;
    max-width: 230px;
    margin-left: auto;
    background: rgba(255, 255, 255, 0.09);
    border-radius: 9999px;
}
.st-key-topbar [role="radiogroup"]::before {
    content: "";
    position: absolute; z-index: 0; pointer-events: none;
    top: 4px; bottom: 4px; left: 4px;
    width: calc((100% - 8px) / 2);
    border-radius: 9999px;
    background: var(--pill);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: transform 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
.st-key-topbar [role="radiogroup"]:has(> :nth-child(2) input:checked)::before { transform: translateX(100%); }
.st-key-topbar [role="radiogroup"] > * { position: relative; z-index: 1; margin: 0 !important; display: flex !important; }
.st-key-topbar [role="radiogroup"] label {
    width: 100%; margin: 0 !important; padding: 7px 10px !important;
    display: flex !important; align-items: center; justify-content: center;
    cursor: pointer; background: transparent !important;
}
.st-key-topbar [role="radiogroup"] label > div { margin: 0 !important; padding: 0 !important; }
.st-key-topbar [role="radiogroup"] label > div > div:first-child:not([data-testid]),
.st-key-topbar [role="radiogroup"] label > span:first-of-type:not(:has(input)) { display: none !important; }
.stApp .st-key-topbar [role="radiogroup"] p {
    margin: 0 !important; font-size: 0.82rem !important; font-weight: 600 !important;
    color: var(--bar-text) !important; transition: color 0.3s ease; white-space: nowrap;
}
.stApp .st-key-topbar [role="radiogroup"] label:has(input:checked) p { color: var(--pill-text) !important; }
.st-key-topbar [data-testid="stWidgetLabel"] { display: none !important; }
.st-key-topbar [data-testid="stElementContainer"] { width: 100%; }

/* Hero */
.apple-hero { margin: 0.4rem 0 1.8rem 0; text-align: center; }
.hero-eyebrow {
    display: inline-block; padding: 6px 14px; border-radius: 9999px;
    background: var(--badge-easy-bg); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
    margin-bottom: 0.9rem;
}
.stApp .hero-eyebrow { color: var(--badge-easy-fg) !important; }
.apple-title {
    font-size: 2.7rem; font-weight: 650; letter-spacing: -0.035em;
    color: var(--text); line-height: 1.1; margin: 0 0 0.6rem 0;
}
.apple-subtitle {
    font-size: 1.04rem; font-weight: 400; color: var(--text-secondary);
    letter-spacing: -0.01em; line-height: 1.5; margin: 0 auto; max-width: 640px;
}

/* Cards */
.apple-card {
    background-color: var(--card-bg);
    border: none;
    border-radius: 24px;
    padding: 1.7rem 1.85rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow);
}
.apple-card-title {
    font-size: 1.15rem; font-weight: 600; color: var(--text);
    letter-spacing: -0.02em; margin: 0 0 0.25rem 0;
}
.apple-card-subtitle {
    font-size: 0.88rem; color: var(--text-secondary);
    margin: 0 0 0.4rem 0; line-height: 1.5;
}

/* Bento Micro-Cards (Canvas View) */
.micro-card {
    background-color: var(--card-bg);
    border: none;
    border-radius: 18px;
    padding: 1rem 1.1rem 0.6rem 1.1rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.micro-title {
    font-size: 0.78rem; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; line-height: 1.2;
}
.micro-unit {
    font-size: 0.72rem; color: var(--text-secondary); opacity: 0.8; margin-bottom: 0.5rem;
}

/* Section headers (Canvas View) */
.section-header {
    margin: 2.6rem 0 1.2rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--track);
}
.section-title {
    font-size: 1.45rem; font-weight: 600; color: var(--text);
    letter-spacing: -0.02em; margin: 0 0 0.25rem 0;
}
.section-subtitle { font-size: 0.92rem; color: var(--text-secondary); margin: 0; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 9999px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    background-color: var(--card-bg) !important;
    box-shadow: var(--shadow) !important;
}
.stApp .stButton > button p, .stApp .stDownloadButton > button p { color: var(--accent) !important; font-weight: 500 !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background-color: var(--accent) !important; }
.stApp .stButton > button:hover p, .stApp .stDownloadButton > button:hover p { color: #FFFFFF !important; }
.stButton > button[kind="primary"] {
    background-color: var(--accent) !important;
    padding: 0.9rem 1.5rem !important;
    box-shadow: 0 8px 22px rgba(20, 87, 58, 0.35) !important;
}
.stApp .stButton > button[kind="primary"] p { color: #FFFFFF !important; font-weight: 600 !important; font-size: 1.02rem !important; }
.stButton > button[kind="primary"]:hover { background-color: var(--accent-hover) !important; box-shadow: 0 10px 26px rgba(20, 87, 58, 0.45) !important; }

/* Inputs */
.stNumberInput > div, .stNumberInput > div > div,
.stTextInput > div > div, .stSelectbox > div > div,
.stTextArea > div > div {
    border-radius: 14px !important;
    border: none !important;
    background-color: var(--card-bg) !important;
}
.stNumberInput [data-baseweb="input"], .stNumberInput [data-testid="stNumberInputContainer"],
.stTextInput [data-baseweb="input"], .stSelectbox [data-baseweb="select"] > div {
    border-radius: 14px !important; background-color: var(--card-bg) !important; box-shadow: var(--shadow);
}
.stNumberInput input, .stTextInput input { color: var(--text) !important; background: transparent !important; }

/* File uploader */
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
    border-radius: 18px !important; border: none !important; background-color: var(--card-bg) !important;
}
[data-testid="stFileUploaderDropzone"] button { border-radius: 9999px !important; border: none !important; }

/* Dataframe */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] > div, [data-testid="stDataFrameResizable"] {
    border-radius: 16px !important; overflow: hidden !important; border: none !important; outline: none !important;
}

/* Alerts + expander */
[data-testid="stAlert"] { border-radius: 14px !important; border: none !important; }
[data-testid="stExpander"], .stExpander { border: none !important; border-radius: 16px !important; background-color: var(--card-bg) !important; box-shadow: var(--shadow); }
.stPyplot img, [data-testid="stImage"] img { border-radius: 16px; }

/* ═══════ SLIDERS (thick track, green fill, bigger handle) ═══════ */
[data-testid="stSlider"] { padding-top: 0.2rem; padding-bottom: 0.2rem; }
[data-testid="stSlider"] [role="group"] > div > div:first-child {
    height: 10px !important;
    border-radius: 9999px !important;
    transform: translateY(-3px);
    filter: hue-rotate(158deg) saturate(0.85) brightness(0.82);
}
[data-testid="stSlider"] [role="group"] > div > div:nth-child(2) {
    width: 26px !important; height: 26px !important;
    box-sizing: border-box;
    background-color: var(--accent) !important;
    border: 4px solid var(--card-bg) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}
[data-testid="stSliderThumbValue"] p { font-size: 0.8rem !important; font-weight: 600 !important; color: var(--text) !important; }

/* BaseWeb fallback for older Streamlit */
.stSlider [data-baseweb="slider"] > div:first-child > div:not([role="slider"]) { height: 10px !important; border-radius: 9999px !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    width: 26px !important; height: 26px !important; background-color: var(--accent) !important;
    border: 4px solid var(--card-bg) !important; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

/* Metric display */
.metric-value { font-size: 3.9rem; font-weight: 700; color: var(--text); letter-spacing: -0.04em; line-height: 1; }
.metric-label {
    font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.4rem;
}

/* Badges */
.apple-badge {
    display: inline-block; padding: 10px 22px; border-radius: 9999px;
    font-size: 0.95rem; font-weight: 600; letter-spacing: -0.01em; text-align: center; border: none;
}
.badge-easy   { background-color: var(--badge-easy-bg); }
.badge-medium { background-color: var(--badge-med-bg); }
.badge-hard   { background-color: var(--badge-hard-bg); }
.stApp .badge-easy   { color: var(--badge-easy-fg) !important; }
.stApp .badge-medium { color: var(--badge-med-fg) !important; }
.stApp .badge-hard   { color: var(--badge-hard-fg) !important; }

/* Gauge */
.gauge-track { width: 100%; height: 12px; background-color: var(--track); border-radius: 9999px; overflow: hidden; margin: 20px 0 8px 0; }
.gauge-fill { height: 100%; border-radius: 9999px; background: linear-gradient(90deg, var(--gauge-a) 0%, var(--gauge-b) 55%, var(--gauge-c) 100%); transition: width 0.6s ease; }
.gauge-labels { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--text-secondary); }

/* Guidance box */
.guidance-box {
    background-color: var(--card-bg-elev); border: none; border-radius: 16px;
    padding: 14px 18px; margin-top: 18px; font-size: 0.92rem; color: var(--text); line-height: 1.5;
}

/* ── FULL SCREEN LOADING OVERLAY ── */
.loading-overlay {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: var(--overlay-bg); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    z-index: 9999999; display: flex; justify-content: center; align-items: center;
}
.loading-box {
    background-color: var(--card-bg); padding: 2.5rem 3rem; border-radius: 28px; text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2); border: none;
    display: flex; flex-direction: column; align-items: center;
}
.loading-dots { display: flex; justify-content: center; gap: 8px; margin-bottom: 1rem; }
.loading-dot { width: 12px; height: 12px; border-radius: 50%; background-color: var(--accent); animation: pulse-dot 1.4s ease-in-out infinite; }
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse-dot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%           { transform: scale(1.1); opacity: 1; }
}
.loading-text { color: var(--text); font-size: 1rem; font-weight: 500; }

/* ═══════ TABLET (tabs become a 2 x 2 pill grid) ═══════ */
@media (max-width: 960px) {
    .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; max-width: 100% !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.8rem !important; }
    [data-testid="stColumn"], [data-testid="column"] {
        min-width: calc(50% - 0.8rem) !important; flex: 1 1 calc(50% - 0.8rem) !important;
    }
    .stTabs [role="tablist"] {
        grid-auto-flow: row !important; grid-template-columns: repeat(2, 1fr) !important; grid-auto-columns: auto !important;
        width: 100% !important; border-radius: 28px !important;
    }
    .stTabs [role="tablist"]::before {
        width: calc((100% - 12px) / 2); height: calc((100% - 12px) / 2); bottom: auto; border-radius: 22px;
    }
    .stTabs [role="tablist"]:has([role="tab"]:nth-child(1)[aria-selected="true"])::before { transform: translate(0, 0); }
    .stTabs [role="tablist"]:has([role="tab"]:nth-child(2)[aria-selected="true"])::before { transform: translate(100%, 0); }
    .stTabs [role="tablist"]:has([role="tab"]:nth-child(3)[aria-selected="true"])::before { transform: translate(0, 100%); }
    .stTabs [role="tablist"]:has([role="tab"]:nth-child(4)[aria-selected="true"])::before { transform: translate(100%, 100%); }
    .stTabs [role="tab"] { padding: 0 12px !important; border-radius: 22px !important; }
}

/* ═══════ PHONE ═══════ */
@media (max-width: 640px) {
    .block-container { padding: 0.8rem 0.8rem 2.5rem 0.8rem !important; }
    [data-testid="stColumn"], [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
    .st-key-topbar { border-radius: 28px; padding: 12px; margin-bottom: 1.4rem; }
    .st-key-topbar [data-testid="stColumn"], .st-key-topbar [data-testid="column"] { min-width: calc(50% - 0.6rem) !important; flex: 1 1 calc(50% - 0.6rem) !important; }
    .st-key-topbar [data-testid="stColumn"]:first-child, .st-key-topbar [data-testid="column"]:first-child { min-width: 100% !important; flex: 1 1 100% !important; }
    .st-key-topbar [role="radiogroup"] { max-width: none; margin-left: 0; }
    .brand-sub { display: none; }
    .apple-title { font-size: 1.85rem; }
    .apple-subtitle { font-size: 0.92rem; }
    .apple-card { padding: 1.15rem 1.15rem; border-radius: 20px; margin-bottom: 1rem; }
    .apple-card-title { font-size: 1.05rem; }
    .section-title { font-size: 1.2rem; }
    .section-header { margin: 1.6rem 0 0.9rem 0; }
    .metric-value { font-size: 2.7rem; }
    .apple-badge { padding: 8px 16px; font-size: 0.85rem; }
    .gauge-labels { font-size: 0.68rem; gap: 4px; }
    .guidance-box { padding: 12px 14px; font-size: 0.88rem; }
    .loading-box { padding: 1.6rem 1.4rem; margin: 0 1.2rem; border-radius: 22px; }
    .loading-text { font-size: 0.9rem; }
    .stButton > button, .stDownloadButton > button { min-height: 2.8rem; }
    [data-testid="stSlider"] [role="group"] > div > div:nth-child(2) { width: 30px !important; height: 30px !important; }
    .stApp .stTabs [role="tab"] p { font-size: 0.82rem !important; }
}
"""

st.markdown(f"<style>{theme_vars}{MAIN_CSS}</style>", unsafe_allow_html=True)

# ── Helpers ──
def render_html(html_str: str):
    cleaned = "\n".join(line.strip() for line in html_str.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

def banner(kind: str, text: str):
    bg = C[f"{kind}_bg"]
    fg = C[f"{kind}_fg"]
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
    val = float(v)
    return "Hard" if val >= HARD_MIN else ("Medium" if val >= EASY_MAX else "Easy")

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
    df = pd.read_csv(FEATURES_PATH)
    if "0" in df.columns:
        return df["0"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()

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
    bounds = {
        feats[i]: {
            "min": float(Xtr[:, i].min()),
            "max": float(Xtr[:, i].max()),
            "median": float(np.median(Xtr[:, i])),
        }
        for i in range(len(feats))
    }
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
    "dip_direction_deg":     ("Dip Direction",       "deg",  1.0),
    "spacing_mode_mm":       ("Joint Spacing",       "mm",   5.0),
    "dip_deg":               ("Dip Angle",           "deg",  1.0),
    "density_t_m3":          ("Rock Density",        "t/m³", 0.01),
    "persistence_m":         ("Persistence",         "m",    0.1),
    "face_azimuth_deg":      ("Face Azimuth",        "deg",  1.0),
    "rotation_pressure_bar": ("Rotation Pressure",   "bar",  1.0),
    "northing":              ("Northing",            "m",    1.0),
}

ordered_feats = [f for f in INPUT_SPEC if f in feature_names] + [f for f in feature_names if f not in INPUT_SPEC]

def widget_range(f):
    b = train_bounds[f]
    lo = float(math.floor(round(float(b["min"]) * 100, 6)) / 100.0)
    hi = float(math.ceil(round(float(b["max"]) * 100, 6)) / 100.0)
    if hi <= lo:
        lo_val = float(round(float(b["min"]), 2))
        return (lo_val, float(round(lo_val + 0.01, 2)))
    return (lo, hi)

def step_for(f):
    if f in INPUT_SPEC:
        return float(INPUT_SPEC[f][2])
    lo, hi = RANGES[f]
    return float(max(round((hi - lo) / 100.0, 4), 0.01))

RANGES = {f: widget_range(f) for f in feature_names}

def clamp(f, v):
    lo, hi = RANGES[f]
    return float(round(float(min(max(float(v), lo), hi)), 2))

# ── Session state for input widgets ──
for f in feature_names:
    b = train_bounds[f]
    median_val = float(b.get("median", (float(b["min"]) + float(b["max"])) / 2.0))
    start = clamp(f, median_val)
    st.session_state.setdefault(f"num_{f}", start)
    st.session_state.setdefault(f"sld_{f}", start)

def sync_from_num(f):
    val = clamp(f, st.session_state[f"num_{f}"])
    st.session_state[f"num_{f}"] = val
    st.session_state[f"sld_{f}"] = val

def sync_from_sld(f):
    val = clamp(f, st.session_state[f"sld_{f}"])
    st.session_state[f"num_{f}"] = val
    st.session_state[f"sld_{f}"] = val

PRESET_HARD = {
    "dip_direction_deg": 240.0, "spacing_mode_mm": 220.0, "dip_deg": 75.0, "density_t_m3": 2.56,
    "persistence_m": 6.0, "face_azimuth_deg": 140.0, "rotation_pressure_bar": 55.0, "northing": 1800.0,
}
PRESET_MEDIUM = {
    "dip_direction_deg": 90.0, "spacing_mode_mm": 80.0, "dip_deg": 35.0, "density_t_m3": 2.46,
    "persistence_m": 1.5, "face_azimuth_deg": 60.0, "rotation_pressure_bar": 35.0, "northing": 800.0,
}

def apply_preset(values):
    for f, v in values.items():
        if f in RANGES:
            clamped = clamp(f, v)
            st.session_state[f"num_{f}"] = clamped
            st.session_state[f"sld_{f}"] = clamped

def reset_to_median():
    for f in feature_names:
        b = train_bounds[f]
        med = float(b.get("median", (float(b["min"]) + float(b["max"])) / 2.0))
        clamped = clamp(f, med)
        st.session_state[f"num_{f}"] = clamped
        st.session_state[f"sld_{f}"] = clamped

def predict_batch(bdf: pd.DataFrame) -> pd.DataFrame:
    X = bdf[feature_names].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().sum()
    if bad.any():
        raise ValueError("Missing or non-numeric values in: " + ", ".join(f"{c} ({n} rows)" for c, n in bad[bad > 0].items()))
    out = bdf.copy()
    out["Predicted_BI"] = np.round(model.predict(X.values), 2)
    out["BI_Category"] = out["Predicted_BI"].apply(classify)
    mins = pd.Series({f: float(train_bounds[f]["min"]) for f in feature_names})
    maxs = pd.Series({f: float(train_bounds[f]["max"]) for f in feature_names})
    outside = (X < mins) | (X > maxs)
    out["Out_of_Range_Inputs"] = outside.apply(lambda r: ", ".join(r.index[r]), axis=1)
    return out

# ═══════════════════════════════════════════════════════════════
# SHARED HELPERS: charts, PDF report, saved runs, result view
# ═══════════════════════════════════════════════════════════════
st.session_state.setdefault("saved_runs", [])

def pretty(f):
    return INPUT_SPEC[f][0] if f in INPUT_SPEC else f

def unit_of(f):
    return INPUT_SPEC[f][1] if f in INPUT_SPEC else ""

def spread_stats(tp):
    if tp is None or len(tp) == 0:
        return None
    tp_arr = np.asarray(tp, dtype=float)
    p5, p95 = (float(v) for v in np.percentile(tp_arr, [5, 95]))
    p25, p75 = (float(v) for v in np.percentile(tp_arr, [25, 75]))
    return {
        "sd": float(tp_arr.std()),
        "p5": p5,
        "p95": p95,
        "n": len(tp_arr),
        "transition": classify(p25) != classify(p75),
    }

def guidance_parts(bi):
    if float(bi) >= HARD_MIN:
        return ("High Blast Resistance:", "Tighter pattern or increased powder factor may be needed to limit oversize.")
    return ("Favorable Blastability:", "A standard powder factor is likely suitable, watching for backbreak and excessive fines.")

# ── Chart primitives ──
def new_fig(P, w, h):
    return plt.subplots(figsize=(w, h), facecolor=P["chart_bg"])

def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)

def style_ax(ax, P, grid=None):
    ax.set_facecolor(P["chart_bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(P["chart_grid"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=P["text_secondary"], labelsize=8)
    ax.tick_params(axis="y", length=0, colors=P["text"], labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=P["chart_grid"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

_CLASS_SEGS = lambda lo, hi: [("Easy", lo, EASY_MAX, "easy"), ("Medium", EASY_MAX, HARD_MIN, "med"), ("Hard", HARD_MIN, hi, "hard")]

def bands_x(ax, P, xlo, xhi, labels=True):
    for name, a, b, k in _CLASS_SEGS(-1e9, 1e9):
        a2, b2 = max(a, xlo), min(b, xhi)
        if b2 <= a2:
            continue
        ax.axvspan(a2, b2, color=P[f"badge_{k}_bg"], zorder=0, lw=0)
        if labels:
            ax.text((a2 + b2) / 2, 0.96, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, color=P[f"badge_{k}_fg"], zorder=1)
    ax.set_xlim(xlo, xhi)

def bands_y(ax, P, ylo, yhi):
    for name, a, b, k in _CLASS_SEGS(-1e9, 1e9):
        a2, b2 = max(a, ylo), min(b, yhi)
        if b2 <= a2:
            continue
        ax.axhspan(a2, b2, color=P[f"badge_{k}_bg"], zorder=0, lw=0)
        ax.text(0.99, (a2 + b2) / 2, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.5, color=P[f"badge_{k}_fg"], zorder=1)
    ax.set_ylim(ylo, yhi)

def draw_gauge(ax, res, P):
    bi = float(res["pred_bi"])
    bands_x(ax, P, GAUGE_MIN, GAUGE_MAX)
    ax.axvline(min(max(bi, GAUGE_MIN), GAUGE_MAX), color=P["accent"], lw=3.5, zorder=3)
    ax.set_yticks([])
    ax.set_xticks([GAUGE_MIN, EASY_MAX, HARD_MIN, GAUGE_MAX])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", colors=P["text_secondary"], labelsize=8, length=0)

def draw_shap(ax, res, P, fs=8.5):
    style_ax(ax, P)
    sv = np.asarray(res["shap_vals"], dtype=float).reshape(-1)
    idx = np.argsort(np.abs(sv))
    vals = sv[idx]
    labels = [pretty(feature_names[i]) for i in idx]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=[P["shap_pos"] if v > 0 else P["shap_neg"] for v in vals], height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs, color=P["text"])
    ax.axvline(0, color=P["chart_grid"], linewidth=1.0)
    lo, hi = min(float(vals.min()), 0.0), max(float(vals.max()), 0.0)
    span = max(hi - lo, 1e-6)
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), yi, f"{float(v):+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=fs - 1, color=P["text_secondary"])
    ax.set_xlim(lo - 0.2 * span, hi + 0.2 * span)
    ax.set_title(f"Baseline Reference: {res['base_val']:.2f}  →  Predicted Output: {res['pred_bi']:.2f}",
                 fontsize=fs + 0.5, color=P["text_secondary"], pad=10)

def draw_tree_hist(ax, res, P):
    style_ax(ax, P)
    tp = np.asarray(res["tree_preds"], dtype=float)
    bi = float(res["pred_bi"])
    ax.hist(tp, bins=min(25, max(8, len(tp) // 4)), color=P["shap_pos"], alpha=0.9,
            edgecolor=P["chart_bg"], linewidth=0.6, zorder=3)
    ax.axvline(bi, color=P["text"], linewidth=1.6, zorder=4)
    bands_x(ax, P, min(float(tp.min()), bi) - 2, max(float(tp.max()), bi) + 2)
    ax.set_xlabel("BI predicted by each tree (line = forest average)", fontsize=8, color=P["text_secondary"])
    ax.set_ylabel("Number of trees", fontsize=8, color=P["text_secondary"])
    ax.tick_params(axis="y", length=0, colors=P["text_secondary"], labelsize=8)

def draw_range_position(ax, res, P):
    style_ax(ax, P)
    snap = res["input_snapshot"]
    feats = list(ordered_feats)
    y = np.arange(len(feats))[::-1]
    ax.barh(y, 1.0, color=P["track"], height=0.3, zorder=1)
    for yi, f in zip(y, feats):
        b = train_bounds[f]
        b_min, b_max = float(b["min"]), float(b["max"])
        span = b_max - b_min
        pos = float(np.clip((snap[f] - b_min) / span, 0, 1)) if span > 0 else 0.5
        med = float(np.clip((float(b.get("median", b_min)) - b_min) / span, 0, 1)) if span > 0 else 0.5
        edge = pos < 0.05 or pos > 0.95
        ax.plot([med, med], [yi - 0.24, yi + 0.24], color=P["text_secondary"], lw=1.4, zorder=2)
        ax.scatter([pos], [yi], s=90, color=P["warning_fg"] if edge else P["shap_pos"],
                   edgecolor=P["chart_bg"], linewidth=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pretty(f)} ({snap[f]:g})" for f in feats], fontsize=8.5, color=P["text"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["training min", "mid", "training max"])
    ax.set_title("Dot = your input  |  tick = training median  |  amber = near the edge of the data",
                 fontsize=8, color=P["text_secondary"], pad=8)

def draw_sensitivity(ax, res, f, P):
    style_ax(ax, P)
    snap = res["input_snapshot"]
    lo, hi = RANGES[f]
    xs = np.linspace(lo, hi, 60)
    vec = np.array([snap[k] for k in feature_names], dtype=float)
    Xs = np.tile(vec, (len(xs), 1))
    Xs[:, feature_names.index(f)] = xs
    ys = np.asarray(model.predict(Xs), dtype=float)
    cur = float(snap[f])
    ylo, yhi = min(float(ys.min()), float(res["pred_bi"])) - 1.5, max(float(ys.max()), float(res["pred_bi"])) + 1.5
    ax.plot(xs, ys, color=P["accent"], linewidth=2.2, zorder=3)
    ax.axvline(cur, color=P["text_secondary"], linestyle=":", linewidth=1.1, zorder=2)
    ax.scatter([cur], [res["pred_bi"]], s=70, color=P["text"], edgecolor=P["chart_bg"], linewidth=1.2, zorder=4)
    bands_y(ax, P, ylo, yhi)
    ax.set_xlim(lo, hi)
    unit = unit_of(f)
    ax.set_xlabel(f"{pretty(f)}" + (f" ({unit})" if unit else ""), fontsize=8.5, color=P["text_secondary"])
    ax.set_ylabel("Predicted BI", fontsize=8.5, color=P["text_secondary"])
    ax.tick_params(axis="y", length=0, colors=P["text_secondary"], labelsize=8)

def draw_history(ax, runs, P):
    style_ax(ax, P, grid="y")
    runs = runs[-15:]
    bis = [float(r["Predicted_BI"]) for r in runs]
    cats = [r["BI_Category"] for r in runs]
    cmap = {"Easy": P["badge_easy_fg"], "Medium": P["badge_med_fg"], "Hard": P["badge_hard_fg"]}
    labels = [(str(r["Label"])[:14] + ("…" if len(str(r["Label"])) > 14 else "")) for r in runs]
    x = np.arange(len(runs))
    ax.bar(x, bis, color=[cmap.get(c, P["accent"]) for c in cats], width=0.6, zorder=3)
    for xi, v in zip(x, bis):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=P["text_secondary"])
    for lvl in (EASY_MAX, HARD_MIN):
        ax.axhline(lvl, color=P["text_secondary"], linestyle="--", linewidth=0.9, zorder=2)
        ax.text(len(runs) - 0.5, lvl + 0.3, f"{lvl:g}", ha="right", va="bottom", fontsize=7.5, color=P["text_secondary"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25 if len(runs) > 4 else 0, ha="right" if len(runs) > 4 else "center", fontsize=8)
    ax.tick_params(axis="x", colors=P["text"])
    ax.set_ylim(0, max(max(bis) * 1.15, HARD_MIN + 5))
    ax.set_ylabel("Predicted BI", fontsize=8.5, color=P["text_secondary"])

# ── PDF report (generated ONLY when user clicks Download PDF report) ──
def build_pdf_report(res):
    P = THEME_LIGHT
    when = res.get("time") or datetime.now()
    snap, bi, cat = res["input_snapshot"], float(res["pred_bi"]), res["category"]
    stats = spread_stats(res["tree_preds"])
    g_title, g_body = guidance_parts(bi)
    footer = "Indicative only. Confirm against site blast trials. Northing is a site coordinate: use this model only at the mine it was trained on."
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.07, 0.945, "Blastability Index Report", fontsize=21, fontweight="bold", color=P["text"])
        fig.text(0.07, 0.922, f"Group 13  |  UMaT BSc Mining Engineering  •  Generated {when:%d %b %Y, %H:%M}",
                 fontsize=9, color=P["text_secondary"])
        fig.text(0.07, 0.865, "ESTIMATED LILLY BI", fontsize=9, fontweight="bold", color=P["text_secondary"])
        fig.text(0.07, 0.805, f"{bi:.2f}", fontsize=46, fontweight="bold", color=P["text"])
        fig.text(0.40, 0.818, f"{cat} Rock Mass", fontsize=17, fontweight="bold", color=P["accent"])
        if stats:
            fig.text(0.40, 0.790, f"Tree spread ±{stats['sd']:.2f}  |  5-95% of {stats['n']} trees: {stats['p5']:.1f} to {stats['p95']:.1f}",
                     fontsize=9, color=P["text_secondary"])
        axg = fig.add_axes([0.07, 0.715, 0.86, 0.045])
        draw_gauge(axg, res, P)
        fig.text(0.07, 0.675, textwrap.fill(f"{g_title} {g_body}", 105), fontsize=9.5, color=P["text"], va="top")
        if stats and stats["transition"]:
            fig.text(0.07, 0.640, textwrap.fill(
                "Transition zone: this estimate sits on the boundary between categories. "
                "Consider planning for both rock classes.", 105
            ), fontsize=9, color=P["warning_fg"], va="top")
        fig.text(0.07, 0.585, "INPUT PARAMETERS", fontsize=9, fontweight="bold", color=P["text_secondary"])
        axt = fig.add_axes([0.07, 0.17, 0.86, 0.40])
        axt.axis("off")
        rows = []
        for f in ordered_feats:
            b = train_bounds[f]
            lo, hi = RANGES[f]
            span = float(b["max"]) - float(b["min"])
            pos = (snap[f] - float(b["min"])) / span * 100 if span > 0 else 50.0
            rows.append([pretty(f), f"{snap[f]:g}", unit_of(f), f"{lo:g} to {hi:g}", f"{pos:.0f}%"])
        tbl = axt.table(cellText=rows, colLabels=["Parameter", "Value", "Unit", "Training range", "Position"],
                        colWidths=[0.30, 0.15, 0.10, 0.28, 0.17], loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.9)
        for (_, _), cell in tbl.get_celld().items():
            cell.set_edgecolor(P["track"])
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(P["accent"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("white" if r % 2 else P["card_bg"])
                cell.set_text_props(color=P["text"])
        fig.text(0.07, 0.04, textwrap.fill(footer, 120), fontsize=7.5, color=P["text_secondary"], va="bottom")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.text(0.07, 0.945, "Explanation", fontsize=18, fontweight="bold", color=P["text"])
        fig.text(0.07, 0.915, "Feature attribution (SHAP)", fontsize=10, fontweight="bold", color=P["text_secondary"])
        draw_shap(fig.add_axes([0.24, 0.72, 0.68, 0.17]), res, P, fs=8.5)
        if res["tree_preds"] is not None:
            fig.text(0.07, 0.675, "Spread of the individual trees", fontsize=10, fontweight="bold", color=P["text_secondary"])
            draw_tree_hist(fig.add_axes([0.10, 0.50, 0.82, 0.15]), res, P)
            next_y = 0.43
        else:
            next_y = 0.65
        fig.text(0.07, next_y, "Where the inputs sit inside the training data", fontsize=10, fontweight="bold", color=P["text_secondary"])
        draw_range_position(fig.add_axes([0.30, 0.08, 0.62, max(next_y - 0.12, 0.25)]), res, P)
        fig.text(0.07, 0.03, textwrap.fill(footer, 120), fontsize=7.5, color=P["text_secondary"], va="bottom")
        pdf.savefig(fig)
        plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "Blastability Index Report"
        info["Author"] = "Group 13, UMaT"
    return buf.getvalue()

def _extract_shap_1d(explainer_obj, input_vector, n_feats):
    raw = explainer_obj.shap_values(input_vector)
    if isinstance(raw, list):
        raw = raw[0]
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        arr = arr[0, :, 0]
    elif arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    if len(arr) != n_feats:
        raise ValueError(f"Expected {n_feats} SHAP values, got {len(arr)} from shape {np.asarray(raw).shape}")
    return arr

# ── Prediction (does NOT build PDF unless requested) ──
def run_prediction(current_inputs, loading_slot):
    with loading_slot:
        show_loading_overlay()
    t0 = time.time()
    try:
        input_vector = np.array([[float(current_inputs[f]) for f in feature_names]], dtype=float)
        pred_bi = float(model.predict(input_vector)[0])
        tree_preds = None
        if hasattr(model, "estimators_"):
            x32 = input_vector.astype(np.float32)
            tree_preds = np.array([float(t.predict(x32)[0]) for t in model.estimators_], dtype=float)
        shap_vals = _extract_shap_1d(explainer, input_vector, len(feature_names))
        ev = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        base_val = float(ev[0])
        res = {
            "pred_bi": pred_bi,
            "category": classify(pred_bi),
            "tree_preds": tree_preds,
            "shap_vals": shap_vals,
            "base_val": base_val,
            "input_snapshot": {k: float(v) for k, v in current_inputs.items()},
            "time": datetime.now(),
        }
        st.session_state["prediction_result"] = res
        st.session_state.pop("generated_pdf_bytes", None)
        st.session_state.pop("generated_pdf_error", None)
    finally:
        elapsed = time.time() - t0
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        loading_slot.empty()

# ── Saved runs ──
def save_current_run():
    res = st.session_state.get("prediction_result")
    if not res:
        return
    runs = st.session_state["saved_runs"]
    label = (st.session_state.get("save_label") or "").strip() or f"Run {len(runs) + 1}"
    rec = {
        "Label": label,
        "Saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Predicted_BI": round(float(res["pred_bi"]), 2),
        "BI_Category": res["category"],
    }
    rec.update({f: round(float(res["input_snapshot"][f]), 4) for f in feature_names})
    runs.append(rec)
    st.session_state["save_label"] = ""
    st.session_state["notice_result"] = f'Saved "{label}". You can compare it under Saved Runs.'

def load_run():
    runs = st.session_state["saved_runs"]
    i = st.session_state.get("run_pick", 0)
    if 0 <= i < len(runs):
        apply_preset({f: runs[i][f] for f in feature_names if f in runs[i]})
        st.session_state["notice_saved"] = f'Loaded "{runs[i]["Label"]}" into the inputs. Go to Prediction and press Predict BI.'

def delete_run():
    runs = st.session_state["saved_runs"]
    i = st.session_state.get("run_pick", 0)
    if 0 <= i < len(runs):
        gone = runs.pop(i)
        st.session_state["run_pick"] = 0
        st.session_state["notice_saved"] = f'Deleted "{gone["Label"]}".'

def clear_runs():
    st.session_state["saved_runs"] = []
    st.session_state["run_pick"] = 0
    st.session_state.pop("import_sig", None)
    st.session_state["notice_saved"] = "All saved runs cleared."

def import_runs_from(file):
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return f"Missing required columns: {missing}", 0
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        return "Some input cells are empty or not numeric.", 0
    df, X = df.head(200), X.head(200)
    preds = np.asarray(model.predict(X.values), dtype=float)
    runs = st.session_state["saved_runs"]
    for i, p in enumerate(preds):
        lab = df["Label"].iloc[i] if "Label" in df.columns and pd.notna(df["Label"].iloc[i]) else f"Imported {i + 1}"
        rec = {
            "Label": str(lab),
            "Saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Predicted_BI": round(float(p), 2),
            "BI_Category": classify(float(p)),
        }
        rec.update({f: round(float(X[f].iloc[i]), 4) for f in feature_names})
        runs.append(rec)
    return None, len(preds)

def render_saved_runs(wide=False):
    mode_tag = "canvas" if wide else "classic"
    runs = st.session_state["saved_runs"]
    render_html("""
    <div class="apple-card">
        <div class="apple-card-title">Saved Predictions</div>
        <div class="apple-card-subtitle">Kept for this browser session only. Export to CSV to keep them, and import the file later to restore them.</div>
    </div>
    """)
    notice = st.session_state.pop("notice_saved", None)
    if notice:
        banner("success", notice)
    if not runs:
        banner("info", "Nothing saved yet. Run a prediction, then press <strong>Save prediction</strong>.")
    else:
        df = pd.DataFrame(runs)
        st.dataframe(df, hide_index=True, use_container_width=True)
        if len(runs) >= 2:
            render_html('<div class="apple-card"> <div class="apple-card-title">Comparison</div> <div class="apple-card-subtitle">Predicted BI of your latest saved runs against the class cut-offs</div> </div>')
            fig, ax = new_fig(C, 12 if wide else 6.8, 3.6 if wide else 3.4)
            draw_history(ax, runs, C)
            plt.tight_layout()
            show_fig(fig)
        if st.session_state.get("run_pick", 0) >= len(runs):
            st.session_state["run_pick"] = 0
        st.selectbox("Select a saved run", list(range(len(runs))), key=f"run_pick_{mode_tag}",
                     on_change=lambda k=f"run_pick_{mode_tag}": st.session_state.update({"run_pick": st.session_state[k]}),
                     format_func=lambda i: f"{runs[i]['Label']}  (BI {runs[i]['Predicted_BI']:.2f}, {runs[i]['BI_Category']})")
        st.session_state["run_pick"] = st.session_state.get(f"run_pick_{mode_tag}", 0)
        b1, b2, b3 = st.columns(3)
        b1.button("Load inputs", on_click=load_run, use_container_width=True, key=f"btn_load_run_{mode_tag}")
        b2.button("Delete", on_click=delete_run, use_container_width=True, key=f"btn_del_run_{mode_tag}")
        b3.button("Clear all", on_click=clear_runs, use_container_width=True, key=f"btn_clear_runs_{mode_tag}")
        st.download_button("Export saved runs (.csv)", df.to_csv(index=False).encode("utf-8"),
                           file_name="Saved_BI_runs.csv", mime="text/csv", key=f"dl_saved_runs_{mode_tag}")
        with st.expander("Import runs from a CSV or Excel sheet"):
            st.caption("Needs the 8 input columns. An optional Label column names each run. The BI is recomputed by the model.")
            up = st.file_uploader("Import file", type=["csv", "xlsx"], key=f"import_file_{mode_tag}", label_visibility="collapsed")
            if up is not None:
                sig = f"{up.name}-{up.size}-{len(runs)}"
                if sig != st.session_state.get("import_sig"):
                    st.session_state["import_sig"] = sig
                    try:
                        err, n = import_runs_from(up)
                    except Exception as e:
                        err, n = f"Could not read the file: {e}", 0
                    if err:
                        banner("error", err)
                    else:
                        st.session_state["notice_saved"] = f"Imported {n} run(s)."
                        st.rerun()

# ── Batch UI helper ──
def render_batch_section(mode_tag: str):
    render_html("""<div class="apple-card"><div class="apple-card-title">Batch Sheet Estimation</div><div class="apple-card-subtitle">Upload a sheet (.xlsx or .csv) with the 8 input columns. Rows with inputs outside the training range are flagged.</div></div>""")
    template = pd.DataFrame([{f: round(float(train_bounds[f].get("median", 0.0)), 2) for f in feature_names}])
    st.download_button(
        "Download blank template (.csv)",
        template.to_csv(index=False).encode("utf-8"),
        file_name="BI_input_template.csv",
        mime="text/csv",
        key=f"dl_template_{mode_tag}",
    )
    uploaded_file = st.file_uploader("Upload sheet", type=["xlsx", "csv"], key=f"batch_upload_{mode_tag}", label_visibility="collapsed")
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
                    banner("warning", f"{n_out} of {len(result)} rows have inputs outside the training range.")
                else:
                    banner("success", f"All {len(result)} rows are inside the training range.")
                show_cols = ["Predicted_BI", "BI_Category", "Out_of_Range_Inputs"] + feature_names
                st.dataframe(result[show_cols], use_container_width=True, height=420)
                st.download_button(
                    "Download predictions (.csv)",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="Estimated_Blastability_Index.csv",
                    mime="text/csv",
                    key=f"dl_batch_preds_{mode_tag}",
                )
        except Exception as e:
            banner("error", f"Processing error: {e}")

# ── Result view ──
def chart_card(title, subtitle):
    render_html(f'<div class="apple-card"> <div class="apple-card-title">{title}</div> <div class="apple-card-subtitle">{subtitle}</div> </div>')

def render_actions(res, mode_tag):
    notice = st.session_state.pop("notice_result", None)
    if notice:
        banner("success", notice)
    st.text_input("Label for this run (optional)", key=f"save_label_{mode_tag}",
                  on_change=lambda k=f"save_label_{mode_tag}": st.session_state.update({"save_label": st.session_state[k]}),
                  placeholder="e.g. Bench 3, north wall")
    st.session_state["save_label"] = st.session_state.get(f"save_label_{mode_tag}", "")
    a, b = st.columns(2)
    a.button("Save prediction", on_click=save_current_run, use_container_width=True, key=f"btn_save_run_{mode_tag}")

    # Only build the PDF when the user explicitly clicks to prepare/download it
    pdf_bytes = st.session_state.get("generated_pdf_bytes")
    if pdf_bytes is None:
        if b.button("Download PDF report", use_container_width=True, key=f"btn_prepare_pdf_{mode_tag}"):
            try:
                st.session_state["generated_pdf_bytes"] = build_pdf_report(res)
                st.rerun()
            except Exception as e:
                st.session_state["generated_pdf_error"] = str(e)
                b.caption("PDF report is unavailable for this run.")
    else:
        b.download_button(
            "Save PDF file",
            pdf_bytes,
            file_name=f"BI_report_{res['time']:%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"btn_save_pdf_file_{mode_tag}",
        )

def render_result(res, current_inputs, wide):
    mode_tag = "canvas" if wide else "classic"
    pred_bi, category, snapshot = float(res["pred_bi"]), res["category"], res["input_snapshot"]
    if wide:
        render_html("""
        <div class="section-header">
            <div class="section-title">Prediction Result</div>
            <div class="section-subtitle">Estimated Blastability Index and operational guidance</div>
        </div>
        """)
    if any(not np.isclose(float(current_inputs[k]), float(snapshot[k]), atol=1e-3) for k in current_inputs):
        banner("warning", "Inputs changed since this prediction. Press <strong>Predict BI</strong> to update.")

    stats = spread_stats(res["tree_preds"])
    spread_html = ""
    if stats:
        spread_html = f'<div style="font-size:0.9rem;color:{C["text_secondary"]};margin-top:12px;">Tree spread ±{stats["sd"]:.2f}  |  5–95% of the {stats["n"]} trees: {stats["p5"]:.1f} – {stats["p95"]:.1f}</div>'
        if stats["transition"]:
            spread_html += f'<div style="font-size:0.86rem;color:{C["warning_fg"]};margin-top:8px;"><strong>Transition Zone:</strong> This estimate sits on the boundary between categories. Consider planning for both rock classes to ensure blast safety.</div>'

    pct = min(max((pred_bi - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN) * 100.0, 0.0), 100.0)
    g_title, g_body = guidance_parts(pred_bi)

    render_html(f"""
    <div class="apple-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.2rem;">
            <div style="flex: 1; min-width: 220px;">
                <div class="metric-label">Estimated Lilly BI</div>
                <div class="metric-value">{pred_bi:.2f}</div>
                {spread_html}
            </div>
            <div><span class="apple-badge {BADGE[category]}">{category} Rock Mass</span></div>
        </div>
        <div class="gauge-track"><div class="gauge-fill" style="width: {pct:.1f}%;"></div></div>
        <div class="gauge-labels">
            <span>Easy (&lt;{EASY_MAX:g})</span><span>Medium ({EASY_MAX:g} – {HARD_MIN:g})</span><span>Hard (&ge;{HARD_MIN:g})</span>
        </div>
        <div class="guidance-box"><strong>{g_title}</strong> {g_body}<div style="font-size:0.8rem;color:{C['text_secondary']};margin-top:8px;">Indicative only. Confirm against site blast trials.</div></div>
    </div>
    """)
    st.caption("Tree spread shows how much the individual trees disagree. It is not a formal confidence interval.")
    render_actions(res, mode_tag)

    # SHAP
    if wide:
        render_html("""
        <div class="section-header">
            <div class="section-title">Attribution & Explanation</div>
            <div class="section-subtitle">SHAP feature contributions pushing the BI higher or lower</div>
        </div>
        """)
    else:
        chart_card("Attribution & Explanation", "SHAP feature contributions pushing BI higher or lower")
    fig, ax = new_fig(C, 12 if wide else 6.8, 4.2 if wide else 3.4)
    draw_shap(ax, res, C, fs=9.5 if wide else 8.5)
    plt.tight_layout()
    show_fig(fig)

    # Deeper analysis
    if wide:
        render_html("""
        <div class="section-header">
            <div class="section-title">Deeper Analysis</div>
            <div class="section-subtitle">How certain the forest is, how typical your inputs are, and what happens if one input changes</div>
        </div>
        """)
        cA, cB = st.columns(2, gap="large")
    else:
        cA = cB = st.container()

    with cA:
        if res["tree_preds"] is not None:
            chart_card("Tree Spread", "How the individual trees vote. A narrow cluster means the forest agrees.")
            fig, ax = new_fig(C, 6 if wide else 6.8, 3.2 if wide else 2.8)
            draw_tree_hist(ax, res, C)
            plt.tight_layout()
            show_fig(fig)

    with cB:
        chart_card("Input Position in Training Data", "Predictions are most reliable well inside the range the model learned from.")
        fig, ax = new_fig(C, 6 if wide else 6.8, 3.2 if wide else 3.4)
        draw_range_position(ax, res, C)
        plt.tight_layout()
        show_fig(fig)

    chart_card("What-If Sweep", "Vary one parameter across its training range while the others stay at the values used for this prediction.")
    sweep = st.selectbox("Parameter to sweep", ordered_feats, format_func=pretty, key=f"whatif_feat_{mode_tag}")
    fig, ax = new_fig(C, 12 if wide else 6.8, 3.4 if wide else 3.0)
    draw_sensitivity(ax, res, sweep, C)
    plt.tight_layout()
    show_fig(fig)

# ═══════════════════════════════════════════════════════════════
# TOP BAR WITH BRAND & TOGGLES
# ═══════════════════════════════════════════════════════════════
def _on_theme_change():
    st.session_state.dark_mode = (st.session_state.theme_radio == "Dark")

def _on_layout_change():
    st.session_state.canvas_mode = (st.session_state.layout_radio == "Canvas")

topbar = st.container()
with topbar:
    st.markdown('<div class="st-key-topbar">', unsafe_allow_html=True)
    brand_col, toggle_col1, toggle_col2 = st.columns([2, 1, 1])

    with brand_col:
        st.markdown("""
        <div class="brand">
            <div class="brand-logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0F1C15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 3v18h18"/>
                    <path d="M7 14l4-4 4 4 5-5"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">Rock Blastability AI</div>
                <div class="brand-sub">Group 13 • UMaT Mining Engineering</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with toggle_col1:
        st.write("")
        st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_radio",
                 on_change=_on_theme_change, label_visibility="collapsed")

    with toggle_col2:
        st.write("")
        st.radio("Layout", ["Classic", "Canvas"], horizontal=True, key="layout_radio",
                 on_change=_on_layout_change, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LAYOUT ROUTER
# ═══════════════════════════════════════════════════════════════
if CANVAS:
    # ═══════════════════════════════════════════════════════════════
    #  BENTO CANVAS VIEW (Scrollable)
    # ═══════════════════════════════════════════════════════════════
    render_html("""
    <div class="apple-hero">
        <div class="hero-eyebrow">Explainable AI • Reduced Input Model</div>
        <div class="apple-title">Blastability Index Estimator</div>
        <div class="apple-subtitle">Eight geotechnical parameters selected by Hybrid PSO-GA. Get instant predictions with SHAP explanations.</div>
    </div>
    """)

    render_html("""
    <div class="section-header">
        <div class="section-title">Geotechnical Parameters</div>
        <div class="section-subtitle">Eight essential inputs, selected by Hybrid PSO-GA. Adjust any value — sliders and number boxes stay perfectly in sync.</div>
    </div>
    """)
    pc1, pc2, pc3 = st.columns([1, 1, 1])
    pc1.button("Hard Domain Preset", on_click=apply_preset, args=(PRESET_HARD,), use_container_width=True, key="preset_hard_canvas")
    pc2.button("Medium Domain Preset", on_click=apply_preset, args=(PRESET_MEDIUM,), use_container_width=True, key="preset_med_canvas")
    pc3.button("Reset to Median", on_click=reset_to_median, use_container_width=True, key="preset_med_val_canvas")

    row1 = st.columns(4, gap="medium")
    row2 = st.columns(4, gap="medium")
    for i, f in enumerate(ordered_feats):
        title, unit, _ = INPUT_SPEC.get(f, (f, "", step_for(f)))
        step = step_for(f)
        lo, hi = RANGES[f]
        col = row1[i] if i < 4 else row2[i - 4]
        with col:
            render_html(f"""
            <div class="micro-card">
                <div class="micro-title">{title}</div>
                <div class="micro-unit">{unit}  •  Range: {lo:g} – {hi:g}</div>
            </div>
            """)
            st.number_input(title, min_value=lo, max_value=hi, step=step, format="%.2f",
                            key=f"num_{f}", on_change=sync_from_num, args=(f,), label_visibility="collapsed")
            st.slider(title, min_value=lo, max_value=hi, step=step, format="%.2f",
                      key=f"sld_{f}", on_change=sync_from_sld, args=(f,), label_visibility="collapsed")

    st.caption("Northing is a site coordinate, not a rock property: use this model only at the mine it was trained on.")
    current_inputs = {f: float(st.session_state[f"num_{f}"]) for f in feature_names}
    predict_clicked = st.button("Predict BI", type="primary", use_container_width=True, key="predict_btn_canvas")
    loading_slot = st.empty()

    if predict_clicked:
        run_prediction(current_inputs, loading_slot)

    if "prediction_result" in st.session_state:
        render_result(st.session_state["prediction_result"], current_inputs, wide=True)

    render_html("""
    <div class="section-header">
        <div class="section-title">Saved Runs</div>
        <div class="section-subtitle">Compare predictions across benches and reload their inputs</div>
    </div>
    """)
    render_saved_runs(wide=True)

    render_html("""
    <div class="section-header">
        <div class="section-title">Advanced Tools</div>
        <div class="section-subtitle">Batch processing and model research metrics</div>
    </div>
    """)
    tool_col1, tool_col2 = st.columns([1.15, 1], gap="large")
    with tool_col1:
        render_batch_section("canvas")

    with tool_col2:
        render_html("""<div class="apple-card"><div class="apple-card-title">Model Performance Comparison</div><div class="apple-card-subtitle">Evaluated on 163 held-out bench records (unseen benches)</div></div>""")
        comp_df = pd.DataFrame([
            {"Model": "Mean Baseline", "Inputs": 0, "R²": -0.0102, "RMSE": 4.5383, "MAE": 3.5889, "Category Match": "74.85%"},
            {"Model": "Linear Regression (Full)", "Inputs": 26, "R²": 0.5546, "RMSE": 3.0136, "MAE": 2.4477, "Category Match": "72.39%"},
            {"Model": "Random Forest (Full)", "Inputs": 26, "R²": 0.8692, "RMSE": 1.6331, "MAE": 1.0623, "Category Match": "93.87%"},
            {"Model": "Random Forest (PSO-GA)", "Inputs": 8, "R²": 0.8862, "RMSE": 1.5231, "MAE": 0.9489, "Category Match": "95.09%"},
            {"Model": "Random Forest (FRFS)", "Inputs": 23, "R²": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Match": "93.25%"},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption("Category match is the share of the 163 test records placed in the correct Easy/Medium/Hard class.")

        render_html(f"""
        <div class="apple-card">
            <div class="apple-card-title">Engineering Stability</div>
            <div class="apple-card-subtitle">30 bootstrap resamples with replacement</div>
            <div style="margin-bottom: 1.2rem;">
                <div class="metric-label">SHAP Ranking Stability (Spearman ρ)</div>
                <div style="font-size: 2rem; font-weight: 700; color: {C['accent']};">0.8280 ± 0.055</div>
                <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">Suggests the feature rankings stay broadly similar across resamples.</div>
            </div>
            <div>
                <div class="metric-label">Selection Frequency (Dip Direction / Spacing)</div>
                <div style="font-size: 2rem; font-weight: 700; color: {C['accent']};">100% / 96.7%</div>
                <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">Dip Direction was selected in 30/30 resamples; Spacing in 29/30.</div>
            </div>
        </div>
        """)

else:
    # ═══════════════════════════════════════════════════════════════
    #  CLASSIC STUDIO VIEW (Split-pane with Tabs)
    # ═══════════════════════════════════════════════════════════════
    render_html("""
    <div class="apple-hero">
        <div class="hero-eyebrow">Explainable AI • Reduced Input Model</div>
        <div class="apple-title">Blastability Index Estimator</div>
        <div class="apple-subtitle">Eight geotechnical parameters selected by Hybrid PSO-GA. Get instant predictions with SHAP explanations.</div>
    </div>
    """)

    tab1, tab_saved, tab2, tab3 = st.tabs(["Prediction & XAI", "Saved Runs", "Batch Upload", "Performance & Stability"])

    with tab1:
        col_input, col_output = st.columns([1, 1.15], gap="large")
        with col_input:
            render_html("""
            <div class="apple-card">
                <div class="apple-card-title">Geotechnical Parameters</div>
                <div class="apple-card-subtitle">Eight essential parameters, carefully selected by PSO-GA. Adjust the numbers or drag the sliders — everything stays perfectly in sync and bounded by the model's training data.</div>
            </div>
            """)
            pcol1, pcol2 = st.columns(2)
            pcol1.button("Hard Domain Preset", on_click=apply_preset, args=(PRESET_HARD,), use_container_width=True, key="preset_hard_classic")
            pcol2.button("Medium Domain Preset", on_click=apply_preset, args=(PRESET_MEDIUM,), use_container_width=True, key="preset_med_classic")

            cols = st.columns(2)
            half = (len(ordered_feats) + 1) // 2
            for i, f in enumerate(ordered_feats):
                label, _, _ = INPUT_SPEC.get(f, (f, "", step_for(f)))
                step = step_for(f)
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
            predict_clicked = st.button("Predict BI", type="primary", use_container_width=True, key="predict_btn_classic")
            loading_slot = st.empty()

            if predict_clicked:
                run_prediction(current_inputs, loading_slot)

        with col_output:
            if "prediction_result" not in st.session_state:
                render_html(f"""
                <div class="apple-card" style="text-align: center; padding: 4rem 2rem;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="{C['accent']}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 0.9rem;">
                        <path d="M3 3v18h18"/>
                        <path d="M7 14l4-4 4 4 5-5"/>
                    </svg>
                    <div class="apple-card-title">Ready to analyze</div>
                    <div class="apple-card-subtitle" style="margin-bottom: 0; max-width: 460px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                        Set your field parameters on the left, then click <strong>Predict BI</strong> to see the estimate, tree spread, and SHAP explanation.
                    </div>
                </div>
                """)
            else:
                render_result(st.session_state["prediction_result"], current_inputs, wide=False)

    with tab_saved:
        render_saved_runs(wide=False)

    with tab2:
        render_batch_section("classic")

    with tab3:
        col_t1, col_t2 = st.columns([1.2, 1], gap="large")
        with col_t1:
            render_html("""<div class="apple-card"><div class="apple-card-title">Model Performance Comparison</div><div class="apple-card-subtitle">Evaluated on 163 held-out bench records (unseen benches)</div></div>""")
            comp_df = pd.DataFrame([
                {"Model": "Mean Baseline", "Inputs": 0, "R²": -0.0102, "RMSE": 4.5383, "MAE": 3.5889, "Category Match": "74.85%"},
                {"Model": "Linear Regression (Full)", "Inputs": 26, "R²": 0.5546, "RMSE": 3.0136, "MAE": 2.4477, "Category Match": "72.39%"},
                {"Model": "Random Forest (Full)", "Inputs": 26, "R²": 0.8692, "RMSE": 1.6331, "MAE": 1.0623, "Category Match": "93.87%"},
                {"Model": "Random Forest (PSO-GA)", "Inputs": 8, "R²": 0.8862, "RMSE": 1.5231, "MAE": 0.9489, "Category Match": "95.09%"},
                {"Model": "Random Forest (FRFS)", "Inputs": 23, "R²": 0.8748, "RMSE": 1.5975, "MAE": 0.9937, "Category Match": "93.25%"},
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            st.caption("Category match is the share of the 163 test records placed in the correct Easy/Medium/Hard class.")

        with col_t2:
            render_html(f"""
            <div class="apple-card">
                <div class="apple-card-title">Engineering Stability</div>
                <div class="apple-card-subtitle">30 bootstrap resamples with replacement</div>
                <div style="margin-bottom: 1.2rem;">
                    <div class="metric-label">SHAP Ranking Stability (Spearman ρ)</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {C['accent']};">0.8280 ± 0.055</div>
                    <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">Suggests the feature rankings stay broadly similar across resamples.</div>
                </div>
                <div>
                    <div class="metric-label">Selection Frequency (Dip Direction / Spacing)</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {C['accent']};">100% / 96.7%</div>
                    <div style="font-size: 0.82rem; color: {C['text_secondary']}; margin-top: 2px;">Dip Direction was selected in 30/30 resamples; Spacing in 29/30.</div>
                </div>
            </div>
            """)