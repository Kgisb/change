
# ---- Compact Status Bar (badges) ----

def _render_status_bar(excluded_count: int, excluded_col: str, rows_in_scope: int, track_val: str):
    import streamlit as st
    html = f"""
    <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:2px 0 4px;">
      <span style="font-size:12px; color:#64748B;">
        <span style="opacity:.85">Excluded</span>
        <span>“1.2 Invalid deal(s)”</span>
        <span style="opacity:.6">·</span>
        <span>{excluded_count:,} rows</span>
        <span style="opacity:.55">({excluded_col})</span>
      </span>
      <span style="font-size:12px; color:#64748B;">
        <span style="opacity:.85">In scope</span>
        <span>{rows_in_scope:,}</span>
      </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
# ---- Global Refresh (reset filters & cache) ----
def _reset_all_filters_and_cache(preserve_nav=True):
    import streamlit as st
    for clear_fn in (
        getattr(getattr(st, "cache_data", object()), "clear", None),
        getattr(getattr(st, "cache_resource", object()), "clear", None),
        getattr(getattr(st, "experimental_memo", object()), "clear", None),
        getattr(getattr(st, "experimental_singleton", object()), "clear", None)
    ):
        try:
            if callable(clear_fn):
                clear_fn()
        except Exception:
            pass
    keep_keys = set()
    if preserve_nav:
        keep_keys |= {"nav_master", "nav_sub", "nav_master_prev"}
    rm_tags = [
        "filter", "selected", "select", "multiselect", "radio", "checkbox",
        "date", "from", "to", "range", "track", "cohort", "pareto",
        "country", "state", "city", "source", "deal", "stage",
        "owner", "counsellor", "counselor", "team",
        "segment", "sku", "plan", "product",
        "data_src_input"
    ]
    to_delete = []
    for k in list(st.session_state.keys()):
        if k in keep_keys:
            continue
        kl = k.lower()
        if any(tag in kl for tag in rm_tags):
            to_delete.append(k)
    for k in to_delete:
        try:
            del st.session_state[k]
        except Exception:
            pass
# ---- Global CSS polish (no logic change) ----
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px !important; padding-top: 1.2rem !important; padding-bottom: 2.0rem !important; }
        .stAltairChart, .stPlotlyChart, .stVegaLiteChart, .stDataFrame, .stTable, .element-container [data-baseweb="table"] {
            border: 1px solid #e7e8ea; border-radius: 16px; padding: 14px; background: #ffffff; box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
        }
        .stDataFrame [role="grid"] { border-radius: 12px; overflow: hidden; border: 1px solid #e7e8ea; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        div[data-testid="stMetric"] { background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%); border: 1px solid #eef0f2; border-radius: 16px; padding: 14px 16px; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        details[data-testid="stExpander"] { border: 1px solid #e7e8ea; border-radius: 14px; background: #ffffff; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        details[data-testid="stExpander"] summary { font-weight: 600; color: #0f172a; }
        button[role="tab"] { border-radius: 999px !important; padding: 8px 14px !important; margin-right: 6px !important; border: 1px solid #e7e8ea !important; }
        button[role="tab"][aria-selected="true"] { background: #111827 !important; color: #ffffff !important; border-color: #111827 !important; }
        div[data-baseweb="select"], .stTextInput > div, .stNumberInput > div, .stDateInput > div { border-radius: 12px !important; box-shadow: 0 1px 4px rgba(16,24,40,.04); }
        .stSlider > div { padding-top: 10px; }
        .stButton > button, .stDownloadButton > button { border-radius: 12px !important; border: 1px solid #11182720 !important; box-shadow: 0 2px 8px rgba(16,24,40,.08) !important; transition: transform .05s ease-in-out; }
        .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0f172a; letter-spacing: 0.1px; }
        .stMarkdown hr { margin: 18px 0; border: none; border-top: 1px dashed #d6d8db; }
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception:
    pass

# app.py — JetLearn: MIS + Predictibility + Trend & Analysis + 80-20 (Merged, de-conflicted)

import streamlit as st

# --- Safe default for optional UI text blobs ---
try:
    DATA_SOURCE_TEXT
except NameError:
    DATA_SOURCE_TEXT = ""
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re
from datetime import date, timedelta

# ======================
# Page & minimal styling
# ======================
st.set_page_config(page_title="JetLearn – MIS + Trend + 80-20", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .stAltairChart {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
      }
      .legend-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
      }
      .pill-total { background: #e5e7eb; }
      .pill-ai    { background: #bfdbfe; }
      .pill-math  { background: #bbf7d0; }

      .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
      }
      .kpi-title { color:#6b7280; font-size:.9rem; margin-bottom:6px; }
      .kpi-value { font-weight:700; font-size:1.4rem; color:#111827; }
      .kpi-sub   { color:#6b7280; font-size:.85rem; }
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .25rem;
      }
      .chip {
        display:inline-block; padding:4px 8px; border-radius:999px;
        background:#f3f4f6; color:#374151; font-size:.8rem; margin-top:.25rem;
      }
      .muted { color:#6b7280; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = {
    "Total": "#6b7280",
    "AI Coding": "#2563eb",
    "Math": "#16a34a",
    "ThresholdLow": "#f3f4f6",
    "ThresholdMid": "#e5e7eb",
    "ThresholdHigh": "#d1d5db",
    "A_actual": "#2563eb",
    "Rem_prev": "#6b7280",
    "Rem_same": "#16a34a",
}

# ======================
# Helpers (shared)
# ======================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=series.index if series is not None else None)
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if s.notna().sum() == 0:
        for unit in ["s", "ms"]:
            try:
                s = pd.to_datetime(series, errors="coerce", unit=unit)
                break
            except Exception:
                pass
    return s

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

# Invalid deals exclusion
INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", flags=re.IGNORECASE)
def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    mask_keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v: return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

# Key-source mapping (Referral / PM buckets)
def normalize_key_source(val: str) -> str:
    if not isinstance(val, str): return "Other"
    v = val.strip().lower()
    if "referr" in v: return "Referral"
    if "pm" in v and "search" in v: return "PM - Search"
    if "pm" in v and "social" in v: return "PM - Social"
    return "Other"

def assign_src_pick(df: pd.DataFrame, source_col: str | None, use_key: bool) -> pd.DataFrame:
    d = df.copy()
    if source_col and source_col in d.columns:
        if use_key:
            d["_src_pick"] = d[source_col].apply(normalize_key_source)
        else:
            d["_src_pick"] = d[source_col].fillna("Unknown").astype(str)
    else:
        d["_src_pick"] = "Other"
    return d

# ======================
# Load data & global sidebar
# ======================
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"  # point to /mnt/data/Master_sheet-DB.csv if needed

if "data_src" not in st.session_state:
    st.session_state["data_src"] = DEFAULT_DATA_PATH

def _update_data_src():
    import streamlit as st
    DEFAULT = globals().get('DEFAULT_DATA_PATH', 'Master_sheet-DB.csv')
    st.session_state['data_src'] = st.session_state.get('data_src_input', DEFAULT)
    try:
        st.rerun()
    except Exception:
        pass
    import streamlit as st
    DEFAULT = globals().get('DEFAULT_DATA_PATH', 'Master_sheet-DB.csv')
    st.session_state['data_src'] = st.session_state.get('data_src_input', DEFAULT)
    try:
        st.rerun()
    except Exception:
        pass

with st.sidebar:
    st.header("JetLearn • Navigation")
    # Master tabs -> Sub tabs (contextual)
    MASTER_SECTIONS = {
        "Performance": ["Cash-in","Dashboard","MIS","Daily Business","Sales Tracker","AC Wise Detail"],
        "Funnel & Movement": ["Funnel","Lead Movement","Stuck deals","Deal Velocity","Deal Decay","Carry Forward"],
        "Insights & Forecast": ["Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph"],
        "Marketing": ["Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement"],
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    # Replace sidebar 'View' radio with session wiring (UI moves to main area)
    sub_views = MASTER_SECTIONS.get(master, [])
    if 'nav_sub' not in st.session_state or st.session_state.get('nav_master_prev') != master:
        st.session_state['nav_sub'] = sub_views[0] if sub_views else ''
    st.session_state['nav_master_prev'] = master
    sub = st.session_state['nav_sub']
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)
    st.caption("Use MIS for status; Predictibility for forecast; Trend & Analysis for grouped drilldowns; 80-20 for Pareto & Mix.")


    st.markdown("<div style=\"height:6px\"></div>", unsafe_allow_html=True)
    st.markdown("<div style=\"height:4px\"></div>", unsafe_allow_html=True)
    try:
        _trk = track if 'track' in locals() else st.session_state.get('track', '')
        if _trk:
            st.caption(f"<span data-testid=\"track-caption-bottom\">Track: <strong>{_trk}</strong></span>", unsafe_allow_html=True)
    except Exception:
        pass
view = sub

st.title("📊 JetLearn – Unified App")





# --- Top-right Refresh button (reset filters + cache) ---
with st.container():
    _cols = st.columns([1,1,1,1,1,1,1,1,1,1,1,2])
    with _cols[-1]:
        st.markdown('<div id="refresh-ctl">', unsafe_allow_html=True)
        if st.button("↻", key="refresh_all_btn"):
            _reset_all_filters_and_cache(preserve_nav=True)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
# --- Breadcrumb path (below title) ---
try:
    _master = master if 'master' in locals() else st.session_state.get('nav_master', '')
    _view = st.session_state.get('nav_sub', locals().get('view', ''))
    if not _view and 'MASTER_SECTIONS' in globals():
        _cands = MASTER_SECTIONS.get(_master, [])
        _view = _cands[0] if _cands else ''
    _track = locals().get('track', st.session_state.get('track', ''))
    track_html = f" <span style='opacity:.5'>&nbsp;•&nbsp;</span> <span style='opacity:.8'>{_track}</span>" if _track else ''
    html = f"""<div style='margin:4px 0 8px; font-size:12.5px; color:#334155;'>
    <span style='opacity:.9'>{_master or ''}</span>
    <span style='opacity:.5'> &nbsp;›&nbsp; </span>
    <strong style='font-weight:600'>{_view or ''}</strong>{track_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)
except Exception:
    pass
# --- Right-side sub-view pills (under the title) ---
sub_views = MASTER_SECTIONS.get(master, [])
cur_sub = st.session_state.get('nav_sub', sub_views[0] if sub_views else '')
if sub_views and cur_sub not in sub_views:
    cur_sub = sub_views[0]
    st.session_state['nav_sub'] = cur_sub
st.markdown("<div style='margin:2px 0 6px; font-size:12px; opacity:.85'>Views</div>", unsafe_allow_html=True)
cols = st.columns(min(4, max(1, len(sub_views)))) if sub_views else []
for i, v in enumerate(sub_views):
    with cols[i % len(cols)]:
        is_active = (v == cur_sub)
        if is_active:
            st.markdown(
    f"""<div class='pill-live' data-pill='{v}' style='display:block; width:100%; text-align:center; padding:8px 12px; border-radius:999px; border:1px solid #1E40AF; background:#1D4ED8; color:#fff; font-weight:600; position:relative; overflow:hidden;'>
      <span class='pill-dot'></span>{v}
      <span class='pill-sheen'></span>
    </div>""",
    unsafe_allow_html=True
)
        else:
            btn = st.button(v, key=f'mainpill_{v}', use_container_width=True)
            if btn:
                st.session_state['nav_sub'] = v
                cur_sub = v
                st.rerun()
view = st.session_state.get('nav_sub', cur_sub)


# Legend pills (for MIS/Trend visuals)
def active_labels(track: str) -> list[str]:
    if track == "AI Coding":
        return ["Total", "AI Coding"]
    if track == "Math":
        return ["Total", "Math"]
    return ["Total", "AI Coding", "Math"]

legend_labels = active_labels(track)
pill_map = {
    "Total": "<span class='legend-pill pill-total'>Total (Both)</span>",
    "AI Coding": "<span class='legend-pill pill-ai'>AI-Coding</span>",
    "Math": "<span class='legend-pill pill-math'>Math</span>",
}
if view == "MIS":
    st.markdown("<div>" + "".join(pill_map[l] for l in legend_labels) + "</div>", unsafe_allow_html=True)

# Data load
data_src = st.session_state["data_src"]
df = load_csv(data_src)

# Column mapping
dealstage_col = find_col(df, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
df, _removed = exclude_invalid_deals(df, dealstage_col)
if dealstage_col:
    _excluded_count, _excluded_col = _removed, dealstage_col
else:
    st.info("Deal Stage column not found — cannot auto-exclude “1.2 Invalid deal(s)”. Check your file.")

create_col = find_col(df, ["Create Date","Create date","Create_Date","Created At"])
pay_col    = find_col(df, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
pipeline_col = find_col(df, ["Pipeline"])
counsellor_col = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor"])
country_col    = find_col(df, ["Country"])
source_col     = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
first_cal_sched_col = find_col(df, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
cal_resched_col     = find_col(df, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
cal_done_col        = find_col(df, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])
calibration_slot_col = find_col(df, ["Calibration Slot (Deal)", "Calibration Slot", "Cal Slot (Deal)", "Cal Slot"])


if not create_col or not pay_col:
    st.error("Could not find required date columns. Need 'Create Date' and 'Payment Received Date' (or close variants).")
    st.stop()

# Clean invalid Create Date
tmp_create_all = coerce_datetime(df[create_col])
missing_create = int(tmp_create_all.isna().sum())
if missing_create > 0:
    df = df.loc[tmp_create_all.notna()].copy()
    st.caption(f"Removed rows with missing/invalid *Create Date: **{missing_create:,}*")

# Presets
today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)
this_m_end_mtd = today

# Global filters for MIS/Pred/Trend
def prep_options(series: pd.Series):
    vals = sorted([str(v) for v in series.dropna().unique()])
    return ["All"] + vals

with st.sidebar.expander("Data & Filters (Global for MIS / Predictibility / Trend & Analysis)", expanded=False):
    st.caption("These filters apply globally across MIS, Predictibility, and Trend & Analysis.")
    
    if counsellor_col:
        sel_counsellors = st.multiselect("Academic Counsellor", options=prep_options(df[counsellor_col]), default=["All"])
    else:
        sel_counsellors = []
        st.info("Academic Counsellor column not found.")
    
    if country_col:
        sel_countries = st.multiselect("Country", options=prep_options(df[country_col]), default=["All"])
    else:
        sel_countries = []
        st.info("Country column not found.")
    
    if source_col:
        sel_sources = st.multiselect("JetLearn Deal Source", options=prep_options(df[source_col]), default=["All"])
    else:
        sel_sources = []
        st.info("JetLearn Deal Source column not found.")
    

    with st.expander("Data file path", expanded=False):
        st.caption("Set/override the CSV path. Kept here to reduce clutter.")
        try:
            _cur_default = st.session_state.get("data_src", DEFAULT_DATA_PATH if "DEFAULT_DATA_PATH" in globals() else "Master_sheet-DB.csv")
        except Exception:
            _cur_default = "Master_sheet-DB.csv"
        st.text_input("CSV path", key="data_src_input", value=_cur_default, on_change=_update_data_src)
def apply_filters(
    df: pd.DataFrame,
    counsellor_col: str | None,
    country_col: str | None,
    source_col: str | None,
    sel_counsellors: list[str],
    sel_countries: list[str],
    sel_sources: list[str],
) -> pd.DataFrame:
    f = df.copy()
    if counsellor_col and sel_counsellors and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and sel_countries and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and sel_sources and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)

if track != "Both":
    if pipeline_col and pipeline_col in df_f.columns:
        _norm = df_f[pipeline_col].map(normalize_pipeline).fillna("Other")
        df_f = df_f.loc[_norm == track].copy()
    else:
        st.warning("Pipeline column not found — the Track filter can’t be applied.", icon="⚠")

_render_status_bar(_excluded_count, _excluded_col, len(df_f), track)

# ======================
# Shared functions for MIS / Trend / Predictibility
# ======================
def prepare_counts_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    month_for_mtd: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col])
    d["_pay_dt"] = coerce_datetime(d[pay_col])

    in_range_pay = d["_pay_dt"].dt.date.between(start_d, end_d)
    m_start, m_end = month_bounds(month_for_mtd)
    in_month_create = d["_create_dt"].dt.date.between(m_start, m_end)

    cohort_df = d.loc[in_range_pay]
    mtd_df = d.loc[in_range_pay & in_month_create]

    if pipeline_col and pipeline_col in d.columns:
        cohort_split = cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        mtd_split = mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        cohort_split = pd.Series([], dtype=object)
        mtd_split = pd.Series([], dtype=object)

    cohort_counts = {
        "Total": int(len(cohort_df)),
        "AI Coding": int((pd.Series(cohort_split) == "AI Coding").sum()),
        "Math": int((pd.Series(cohort_split) == "Math").sum()),
    }
    mtd_counts = {
        "Total": int(len(mtd_df)),
        "AI Coding": int((pd.Series(mtd_split) == "AI Coding").sum()),
        "Math": int((pd.Series(mtd_split) == "Math").sum()),
    }
    return mtd_counts, cohort_counts

def deals_created_mask_range(df: pd.DataFrame, denom_start: date, denom_end: date, create_col: str) -> pd.Series:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    return d["_create_dt"].between(denom_start, denom_end)

def prepare_conversion_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    *,
    denom_start: date,
    denom_end: date
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    d["_pay_dt"] = coerce_datetime(d[pay_col]).dt.date

    denom_mask = deals_created_mask_range(d, denom_start, denom_end, create_col)

    if pipeline_col and pipeline_col in d.columns:
        pl = d[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        pl = pd.Series(["Other"] * len(d), index=d.index)

    den_total = int(denom_mask.sum()); den_ai = int((denom_mask & (pl == "AI Coding")).sum()); den_math = int((denom_mask & (pl == "Math")).sum())
    denoms = {"Total": den_total, "AI Coding": den_ai, "Math": den_math}

    pay_mask = d["_pay_dt"].between(start_d, end_d)

    mtd_mask = pay_mask & denom_mask
    mtd_total = int(mtd_mask.sum()); mtd_ai = int((mtd_mask & (pl == "AI Coding")).sum()); mtd_math = int((mtd_mask & (pl == "Math")).sum())

    coh_mask = pay_mask
    coh_total = int(coh_mask.sum()); coh_ai = int((coh_mask & (pl == "AI Coding")).sum()); coh_math = int((coh_mask & (pl == "Math")).sum())

    def pct(n, d):
        if d == 0: return 0.0
        return max(0.0, min(100.0, round(100.0 * n / d, 1)))

    mtd_pct = {"Total": pct(mtd_total, den_total), "AI Coding": pct(mtd_ai, den_ai), "Math": pct(mtd_math, den_math)}
    coh_pct = {"Total": pct(coh_total, den_total), "AI Coding": pct(coh_ai, den_ai), "Math": pct(coh_math, den_math)}
    numerators = {"mtd": {"Total": mtd_total, "AI Coding": mtd_ai, "Math": mtd_math}, "cohort": {"Total": coh_total, "AI Coding": coh_ai, "Math": coh_math}}
    return mtd_pct, coh_pct, denoms, numerators

def bubble_chart_counts(title: str, total: int, ai_cnt: int, math_cnt: int, labels: list[str] = None):
    all_rows = [
        {"Label": "Total",     "Value": total,   "Row": 0, "Col": 0.5},
        {"Label": "AI Coding", "Value": ai_cnt,  "Row": 1, "Col": 0.33},
        {"Label": "Math",      "Value": math_cnt,"Row": 1, "Col": 0.66},
    ]
    if labels is None:
        labels = ["Total", "AI Coding", "Math"]
    data = pd.DataFrame([r for r in all_rows if r["Label"] in labels])

    color_domain = labels
    color_range_map = {"Total": PALETTE["Total"], "AI Coding": PALETTE["AI Coding"], "Math": PALETTE["Math"]}
    color_range = [color_range_map[l] for l in labels]

    base = alt.Chart(data).encode(
        x=alt.X("Col:Q", axis=None, scale=alt.Scale(domain=(0, 1))),
        y=alt.Y("Row:Q", axis=None, scale=alt.Scale(domain=(-0.2, 1.2))),
        tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Value:Q")],
    )
    circles = base.mark_circle(opacity=0.85).encode(
        size=alt.Size("Value:Q", scale=alt.Scale(range=[400, 8000]), legend=None),
        color=alt.Color("Label:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
    )
    text = base.mark_text(fontWeight="bold", dy=0, color="#111827").encode(text=alt.Text("Value:Q"))
    return (circles + text).properties(height=360, title=title)

def conversion_kpis_only(title: str, pcts: dict, nums: dict, denoms: dict, labels: list[str]):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    order = [l for l in ["Total", "AI Coding", "Math"] if l in labels]
    cols = st.columns(len(order))
    for i, label in enumerate(order):
        color = {"Total":"#111827","AI Coding":PALETTE["AI Coding"],"Math":PALETTE["Math"]}[label]
        with cols[i]:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>{label}</div>"
                f"<div class='kpi-value' style='color:{color}'>{pcts[label]:.1f}%</div>"
                f"<div class='kpi-sub'>Den: {denoms.get(label,0):,} • Num: {nums.get(label,0):,}</div></div>",
                unsafe_allow_html=True,
            )

def trend_timeseries(
    df: pd.DataFrame,
    payments_start: date,
    payments_end: date,
    *,
    denom_start: date,
    denom_end: date,
    create_col: str = "",
    pay_col: str = ""
):
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col]).dt.date
    df["_pay_dt"] = coerce_datetime(df[pay_col]).dt.date

    base_start = min(payments_start, denom_start)
    base_end = max(payments_end, denom_end)
    denom_mask = df["_create_dt"].between(denom_start, denom_end)

    all_days = pd.date_range(base_start, base_end, freq="D").date

    leads = (
        df.loc[denom_mask]
          .groupby("_create_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Leads")
    )
    pay_mask = df["_pay_dt"].between(payments_start, payments_end)
    cohort = (
        df.loc[pay_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Cohort")
    )
    mtd = (
        df.loc[pay_mask & denom_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("MTD")
    )

    ts = pd.concat([leads, mtd, cohort], axis=1).fillna(0).reset_index()
    ts = ts.rename(columns={"index": "Date"})
    return ts

def trend_chart(ts: pd.DataFrame, title: str):
    base = alt.Chart(ts).encode(x=alt.X("Date:T", axis=alt.Axis(title=None)))
    bars = base.mark_bar(opacity=0.75).encode(
        y=alt.Y("Leads:Q", axis=alt.Axis(title="Leads (deals created)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Leads:Q")]
    ).properties(height=260)
    line_mtd = base.mark_line(point=True).encode(
        y=alt.Y("MTD:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["AI Coding"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("MTD:Q", title="MTD Enrolments")]
    )
    line_coh = base.mark_line(point=True).encode(
        y=alt.Y("Cohort:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["Math"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cohort:Q", title="Cohort Enrolments")]
    )
    return alt.layer(bars, line_mtd, line_coh).resolve_scale(y='independent').properties(title=title)

# ======================
# MIS rendering
# ======================
def render_period_block(
    df_scope: pd.DataFrame,
    title: str,
    range_start: date,
    range_end: date,
    running_month_anchor: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    track: str
):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    labels = active_labels(track)

    # Counts
    mtd_counts, coh_counts = prepare_counts_for_range(
        df_scope, range_start, range_end, running_month_anchor, create_col, pay_col, pipeline_col
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)",
                                            mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"],
                                            labels=labels), use_container_width=True)
    with c2:
        st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)",
                                            coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"],
                                            labels=labels), use_container_width=True)

    # Conversion% (denominator = create dates within selected window) — KPI only
    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
        df_scope, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_start=range_start, denom_end=range_end
    )
    st.caption("Denominators (selected window create dates) — " +
               " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in labels]))

    conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=labels)
    conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=labels)

    # Trend uses SAME population rule
    ts = trend_timeseries(df_scope, range_start, range_end,
                          denom_start=range_start, denom_end=range_end,
                          create_col=create_col, pay_col=pay_col)
    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# ======================
# Predictibility helpers
# ======================
def add_month_cols(df: pd.DataFrame, create_col: str, pay_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(df[create_col])
    d["_pay_dt"]    = coerce_datetime(df[pay_col])
    d["_create_m"]  = d["_create_dt"].dt.to_period("M")
    d["_pay_m"]     = d["_pay_dt"].dt.to_period("M")
    d["_same_month"] = (d["_create_m"] == d["_pay_m"])
    return d

def per_source_monthly_counts(d_hist: pd.DataFrame, source_col: str):
    if d_hist.empty:
        return pd.DataFrame(columns=["_pay_m", source_col, "cnt_same", "cnt_prev", "days_in_month"])
    g = d_hist.groupby(["_pay_m", source_col])
    by = g["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by["days_in_month"] = by["_pay_m"].apply(lambda p: monthrange(p.year, p.month)[1])
    return by

def daily_rates_from_lookback(d_hist: pd.DataFrame, source_col: str, lookback: int, weighted: bool):
    if d_hist.empty:
        return {}, {}, 0.0, 0.0

    months = sorted(d_hist["_pay_m"].unique())
    months = months[-lookback:] if len(months) > lookback else months
    d_hist = d_hist[d_hist["_pay_m"].isin(months)].copy()

    by = per_source_monthly_counts(d_hist, source_col)
    month_to_w = {m: (i+1 if weighted else 1.0) for i, m in enumerate(sorted(months))}

    rates_same, rates_prev = {}, {}
    for src, sub in by.groupby(source_col):
        w = sub["_pay_m"].map(month_to_w)
        num_same = (sub["cnt_same"] / sub["days_in_month"] * w).sum()
        num_prev = (sub["cnt_prev"] / sub["days_in_month"] * w).sum()
        den = w.sum()
        rates_same[str(src)] = float(num_same/den) if den > 0 else 0.0
        rates_prev[str(src)] = float(num_prev/den) if den > 0 else 0.0

    by_overall = d_hist.groupby("_pay_m")["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by_overall["days_in_month"] = by_overall["_pay_m"].apply(lambda p: monthrange(p.year, p.month)[1])
    w_all = by_overall["_pay_m"].map(month_to_w)
    num_same_o = (by_overall["cnt_same"] / by_overall["days_in_month"] * w_all).sum()
    num_prev_o = (by_overall["cnt_prev"] / by_overall["days_in_month"] * w_all).sum()
    den_o = w_all.sum()
    overall_same_rate = float(num_same_o/den_o) if den_o > 0 else 0.0
    overall_prev_rate = float(num_prev_o/den_o) if den_o > 0 else 0.0
    return rates_same, rates_prev, overall_same_rate, overall_prev_rate

def predict_running_month(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                          lookback: int, weighted: bool, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()
        # include blank/NaN deal sources as "Unknown" so they are counted
        df_work[source_col] = df_work[source_col].fillna("Unknown").astype(str)

    d = add_month_cols(df_work, create_col, pay_col)

    cur_start, cur_end = month_bounds(today)
    cur_period = pd.Period(today, freq="M")

    d_cur = d[d["_pay_m"] == cur_period].copy()
    if d_cur.empty:
        realized_by_src = pd.DataFrame(columns=[source_col, "A"])
    else:
        # include Unknown deal source in Actual-to-date
        realized_by_src = (
            d_cur.assign({source_col: d_cur[source_col].fillna("Unknown").astype(str)})
                .groupby(source_col).size().rename("A").reset_index()
        )

    d_hist = d[d["_pay_m"] < cur_period].copy()
    rates_same, rates_prev, overall_same_rate, overall_prev_rate = daily_rates_from_lookback(
        d_hist, source_col, lookback, weighted
    )

    elapsed_days = (today - cur_start).days + 1
    total_days   = (cur_end - cur_start).days + 1
    remaining_days = max(0, total_days - elapsed_days)

    src_realized = set(d_cur[source_col].fillna("Unknown").astype(str)) if not d_cur.empty else set()
    src_hist = set(list(rates_same.keys()) + list(rates_prev.keys()))
    all_sources = sorted(src_realized | src_hist | ({"All"} if source_col == "_Source" else set()))

    A_tot = B_tot = C_tot = 0.0
    rows = []
    a_map = dict(zip(realized_by_src[source_col], realized_by_src["A"])) if not realized_by_src.empty else {}

    for src in all_sources:
        a_val = float(a_map.get(src, 0.0))
        rate_same = rates_same.get(src, overall_same_rate)
        rate_prev = rates_prev.get(src, overall_prev_rate)

        b_val = float(rate_same * remaining_days)
        c_val = float(rate_prev * remaining_days)

        rows.append({
            "Source": src,
            "A_Actual_ToDate": a_val,
            "B_Remaining_SameMonth": b_val,
            "C_Remaining_PrevMonths": c_val,
            "Projected_MonthEnd_Total": a_val + b_val + c_val,
            "Rate_Same_Daily": rate_same,
            "Rate_Prev_Daily": rate_prev,
            "Remaining_Days": remaining_days
        })
        A_tot += a_val
        B_tot += b_val
        C_tot += c_val

    tbl = pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)
    totals = {
        "A_Actual_ToDate": A_tot,
        "B_Remaining_SameMonth": B_tot,
        "C_Remaining_PrevMonths": C_tot,
        "Projected_MonthEnd_Total": A_tot + B_tot + C_tot,
        "Remaining_Days": remaining_days
    }
    return tbl, totals



def predict_chart_stacked(tbl: pd.DataFrame):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    melt = tbl.melt(
        id_vars=["Source"],
        value_vars=["A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths"],
        var_name="Component",
        value_name="Value"
    )
    color_map = {"A_Actual_ToDate": PALETTE["A_actual"], "B_Remaining_SameMonth": PALETTE["Rem_same"], "C_Remaining_PrevMonths": PALETTE["Rem_prev"]}
    chart = alt.Chart(melt).mark_bar().encode(
        x=alt.X("Source:N", sort=alt.SortField("Source")),
        y=alt.Y("Value:Q", stack=True),
        color=alt.Color("Component:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Component", orient="top", labelLimit=240)),
        tooltip=[alt.Tooltip("Source:N"), alt.Tooltip("Component:N"), alt.Tooltip("Value:Q", format=",.1f")]
    ).properties(height=360, title="Predictibility (A + B + C = Projected Month-End)")
    return chart

def month_list_before(period_end: pd.Period, k: int):
    months = []
    p = period_end
    for _ in range(k):
        p = (p - 1)
        months.append(p)
    months.reverse()
    return months

def backtest_accuracy(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                      lookback: int, weighted: bool, backtest_months: int, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()

    d = add_month_cols(df_work, create_col, pay_col)
    current_period = pd.Period(today, freq="M")

    months_to_eval = month_list_before(current_period, backtest_months)
    rows = []
    for m in months_to_eval:
        train_months = month_list_before(m, lookback)
        d_train = d[d["_pay_m"].isin(train_months)]
        if d_train.empty:
            same_rates, prev_rates, same_rate_o, prev_rate_o = {}, {}, 0.0, 0.0
        else:
            same_rates, prev_rates, same_rate_o, prev_rate_o = daily_rates_from_lookback(
                d_train, source_col, lookback=len(train_months), weighted=weighted
            )

        d_m = d[d["_pay_m"] == m]
        actual_total = int(len(d_m))
        days_in_m = monthrange(m.year, m.month)[1]

        sources = set(list(same_rates.keys()) + list(prev_rates.keys()))
        if not sources and source_col != "_Source":
            sources = set(d_m[source_col].dropna().astype(str).unique().tolist())
        if not sources:
            sources = {"All"}

        forecast = 0.0
        for src in sources:
            r_same = same_rates.get(src, same_rate_o)
            r_prev = prev_rates.get(src, prev_rate_o)
            forecast += (r_same + r_prev) * days_in_m

        err = forecast - actual_total
        rows.append({
            "Month": str(m), "Days": days_in_m,
            "Forecast": float(forecast), "Actual": float(actual_total),
            "Error": float(err), "AbsError": float(abs(err)),
            "SqError": float(err**2),
            "APE": float(abs(err) / actual_total) if actual_total > 0 else np.nan
        })

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {"MAPE": np.nan, "WAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = bt["AbsError"].mean()
    rmse = (bt["SqError"].mean())**0.5
    wape = (bt["AbsError"].sum() / bt["Actual"].sum()) if bt["Actual"].sum() > 0 else np.nan
    mape = bt["APE"].dropna().mean() if bt["APE"].notna().any() else np.nan
    ss_res = ((bt["Actual"] - bt["Forecast"])**2).sum()
    ss_tot = ((bt["Actual"] - bt["Actual"].mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return bt, {"MAPE": mape, "WAPE": wape, "MAE": mae, "RMSE": rmse, "R2": r2}

def accuracy_scatter(bt: pd.DataFrame):
    if bt.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    chart = alt.Chart(bt).mark_circle(size=120, opacity=0.8).encode(
        x=alt.X("Actual:Q", title="Actual (month total)"),
        y=alt.Y("Forecast:Q", title="Forecast (start-of-month)"),
        tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Actual:Q"), alt.Tooltip("Forecast:Q"), alt.Tooltip("Error:Q")],
    ).properties(height=360, title="Forecast vs Actual (by month)")
    line = alt.Chart(pd.DataFrame({"x":[bt["Actual"].min(), bt["Actual"].max()],
                                   "y":[bt["Actual"].min(), bt["Actual"].max()]})).mark_line()
    return chart + line

# ======================
# 80-20 (Pareto + Trajectory + Mix) helpers
# ======================
def build_pareto(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame(columns=[label, "Count", "CumCount", "CumPct", "Tag"])
    tmp = (
        df.assign(_grp=df[group_col].fillna("Unknown").astype(str))
          .groupby("_grp").size().sort_values(ascending=False).rename("Count").reset_index()
          .rename(columns={"_grp": label})
    )
    if tmp.empty:
        return tmp
    tmp["CumCount"] = tmp["Count"].cumsum()
    total = tmp["Count"].sum()
    tmp["CumPct"] = (tmp["CumCount"] / total) * 100.0
    tmp["Tag"] = np.where(tmp["CumPct"] <= 80.0, "Top 80%", "Bottom 20%")
    return tmp

def pareto_chart(tbl: pd.DataFrame, label: str, title: str):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    base = alt.Chart(tbl).encode(x=alt.X(f"{label}:N", sort=list(tbl[label])))
    bars = base.mark_bar(opacity=0.85).encode(
        y=alt.Y("Count:Q", axis=alt.Axis(title="Enrollments (count)")),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("Count:Q")]
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("CumPct:Q", axis=alt.Axis(title="Cumulative %", orient="right")),
        color=alt.value("#16a34a"),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("CumPct:Q", format=".1f")]
    )
    rule80 = alt.Chart(pd.DataFrame({"y":[80.0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    return alt.layer(bars, line, rule80).resolve_scale(y='independent').properties(title=title, height=360)

def months_back_list(end_d: date, k: int):
    p_end = pd.Period(end_d, freq="M")
    return [p_end - i for i in range(k-1, -1, -1)]

# ======================
# RENDER: Views
# ======================
if view == "MIS":
    show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)
    if show_all:
        st.subheader("Preset Periods")
        colA, colB = st.columns(2)
        with colA:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with colB:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "This Month (MTD)", this_m_start, this_m_end_mtd, this_m_start, create_col, pay_col, pipeline_col, track)
    else:
        tabs = st.tabs(["Yesterday", "Today", "Last Month", "This Month (MTD)", "Custom"])
        with tabs[0]:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
        with tabs[1]:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
        with tabs[2]:
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[3]:
            render_period_block(df_f, "This Month (MTD)", this_m_start, this_m_end_mtd, this_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[4]:
            st.markdown("Select a *payments period* and choose the *Conversion% denominator* mode.")
            colc1, colc2 = st.columns(2)
            with colc1: custom_start = st.date_input("Payments period start", value=this_m_start, key="mis_cust_pay_start")
            with colc2: custom_end   = st.date_input("Payments period end (inclusive)", value=this_m_end, key="mis_cust_pay_end")
            if custom_end < custom_start:
                st.error("Payments period end cannot be before start.")
            else:
                denom_mode = st.radio("Denominator for Conversion%", ["Anchor month", "Custom range"], index=0, horizontal=True, key="mis_dmode")
                if denom_mode == "Anchor month":
                    anchor = st.date_input("Running-month anchor (denominator month)", value=custom_start, key="mis_anchor")
                    mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor, create_col, pay_col, pipeline_col)
                    c1, c2 = st.columns(2)
                    with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                    with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)

                    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                        df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                        denom_start=anchor.replace(day=1),
                        denom_end=month_bounds(anchor)[1]
                    )
                    st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                    conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                    conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))

                    ts = trend_timeseries(df_f, custom_start, custom_end,
                                          denom_start=anchor.replace(day=1), denom_end=month_bounds(anchor)[1],
                                          create_col=create_col, pay_col=pay_col)
                    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)
                else:
                    cold1, cold2 = st.columns(2)
                    with cold1: denom_start = st.date_input("Denominator start (deals created from)", value=custom_start, key="mis_den_start")
                    with cold2: denom_end   = st.date_input("Denominator end (deals created to)",   value=custom_end,   key="mis_den_end")
                    if denom_end < denom_start:
                        st.error("Denominator end cannot be before start.")
                    else:
                        anchor_for_counts = custom_start
                        mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor_for_counts, create_col, pay_col, pipeline_col)
                        c1, c2 = st.columns(2)
                        with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                        with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)

                        mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                            df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                            denom_start=denom_start, denom_end=denom_end
                        )
                        st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                        conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                        conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))

                        ts = trend_timeseries(df_f, custom_start, custom_end,
                                              denom_start=denom_start, denom_end=denom_end,
                                              create_col=create_col, pay_col=pay_col)
                        st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)



elif view == "Trend & Analysis":
    def _trend_and_analysis_tab():
        st.subheader("Trend & Analysis – Grouped Drilldowns (Final rules)")

        # ----------------------------
        # Group-by fields (unchanged)
        # ----------------------------
        available_groups, group_map = [], {}
        if counsellor_col: available_groups.append("Academic Counsellor"); group_map["Academic Counsellor"] = counsellor_col
        if country_col:    available_groups.append("Country");            group_map["Country"] = country_col
        if source_col:     available_groups.append("JetLearn Deal Source"); group_map["JetLearn Deal Source"] = source_col

        sel_group_labels = st.multiselect(
            "Group by (pick one or more)",
            options=available_groups,
            default=available_groups[:1] if available_groups else []
        )
        group_cols = [group_map[l] for l in sel_group_labels if l in group_map]

        # ----------------------------
        # Mode (unchanged)
        # ----------------------------
        level = st.radio("Mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ta_mode")

        # ================================
        # 4-BOX KPI STRIP (ADDED & EXPANDED)
        # ================================
        try:
            _ref_intent_col = find_col(df, ["Referral Intent Source", "Referral intent source"])
            _src_col        = source_col if (source_col and source_col in df_f.columns) \
                              else find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
            _create_ok      = create_col and (create_col in df_f.columns)
            _pay_ok         = pay_col and (pay_col in df_f.columns)

            # Optional calibration columns
            _first_ok = first_cal_sched_col and (first_cal_sched_col in df_f.columns)
            _resch_ok = cal_resched_col and (cal_resched_col in df_f.columns)
            _done_ok  = cal_done_col and (cal_done_col in df_f.columns)

            if not (_create_ok and _pay_ok):
                st.warning("Create/Payment columns are needed for the KPI strip. Please map them in the sidebar.", icon="⚠️")
            else:
                # Normalize base fields
                _C = coerce_datetime(df_f[create_col]).dt.date
                _P = coerce_datetime(df_f[pay_col]).dt.date
                _SRC  = (df_f[_src_col].fillna("Unknown").astype(str).str.strip()) if _src_col else pd.Series("Unknown", index=df_f.index)
                _REFI = (df_f[_ref_intent_col].fillna("Unknown").astype(str).str.strip()) if (_ref_intent_col and _ref_intent_col in df_f.columns) else pd.Series("Unknown", index=df_f.index)

                # Optional: calibration dates
                _FDT = coerce_datetime(df_f[first_cal_sched_col]).dt.date if _first_ok else pd.Series(pd.NaT, index=df_f.index)
                _RDT = coerce_datetime(df_f[cal_resched_col]).dt.date     if _resch_ok else pd.Series(pd.NaT, index=df_f.index)
                _DDT = coerce_datetime(df_f[cal_done_col]).dt.date        if _done_ok else pd.Series(pd.NaT, index=df_f.index)

                # Windows
                tm_start, tm_end = month_bounds(today)
                lm_start, lm_end = last_month_bounds(today)
                yd = today - timedelta(days=1)
                windows = [
                    ("Yesterday", yd, yd),
                    ("Today", today, today),
                    ("Last month", lm_start, lm_end),
                    ("This month", tm_start, tm_end),
                ]

                # Helpers
                def _is_referral_created(sr: pd.Series) -> pd.Series:
                    s = sr.fillna("").astype(str)
                    return s.str.contains("referr", case=False, na=False)

                def _is_sales_generated_intent(sr: pd.Series) -> pd.Series:
                    s = sr.fillna("").astype(str)
                    return s.str.contains(r"\bsales\s*generated\b", case=False, na=False)

                def _event_count_in_window(event_dates: pd.Series, c_in: pd.Series, start_d: date, end_d: date, mode: str) -> int:
                    """Count events in [start_d, end_d]; MTD requires create-date also in window."""
                    if event_dates is None or event_dates.isna().all():
                        return 0
                    e_in = event_dates.between(start_d, end_d)
                    if mode == "MTD":
                        return int((e_in & c_in).sum())
                    return int(e_in.sum())

                # Counters per window
                def _counts_for_window(start_d: date, end_d: date, mode: str) -> dict:
                    c_in = _C.between(start_d, end_d)
                    p_in = _P.between(start_d, end_d)

                    deals_created = int(c_in.sum())
                    enrolments    = int((c_in & p_in).sum()) if mode == "MTD" else int(p_in.sum())

                    # Referral Created — create-date based
                    if _src_col:
                        referral_created = int((c_in & _is_referral_created(_SRC)).sum())
                    else:
                        referral_created = 0

                    # Sales Generated (Intent) — create-date based
                    if _ref_intent_col and _ref_intent_col in df_f.columns:
                        sales_generated_intent = int((c_in & _is_sales_generated_intent(_REFI)).sum())
                    else:
                        sales_generated_intent = 0

                    # Calibration counts
                    first_cal_cnt = _event_count_in_window(_FDT, c_in, start_d, end_d, mode)
                    resched_cnt   = _event_count_in_window(_RDT, c_in, start_d, end_d, mode)
                    done_cnt      = _event_count_in_window(_DDT, c_in, start_d, end_d, mode)

                    return {
                        "Deals Created": deals_created,
                        "Enrolments": enrolments,
                        "Referral Created": referral_created,
                        "Sales Generated (Intent)": sales_generated_intent,
                        "First Cal Scheduled": first_cal_cnt,
                        "Cal Rescheduled": resched_cnt,
                        "Cal Done": done_cnt,
                    }

                kpis = [(label, _counts_for_window(s, e, level)) for (label, s, e) in windows]

                # Render
                st.markdown(
                    """
                    <style>
                      .kpi4-grid {display:grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-top: 8px;}
                      .kpi4-card {border:1px solid #e5e7eb; border-radius:14px; padding:10px 12px; background:#ffffff;}
                      .kpi4-title {font-weight:700; font-size:0.95rem; margin-bottom:6px;}
                      .kpi4-row {display:flex; justify-content:space-between; font-size:0.9rem; padding:2px 0;}
                      .kpi4-key {color:#6b7280;}
                      .kpi4-val {font-weight:700;}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                _metric_order = [
                    "Deals Created",
                    "Enrolments",
                    "Referral Created",
                    "Sales Generated (Intent)",
                    "First Cal Scheduled",
                    "Cal Rescheduled",
                    "Cal Done",
                ]
                _html = ['<div class="kpi4-grid">']
                for label, vals in kpis:
                    _html.append(f'<div class="kpi4-card"><div class="kpi4-title">{label}</div>')
                    for k in _metric_order:
                        _html.append(f'<div class="kpi4-row"><div class="kpi4-key">{k}</div><div class="kpi4-val">{vals.get(k,0):,}</div></div>')
                    _html.append("</div>")
                _html.append("</div>")
                st.markdown("".join(_html), unsafe_allow_html=True)
        except Exception as _kpi_err:
            st.warning(f"4-box strip could not render: {_kpi_err}", icon="⚠️")
        # ===== END 4-BOX KPI STRIP =====

        # ----------------------------
        # Date scope (unchanged)
        # ----------------------------
        date_mode = st.radio("Date scope", ["This month", "Last month", "Custom date range"], index=0, horizontal=True, key="ta_dscope")
        if date_mode == "This month":
            range_start, range_end = month_bounds(today)
            st.caption(f"Scope: **This month** ({range_start} → {range_end})")
        elif date_mode == "Last month":
            range_start, range_end = last_month_bounds(today)
            st.caption(f"Scope: **Last month** ({range_start} → {range_end})")
        else:
            col_d1, col_d2 = st.columns(2)
            with col_d1: range_start = st.date_input("Start date", value=today.replace(day=1), key="ta_custom_start")
            with col_d2: range_end   = st.date_input("End date", value=month_bounds(today)[1], key="ta_custom_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                st.stop()
            st.caption(f"Scope: **Custom** ({range_start} → {range_end})")

        # ================================
        # DYNAMIC BOX for selected range
        # ================================
        try:
            _ref_intent_col2 = find_col(df, ["Referral Intent Source", "Referral intent source"])
            _src_col2        = source_col if (source_col and source_col in df_f.columns) \
                               else find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
            _create_ok2      = create_col and (create_col in df_f.columns)
            _pay_ok2         = pay_col and (pay_col in df_f.columns)

            _first_ok2 = first_cal_sched_col and (first_cal_sched_col in df_f.columns)
            _resch_ok2 = cal_resched_col and (cal_resched_col in df_f.columns)
            _done_ok2  = cal_done_col and (cal_done_col in df_f.columns)

            if _create_ok2 and _pay_ok2:
                _C2 = coerce_datetime(df_f[create_col]).dt.date
                _P2 = coerce_datetime(df_f[pay_col]).dt.date
                _SRC2  = (df_f[_src_col2].fillna("Unknown").astype(str).str.strip()) if _src_col2 else pd.Series("Unknown", index=df_f.index)
                _REFI2 = (df_f[_ref_intent_col2].fillna("Unknown").astype(str).str.strip()) if (_ref_intent_col2 and _ref_intent_col2 in df_f.columns) else pd.Series("Unknown", index=df_f.index)
                _FDT2 = coerce_datetime(df_f[first_cal_sched_col]).dt.date if _first_ok2 else pd.Series(pd.NaT, index=df_f.index)
                _RDT2 = coerce_datetime(df_f[cal_resched_col]).dt.date     if _resch_ok2 else pd.Series(pd.NaT, index=df_f.index)
                _DDT2 = coerce_datetime(df_f[cal_done_col]).dt.date        if _done_ok2 else pd.Series(pd.NaT, index=df_f.index)

                def _is_referral_created2(sr: pd.Series) -> pd.Series:
                    s = sr.fillna("").astype(str)
                    return s.str.contains("referr", case=False, na=False)

                def _is_sales_generated_intent2(sr: pd.Series) -> pd.Series:
                    s = sr.fillna("").astype(str)
                    return s.str.contains(r"\bsales\s*generated\b", case=False, na=False)

                def _event_count2(event_dates: pd.Series, c_in: pd.Series) -> pd.Series:
                    if event_dates is None or event_dates.isna().all():
                        return pd.Series(False, index=c_in.index)
                    return event_dates.between(range_start, range_end)

                c_in2 = _C2.between(range_start, range_end)
                p_in2 = _P2.between(range_start, range_end)

                deals_created2 = int(c_in2.sum())
                enrolments2    = int((c_in2 & p_in2).sum()) if level == "MTD" else int(p_in2.sum())
                referral_created2 = int((c_in2 & _is_referral_created2(_SRC2)).sum()) if _src_col2 else 0
                sales_generated_intent2 = int((c_in2 & _is_sales_generated_intent2(_REFI2)).sum()) if (_ref_intent_col2 and _ref_intent_col2 in df_f.columns) else 0

                f_in2 = _event_count2(_FDT2, c_in2)
                r_in2 = _event_count2(_RDT2, c_in2)
                d_in2 = _event_count2(_DDT2, c_in2)
                first_cnt2 = int((f_in2 & c_in2).sum()) if level == "MTD" else int(f_in2.sum())
                resch_cnt2 = int((r_in2 & c_in2).sum()) if level == "MTD" else int(r_in2.sum())
                done_cnt2  = int((d_in2 & c_in2).sum()) if level == "MTD" else int(d_in2.sum())

                st.markdown("#### Summary for Selected Range")
                st.markdown(
                    """
                    <style>
                      .kpi1-card {border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#ffffff; margin-bottom:8px;}
                      .kpi1-title {font-weight:700; font-size:0.95rem; margin-bottom:6px;}
                      .kpi1-row {display:flex; justify-content:space-between; font-size:0.9rem; padding:2px 0;}
                      .kpi1-key {color:#6b7280;}
                      .kpi1-val {font-weight:700;}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                _metric_order2 = [
                    "Deals Created",
                    "Enrolments",
                    "Referral Created",
                    "Sales Generated (Intent)",
                    "First Cal Scheduled",
                    "Cal Rescheduled",
                    "Cal Done",
                ]
                _vals2 = {
                    "Deals Created": deals_created2,
                    "Enrolments": enrolments2,
                    "Referral Created": referral_created2,
                    "Sales Generated (Intent)": sales_generated_intent2,
                    "First Cal Scheduled": first_cnt2,
                    "Cal Rescheduled": resch_cnt2,
                    "Cal Done": done_cnt2,
                }
                _box = [f'<div class="kpi1-card"><div class="kpi1-title">{range_start} → {range_end} ({level})</div>']
                for k in _metric_order2:
                    _box.append(f'<div class="kpi1-row"><div class="kpi1-key">{k}</div><div class="kpi1-val">{_vals2[k]:,}</div></div>')
                _box.append("</div>")
                st.markdown("".join(_box), unsafe_allow_html=True)
            else:
                st.info("Map Create/Payment columns to see the dynamic summary box.", icon="ℹ️")
        except Exception as _dyn_err:
            st.warning(f"Dynamic box could not render: {_dyn_err}", icon="⚠️")

        # ----------------------------
        # Metric picker (unchanged)
        # ----------------------------
        all_metrics = [
            "Payment Received Date — Count",
            "First Calibration Scheduled Date — Count",
            "Calibration Rescheduled Date — Count",
            "Calibration Done Date — Count",
            "Create Date (deals) — Count",
            "Future Calibration Scheduled — Count",
        ]
        metrics_selected = st.multiselect("Metrics to show", options=all_metrics, default=all_metrics, key="ta_metrics")

        metric_cols = {
            "Payment Received Date — Count": pay_col,
            "First Calibration Scheduled Date — Count": first_cal_sched_col,
            "Calibration Rescheduled Date — Count": cal_resched_col,
            "Calibration Done Date — Count": cal_done_col,
            "Create Date (deals) — Count": create_col,
            "Future Calibration Scheduled — Count": None,  # derived
        }

        # Missing column warnings (unchanged)
        miss = []
        for m in metrics_selected:
            if m == "Future Calibration Scheduled — Count":
                if (first_cal_sched_col is None or first_cal_sched_col not in df_f.columns) and \
                   (cal_resched_col is None or cal_resched_col not in df_f.columns):
                    miss.append("Future Calibration Scheduled (needs First and/or Rescheduled)")
            elif m != "Create Date (deals) — Count":
                if (metric_cols.get(m) is None) or (metric_cols.get(m) not in df_f.columns):
                    miss.append(m)
        if miss:
            st.warning("Missing columns for: " + ", ".join(miss) + ". Those counts will show as 0.", icon="⚠️")

        # ----------------------------
        # Build table (unchanged)
        # ----------------------------
        def ta_count_table(
            df_scope: pd.DataFrame,
            group_cols: list[str],
            mode: str,
            range_start: date,
            range_end: date,
            create_col: str,
            metric_cols: dict,
            metrics_selected: list[str],
            *,
            first_cal_col: str | None,
            cal_resched_col: str | None,
        ) -> pd.DataFrame:

            if not group_cols:
                df_work = df_scope.copy()
                df_work["_GroupDummy"] = "All"
                group_cols_local = ["_GroupDummy"]
            else:
                df_work = df_scope.copy()
                group_cols_local = group_cols

            create_dt = coerce_datetime(df_work[create_col]).dt.date

            if first_cal_col and first_cal_col in df_work.columns:
                first_dt = coerce_datetime(df_work[first_cal_col])
            else:
                first_dt = pd.Series(pd.NaT, index=df_work.index)
            if cal_resched_col and cal_resched_col in df_work.columns:
                resch_dt = coerce_datetime(df_work[cal_resched_col])
            else:
                resch_dt = pd.Series(pd.NaT, index=df_work.index)

            eff_cal = resch_dt.copy().fillna(first_dt)
            eff_cal_date = eff_cal.dt.date

            pop_mask_mtd = create_dt.between(range_start, range_end)

            outs = []
            for disp in metrics_selected:
                col = metric_cols.get(disp)

                if disp == "Create Date (deals) — Count":
                    idx = pop_mask_mtd if mode == "MTD" else create_dt.between(range_start, range_end)
                    gdf = df_work.loc[idx, group_cols_local].copy()
                    agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
                    outs.append(agg)
                    continue

                if disp == "Future Calibration Scheduled — Count":
                    if eff_cal_date is None:
                        base_idx = pop_mask_mtd if mode == "MTD" else slice(None)
                        target = df_work.loc[base_idx, group_cols_local] if mode == "MTD" else df_work[group_cols_local]
                        agg = target.assign(**{disp:0}).groupby(group_cols_local)[disp].sum().reset_index() if not target.empty else pd.DataFrame(columns=group_cols_local+[disp])
                        outs.append(agg)
                        continue
                    future_mask = eff_cal_date > range_end
                    idx = (pop_mask_mtd & future_mask) if mode == "MTD" else future_mask
                    gdf = df_work.loc[idx, group_cols_local].copy()
                    agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
                    outs.append(agg)
                    continue

                if (not col) or (col not in df_work.columns):
                    base_idx = pop_mask_mtd if mode == "MTD" else slice(None)
                    target = df_work.loc[base_idx, group_cols_local] if mode == "MTD" else df_work[group_cols_local]
                    agg = target.assign(**{disp:0}).groupby(group_cols_local)[disp].sum().reset_index() if not target.empty else pd.DataFrame(columns=group_cols_local+[disp])
                    outs.append(agg)
                    continue

                ev_date = coerce_datetime(df_work[col]).dt.date
                ev_in_range = ev_date.between(range_start, range_end)

                if mode == "MTD":
                    idx = pop_mask_mtd & ev_in_range
                else:
                    idx = ev_in_range

                gdf = df_work.loc[idx, group_cols_local].copy()
                agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
                outs.append(agg)

            if outs:
                result = outs[0]
                for f in outs[1:]:
                    result = result.merge(f, on=group_cols_local, how="outer")
            else:
                result = pd.DataFrame(columns=group_cols_local)

            for m in metrics_selected:
                if m not in result.columns:
                    result[m] = 0
            result[metrics_selected] = result[metrics_selected].fillna(0).astype(int)
            if metrics_selected:
                result = result.sort_values(metrics_selected[0], ascending=False)
            return result.reset_index(drop=True)

        tbl = ta_count_table(
            df_scope=df_f,
            group_cols=group_cols,
            mode=level,
            range_start=range_start,
            range_end=range_end,
            create_col=create_col,
            metric_cols=metric_cols,
            metrics_selected=metrics_selected,
            first_cal_col=first_cal_sched_col,
            cal_resched_col=cal_resched_col,
        )

        st.markdown("### Output")
        if tbl.empty:
            st.info("No rows match the selected filters and date range.")
        else:
            rename_map = {group_map.get(lbl): lbl for lbl in sel_group_labels}
            show = tbl.rename(columns=rename_map)
            st.dataframe(show, use_container_width=True)

            csv = show.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV (Trend & Analysis)", data=csv, file_name="trend_analysis_final.csv", mime="text/csv")

        # ---------------------------------------------------------------------
        # Referral business — Created vs Converted by Referral Intent Source
        # ---------------------------------------------------------------------
        st.markdown("### Referral business — Created vs Converted by Referral Intent Source")
        referral_intent_col = find_col(df, ["Referral Intent Source", "Referral intent source"])

        if (not referral_intent_col) or (referral_intent_col not in df_f.columns):
            st.info("Referral Intent Source column not found.")
        else:
            d_ref = df_f.copy()
            d_ref["_ref"] = d_ref[referral_intent_col].fillna("Unknown").astype(str).str.strip()

            exclude_unknown = st.checkbox("Exclude 'Unknown' (Referral Intent Source)", value=False, key="ta_ref_exclude")
            if exclude_unknown:
                d_ref = d_ref[d_ref["_ref"] != "Unknown"]

            # Normalize dates
            _cdate_r = coerce_datetime(d_ref[create_col]).dt.date if create_col in d_ref.columns else pd.Series(pd.NaT, index=d_ref.index)
            _pdate_r = coerce_datetime(d_ref[pay_col]).dt.date    if pay_col in d_ref.columns    else pd.Series(pd.NaT, index=d_ref.index)

            m_created_r = _cdate_r.between(range_start, range_end) if _cdate_r.notna().any() else pd.Series(False, index=d_ref.index)
            m_paid_r    = _pdate_r.between(range_start, range_end) if _pdate_r.notna().any() else pd.Series(False, index=d_ref.index)

            if level == "MTD":
                # Count payments only from deals whose Create Date is in scope
                created_mask_r   = m_created_r
                converted_mask_r = m_created_r & m_paid_r
            else:
                # Cohort: payments in scope regardless of create-month
                created_mask_r   = m_created_r
                converted_mask_r = m_paid_r

            ref_tbl = pd.DataFrame({
                "Referral Intent Source": d_ref["_ref"],
                "Created":  created_mask_r.astype(int),
                "Converted": converted_mask_r.astype(int),
            })
            grp = (
                ref_tbl.groupby("Referral Intent Source", as_index=False)
                       .sum(numeric_only=True)
                       .sort_values("Created", ascending=False)
            )

            # Controls
            col_r1, col_r2 = st.columns([1,1])
            with col_r1:
                top_k_ref = st.number_input("Show top N Referral Intent Sources", min_value=1, max_value=max(1, len(grp)),
                                            value=min(20, len(grp)) if len(grp) else 1, step=1, key="ta_ref_topn")
            with col_r2:
                sort_metric_ref = st.selectbox("Sort by", ["Created (desc)", "Converted (desc)", "A–Z"], index=0, key="ta_ref_sort")

            if sort_metric_ref == "Converted (desc)":
                grp = grp.sort_values("Converted", ascending=False)
            elif sort_metric_ref == "A–Z":
                grp = grp.sort_values("Referral Intent Source", ascending=True)
            else:
                grp = grp.sort_values("Created", ascending=False)

            grp_show = grp.head(int(top_k_ref)) if len(grp) > int(top_k_ref) else grp

            # Chart
            melt_ref = grp_show.melt(
                id_vars=["Referral Intent Source"],
                value_vars=["Created", "Converted"],
                var_name="Metric",
                value_name="Count"
            )
            chart_ref = (
                alt.Chart(melt_ref)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Referral Intent Source:N", sort=grp_show["Referral Intent Source"].tolist(), title="Referral Intent Source"),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.Color("Metric:N", title="", legend=alt.Legend(orient="bottom")),
                    xOffset=alt.XOffset("Metric:N"),
                    tooltip=[alt.Tooltip("Referral Intent Source:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]
                )
                .properties(height=360, title=f"Created vs Converted by Referral Intent Source ({level})")
            )
            st.altair_chart(chart_ref, use_container_width=True)

            # Table + download
            st.dataframe(grp_show, use_container_width=True)
            st.download_button(
                "Download CSV — Referral business (Created vs Converted)",
                data=grp_show.to_csv(index=False).encode("utf-8"),
                file_name=f"trend_referral_business_{level.lower()}_{range_start}_{range_end}.csv",
                mime="text/csv",
                key="ta_ref_business_dl"
            )

    # run the wrapped tab to avoid outer indentation issues
    _trend_and_analysis_tab()





elif view == "80-20":
    st.subheader("80-20 Pareto + Trajectory + Conversion% + Mix Analyzer")
    ...


    # Precompute for this module
    df80 = df.copy()
    df80["_pay_dt"] = coerce_datetime(df80[pay_col])
    df80["_create_dt"] = coerce_datetime(df80[create_col])
    df80["_pay_m"] = df80["_pay_dt"].dt.to_period("M")

    # ✅ Apply Track filter to 80-20 too
    if track != "Both":
        if pipeline_col and pipeline_col in df80.columns:
            _norm80 = df80[pipeline_col].map(normalize_pipeline).fillna("Other")
            before_ct = len(df80)
            df80 = df80.loc[_norm80 == track].copy()
            st.caption(f"80-20 scope after Track filter ({track}): **{len(df80):,}** rows (was {before_ct:,}).")
        else:
            st.warning("Pipeline column not found — the Track filter can’t be applied in 80-20.", icon="⚠️")

    if source_col:
        df80["_src_raw"] = df80[source_col].fillna("Unknown").astype(str)
    else:
        df80["_src_raw"] = "Other"

    # ---- Cohort scope / date pickers (in-tab)
    st.markdown("#### Cohort scope (Payment Received)")
    unique_months = df80["_pay_dt"].dropna().dt.to_period("M").drop_duplicates().sort_values()
    month_labels = [str(p) for p in unique_months]
    use_custom = st.toggle("Use custom date range", value=False, key="eighty_use_custom")

    if not use_custom and len(month_labels) > 0:
        month_pick = st.selectbox("Cohort month", month_labels, index=len(month_labels)-1, key="eighty_month_pick")
        y, m = map(int, month_pick.split("-"))
        start_d = date(y, m, 1)
        end_d = date(y, m, monthrange(y, m)[1])
    else:
        default_start = df80["_pay_dt"].min().date() if df80["_pay_dt"].notna().any() else date.today().replace(day=1)
        default_end   = df80["_pay_dt"].max().date() if df80["_pay_dt"].notna().any() else date.today()
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start date", value=default_start, key="eighty_start")
        with c2: end_d   = st.date_input("End date", value=default_end, key="eighty_end")
        if end_d < start_d:
            st.error("End date cannot be before start date.")
            st.stop()

    # Source include list (Pareto/Cohort) using _src_raw (includes Unknown)
    st.markdown("#### Source filter (for Pareto & Cohort views)")
    if source_col:
        all_sources = sorted(df80['_src_raw'].unique().tolist())
        excl_ref = st.checkbox('Exclude Referral (for Pareto view)', value=False, key='eighty_excl_ref')
        sources_for_pick = [s for s in all_sources if not (excl_ref and 'referr' in s.lower())]
        picked_sources = st.multiselect('Include Deal Sources (Pareto)', options=['All'] + sources_for_pick, default=['All'], key='eighty_picked_src')
        # Normalize: if 'All' or empty => use full filtered list
        if (not picked_sources) or ('All' in picked_sources):
            picked_sources = sources_for_pick
    else:
        picked_sources = []

    # ---- Range KPI (Created vs Enrolments)
    in_create_window = df80["_create_dt"].dt.date.between(start_d, end_d)
    deals_created = int(in_create_window.sum())

    in_pay_window = df80["_pay_dt"].dt.date.between(start_d, end_d)
    enrolments = int(in_pay_window.sum())

    conv_pct_simple = (enrolments / deals_created * 100.0) if deals_created > 0 else 0.0

    st.markdown("<div class='section-title'>Range KPI — Deals Created vs Enrolments</div>", unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div><div class='kpi-value'>{deals_created:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with cB:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div><div class='kpi-value'>{enrolments:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with cC:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div><div class='kpi-value'>{conv_pct_simple:.1f}%</div><div class='kpi-sub'>Num: {enrolments:,} • Den: {deals_created:,}</div></div>", unsafe_allow_html=True)

    # ---- Build cohort df for 80-20 section (respect picked_sources)
    scope_mask = df80["_pay_dt"].dt.date.between(start_d, end_d)
    df_cohort = df80.loc[scope_mask].copy()
    if picked_sources is not None and source_col:
        df_cohort = df_cohort[df_cohort["_src_raw"].isin(picked_sources)]

    # ---- Cohort KPIs
    st.markdown("<div class='section-title'>Cohort KPIs</div>", unsafe_allow_html=True)
    total_enr = int(len(df_cohort))
    if source_col and source_col in df_cohort.columns:
        ref_cnt = int(df_cohort[source_col].fillna("").str.contains("referr", case=False).sum())
    else:
        ref_cnt = 0
    ref_pct = (ref_cnt/total_enr*100.0) if total_enr > 0 else 0.0

    src_tbl = build_pareto(df_cohort, source_col, "Deal Source") if total_enr > 0 else pd.DataFrame()
    cty_tbl = build_pareto(df_cohort, country_col, "Country") if total_enr > 0 else pd.DataFrame()
    n_sources_80 = int((src_tbl["CumPct"] <= 80).sum()) if not src_tbl.empty else 0
    n_countries_80 = int((cty_tbl["CumPct"] <= 80).sum()) if not cty_tbl.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Cohort Enrolments</div><div class='kpi-value'>{total_enr:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Referral % (cohort)</div><div class='kpi-value'>{ref_pct:.1f}%</div><div class='kpi-sub'>{ref_cnt:,} of {total_enr:,}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Sources for 80%</div><div class='kpi-value'>{n_sources_80}</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Countries for 80%</div><div class='kpi-value'>{n_countries_80}</div></div>", unsafe_allow_html=True)

    # ---- 80-20 Charts
    c1, c2 = st.columns([2,1])
    # Guard: avoid Altair schema errors on empty data
    if src_tbl.empty:
        st.info('No data for selected deal sources / date range.')
    else:
        with c1: st.altair_chart(pareto_chart(src_tbl, "Deal Source", "Pareto – Enrolments by Deal Source"), use_container_width=True)
    with c2:
        # Donut: Referral vs Non-Referral in cohort
        if source_col and source_col in df_cohort.columns and not df_cohort.empty:
            s = df_cohort[source_col].fillna("Unknown").astype(str)
            is_ref = s.str.contains("referr", case=False, na=False)
            pie = pd.DataFrame({"Category": ["Referral", "Non-Referral"], "Value": [int(is_ref.sum()), int((~is_ref).sum())]})
            donut = alt.Chart(pie).mark_arc(innerRadius=70).encode(
                theta="Value:Q",
                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                tooltip=["Category:N", "Value:Q"]
            ).properties(title="Referral vs Non-Referral (cohort)")
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("Referral split not available (missing source column or empty cohort).")
    st.altair_chart(pareto_chart(cty_tbl, "Country", "Pareto – Enrolments by Country"), use_container_width=True)

    # ---- Conversion% by Key Source
    st.markdown("### Conversion% by Key Source (range-based)")
    def conversion_stats(df_all: pd.DataFrame, start_d: date, end_d: date):
        if create_col is None or pay_col is None:
            return pd.DataFrame(columns=["KeySource","Den","Num","Pct"])
        d = df_all.copy()
        d["_cdate"] = coerce_datetime(d[create_col]).dt.date
        d["_pdate"] = coerce_datetime(d[pay_col]).dt.date
        d["_key_source"] = d[source_col].apply(normalize_key_source) if source_col else "Other"

        denom_mask = d["_cdate"].between(start_d, end_d)
        num_mask = d["_pdate"].between(start_d, end_d)

        rows = []
        for src in ["Referral", "PM - Search", "PM - Social"]:
            src_mask = (d["_key_source"] == src)
            den = int((denom_mask & src_mask).sum())
            num = int((num_mask & src_mask).sum())
            pct = (num/den*100.0) if den > 0 else 0.0
            rows.append({"KeySource": src, "Den": den, "Num": num, "Pct": pct})
        return pd.DataFrame(rows)

    bysrc_conv = conversion_stats(df80, start_d, end_d)
    if not bysrc_conv.empty:
        conv_chart = alt.Chart(bysrc_conv).mark_bar(opacity=0.9).encode(
            x=alt.X("KeySource:N", sort=["Referral","PM - Search","PM - Social"], title="Source"),
            y=alt.Y("Pct:Q", title="Conversion%"),
            tooltip=[alt.Tooltip("KeySource:N"), alt.Tooltip("Den:Q", title="Deals (Created)"),
                     alt.Tooltip("Num:Q", title="Enrolments (Payments)"), alt.Tooltip("Pct:Q", title="Conversion%", format=".1f")]
        ).properties(height=300, title=f"Conversion% (Payments / Created) • {start_d} → {end_d}")
        st.altair_chart(conv_chart, use_container_width=True)
    else:
        st.info("No data to compute Conversion% by key source for this window.")

    # ---- Trajectory – Top Countries × (Key or Raw Deal Sources)
    st.markdown("### Trajectory – Top Countries × Referral / PM - Search / PM - Social (or All Raw Sources)")
    col_t1, col_t2, col_tg, col_t3 = st.columns([1, 1, 1.4, 1.6])
    with col_t1:
        trailing_k = st.selectbox("Trailing window (months)", [3, 6, 12], index=0, key="eighty_trailing")
    with col_t2:
        top_k = st.selectbox("Top countries (by cohort enrolments)", [5, 7], index=0, key="eighty_topk")
    with col_tg:
        traj_grouping = st.radio(
            "Source grouping",
            ["Key (Referral/PM-Search/PM-Social/Other)", "Raw (all)"],
            index=0, horizontal=False, key="eighty_grouping"
        )

    months_list = months_back_list(end_d, trailing_k)
    months_str = [str(p) for p in months_list]
    df_trail = df80[df80["_pay_m"].isin(months_list)].copy()

    if traj_grouping.startswith("Key"):
        df_trail["_traj_source"] = df_trail[source_col].apply(normalize_key_source) if source_col else "Other"
        traj_source_options = ["All sources", "Referral", "PM - Search", "PM - Social", "Other"]
    else:
        df_trail["_traj_source"] = df_trail[source_col].fillna("Unknown").astype(str) if source_col else "Other"
        unique_raw = sorted(df_trail["_traj_source"].dropna().unique().tolist())
        traj_source_options = ["All sources"] + unique_raw

    with col_t3:
        traj_src_pick = st.selectbox("Deal Source for Top Countries", options=traj_source_options, index=0, key="eighty_srcpick")

    if traj_src_pick != "All sources":
        df_trail_src = df_trail[df_trail["_traj_source"] == traj_src_pick].copy()
    else:
        df_trail_src = df_trail.copy()

    if country_col and not df_trail_src.empty:
        cty_counts = df_trail_src.groupby(country_col).size().sort_values(ascending=False)
        top_countries = cty_counts.head(top_k).index.astype(str).tolist()
    else:
        top_countries = []

    monthly_total = df_trail.groupby("_pay_m").size().rename("TotalAll").reset_index()

    if top_countries and source_col and country_col:
        mcs = (
        df_trail_src[df_trail_src[country_col].astype(str).isin(top_countries)]
        .groupby(["_pay_m", country_col, "_traj_source"]).size().rename("Cnt").reset_index()
    )
    else:
        mcs = pd.DataFrame(columns=["_pay_m", country_col if country_col else "Country", "_traj_source", "Cnt"])

    if not mcs.empty:
        mcs = mcs.merge(monthly_total, on="_pay_m", how="left")
        mcs["PctOfOverall"] = np.where(mcs["TotalAll"]>0, mcs["Cnt"]/mcs["TotalAll"]*100.0, 0.0)
        mcs["_pay_m_str"] = pd.Categorical(mcs["_pay_m"].astype(str), categories=months_str, ordered=True)
        # safe categorical cleanup
        mcs["_pay_m_str"] = mcs["_pay_m_str"].cat.remove_unused_categories()

    if not mcs.empty:
        # sort legend by frequency
        src_order = mcs["_traj_source"].value_counts().index.tolist()
        title_suffix = f"{traj_src_pick}" if traj_src_pick != "All sources" else "All sources"
        grouping_suffix = "Key" if traj_grouping.startswith("Key") else "Raw"

        facet_chart = alt.Chart(mcs).mark_bar(opacity=0.9).encode(
            x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
            y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("_traj_source:N", title="Source", sort=src_order),
            tooltip=[
                alt.Tooltip("_pay_m_str:N", title="Month"),
                alt.Tooltip(f"{country_col}:N", title="Country") if country_col else alt.Tooltip("_pay_m_str:N"),
                alt.Tooltip("_traj_source:N", title="Source"),
                alt.Tooltip("Cnt:Q", title="Count"),
                alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
            ],
        ).properties(
            height=220,
            title=f"Top Countries • Basis: {title_suffix} • Grouping: {grouping_suffix}",
        ).facet(
            column=alt.Column(f"{country_col}:N", title="Top Countries", sort=top_countries)
        )
        st.altair_chart(facet_chart, use_container_width=True)

        # Overall contribution lines (only within chosen top countries)
        overall = (
            mcs
            .assign(_pay_m_str=mcs["_pay_m_str"].astype(str))
            .groupby(["_pay_m_str","_traj_source"], observed=True, as_index=False)
            .agg(Cnt=("Cnt","sum"), TotalAll=("TotalAll","first"))
        )
        overall["PctOfOverall"] = np.where(overall["TotalAll"]>0, overall["Cnt"]/overall["TotalAll"]*100.0, 0.0)

        lines = alt.Chart(overall).mark_line(point=True).encode(
            x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
            y=alt.Y("PctOfOverall:Q", title="% of overall business (Top countries)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("_traj_source:N", title="Source", sort=src_order),
            tooltip=[
                alt.Tooltip("_pay_m_str:N", title="Month"),
                alt.Tooltip("_traj_source:N", title="Source"),
                alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
            ],
        ).properties(
            title=f"Overall contribution by source (Top countries • Basis: {title_suffix} • Grouping: {grouping_suffix})",
            height=320,
        )
        st.altair_chart(lines, use_container_width=True)
    else:
        st.info("No data for the selected trailing window to build the trajectory.", icon="ℹ️")

    # =========================
    # Interactive Mix Analyzer
    # =========================
    st.markdown("### Interactive Mix Analyzer — % of overall business from your selection")

    col_im1, col_im2 = st.columns([1.6, 1])
    with col_im1:
        use_key_sources = st.checkbox(
            "Use key-source mapping (Referral / PM - Search / PM - Social)",
            value=True,
            key="eighty_use_key_sources",
            help="On = group sources into 3 key buckets. Off = raw deal source names.",
        )

    # Cohort within window (payments inside window)
    cohort_now = df80[df80["_pay_dt"].dt.date.between(start_d, end_d)].copy()
    cohort_now = assign_src_pick(cohort_now, source_col, use_key_sources)

    # Source option list
    if source_col and source_col in cohort_now.columns:
        if use_key_sources:
            src_options = ["Referral", "PM - Search", "PM - Social", "Other"]
            default_srcs = ["Referral"]
        else:
            src_options = sorted(cohort_now["_src_pick"].unique().tolist())
            default_srcs = src_options[:1] if src_options else []
        picked_srcs = st.multiselect(
            "Select Deal Sources",
            options=src_options,
            default=[s for s in default_srcs if s in src_options],
            key="eighty_mix_sources_pick",
            help="Pick one or more sources. Each source gets its own Country control below.",
        )
    else:
        picked_srcs = []
        st.info("Deal Source column not found, source filtering disabled for Mix Analyzer.")

    # Session keys helpers
    def _mode_key(src): return f"eighty_src_mode::{src}"
    def _countries_key(src): return f"eighty_src_countries::{src}"

    DISPLAY_ANY = "Any country (all)"
    per_source_config = {}  # src -> dict(mode, countries, available)

    for src in picked_srcs:
        available = (
            cohort_now.loc[cohort_now["_src_pick"] == src, country_col]
            .astype(str).fillna("Unknown").value_counts().index.tolist()
            if country_col and country_col in cohort_now.columns else []
        )
        if _mode_key(src) not in st.session_state:
            st.session_state[_mode_key(src)] = "All"
        if _countries_key(src) not in st.session_state:
            st.session_state[_countries_key(src)] = available.copy()

        if st.session_state[_mode_key(src)] == "Specific":
            prev = st.session_state[_countries_key(src)]
            st.session_state[_countries_key(src)] = [c for c in prev if (c in available) or (c == DISPLAY_ANY)]
            if not st.session_state[_countries_key(src)] and available:
                st.session_state[_countries_key(src)] = available[:5]

        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Source:** {src}")
                mode = st.radio(
                    "Country scope",
                    options=["All", "None", "Specific"],
                    index=["All", "None", "Specific"].index(st.session_state[_mode_key(src)]),
                    key=_mode_key(src),
                    horizontal=True,
                )
            with c2:
                if mode == "Specific":
                    options = [DISPLAY_ANY] + available
                    st.multiselect(
                        f"Countries for {src}",
                        options=options,
                        default=st.session_state[_countries_key(src)],
                        key=_countries_key(src),
                        help="Pick countries or choose 'Any country (all)' to include all countries for this source.",
                    )
                elif mode == "All":
                    st.caption(f"All countries for **{src}** ({len(available)}).")
                else:
                    st.caption(f"Excluded **{src}** (no countries).")

        per_source_config[src] = {
            "mode": st.session_state[_mode_key(src)],
            "countries": st.session_state[_countries_key(src)],
            "available": available,
        }

    # Build masks from per-source config
    def make_union_mask(df_in: pd.DataFrame, per_cfg: dict, use_key: bool) -> pd.Series:
        d = assign_src_pick(df_in, source_col, use_key)
        base = pd.Series(False, index=d.index)
        if not per_cfg:
            return base
        if country_col and country_col in d.columns:
            c_series = d[country_col].astype(str).fillna("Unknown")
        else:
            c_series = pd.Series("Unknown", index=d.index)

        for src, info in per_cfg.items():
            mode = info["mode"]
            if mode == "None":
                continue
            src_mask = (d["_src_pick"] == src)
            if mode == "All":
                base = base | src_mask
            else:  # Specific
                chosen = set(info["countries"])
                if not chosen:
                    continue
                if DISPLAY_ANY in chosen:
                    base = base | src_mask
                else:
                    base = base | (src_mask & c_series.isin(chosen))
        return base

    def active_sources(per_cfg: dict) -> list[str]:
        return [s for s, v in per_cfg.items() if v["mode"] != "None"]

    mix_view = st.radio(
        "Mix view",
        ["Aggregate (range total)", "Month-wise"],
        index=0,
        horizontal=True,
        key="eighty_mix_view",
        help="Aggregate = single % for whole range. Month-wise = monthly % time series with one line per picked source.",
    )

    total_payments = int(len(cohort_now))
    if total_payments == 0:
        st.warning("No payments (enrolments) in the selected window.", icon="⚠️")
    else:
        sel_mask = make_union_mask(cohort_now, per_source_config, use_key_sources)
        if not sel_mask.any():
            st.info("No selection applied (pick at least one source in All/Specific).")
        else:
            selected_payments = int(sel_mask.sum())
            pct_of_overall = (selected_payments / total_payments * 100.0) if total_payments > 0 else 0.0

            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-title'>Contribution of your selection ({start_d} → {end_d})</div>"
                f"<div class='kpi-value'>{pct_of_overall:.1f}%</div>"
                f"<div class='kpi-sub'>Enrolments in selection: {selected_payments:,} • Total: {total_payments:,}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Quick breakdown by source
            dsel = cohort_now.loc[sel_mask].copy()
            if not dsel.empty:
                bysrc = dsel.groupby("_src_pick").size().rename("SelCnt").reset_index()
                bysrc["PctOfOverall"] = bysrc["SelCnt"] / total_payments * 100.0
                chart = alt.Chart(bysrc).mark_bar(opacity=0.9).encode(
                    x=alt.X("_src_pick:N", title="Source"),
                    y=alt.Y("PctOfOverall:Q", title="% of overall business"),
                    tooltip=[
                        alt.Tooltip("_src_pick:N", title="Source"),
                        alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                        alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
                    ],
                    color=alt.Color("_src_pick:N", legend=alt.Legend(orient="bottom")),
                ).properties(height=320, title="Selection breakdown by source — % of overall")
                st.altair_chart(chart, use_container_width=True)

            # Month-wise lines
            if mix_view == "Month-wise":
                cohort_now["_pay_m"] = cohort_now["_pay_dt"].dt.to_period("M")
                months_in_range = (
                    cohort_now["_pay_m"].dropna().sort_values().unique().astype(str).tolist()
                )

                # Overall monthly totals
                overall_m = cohort_now.groupby("_pay_m").size().rename("TotalAll").reset_index()
                overall_m["Month"] = overall_m["_pay_m"].astype(str)

                # All Selected monthly counts using union mask
                all_sel_m = cohort_now.loc[sel_mask].groupby("_pay_m").size().rename("SelCnt").reset_index()
                all_sel_m["Month"] = all_sel_m["_pay_m"].astype(str)

                all_line = overall_m.merge(all_sel_m[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
                all_line["PctOfOverall"] = np.where(all_line["TotalAll"]>0, all_line["SelCnt"]/all_line["TotalAll"]*100.0, 0.0)
                all_line["Series"] = "All Selected"
                all_line = all_line[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
                all_line["Month"] = pd.Categorical(all_line["Month"], categories=months_in_range, ordered=True)

                # Per-source monthly lines honoring each source's country selection
                per_src_frames = []
                for src in active_sources(per_source_config):
                    one_cfg = {src: per_source_config[src]}
                    smask = make_union_mask(cohort_now, one_cfg, use_key_sources)
                    s_cnt = cohort_now.loc[smask].groupby("_pay_m").size().rename("SelCnt").reset_index()
                    if s_cnt.empty:
                        continue
                    s_cnt["Month"] = s_cnt["_pay_m"].astype(str)
                    s_join = overall_m.merge(s_cnt[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
                    s_join["PctOfOverall"] = np.where(s_join["TotalAll"]>0, s_join["SelCnt"]/s_join["TotalAll"]*100.0, 0.0)
                    s_join["Series"] = src
                    s_join = s_join[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
                    s_join["Month"] = pd.Categorical(s_join["Month"], categories=months_in_range, ordered=True)
                    per_src_frames.append(s_join)

                if per_src_frames:
                    lines_df = pd.concat([all_line] + per_src_frames, ignore_index=True)
                else:
                    lines_df = all_line.copy()

                avg_monthly_pct = lines_df.loc[lines_df["Series"]=="All Selected", "PctOfOverall"].mean() if not lines_df.empty else 0.0
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-title'>Month-wise: average % contribution (All Selected)</div>"
                    f"<div class='kpi-value'>{avg_monthly_pct:.1f}%</div>"
                    f"<div class='kpi-sub'>Months: {lines_df['Month'].nunique() if not lines_df.empty else 0}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                stroke_width = alt.condition("datum.Series == 'All Selected'", alt.value(4), alt.value(2))
                chart = alt.Chart(lines_df).mark_line(point=True).encode(
                    x=alt.X("Month:N", sort=months_in_range, title="Month"),
                    y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Series:N", title="Series"),
                    strokeWidth=stroke_width,
                    tooltip=[
                        alt.Tooltip("Month:N"),
                        alt.Tooltip("Series:N"),
                        alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                        alt.Tooltip("TotalAll:Q", title="Total enrolments"),
                        alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
                    ],
                ).properties(height=360, title="Month-wise % of overall — All Selected vs each picked source")
                st.altair_chart(chart, use_container_width=True)

    # =========================
    # Deals vs Enrolments — current selection
    # =========================
    st.markdown("### Deals vs Enrolments — for your current selection")
    def _build_created_paid_monthly(df_all: pd.DataFrame, start_d: date, end_d: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = df_all.copy()
        d["_cdate"] = coerce_datetime(d[create_col]).dt.date
        d["_pdate"] = coerce_datetime(d[pay_col]).dt.date
        d["_cmonth"] = coerce_datetime(d[create_col]).dt.to_period("M")
        d["_pmonth"] = coerce_datetime(d[pay_col]).dt.to_period("M")

        cwin = d["_cdate"].between(start_d, end_d)
        pwin = d["_pdate"].between(start_d, end_d)

        month_index = pd.period_range(start=start_d.replace(day=1), end=end_d.replace(day=1), freq="M")

        created_m = (
            d.loc[cwin].groupby("_cmonth").size()
              .reindex(month_index, fill_value=0)
              .rename_axis(index="_month").reset_index(name="CreatedCnt")
        )
        paid_m = (
            d.loc[pwin].groupby("_pmonth").size()
              .reindex(month_index, fill_value=0)
              .rename_axis(index="_month").reset_index(name="PaidCnt")
        )

        monthly = created_m.merge(paid_m, on="_month", how="outer").fillna(0)
        monthly["Month"] = monthly["_month"].astype(str)
        monthly = monthly[["Month", "CreatedCnt", "PaidCnt"]]
        monthly["ConvPct"] = np.where(monthly["CreatedCnt"] > 0,
                                      monthly["PaidCnt"] / monthly["CreatedCnt"] * 100.0, 0.0)

        total_created = int(monthly["CreatedCnt"].sum())
        total_paid    = int(monthly["PaidCnt"].sum())
        agg = pd.DataFrame({
            "CreatedCnt": [total_created],
            "PaidCnt":    [total_paid],
            "ConvPct":    [float((total_paid / total_created * 100.0) if total_created > 0 else 0.0)]
        })
        return monthly, agg

    if picked_srcs:
        union_mask_all = make_union_mask(df80, per_source_config, use_key_sources)
    else:
        union_mask_all = pd.Series(False, index=df80.index)

    df_sel_all = df80.loc[union_mask_all].copy()
    monthly_sel, agg_sel = _build_created_paid_monthly(df_sel_all, start_d, end_d)

    kpa, kpb, kpc = st.columns(3)
    with kpa:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Deals (Created)</div>"
            f"<div class='kpi-value'>{int(agg_sel['CreatedCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div>"
            f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with kpb:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div>"
            f"<div class='kpi-value'>{int(agg_sel['PaidCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div>"
            f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with kpc:
        conv_val = float(agg_sel['ConvPct'].iloc[0]) if not agg_sel.empty else 0.0
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div>"
            f"<div class='kpi-value'>{conv_val:.1f}%</div>"
            f"<div class='kpi-sub'>Num: {int(agg_sel['PaidCnt'].iloc[0]) if not agg_sel.empty else 0:,} • Den: {int(agg_sel['CreatedCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div></div>",
            unsafe_allow_html=True)

    show_conv_line = st.checkbox("Overlay Conversion% line on bars", value=True, key="eighty_mix_conv_line")

    if not monthly_sel.empty:
        bar_df = monthly_sel.melt(
            id_vars=["Month"],
            value_vars=["CreatedCnt", "PaidCnt"],
            var_name="Metric",
            value_name="Count"
        )
        bar_df["Metric"] = bar_df["Metric"].map({"CreatedCnt": "Deals Created", "PaidCnt": "Enrolments"})

        bars = alt.Chart(bar_df).mark_bar(opacity=0.9).encode(
            x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Metric:N", title=""),
            xOffset=alt.XOffset("Metric:N"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")],
        ).properties(height=360, title="Month-wise — Deals & Enrolments (bars)")

        if show_conv_line:
            line = alt.Chart(monthly_sel).mark_line(point=True).encode(
                x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
                y=alt.Y("ConvPct:Q", title="Conversion%", axis=alt.Axis(orient="right")),
                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("ConvPct:Q", title="Conversion%", format=".1f")],
                color=alt.value("#16a34a"),
            )
            st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent'), use_container_width=True)
        else:
            st.altair_chart(bars, use_container_width=True)

        with st.expander("Download: Month-wise Deals / Enrolments / Conversion% (selection)"):
            out_tbl = monthly_sel.rename(columns={
                "CreatedCnt": "Deals Created",
                "PaidCnt": "Enrolments",
                "ConvPct": "Conversion %"
            })
            st.dataframe(out_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Month-wise Deals/Enrolments/Conversion",
                data=out_tbl.to_csv(index=False).encode("utf-8"),
                file_name="selection_deals_enrolments_conversion_monthwise.csv",
                mime="text/csv",
                key="eighty_download_monthwise",
            )
    else:
        st.info("No month-wise data to plot for the current selection. Pick at least one source in All/Specific.")

    # ----------------------------
    # Tables + Downloads
    # ----------------------------
    st.markdown("<div class='section-title'>Tables</div>", unsafe_allow_html=True)
    tabs80 = st.tabs(["Deal Source 80-20", "Country 80-20", "Cohort Rows", "Trajectory table", "Conversion by Source"])

    with tabs80[0]:
        if src_tbl.empty:
            st.info("No enrollments in scope.")
        else:
            st.dataframe(src_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Deal Source Pareto",
                src_tbl.to_csv(index=False).encode("utf-8"),
                "pareto_deal_source.csv",
                "text/csv",
                key="eighty_dl_srcpareto",
            )

    with tabs80[1]:
        if cty_tbl.empty:
            st.info("No enrollments in scope.")
        else:
            st.dataframe(cty_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Country Pareto",
                cty_tbl.to_csv(index=False).encode("utf-8"),
                "pareto_country.csv",
                "text/csv",
                key="eighty_dl_ctypareto",
            )

    with tabs80[2]:
        show_cols = []
        if create_col: show_cols.append(create_col)
        if pay_col: show_cols.append(pay_col)
        if source_col: show_cols.append(source_col)
        if country_col: show_cols.append(country_col)
        preview = df_cohort[show_cols].copy() if show_cols else df_cohort.copy()
        st.dataframe(preview.head(1000), use_container_width=True)
        st.download_button(
            "Download CSV – Cohort subset",
            preview.to_csv(index=False).encode("utf-8"),
            "cohort_subset.csv",
            "text/csv",
            key="eighty_dl_cohort",
        )

    with tabs80[3]:
        if 'mcs' in locals() and not mcs.empty:
            show = mcs.rename(columns={country_col: "Country"})[["Country","_pay_m_str","_traj_source","Cnt","TotalAll","PctOfOverall"]]
            show = show.sort_values(["Country","_pay_m_str","_traj_source"])
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Download CSV – Trajectory",
                show.to_csv(index=False).encode("utf-8"),
                "trajectory_top_countries_sources.csv",
                "text/csv",
                key="eighty_dl_traj",
            )
        else:
            st.info("No trajectory table for the current selection.")

    with tabs80[4]:
        if not bysrc_conv.empty:
            st.dataframe(bysrc_conv, use_container_width=True)
            st.download_button(
                "Download CSV – Conversion by Key Source",
                bysrc_conv.to_csv(index=False).encode("utf-8"),
                "conversion_by_key_source.csv",
                "text/csv",
                key="eighty_dl_conv",
            )
        else:
            st.info("No conversion table for the current selection.")

# =========================
elif view == "Stuck deals":
    import pandas as pd, numpy as np, altair as alt
    from datetime import date, timedelta
    from calendar import monthrange

    st.subheader("Stuck deals – Funnel & Propagation (Created → Trial → Cal Done → Payment)")

    # ---------------------------
    # Helper: months_back_list()
    # ---------------------------
    def months_back_list(end_d: date, k: int):
        """
        Return a chronological list of k monthly Periods ending at end_d's month.

        Parameters
        ----------
        end_d : datetime.date
            Anchor date; the last period in the list is end_d's calendar month.
        k : int
            Number of months to include (>=1).

        Returns
        -------
        List[pandas.Period]
            Monthly periods in ascending order, length k, ending at end_d's month.

        Example
        -------
        >>> months_back_list(date(2025, 3, 15), 3)
        [Period('2025-01', 'M'), Period('2025-02', 'M'), Period('2025-03', 'M')]
        """
        if k <= 0:
            return []
        end_per = pd.Period(end_d, freq="M")
        return list(pd.period_range(end=end_per, periods=k, freq="M"))

    # Small style for KPI cards
    st.markdown(
        """
        <style>
          .kpi-card { border:1px solid #e5e7eb; border-radius:14px; padding:10px 12px; background:#fff; }
          .kpi-title { font-size:0.9rem; color:#6b7280; margin-bottom:6px; }
          .kpi-value { font-size:1.4rem; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ==== Column presence (warn but never stop)
    missing_cols = []
    for col_label, col_var in [
        ("Create Date", create_col),
        ("First Calibration Scheduled Date", first_cal_sched_col),
        ("Calibration Rescheduled Date", cal_resched_col),
        ("Calibration Done Date", cal_done_col),
        ("Payment Received Date", pay_col),
    ]:
        if not col_var or col_var not in df_f.columns:
            missing_cols.append(col_label)
    if missing_cols:
        st.warning(
            "Missing columns: " + ", ".join(missing_cols) +
            ". Funnel/metrics will skip the missing stages where applicable.",
            icon="⚠️"
        )

    # Try to find the Slot column if not already mapped
    if ("calibration_slot_col" not in locals()) or (not calibration_slot_col) or (calibration_slot_col not in df_f.columns):
        calibration_slot_col = find_col(df_f, [
            "Calibration Slot (Deal)", "Calibration Slot", "Book Slot", "Trial Slot"
        ])

    # ==== Scope controls
    scope_mode = st.radio(
        "Scope",
        ["Month", "Trailing days"],
        horizontal=True,
        index=0,
        help="Month = a single calendar month. Trailing days = rolling window ending today."
    )

    if scope_mode == "Month":
        # Build month list from whatever date columns exist
        candidates = []
        if create_col:
            candidates.append(coerce_datetime(df_f[create_col]))
        if first_cal_sched_col and first_cal_sched_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[first_cal_sched_col]))
        if cal_resched_col and cal_resched_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[cal_resched_col]))
        if cal_done_col and cal_done_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[cal_done_col]))
        if pay_col:
            candidates.append(coerce_datetime(df_f[pay_col]))

        if candidates:
            all_months = (
                pd.to_datetime(pd.concat(candidates, ignore_index=True))
                  .dropna()
                  .dt.to_period("M")
                  .sort_values()
                  .unique()
                  .astype(str)
                  .tolist()
            )
        else:
            all_months = []

        # Ensure at least the running month is present
        if not all_months:
            all_months = [str(pd.Period(date.today(), freq="M"))]

        # Preselect running month if present; else fallback to last available month
        running_period = str(pd.Period(date.today(), freq="M"))
        default_idx = all_months.index(running_period) if running_period in all_months else len(all_months) - 1

        sel_month = st.selectbox("Select month (YYYY-MM)", options=all_months, index=default_idx)
        yy, mm = map(int, sel_month.split("-"))

        # Month bounds as timestamps (avoid dtype mismatch)
        range_start = pd.Timestamp(date(yy, mm, 1))
        range_end   = pd.Timestamp(date(yy, mm, monthrange(yy, mm)[1]))

        st.caption(f"Scope: **{range_start.date()} → {range_end.date()}**")
    else:
        trailing = st.slider("Trailing window (days)", min_value=7, max_value=60, value=15, step=1)
        range_end = pd.Timestamp(date.today())
        range_start = range_end - pd.Timedelta(days=trailing - 1)
        st.caption(f"Scope: **{range_start.date()} → {range_end.date()}** (last {trailing} days)")

    # ==== Prepare normalized datetime columns
    d = df_f.copy()
    d["_c"]  = coerce_datetime(d[create_col]) if create_col else pd.Series(pd.NaT, index=d.index)
    d["_f"]  = coerce_datetime(d[first_cal_sched_col]) if first_cal_sched_col and first_cal_sched_col in d.columns else pd.Series(pd.NaT, index=d.index)
    d["_r"]  = coerce_datetime(d[cal_resched_col])     if cal_resched_col and cal_resched_col in d.columns     else pd.Series(pd.NaT, index=d.index)
    d["_fd"] = coerce_datetime(d[cal_done_col])        if cal_done_col and cal_done_col in d.columns          else pd.Series(pd.NaT, index=d.index)
    d["_p"]  = coerce_datetime(d[pay_col]) if pay_col else pd.Series(pd.NaT, index=d.index)

    # Effective trial date = min(First Cal, Rescheduled), NaT-safe
    d["_trial"] = d[["_f", "_r"]].min(axis=1, skipna=True)

    # ==== Filter: Booking type (Pre-Book vs Self-Book) based on Trial + Slot
    # Rule:
    #   Pre-Book  = has a Trial date AND Calibration Slot (Deal) is non-empty
    #   Self-Book = everything else (no trial OR empty slot)
    if calibration_slot_col and calibration_slot_col in d.columns:
        slot_series = d[calibration_slot_col].astype(str)
        _s = slot_series.str.strip().str.lower()
        has_slot = _s.ne("") & _s.ne("nan") & _s.ne("none")

        is_prebook = d["_trial"].notna() & has_slot
        d["_booking_type"] = np.where(is_prebook, "Pre-Book", "Self-Book")

        booking_choice = st.radio(
            "Booking type",
            options=["All", "Pre-Book", "Self-Book"],
            index=0,
            horizontal=True,
            help="Pre-Book = Trial present AND slot filled. Self-Book = otherwise."
        )
        if booking_choice != "All":
            d = d[d["_booking_type"] == booking_choice].copy()
            st.caption(f"Booking type filter: **{booking_choice}** • Rows now: **{len(d):,}**")
    else:
        st.info("Calibration Slot (Deal) column not found — booking type filter not applied.")

    # Use Timestamp boundaries for between() to avoid dtype mismatch
    rs, re = pd.Timestamp(range_start), pd.Timestamp(range_end)

    # ==== Cohort: deals CREATED within scope
    mask_created = d["_c"].between(rs, re)
    cohort = d.loc[mask_created].copy()
    total_created = int(len(cohort))

    # Stage 2: Trial in SAME scope & same cohort
    trial_mask = cohort["_trial"].between(rs, re)
    trial_df = cohort.loc[trial_mask].copy()
    total_trial = int(len(trial_df))

    # Stage 3: Cal Done in SAME scope from those that had Trial in scope
    caldone_mask = trial_df["_fd"].between(rs, re)
    caldone_df = trial_df.loc[caldone_mask].copy()
    total_caldone = int(len(caldone_df))

    # Stage 4: Payment in SAME scope from those that had Cal Done in scope
    pay_mask = caldone_df["_p"].between(rs, re)
    pay_df = caldone_df.loc[pay_mask].copy()
    total_pay = int(len(pay_df))

    # ==== Funnel summary
    funnel_rows = [
        {"Stage": "Created (T)",            "Count": total_created, "FromPrev_pct": 100.0},
        {"Stage": "Trial (First/Resched)",  "Count": total_trial,   "FromPrev_pct": (total_trial / total_created * 100.0) if total_created > 0 else 0.0},
        {"Stage": "Calibration Done",       "Count": total_caldone, "FromPrev_pct": (total_caldone / total_trial * 100.0) if total_trial > 0 else 0.0},
        {"Stage": "Payment Received",       "Count": total_pay,     "FromPrev_pct": (total_pay / total_caldone * 100.0) if total_caldone > 0 else 0.0},
    ]
    funnel_df = pd.DataFrame(funnel_rows)

    bar = alt.Chart(funnel_df).mark_bar(opacity=0.9).encode(
        x=alt.X("Count:Q", title="Count"),
        y=alt.Y("Stage:N", sort=list(funnel_df["Stage"])[::-1], title=""),
        tooltip=[
            alt.Tooltip("Stage:N"),
            alt.Tooltip("Count:Q"),
            alt.Tooltip("FromPrev_pct:Q", title="% from previous", format=".1f"),
        ],
        color=alt.Color("Stage:N", legend=None),
    ).properties(height=240, title="Funnel (same cohort within scope)")
    txt = alt.Chart(funnel_df).mark_text(align="left", dx=5).encode(
        x="Count:Q",
        y=alt.Y("Stage:N", sort=list(funnel_df["Stage"])[::-1]),
        text=alt.Text("Count:Q"),
    )
    st.altair_chart(bar + txt, use_container_width=True)

    st.caption(
        f"Created: {total_created} • Trial: {total_trial} • Cal Done: {total_caldone} • Payments: {total_pay}"
    )

    # ==== Propagation (average days) – computed only on the same filtered sets
    def avg_days(src_series, dst_series) -> float:
        s = (dst_series - src_series).dt.days
        s = s.dropna()
        return float(s.mean()) if len(s) else np.nan

    avg_ct = avg_days(trial_df["_c"], trial_df["_trial"]) if not trial_df.empty else np.nan
    avg_tc = avg_days(caldone_df["_trial"], caldone_df["_fd"]) if not caldone_df.empty else np.nan
    avg_dp = avg_days(pay_df["_fd"], pay_df["_p"]) if not pay_df.empty else np.nan

    def fmtd(x): return "–" if pd.isna(x) else f"{x:.1f} days"
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Created → Trial</div><div class='kpi-value'>{fmtd(avg_ct)}</div></div>",
            unsafe_allow_html=True
        )
    with g2:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Trial → Cal Done</div><div class='kpi-value'>{fmtd(avg_tc)}</div></div>",
            unsafe_allow_html=True
        )
    with g3:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Cal Done → Payment</div><div class='kpi-value'>{fmtd(avg_dp)}</div></div>",
            unsafe_allow_html=True
        )

    # ==== Month-on-Month comparison
    st.markdown("### Month-on-Month comparison")
    compare_k = st.slider("Compare last N months (ending at selected month or current)", 2, 12, 6, step=1)

    # Decide anchor month
    anchor_day = rs.date() if scope_mode == "Month" else date.today()
    months = months_back_list(anchor_day, compare_k)  # list of monthly Periods

    def month_funnel(m_period: pd.Period):
        ms = pd.Timestamp(date(m_period.year, m_period.month, 1))
        me = pd.Timestamp(date(m_period.year, m_period.month, monthrange(m_period.year, m_period.month)[1]))

        coh = d[d["_c"].between(ms, me)].copy()
        ct = int(len(coh))

        coh_tr = coh[coh["_trial"].between(ms, me)].copy()
        tr = int(len(coh_tr))

        coh_cd = coh_tr[coh_tr["_fd"].between(ms, me)].copy()
        cd = int(len(coh_cd))

        py = int(coh_cd["_p"].between(ms, me).sum())

        # propagation avgs
        avg1 = avg_days(coh_tr["_c"], coh_tr["_trial"]) if not coh_tr.empty else np.nan
        avg2 = avg_days(coh_cd["_trial"], coh_cd["_fd"]) if not coh_cd.empty else np.nan
        avg3 = avg_days(coh_cd["_fd"], coh_cd["_p"]) if not coh_cd.empty else np.nan

        return {
            "Month": str(m_period),
            "Created": ct,
            "Trial": tr,
            "CalDone": cd,
            "Paid": py,
            "Trial_from_Created_pct": (tr / ct * 100.0) if ct > 0 else 0.0,
            "CalDone_from_Trial_pct": (cd / tr * 100.0) if tr > 0 else 0.0,
            "Paid_from_CalDone_pct": (py / cd * 100.0) if cd > 0 else 0.0,
            "Avg_Created_to_Trial_days": avg1,
            "Avg_Trial_to_CalDone_days": avg2,
            "Avg_CalDone_to_Payment_days": avg3,
        }

    mom_tbl = pd.DataFrame([month_funnel(m) for m in months])

    if mom_tbl.empty:
        st.info("Not enough historical data to build month-on-month comparison.")
    else:
        # Conversion step lines
        conv_long = mom_tbl.melt(
            id_vars=["Month"],
            value_vars=["Trial_from_Created_pct", "CalDone_from_Trial_pct", "Paid_from_CalDone_pct"],
            var_name="Step",
            value_name="Pct",
        )
        conv_chart = alt.Chart(conv_long).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=mom_tbl["Month"].tolist()),
            y=alt.Y("Pct:Q", title="Step conversion %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Step:N", title="Step"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Step:N"), alt.Tooltip("Pct:Q", format=".1f")],
        ).properties(height=320, title="Step conversion% (MoM)")
        st.altair_chart(conv_chart, use_container_width=True)

        # Propagation lines
        lag_long = mom_tbl.melt(
            id_vars=["Month"],
            value_vars=["Avg_Created_to_Trial_days", "Avg_Trial_to_CalDone_days", "Avg_CalDone_to_Payment_days"],
            var_name="Lag",
            value_name="Days",
        )
        lag_chart = alt.Chart(lag_long).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=mom_tbl["Month"].tolist()),
            y=alt.Y("Days:Q", title="Avg days"),
            color=alt.Color("Lag:N", title="Propagation"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Lag:N"), alt.Tooltip("Days:Q", format=".1f")],
        ).properties(height=320, title="Average propagation (MoM)")
        st.altair_chart(lag_chart, use_container_width=True)

        with st.expander("Month-on-Month table"):
            st.dataframe(mom_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – MoM Funnel & Propagation",
                data=mom_tbl.to_csv(index=False).encode("utf-8"),
                file_name="stuck_deals_mom_funnel_propagation.csv",
                mime="text/csv",
            )


elif view == "Lead Movement":
    st.subheader("Lead Movement — inactivity by Last Connected / Lead Activity (Create-date scoped)")

    # ---- Column mapping
    lead_activity_col = find_col(df, [
        "Lead Activity Date", "Lead activity date", "Last Activity Date", "Last Activity"
    ])
    last_connected_col = find_col(df, [
        "Last Connected", "Last connected", "Last Contacted", "Last Contacted Date"
    ])

    if not create_col:
        st.error("Create Date column not found — this tab scopes the population by Create Date.")
        st.stop()

    # ---- Optional Deal Stage filter (applies to population)
    d = df_f.copy()
    if dealstage_col and dealstage_col in d.columns:
        stage_vals = ["All"] + sorted(d[dealstage_col].dropna().astype(str).unique().tolist())
        sel_stages = st.multiselect(
            "Deal Stage (optional filter on population)",
            stage_vals, default=["All"], key="lm_stage_filter"
        )
        if "All" not in sel_stages:
            d = d[d[dealstage_col].astype(str).isin(sel_stages)].copy()
    else:
        st.caption("Deal Stage column not found — stage filter disabled.")

    if d.empty:
        st.info("No rows after filters.")
        st.stop()

    # ---- Date scope (population by Create Date)
    st.markdown("**Date scope (based on Create Date)**")
    c1, c2 = st.columns(2)
    scope_pick = st.radio(
        "Presets",
        ["Yesterday", "Today", "This month", "Last month", "Custom"],
        index=2, horizontal=True, key="lm_scope"
    )
    if scope_pick == "Yesterday":
        scope_start, scope_end = yday, yday
    elif scope_pick == "Today":
        scope_start, scope_end = today, today
    elif scope_pick == "This month":
        scope_start, scope_end = month_bounds(today)
    elif scope_pick == "Last month":
        scope_start, scope_end = last_month_bounds(today)
    else:
        with c1:
            scope_start = st.date_input("Start (Create Date)", value=today.replace(day=1), key="lm_cstart")
        with c2:
            scope_end = st.date_input("End (Create Date)", value=month_bounds(today)[1], key="lm_cend")
        if scope_end < scope_start:
            st.error("End date cannot be before start date.")
            st.stop()

    st.caption(f"Create-date scope: **{scope_start} → {scope_end}**")

    # ---- Choose reference date for inactivity
    ref_pick = st.radio(
        "Reference date (for inactivity days)",
        ["Last Connected", "Lead Activity Date"],
        index=0, horizontal=True, key="lm_ref_pick"
    )
    if ref_pick == "Last Connected":
        ref_col = last_connected_col if (last_connected_col and last_connected_col in d.columns) else None
    else:
        ref_col = lead_activity_col if (lead_activity_col and lead_activity_col in d.columns) else None

    if not ref_col:
        st.warning(f"Selected reference column for '{ref_pick}' not found in data.")
        st.stop()

    # ---- Build in-scope dataset (population by Create Date)
    d["_cdate"] = coerce_datetime(d[create_col]).dt.date
    pop_mask = d["_cdate"].between(scope_start, scope_end)
    d_work = d.loc[pop_mask].copy()

    # Compute inactivity days from chosen reference column
    d_work["_ref_dt"] = coerce_datetime(d_work[ref_col])
    d_work["_days_since"] = (pd.Timestamp(today) - d_work["_ref_dt"]).dt.days  # NaT-safe diff

    # ---- Slider (inactivity range)
    valid_days = d_work["_days_since"].dropna()
    if valid_days.empty:
        min_d, max_d = 0, 90
    else:
        min_d, max_d = int(valid_days.min()), int(valid_days.max())
        min_d = min(0, min_d)
        max_d = max(1, max_d)
    days_low, days_high = st.slider(
        "Inactivity range (days)",
        min_value=int(min_d), max_value=int(max_d),
        value=(min(7, max(0, min_d)), min(30, max_d)),
        step=1, key="lm_range"
    )
    range_mask = d_work["_days_since"].between(days_low, days_high)

    # ---- Bucketize inactivity for stacked charts
    def bucketize(n):
        if pd.isna(n):
            return "Unknown"
        n = int(n)
        if n <= 1:   return "0–1"
        if n <= 3:   return "2–3"
        if n <= 7:   return "4–7"
        if n <= 14:  return "8–14"
        if n <= 30:  return "15–30"
        if n <= 60:  return "31–60"
        if n <= 90:  return "61–90"
        return "90+"

    d_work["Bucket"] = d_work["_days_since"].apply(bucketize)
    bucket_order = ["0–1","2–3","4–7","8–14","15–30","31–60","61–90","90+","Unknown"]

    # ---- Stacked by Deal Source
    st.markdown("### Inactivity distribution — stacked by JetLearn Deal Source")
    if source_col and source_col in d_work.columns:
        d_work["_source"] = d_work[source_col].fillna("Unknown").astype(str)
        by_src = (
            d_work.groupby(["Bucket","_source"])
                  .size().reset_index(name="Count")
        )
        chart_src = (
            alt.Chart(by_src)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_source:N", title="Deal Source", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_source:N", title="Deal Source"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Deal Source")
        )
        st.altair_chart(chart_src, use_container_width=True)
    else:
        st.info("Deal Source column not found — skipping source-wise stack.")

    # ---- Stacked by Country (Top-5 toggle)
    st.markdown("### Inactivity distribution — stacked by Country")
    if country_col and country_col in d_work.columns:
        d_work["_country"] = d_work[country_col].fillna("Unknown").astype(str)
        totals_country = d_work.groupby("_country").size().sort_values(ascending=False)
        show_all_countries = st.checkbox(
            "Show all countries (uncheck to show Top 5 only)",
            value=False, key="lm_show_all_cty"
        )
        if show_all_countries:
            keep_countries = totals_country.index.tolist()
            title_suffix = "All countries"
        else:
            keep_countries = totals_country.head(5).index.tolist()
            title_suffix = "Top 5 countries"

        by_cty = (
            d_work[d_work["_country"].isin(keep_countries)]
            .groupby(["Bucket","_country"]).size().reset_index(name="Count")
        )
        chart_cty = (
            alt.Chart(by_cty)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_country:N", title="Country", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_country:N", title="Country"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Country ({title_suffix})")
        )
        st.altair_chart(chart_cty, use_container_width=True)
    else:
        st.info("Country column not found — skipping country-wise stack.")

    # ---- Deal Stage detail for selected inactivity range
    st.markdown("### Deal Stage detail — for selected inactivity range")
    if dealstage_col and dealstage_col in d_work.columns:
        stage_counts = (
            d_work.loc[range_mask, dealstage_col]
                  .fillna("Unknown").astype(str)
                  .value_counts().reset_index()
        )
        stage_counts.columns = ["Deal Stage", "Count"]
        st.dataframe(stage_counts, use_container_width=True)
        st.download_button(
            "Download CSV — Deal Stage counts (selected inactivity range)",
            data=stage_counts.to_csv(index=False).encode("utf-8"),
            file_name="lead_movement_dealstage_counts.csv",
            mime="text/csv",
            key="lm_stage_dl"
        )

        with st.expander("Show matching rows (first 1000)"):
            cols_show = []
            for c in [create_col, dealstage_col, ref_col, country_col, source_col]:
                if c and c in d_work.columns:
                    cols_show.append(c)
            preview = d_work.loc[range_mask, cols_show].head(1000)
            st.dataframe(preview, use_container_width=True)
            st.download_button(
                "Download CSV — Matching rows",
                data=d_work.loc[range_mask, cols_show].to_csv(index=False).encode("utf-8"),
                file_name="lead_movement_matching_rows.csv",
                mime="text/csv",
                key="lm_rows_dl"
            )
    else:
        st.info("Deal Stage column not found — cannot show stage detail.")

    # ---- Quick KPIs
    total_in_scope = int(len(d_work))
    missing_ref = int(d_work["_ref_dt"].isna().sum())
    selected_cnt = int(range_mask.sum())
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-title'>In-scope leads (Create Date {scope_start} → {scope_end})</div>"
        f"<div class='kpi-value'>{total_in_scope:,}</div>"
        f"<div class='kpi-sub'>Missing {ref_pick}: {missing_ref:,} • In range {days_low}–{days_high} days: {selected_cnt:,}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # =====================================================================
    #        📊 Inactivity distribution — Deal Owner / Academic Counselor
    # =====================================================================
    st.markdown("---")
    st.markdown("### 📊 Inactivity distribution — Deal Owner (Academic Counselor)")

    # Detect both fields separately
    deal_owner_raw = find_col(df, ["Deal Owner", "Owner"])
    acad_couns_raw = find_col(df, [
        "Student/Academic Counselor", "Student/Academic Counsellor",
        "Academic Counselor", "Academic Counsellor",
        "Counselor", "Counsellor"
    ])

    # Owner field selection: choose one or combine
    owner_mode = st.selectbox(
        "Owner dimension for analysis",
        [
            "Deal Owner",
            "Student/Academic Counselor",
            "Combine (Deal Owner → Student/Academic Counselor)",
            "Combine (Student/Academic Counselor → Deal Owner)",
        ],
        index=0,
        key="lm_owner_mode"
    )

    # Validate availability
    def _series_or_none(colname):
        return d_work[colname] if (colname and colname in d_work.columns) else None

    s_owner = _series_or_none(deal_owner_raw)
    s_acad  = _series_or_none(acad_couns_raw)

    if owner_mode == "Deal Owner" and s_owner is None:
        st.info("‘Deal Owner’ column not found in the current dataset.")
        st.stop()
    if owner_mode == "Student/Academic Counselor" and s_acad is None:
        st.info("‘Student/Academic Counselor’ column not found in the current dataset.")
        st.stop()
    if "Combine" in owner_mode and (s_owner is None and s_acad is None):
        st.info("Neither ‘Deal Owner’ nor ‘Student/Academic Counselor’ columns are present.")
        st.stop()

    # Build the owner dimension
    if owner_mode == "Deal Owner":
        d_work["_owner"] = s_owner.fillna("Unknown").replace("", "Unknown").astype(str)
    elif owner_mode == "Student/Academic Counselor":
        d_work["_owner"] = s_acad.fillna("Unknown").replace("", "Unknown").astype(str)
    elif owner_mode == "Combine (Deal Owner → Student/Academic Counselor)":
        # Prefer Deal Owner, fallback to Academic Counselor
        d_work["_owner"] = (
            (s_owner.fillna("").astype(str))
            .mask(lambda x: x.str.strip().eq("") & s_acad.notna(), s_acad.astype(str))
            .replace("", "Unknown")
            .fillna("Unknown")
            .astype(str)
        )
    else:  # Combine (Student/Academic Counselor → Deal Owner)
        d_work["_owner"] = (
            (s_acad.fillna("").astype(str))
            .mask(lambda x: x.str.strip().eq("") & (s_owner.notna()), s_owner.astype(str))
            .replace("", "Unknown")
            .fillna("Unknown")
            .astype(str)
        )

    # Controls: Aggregate vs Split + Top-N owners
    col_oview, col_topn = st.columns([1.2, 1])
    with col_oview:
        owner_view = st.radio(
            "View mode",
            ["Aggregate (overall)", "Split by Academic Counselor"],
            index=1, horizontal=False, key="lm_owner_view"
        )
    with col_topn:
        owner_counts_all = d_work["_owner"].value_counts()
        max_top = min(30, max(5, len(owner_counts_all)))
        top_n = st.number_input("Top N owners for charts", min_value=5, max_value=max_top, value=min(12, max_top), step=1, key="lm_owner_topn")

    # Limit to Top-N for readability
    top_owners = owner_counts_all.head(int(top_n)).index.tolist()
    d_top = d_work[d_work["_owner"].isin(top_owners)].copy()

    # Aggregate mode: bucket totals overall (no owner split)
    if owner_view == "Aggregate (overall)":
        agg_bucket = (
            d_top.groupby("Bucket")
                 .size().reset_index(name="Count")
        )
        chart_owner_agg = (
            alt.Chart(agg_bucket)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", title="Count"),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("Count:Q")]
            )
            .properties(height=320, title=f"Inactivity by {ref_pick} — Aggregate (Top {len(top_owners)} owners)")
        )
        st.altair_chart(chart_owner_agg, use_container_width=True)

    else:
        # Split mode: stacked by owner across buckets (Bucket on x, colors = owner)
        by_owner_bucket = (
            d_top.groupby(["Bucket", "_owner"])
                 .size().reset_index(name="Count")
        )
        chart_owner_split = (
            alt.Chart(by_owner_bucket)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_owner:N", title="Academic Counselor", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_owner:N", title="Academic Counselor"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Academic Counselor (Top {len(top_owners)})")
        )
        st.altair_chart(chart_owner_split, use_container_width=True)

    # ================= Option: Exclude Unknown on Owner-on-X chart =================
    st.markdown("#### Inactivity distribution — stacked by Bucket (Owner on X-axis)")
    exclude_unknown_owner = st.checkbox(
        "Exclude ‘Unknown’ owners from this chart",
        value=True,
        key="lm_owner_exclude_unknown_xaxis"
    )

    owner_x_df = d_top.copy()
    if exclude_unknown_owner:
        owner_x_df = owner_x_df[owner_x_df["_owner"] != "Unknown"]

    owner_x_bucket = (
        owner_x_df.groupby(["_owner", "Bucket"])
                  .size().reset_index(name="Count")
    )

    chart_owner_x = (
        alt.Chart(owner_x_bucket)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("_owner:N", title="Academic Counselor", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Count:Q", stack=True, title="Count"),
            color=alt.Color("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
            tooltip=[alt.Tooltip("_owner:N", title="Academic Counselor"),
                     alt.Tooltip("Bucket:N", title="Bucket"),
                     alt.Tooltip("Count:Q")]
        )
        .properties(height=380, title=f"Inactivity by {ref_pick} — Academic Counselor on X-axis (Top {len(top_owners)})")
    )
    st.altair_chart(chart_owner_x, use_container_width=True)

    # Owner table for currently selected inactivity range (actionable)
    st.markdown("#### Owners in selected inactivity range")
    owner_range = (
        d_work.loc[range_mask, "_owner"]
             .fillna("Unknown").astype(str)
             .value_counts().reset_index()
    )
    owner_range.columns = ["Academic Counselor", "Count"]
    owner_range["Share %"] = (owner_range["Count"] / max(int(range_mask.sum()), 1) * 100).round(1)
    st.dataframe(owner_range, use_container_width=True)

    st.download_button(
        "Download CSV — Owners (selected inactivity range)",
        data=owner_range.to_csv(index=False).encode("utf-8"),
        file_name="lead_movement_owners_selected_range.csv",
        mime="text/csv",
        key="lm_owner_dl"
    )

    with st.expander("Show matching rows by owner (first 1000)"):
        cols_show_owner = []
        for c in [create_col, ref_col, dealstage_col, country_col, source_col, deal_owner_raw, acad_couns_raw]:
            if c and c in d_work.columns:
                cols_show_owner.append(c)
        preview_owner = d_work.loc[range_mask, cols_show_owner].head(1000)
        st.dataframe(preview_owner, use_container_width=True)



elif view == "AC Wise Detail":
    st.subheader("AC Wise Detail – Create-date scoped counts & % conversions")

    # ---- Required cols & special columns
    referral_intent_col = find_col(df, ["Referral Intent Source", "Referral intent source"])
    if not create_col or not counsellor_col:
        st.error("Missing required columns (Create Date and Academic Counsellor).")
        st.stop()

    # ---- Date scope (population by Create Date) & Counting mode
    st.markdown("**Date scope (based on Create Date) & Counting mode**")
    c1, c2 = st.columns(2)
    scope_pick = st.radio(
        "Presets",
        ["Yesterday", "Today", "This month", "Last month", "Custom"],
        index=2, horizontal=True, key="ac_scope"
    )
    if scope_pick == "Yesterday":
        scope_start, scope_end = yday, yday
    elif scope_pick == "Today":
        scope_start, scope_end = today, today
    elif scope_pick == "This month":
        scope_start, scope_end = month_bounds(today)
    elif scope_pick == "Last month":
        scope_start, scope_end = last_month_bounds(today)
    else:
        with c1:
            scope_start = st.date_input("Start (Create Date)", value=today.replace(day=1), key="ac_cstart")
        with c2:
            scope_end = st.date_input("End (Create Date)", value=month_bounds(today)[1], key="ac_cend")
        if scope_end < scope_start:
            st.error("End date cannot be before start date.")
            st.stop()

    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ac_mode")
    st.caption(f"Create-date scope: **{scope_start} → {scope_end}** • Mode: **{mode}**")

    # ---- Start from globally filtered df_f, optional Deal Stage filter
    d = df_f.copy()

    if dealstage_col and dealstage_col in d.columns:
        stage_vals = ["All"] + sorted(d[dealstage_col].dropna().astype(str).unique().tolist())
        sel_stages = st.multiselect(
            "Deal Stage (optional filter on population)",
            stage_vals, default=["All"], key="ac_stage"
        )
        if "All" not in sel_stages:
            d = d[d[dealstage_col].astype(str).isin(sel_stages)].copy()
    else:
        st.caption("Deal Stage column not found — stage filter disabled.")

    if d.empty:
        st.info("No rows after filters.")
        st.stop()

    # ---- Normalize helper columns
    d["_ac"] = d[counsellor_col].fillna("Unknown").astype(str)

    _cdate = coerce_datetime(d[create_col]).dt.date
    _first = coerce_datetime(d[first_cal_sched_col]).dt.date if first_cal_sched_col and first_cal_sched_col in d.columns else pd.Series(pd.NaT, index=d.index)
    _resch = coerce_datetime(d[cal_resched_col]).dt.date     if cal_resched_col     and cal_resched_col     in d.columns else pd.Series(pd.NaT, index=d.index)
    _done  = coerce_datetime(d[cal_done_col]).dt.date        if cal_done_col        and cal_done_col        in d.columns else pd.Series(pd.NaT, index=d.index)
    _paid  = coerce_datetime(d[pay_col]).dt.date             if pay_col             and pay_col             in d.columns else pd.Series(pd.NaT, index=d.index)

    # Masks
    pop_mask = _cdate.between(scope_start, scope_end)  # population by Create Date
    m_first = _first.between(scope_start, scope_end) if _first.notna().any() else pd.Series(False, index=d.index)
    m_resch = _resch.between(scope_start, scope_end) if _resch.notna().any() else pd.Series(False, index=d.index)
    m_done  = _done.between(scope_start, scope_end)  if _done.notna().any()  else pd.Series(False, index=d.index)
    m_paid  = _paid.between(scope_start, scope_end)  if _paid.notna().any()  else pd.Series(False, index=d.index)

    # Apply mode to event indicators
    if mode == "MTD":
        ind_create = pop_mask
        ind_first  = pop_mask & m_first
        ind_resch  = pop_mask & m_resch
        ind_done   = pop_mask & m_done
        ind_paid   = pop_mask & m_paid
    else:  # Cohort
        ind_create = pop_mask
        ind_first  = m_first
        ind_resch  = m_resch
        ind_done   = m_done
        ind_paid   = m_paid

    # ---------- Referral Intent Source = "Sales Generated" only ----------
    if referral_intent_col and referral_intent_col in d.columns:
        _ref = d[referral_intent_col].astype(str).str.strip().str.lower()
        sales_generated_mask = (_ref == "sales generated")
    else:
        sales_generated_mask = pd.Series(False, index=d.index)
    ind_ref_sales = pop_mask & sales_generated_mask

    # ---------- Aggregate toggle (All Academic Counsellors) ----------
    st.markdown("#### Display mode")
    show_all_ac = st.checkbox("Aggregate all Academic Counsellors (show totals only)", value=False, key="ac_all_toggle")

    # ---- AC-wise table
    col_label_ref = "Referral Intent Source = Sales Generated — Count"

    base_sub = pd.DataFrame({
        "Academic Counsellor": d["_ac"],
        "Create Date — Count": ind_create.astype(int),
        "First Cal — Count": ind_first.astype(int),
        "Cal Rescheduled — Count": ind_resch.astype(int),
        "Cal Done — Count": ind_done.astype(int),
        "Payment Received — Count": ind_paid.astype(int),
        col_label_ref:           ind_ref_sales.astype(int),
    })

    if show_all_ac:
        agg = (
            base_sub.drop(columns=["Academic Counsellor"])
                    .sum(numeric_only=True)
                    .to_frame().T
        )
        agg.insert(0, "Academic Counsellor", "All ACs (Total)")
    else:
        agg = (
            base_sub.groupby("Academic Counsellor", as_index=False)
                    .sum(numeric_only=True)
                    .sort_values("Create Date — Count", ascending=False)
        )

    st.markdown("### AC-wise counts")
    st.dataframe(agg, use_container_width=True)
    st.download_button(
        "Download CSV — AC-wise counts",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name=f"ac_wise_counts_{'all' if show_all_ac else 'by_ac'}_{mode.lower()}.csv",
        mime="text/csv",
        key="ac_dl_counts"
    )

    # ---- % Conversion between two chosen metrics
    st.markdown("### Conversion % between two metrics")
    metric_labels = [
        "Create Date — Count",
        "First Cal — Count",
        "Cal Rescheduled — Count",
        "Cal Done — Count",
        "Payment Received — Count",
        col_label_ref,
    ]
    c3, c4 = st.columns(2)
    with c3:
        denom_label = st.selectbox("Denominator", metric_labels, index=0, key="ac_pct_denom")
    with c4:
        numer_label = st.selectbox("Numerator",  metric_labels, index=3, key="ac_pct_numer")

    pct_tbl = agg[["Academic Counsellor", denom_label, numer_label]].copy()
    pct_tbl["%"] = np.where(
        pct_tbl[denom_label] > 0,
        (pct_tbl[numer_label] / pct_tbl[denom_label]) * 100.0,
        0.0
    ).round(1)
    pct_tbl = pct_tbl.sort_values("%", ascending=False) if not show_all_ac else pct_tbl

    st.dataframe(pct_tbl, use_container_width=True)
    st.download_button(
        "Download CSV — Conversion %",
        data=pct_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"ac_conversion_percent_{'all' if show_all_ac else 'by_ac'}_{mode.lower()}.csv",
        mime="text/csv",
        key="ac_dl_pct"
    )

    # Overall KPI
    den_sum = int(pct_tbl[denom_label].sum())
    num_sum = int(pct_tbl[numer_label].sum())
    overall_pct = (num_sum / den_sum * 100.0) if den_sum > 0 else 0.0
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Overall {numer_label} / {denom_label} ({mode})</div>"
        f"<div class='kpi-value'>{overall_pct:.1f}%</div>"
        f"<div class='kpi-sub'>Num: {num_sum:,} • Den: {den_sum:,}</div></div>",
        unsafe_allow_html=True
    )

    # ---- Breakdown: AC × (Deal Source or Country)
    st.markdown("### Breakdown")
    grp_mode = st.radio("Group by", ["JetLearn Deal Source", "Country"], index=0, horizontal=True, key="ac_grp_mode")

    have_grp = False
    if grp_mode == "JetLearn Deal Source":
        if not source_col or source_col not in d.columns:
            st.info("Deal Source column not found.")
        else:
            d["_grp"] = d[source_col].fillna("Unknown").astype(str)
            have_grp = True
    else:
        if not country_col or country_col not in d.columns:
            st.info("Country column not found.")
        else:
            d["_grp"] = d[country_col].fillna("Unknown").astype(str)
            have_grp = True

    if have_grp:
        sub2 = pd.DataFrame({
            "Academic Counsellor": d["_ac"],
            "_grp": d["_grp"],
            "Create Date — Count": ind_create.astype(int),
            "First Cal — Count": ind_first.astype(int),
            "Cal Rescheduled — Count": ind_resch.astype(int),
            "Cal Done — Count": ind_done.astype(int),
            "Payment Received — Count": ind_paid.astype(int),
            col_label_ref:           ind_ref_sales.astype(int),
        })

        if show_all_ac:
            gb = (
                sub2.drop(columns=["Academic Counsellor"])
                    .groupby("_grp", as_index=False)
                    .sum(numeric_only=True)
                    .rename(columns={"_grp": grp_mode})
                    .sort_values("Create Date — Count", ascending=False)
            )
        else:
            gb = (
                sub2.groupby(["Academic Counsellor","_grp"], as_index=False)
                    .sum(numeric_only=True)
                    .rename(columns={"_grp": grp_mode})
                    .sort_values(["Academic Counsellor","Create Date — Count"], ascending=[True, False])
            )

        st.dataframe(gb, use_container_width=True)
        st.download_button(
            f"Download CSV — {'Totals × ' if show_all_ac else 'AC × '}{grp_mode} breakdown ({mode})",
            data=gb.to_csv(index=False).encode("utf-8"),
            file_name=f"{'totals' if show_all_ac else 'ac'}_breakdown_by_{'deal_source' if grp_mode.startswith('JetLearn') else 'country'}_{mode.lower()}.csv",
            mime="text/csv",
            key="ac_dl_breakdown"
        )

    # ==== AC × Deal Source — Stacked charts (Payments, Deals Created, and Conversion%) ====
    st.markdown("### AC × Deal Source — Stacked charts (Payments, Deals Created & Conversion %)")

    if (not source_col) or (source_col not in d.columns):
        st.info("Deal Source column not found — cannot draw stacked charts.")
    else:
        _idx = d.index
        ac_series  = (pd.Series("All ACs (Total)", index=_idx) if show_all_ac else d["_ac"])
        src_series = d[source_col].fillna("Unknown").astype(str)

        ind_paid_series   = pd.Series(ind_paid, index=_idx).astype(bool)
        ind_create_series = pd.Series(ind_create, index=_idx).astype(bool)

        # Payments stacked
        df_pay = pd.DataFrame({
            "Academic Counsellor": ac_series,
            "Deal Source": src_series,
            "Count": ind_paid_series.astype(int)
        })
        g_pay = df_pay.groupby(["Academic Counsellor", "Deal Source"], as_index=False)["Count"].sum()
        totals_pay = g_pay.groupby("Academic Counsellor", as_index=False)["Count"].sum().rename(columns={"Count": "Total"})

        # Deals Created stacked
        df_create = pd.DataFrame({
            "Academic Counsellor": ac_series,
            "Deal Source": src_series,
            "Count": ind_create_series.astype(int)
        })
        g_create = df_create.groupby(["Academic Counsellor", "Deal Source"], as_index=False)["Count"].sum()
        totals_create = g_create.groupby("Academic Counsellor", as_index=False)["Count"].sum().rename(columns={"Count": "Total"})

        # --- Options (added Conversion % sort)
        col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
        with col_opt1:
            normalize_pct = st.checkbox(
                "Show Payments/Created as % of AC total (for the first two charts)",
                value=False, key="ac_stack_pct"
            )
        with col_opt2:
            sort_mode = st.selectbox(
                "Sort ACs by",
                ["Payments (desc)", "Deals Created (desc)", "Conversion % (desc)", "A–Z"],
                index=0, key="ac_stack_sort"
            )
        with col_opt3:
            top_n = st.number_input("Max ACs to show", min_value=1, max_value=500, value=30, step=1, key="ac_stack_topn")

        # --- Build AC ordering, including Conversion % option ---
        if sort_mode == "Payments (desc)":
            order_src = totals_pay.copy().sort_values("Total", ascending=False)

        elif sort_mode == "Deals Created (desc)":
            order_src = totals_create.copy().sort_values("Total", ascending=False)

        elif sort_mode == "Conversion % (desc)":
            # AC-level conversion% = (sum Paid) / (sum Created) * 100
            ac_conv = (
                totals_pay.rename(columns={"Total": "Paid"})
                .merge(totals_create.rename(columns={"Total": "Created"}), on="Academic Counsellor", how="outer")
                .fillna({"Paid": 0, "Created": 0})
            )
            ac_conv["ConvPct"] = np.where(ac_conv["Created"] > 0, ac_conv["Paid"] / ac_conv["Created"] * 100.0, 0.0)
            order_src = ac_conv.sort_values("ConvPct", ascending=False)[["Academic Counsellor"]]

        else:  # "A–Z"
            base_totals = totals_pay if not totals_pay.empty else totals_create
            order_src = base_totals[["Academic Counsellor"]].copy().sort_values("Academic Counsellor", ascending=True)

        ac_order = order_src["Academic Counsellor"].head(int(top_n)).tolist() if not order_src.empty else []

        def prep_for_chart(g_df, totals_df):
            g = g_df.merge(totals_df, on="Academic Counsellor", how="left")
            if ac_order:
                g = g[g["Academic Counsellor"].isin(ac_order)].copy()
                g["Academic Counsellor"] = pd.Categorical(g["Academic Counsellor"], categories=ac_order, ordered=True)
            else:
                g["Academic Counsellor"] = g["Academic Counsellor"].astype(str)
            if normalize_pct:
                g["Pct"] = np.where(g["Total"] > 0, g["Count"] / g["Total"] * 100.0, 0.0)
            return g

        g_pay_c    = prep_for_chart(g_pay, totals_pay)
        g_create_c = prep_for_chart(g_create, totals_create)

        def stacked_chart(g, title, use_pct):
            if g.empty:
                return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

            y_field = alt.Y(
                ("Pct:Q" if use_pct else "Count:Q"),
                title=("% of AC total" if use_pct else "Count"),
                stack=True,
                scale=(alt.Scale(domain=[0, 100]) if use_pct else alt.Undefined)
            )
            tooltips = [
                alt.Tooltip("Academic Counsellor:N"),
                alt.Tooltip("Deal Source:N"),
                alt.Tooltip("Count:Q", title="Count"),
                alt.Tooltip("Total:Q", title="AC Total"),
            ]
            if use_pct:
                tooltips.append(alt.Tooltip("Pct:Q", title="% of AC", format=".1f"))

            chart = (
                alt.Chart(g)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Academic Counsellor:N", sort=ac_order, title="Academic Counsellor"),
                    y=y_field,
                    color=alt.Color("Deal Source:N", legend=alt.Legend(orient="bottom", title="Deal Source")),
                    tooltip=tooltips,
                )
                .properties(height=360, title=title)
            )
            return chart

        # ---- Conversion% stacked (Payments / Created within AC × Source)
        g_merge = (
            g_create.rename(columns={"Count": "Created"})
                    .merge(g_pay.rename(columns={"Count": "Paid"}),
                           on=["Academic Counsellor", "Deal Source"], how="outer")
                    .fillna({"Created": 0, "Paid": 0})
        )
        # keep AC order and top_n selection
        if ac_order:
            g_merge = g_merge[g_merge["Academic Counsellor"].isin(ac_order)].copy()
            g_merge["Academic Counsellor"] = pd.Categorical(g_merge["Academic Counsellor"], categories=ac_order, ordered=True)

        g_merge["ConvPct"] = np.where(g_merge["Created"] > 0, g_merge["Paid"] / g_merge["Created"] * 100.0, 0.0)

        def conversion_chart(g):
            if g.empty:
                return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()
            tooltips = [
                alt.Tooltip("Academic Counsellor:N"),
                alt.Tooltip("Deal Source:N"),
                alt.Tooltip("Created:Q"),
                alt.Tooltip("Paid:Q"),
                alt.Tooltip("ConvPct:Q", title="Conversion %", format=".1f"),
            ]
            return (
                alt.Chart(g)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Academic Counsellor:N", sort=ac_order, title="Academic Counsellor"),
                    y=alt.Y("ConvPct:Q", title="Conversion % (Paid / Created)", scale=alt.Scale(domain=[0, 100]), stack=True),
                    color=alt.Color("Deal Source:N", legend=alt.Legend(orient="bottom", title="Deal Source")),
                    tooltip=tooltips,
                )
                .properties(height=360, title="Conversion % — stacked by Deal Source")
            )

        col_pay, col_create, col_conv = st.columns(3)
        with col_pay:
            st.altair_chart(
                stacked_chart(g_pay_c, "Payments (Payment Received — stacked by Deal Source)", use_pct=normalize_pct),
                use_container_width=True
            )
        with col_create:
            st.altair_chart(
                stacked_chart(g_create_c, "Deals Created (Create Date — stacked by Deal Source)", use_pct=normalize_pct),
                use_container_width=True
            )
        with col_conv:
            st.altair_chart(conversion_chart(g_merge), use_container_width=True)

        with st.expander("Download data used in stacked charts"):
            st.download_button(
                "Download CSV — Payments by AC × Deal Source",
                data=g_pay_c.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_payments.csv",
                mime="text/csv",
                key="ac_stack_dl_pay"
            )
            st.download_button(
                "Download CSV — Deals Created by AC × Deal Source",
                data=g_create_c.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_created.csv",
                mime="text/csv",
                key="ac_stack_dl_created"
            )
            st.download_button(
                "Download CSV — Conversion% by AC × Deal Source",
                data=g_merge.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_conversion_pct.csv",
                mime="text/csv",
                key="ac_stack_dl_conv"
            )
elif view == "Dashboard":
    st.subheader("Dashboard – Key Business Snapshot")

    # ---- Resolve core columns (Create / Payment) defensively
    _create = create_col if create_col in df_f.columns else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
    _pay    = pay_col    if pay_col    in df_f.columns else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])

    # Optional calibration columns (will be used in Day-wise Explorer if present)
    _first_cal = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First Calibration Scheduled","Calibration First Scheduled"])
    _resched   = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Calibration Rescheduled"])
    _done      = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Calibration Done"])

    if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
        st.warning("Create/Payment columns are missing. Please map 'Create Date' and 'Payment Received Date' in your sidebar.", icon="⚠️")
    else:
        # ---- Date presets
        preset = st.selectbox(
            "Range",
            ["Today", "Yesterday", "This month", "Last 7 days", "Custom"],
            index=2,
            help="Pick the window for KPIs."
        )

        today_d = date.today()
        if preset == "Today":
            start_d, end_d = today_d, today_d
        elif preset == "Yesterday":
            yd = today_d - timedelta(days=1)
            start_d, end_d = yd, yd
        elif preset == "This month":
            mstart, mend = month_bounds(today_d)
            start_d, end_d = mstart, mend
        elif preset == "Last 7 days":
            start_d, end_d = today_d - timedelta(days=6), today_d
        else:
            # Custom
            d1, d2 = st.date_input(
                "Custom range",
                value=(today_d.replace(day=1), today_d),
                help="Select start and end (inclusive)."
            )
            if isinstance(d1, tuple) or isinstance(d2, tuple):  # safety
                d1, d2 = d1[0], d2[0]
            start_d, end_d = (min(d1, d2), max(d1, d2))

        # ---- Mode toggle (default Cohort)
        mode = st.radio(
            "Mode",
            ["Cohort (Payment Month)", "MTD (Created Cohort)"],
            horizontal=True,
            index=0,
            help=(
                "Cohort: numerator = payments in window (Create can be any time); "
                "MTD: numerator = payments in window AND created in window."
            )
        )

        # ---- Normalize timestamps (for KPI cards)
        c_ts = coerce_datetime(df_f[_create]).dt.date
        p_ts = coerce_datetime(df_f[_pay]).dt.date

        # Inclusive window mask helpers
        def between_date(s, a, b):  # s is date series
            return s.notna() & (s >= a) & (s <= b)

        # Denominator: deals created in window (common to both modes)
        mask_created_in_win = between_date(c_ts, start_d, end_d)
        denom_created = int(mask_created_in_win.sum())

        # Numerator logic for KPI cards
        if mode.startswith("MTD"):
            # Payments in window AND created in window
            num_mask = mask_created_in_win & between_date(p_ts, start_d, end_d)
        else:
            # Cohort: payments in window (ignore Create Date)
            num_mask = between_date(p_ts, start_d, end_d)

        numerator_payments = int(num_mask.sum())
        conv_pct = (numerator_payments / denom_created * 100.0) if denom_created > 0 else np.nan

        # ---- KPIs (kept intact)
        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 14px; background: #ffffff; }
              .kpi-title { font-size: 0.85rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.6rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div>"
                f"<div class='kpi-value'>{denom_created:,}</div>"
                f"<div class='kpi-sub'>{start_d.isoformat()} → {end_d.isoformat()}</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div>"
                f"<div class='kpi-value'>{numerator_payments:,}</div>"
                f"<div class='kpi-sub'>Mode: {mode}</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            conv_txt = "–" if np.isnan(conv_pct) else f"{conv_pct:.1f}%"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Conversion%</div>"
                f"<div class='kpi-value'>{conv_txt}</div>"
                f"<div class='kpi-sub'>Numerator/Denominator per selected Mode</div></div>",
                unsafe_allow_html=True,
            )

        # ---- Optional detail tables (kept intact)
        with st.expander("Breakout by Deal Source"):
            src_col = source_col if source_col in df_f.columns else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
            if not src_col or src_col not in df_f.columns:
                st.info("Deal Source column not found.")
            else:
                view_cols = [src_col]
                # Created in window
                created_by_src = (
                    df_f.loc[mask_created_in_win, view_cols]
                    .assign(_ones=1)
                    .groupby(src_col, dropna=False)["_ones"].sum()
                    .rename("DealsCreated")
                    .reset_index()
                )
                # Payments per mode
                paid_by_src = (
                    df_f.loc[num_mask, view_cols]
                    .assign(_ones=1)
                    .groupby(src_col, dropna=False)["_ones"].sum()
                    .rename("Enrolments")
                    .reset_index()
                )
                out = created_by_src.merge(paid_by_src, on=src_col, how="outer").fillna(0)
                out["Conversion%"] = np.where(out["DealsCreated"] > 0, (out["Enrolments"] / out["DealsCreated"]) * 100.0, np.nan)
                out = out.sort_values("Enrolments", ascending=False)
                st.dataframe(out, use_container_width=True)

        with st.expander("Breakout by Country"):
            ctry_col = country_col if country_col in df_f.columns else find_col(df_f, ["Country","Student Country","Deal Country"])
            if not ctry_col or ctry_col not in df_f.columns:
                st.info("Country column not found.")
            else:
                view_cols = [ctry_col]
                created_by_ctry = (
                    df_f.loc[mask_created_in_win, view_cols]
                    .assign(_ones=1)
                    .groupby(ctry_col, dropna=False)["_ones"].sum()
                    .rename("DealsCreated")
                    .reset_index()
                )
                paid_by_ctry = (
                    df_f.loc[num_mask, view_cols]
                    .assign(_ones=1)
                    .groupby(ctry_col, dropna=False)["_ones"].sum()
                    .rename("Enrolments")
                    .reset_index()
                )
                outc = created_by_ctry.merge(paid_by_ctry, on=ctry_col, how="outer").fillna(0)
                outc["Conversion%"] = np.where(outc["DealsCreated"] > 0, (outc["Enrolments"] / outc["DealsCreated"]) * 100.0, np.nan)
                outc = outc.sort_values("Enrolments", ascending=False)
                st.dataframe(outc, use_container_width=True)

        # =====================================================================
        # Day-wise Explorer (ADDED) — multiple metrics + group + chart options
        # =====================================================================
        st.markdown("### Day-wise Explorer")

        # Controls
        metric_options = ["Deals Created", "Enrolments (Payments)"]
        # add calibration metrics only if we have their columns
        if _first_cal: metric_options.append("First Calibration Scheduled")
        if _resched:   metric_options.append("Calibration Rescheduled")
        if _done:      metric_options.append("Calibration Done")

        metrics_picked = st.multiselect(
            "Metrics (select one or more)",
            options=metric_options,
            default=[m for m in metric_options[:2]],  # default first two
            help="You can plot multiple metrics; each will render as its own chart and totals."
        )

        group_by = st.selectbox(
            "Group (optional)",
            ["(None)", "JetLearn Deal Source", "Country", "Referral Intent Source"],
            index=0,
            help="Stack or breakout by a category (optional)."
        )
        chart_type = st.selectbox(
            "Chart type",
            ["Line", "Bars", "Stacked Bars"],
            index=0
        )

        # Resolve grouping column
        grp_col = None
        if group_by == "JetLearn Deal Source":
            grp_col = source_col if source_col in df_f.columns else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        elif group_by == "Country":
            grp_col = country_col if country_col in df_f.columns else find_col(df_f, ["Country","Student Country","Deal Country"])
        elif group_by == "Referral Intent Source":
            grp_col = find_col(df_f, ["Referral Intent Source","Referral intent source"])

        # Helpers to get the event-date series for a metric
        def metric_date_series(df_in: pd.DataFrame, metric_name: str) -> pd.Series:
            if metric_name == "Deals Created":
                return coerce_datetime(df_in[_create]).dt.date
            if metric_name == "Enrolments (Payments)":
                return coerce_datetime(df_in[_pay]).dt.date
            if metric_name == "First Calibration Scheduled" and _first_cal:
                return coerce_datetime(df_in[_first_cal]).dt.date
            if metric_name == "Calibration Rescheduled" and _resched:
                return coerce_datetime(df_in[_resched]).dt.date
            if metric_name == "Calibration Done" and _done:
                return coerce_datetime(df_in[_done]).dt.date
            # fallback: all NaT
            return pd.Series(pd.NaT, index=df_in.index)

        # Build & render per-metric charts
        for metric_name in metrics_picked:
            # Base frame with Create/Pay already normalized
            base = df_f.copy()
            base["_cdate"] = c_ts
            m_dates = metric_date_series(base, metric_name)
            base["_mdate"] = m_dates

            # Window masks per mode:
            # - MTD: count rows where CreateDate in window AND metric-date in window
            # - Cohort: count rows where metric-date in window (Create may be anywhere)
            m_in = between_date(base["_mdate"], start_d, end_d)
            if mode.startswith("MTD"):
                use_mask = between_date(base["_cdate"], start_d, end_d) & m_in
            else:
                use_mask = m_in

            df_metric = base.loc[use_mask].copy()
            if df_metric.empty:
                st.info(f"No rows in range for **{metric_name}**.")
                continue

            # Day column
            df_metric["_day"] = df_metric["_mdate"]

            # Aggregate
            if grp_col and grp_col in df_metric.columns:
                df_metric["_grp"] = df_metric[grp_col].fillna("Unknown").astype(str).str.strip()
                g = (
                    df_metric.groupby(["_day","_grp"], dropna=False)
                             .size().rename("Count").reset_index()
                             .rename(columns={"_day":"Date","_grp":"Group"})
                             .sort_values(["Date","Count"], ascending=[True,False])
                )
            else:
                g = (
                    df_metric.groupby(["_day"], dropna=False)
                             .size().rename("Count").reset_index()
                             .rename(columns={"_day":"Date"})
                             .sort_values(["Date"], ascending=True)
                )

            # Chart for this metric
            st.markdown(f"#### {metric_name}")
            g["Date"] = pd.to_datetime(g["Date"])
            if chart_type == "Line":
                if "Group" in g.columns:
                    ch = (
                        alt.Chart(g)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Count:Q", title=metric_name),
                            color=alt.Color("Group:N", title="Group"),
                            tooltip=["Date:T","Group:N","Count:Q"]
                        )
                        .properties(height=360, title=f"{metric_name} — Day-wise")
                    )
                else:
                    ch = (
                        alt.Chart(g)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Count:Q", title=metric_name),
                            tooltip=["Date:T","Count:Q"]
                        )
                        .properties(height=360, title=f"{metric_name} — Day-wise")
                    )
            elif chart_type == "Bars":
                if "Group" in g.columns:
                    ch = (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Count:Q", title=metric_name),
                            color=alt.Color("Group:N", title="Group"),
                            column=alt.Column("Group:N", title=""),
                            tooltip=["Date:T","Group:N","Count:Q"]
                        )
                        .properties(height=280, title=f"{metric_name} — Day-wise (bars)")
                    )
                else:
                    ch = (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Count:Q", title=metric_name),
                            tooltip=["Date:T","Count:Q"]
                        )
                        .properties(height=360, title=f"{metric_name} — Day-wise (bars)")
                    )
            else:  # Stacked Bars
                if "Group" not in g.columns:
                    st.info("Pick a Group to enable stacked bars.")
                    ch = (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("Count:Q", title=metric_name),
                            tooltip=["Date:T","Count:Q"]
                        )
                        .properties(height=360, title=f"{metric_name} — Day-wise")
                    )
                else:
                    ch = (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y("sum(Count):Q", title=metric_name),
                            color=alt.Color("Group:N", title="Group"),
                            tooltip=["Date:T","Group:N","Count:Q"]
                        )
                        .properties(height=360, title=f"{metric_name} — Day-wise (stacked)")
                    )

            st.altair_chart(ch, use_container_width=True)

            # === Totals for this metric (same section as before, but per metric) ===
            st.markdown("##### Totals")
            total_cnt = int(g["Count"].sum())
            unique_days = g["Date"].dt.date.nunique() if "Date" in g.columns else 0
            avg_per_day = (total_cnt / unique_days) if unique_days > 0 else 0

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:6px 0;">
                  <div class='kpi-card'>
                    <div class='kpi-title'>Total {metric_name}</div>
                    <div class='kpi-value'>{total_cnt:,}</div>
                    <div class='kpi-sub'>{start_d} → {end_d} • Mode: {mode}</div>
                  </div>
                  <div class='kpi-card'>
                    <div class='kpi-title'>Avg per day</div>
                    <div class='kpi-value'>{avg_per_day:.2f}</div>
                    <div class='kpi-sub'>{unique_days} day(s) in range</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "Group" in g.columns:
                grp_tot = (
                    g.groupby("Group", dropna=False)["Count"]
                     .sum()
                     .reset_index()
                     .sort_values("Count", ascending=False)
                )
                grp_tot["Share %"] = np.where(
                    total_cnt > 0, grp_tot["Count"] / total_cnt * 100.0, 0.0
                )
                st.dataframe(grp_tot, use_container_width=True)
                st.download_button(
                    f"Download CSV — Group totals ({metric_name})",
                    data=grp_tot.to_csv(index=False).encode("utf-8"),
                    file_name=f"dashboard_daywise_group_totals_{metric_name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                    key=f"dash_daywise_group_totals_dl_{metric_name}",
                )

            with st.expander(f"Show / Download day-wise data — {metric_name}"):
                st.dataframe(g, use_container_width=True)
                st.download_button(
                    f"Download CSV — Day-wise data ({metric_name})",
                    data=g.to_csv(index=False).encode("utf-8"),
                    file_name=f"dashboard_daywise_data_{metric_name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                    key=f"dash_daywise_dl_{metric_name}"
                )

elif view == "Predictibility":
    import pandas as pd, numpy as np
    from datetime import date
    from calendar import monthrange
    import altair as alt

    st.subheader("Predictibility – Running Month Enrolment Forecast (row counts)")

    # ---------- Resolve columns (Create / Payment / Source) ----------
    def _pick(df, preferred, cands):
        if preferred and preferred in df.columns: return preferred
        for c in cands:
            if c in df.columns: return c
        return None

    _create = _pick(df_f, globals().get("create_col"),
                    ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _pick(df_f, globals().get("pay_col"),
                    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _src    = _pick(df_f, globals().get("source_col"),
                    ["JetLearn Deal Source","Deal Source","Source","_src_raw","Lead Source"])

    if not _create or not _pay:
        st.warning("Predictibility needs 'Create Date' and 'Payment Received Date' columns. Please map them.", icon="⚠️")
    else:
        # ---------- Controls ----------
        c1, c2 = st.columns(2)
        with c1:
            lookback = st.selectbox("Lookback months (exclude current)", [3, 6, 12], index=0)
        with c2:
            weighted = st.checkbox("Recency-weighted learning", value=True, help="Weights recent months higher when estimating daily averages.")

        # ---------- Prep dataframe ----------
        dfp = df_f.copy()
        # If your data is MM/DD/YYYY set dayfirst=False
        dfp["_C"] = pd.to_datetime(dfp[_create], errors="coerce", dayfirst=True)
        dfp["_P"] = pd.to_datetime(dfp[_pay],    errors="coerce", dayfirst=True)
        dfp["_SRC"] = (dfp[_src].fillna("Unknown").astype(str)) if _src else "All"

        # ---------- Current month window ----------
        today_d = date.today()
        mstart  = date(today_d.year, today_d.month, 1)
        mlen    = monthrange(today_d.year, today_d.month)[1]
        mend    = date(today_d.year, today_d.month, mlen)
        days_elapsed = (today_d - mstart).days + 1
        days_left    = max(0, mlen - days_elapsed)

        # =========================================================
        # A = sum of per-day payment counts from 1st → today (ROW COUNT)
        # =========================================================
        mask_pay_cur_mtd = dfp["_P"].dt.date.between(mstart, today_d)
        daily_counts = (
            dfp.loc[mask_pay_cur_mtd, "_P"].dt.date
               .value_counts()
               .sort_index()
        )
        A = int(daily_counts.sum())

        # =========================================================
        # Learn historical DAILY averages (row counts) for SAME vs PREV
        #   SAME: payments whose CreateMonth == PayMonth
        #   PREV: payments whose CreateMonth <  PayMonth
        # Over last K full months (exclude current). Optionally recency-weighted.
        # =========================================================
        cur_per  = pd.Period(today_d, freq="M")
        months   = [cur_per - i for i in range(1, lookback+1)]
        hist_rows = []
        C_per_series = dfp["_C"].dt.to_period("M")

        for per in months:
            ms = date(per.year, per.month, 1)
            ml = monthrange(per.year, per.month)[1]
            me = date(per.year, per.month, ml)

            pay_mask = dfp["_P"].dt.date.between(ms, me)
            if not pay_mask.any():
                hist_rows.append({"per": per, "days": ml, "same": 0, "prev": 0})
                continue

            same_rows = int((pay_mask & (C_per_series == per)).sum())
            prev_rows = int((pay_mask & (C_per_series <  per)).sum())
            hist_rows.append({"per": per, "days": ml, "same": same_rows, "prev": prev_rows})

        hist = pd.DataFrame(hist_rows)

        if hist.empty:
            daily_same = 0.0
            daily_prev = 0.0
        else:
            if weighted:
                hist = hist.sort_values("per")
                hist["w"] = np.arange(1, len(hist)+1)           # 1..K (newest gets highest weight)
                w_days = (hist["days"] * hist["w"]).sum()
                w_same = (hist["same"] * hist["w"]).sum()
                w_prev = (hist["prev"] * hist["w"]).sum()
            else:
                w_days = hist["days"].sum()
                w_same = hist["same"].sum()
                w_prev = hist["prev"].sum()

            daily_same = (w_same / w_days) if w_days > 0 else 0.0
            daily_prev = (w_prev / w_days) if w_days > 0 else 0.0

        # =========================================================
        # Forecast remaining (row counts)
        #   B = daily_same * days_left
        #   C = daily_prev * days_left
        # =========================================================
        B = float(daily_same * days_left)
        C = float(daily_prev * days_left)
        Projected_Total = float(A + B + C)

        # ---------- KPIs ----------
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("A · Actual to date (row count)", f"{A:,}", help=f"Payments between {mstart.isoformat()} and {today_d.isoformat()}")
        k2.metric("B · Remaining (same-month creates)", f"{B:.1f}", help=f"Expected in remaining {days_left} day(s) from deals created this month")
        k3.metric("C · Remaining (prev-months creates)", f"{C:.1f}", help="Expected in remaining days from deals created before this month")
        k4.metric("Projected Month-End", f"{Projected_Total:.1f}", help="A + B + C")

        st.caption(
            f"Month **{mstart.strftime('%b %Y')}** • Days elapsed **{days_elapsed}/{mlen}** • "
            f"Hist daily (same, prev): **{daily_same:.2f}**, **{daily_prev:.2f}** "
            f"(lookback={lookback}{', weighted' if weighted else ''})"
        )

        # ---------- Optional: Per-source breakdown (row counts) ----------
        a_by_src = (
            dfp.loc[mask_pay_cur_mtd, ["_SRC"]]
               .assign(_ones=1)
               .groupby("_SRC")["_ones"].sum()
               .rename("A_Actual_ToDate")
               .reset_index()
        )

        def _hist_dist(component: str):
            parts = []
            for per in months:
                ms = date(per.year, per.month, 1)
                ml = monthrange(per.year, per.month)[1]
                me = date(per.year, per.month, ml)

                pay_mask = dfp["_P"].dt.date.between(ms, me)
                if not pay_mask.any(): 
                    continue

                if component == "same":
                    subset_idx = pay_mask & (C_per_series == per)
                else:
                    subset_idx = pay_mask & (C_per_series <  per)

                if not subset_idx.any(): 
                    continue

                grp = dfp.loc[subset_idx].groupby("_SRC").size().rename("cnt").reset_index()
                grp["per"] = per
                parts.append(grp)

            if not parts:
                return pd.DataFrame(columns=["_SRC","cnt"])

            dist = pd.concat(parts, ignore_index=True)
            if weighted and "per" in dist.columns:
                per_to_w = {p: (i+1) for i, p in enumerate(sorted(months))}
                dist["w"] = dist["per"].map(per_to_w).fillna(1)
                dist["wcnt"] = dist["cnt"] * dist["w"]
                out = dist.groupby("_SRC")["wcnt"].sum().rename("cnt").reset_index()
            else:
                out = dist.groupby("_SRC")["cnt"].sum().reset_index()
            return out

        same_dist = _hist_dist("same")
        prev_dist = _hist_dist("prev")

        all_srcs = sorted(set(a_by_src["_SRC"]).union(set(same_dist["_SRC"])).union(set(prev_dist["_SRC"])) or {"All"})
        out = pd.DataFrame({"Source": all_srcs})
        out = out.merge(a_by_src.rename(columns={"_SRC":"Source"}), on="Source", how="left").fillna({"A_Actual_ToDate":0})

        def _alloc(total, dist_df, fallback_series):
            if dist_df.empty:
                weights = fallback_series.copy()
            else:
                weights = dist_df.set_index("_SRC")["cnt"].reindex(all_srcs).fillna(0.0)
            if (weights > 0).any():
                w = weights / weights.sum()
            else:
                if (fallback_series > 0).any():
                    w = fallback_series / fallback_series.sum()
                else:
                    w = pd.Series(1.0/len(all_srcs), index=all_srcs)
            return (total * w).reindex(all_srcs).values

        fallback = out.set_index("Source")["A_Actual_ToDate"].astype(float)
        out["B_Remaining_SameMonth"]    = _alloc(B,  same_dist, fallback)
        out["C_Remaining_PrevMonths"]   = _alloc(C,  prev_dist, fallback)
        out["Projected_MonthEnd_Total"] = out["A_Actual_ToDate"] + out["B_Remaining_SameMonth"] + out["C_Remaining_PrevMonths"]
        out = out.sort_values("Projected_MonthEnd_Total", ascending=False)

        # Chart
        chart_df = out.melt(id_vars=["Source"],
                            value_vars=["A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths"],
                            var_name="Component", value_name="Count")
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Source:N", sort="-y"),
                y=alt.Y("Count:Q"),
                color=alt.Color("Component:N"),
                tooltip=["Source","Component","Count"]
            ).properties(height=340),
            use_container_width=True
        )

        # Table + download
        with st.expander("Detailed table (by source)"):
            show_cols = ["Source","A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths","Projected_MonthEnd_Total"]
            tbl = out[show_cols].copy()
            for c in show_cols[1:]:
                tbl[c] = tbl[c].astype(float).round(3)
            st.dataframe(tbl, use_container_width=True)
            st.download_button("Download CSV", tbl.to_csv(index=False).encode("utf-8"),
                               file_name="predictibility_by_source.csv", mime="text/csv")

        # (Optional) quick sanity
        with st.expander("Sanity checks"):
            st.write({
                "Create col": _create, "Payment col": _pay, "Source col": _src or "All",
                "A_rows_mtd": A, "days_left": days_left,
                "daily_same_hist": round(daily_same, 3), "daily_prev_hist": round(daily_prev, 3),
                "lookback": lookback, "weighted": weighted,
            })

        # =========================================================
        # ADD-ON: Accuracy % — Prediction vs Actual (MTD, created-to-date population)
        # =========================================================
        st.markdown("### Accuracy % — Prediction vs Actual (Current Month-to-Date)")

        # Helper masks
        def _between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        # Population: deals created THIS MONTH up to today
        pop_mask = _between_date(dfp["_C"].dt.date, mstart, today_d)
        # Actual positive: those in population that PAID in MTD
        actual_pos = pop_mask & _between_date(dfp["_P"].dt.date, mstart, today_d)

        if pop_mask.sum() == 0:
            st.info("No deals created in the current month-to-date — cannot compute Accuracy%.")
        else:
            cols = dfp.columns.tolist()

            # Try to auto-detect probability-like columns (0..1)
            prob_candidates = []
            for c in cols:
                s = pd.to_numeric(dfp[c], errors="coerce")
                if s.notna().sum() >= 10 and (s.between(0, 1).mean() > 0.8):
                    prob_candidates.append(c)

            # Try to auto-detect label-like columns
            def _is_binary_label(series: pd.Series) -> bool:
                v = series.dropna().astype(str).str.strip().str.lower().unique().tolist()
                v = [x for x in v if x != ""]
                ok = {"yes","no","true","false","1","0","paid","not paid","enrolled","not enrolled","positive","negative"}
                return len(v) <= 6 and any(x in ok for x in v)

            label_candidates = [c for c in cols if _is_binary_label(dfp[c])]

            with st.expander("Prediction input (choose column & threshold)"):
                pred_type = st.radio("Prediction type", ["Probability (0–1)", "Label (Yes/No)"],
                                     index=0 if prob_candidates else 1, horizontal=True, key="pred_mtd_type")

                if pred_type.startswith("Probability"):
                    prob_col = st.selectbox("Probability column", options=(prob_candidates or cols),
                                            index=0 if prob_candidates else 0,
                                            help="Numeric 0..1 scores; values coerced to [0,1].")
                    thresh = st.slider("Decision threshold (≥)", 0.0, 1.0, 0.5, 0.01)
                    prob = pd.to_numeric(dfp[prob_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    pred_pos = prob >= thresh
                else:
                    lbl_col = st.selectbox("Label column (Yes/No)",
                                           options=(label_candidates or cols),
                                           index=0 if label_candidates else 0)
                    s = dfp[lbl_col].astype(str).str.strip().str.lower()
                    pred_pos = s.isin(["yes","true","1","paid","enrolled","positive"])

            # Restrict to population rows
            y_true = actual_pos[pop_mask]
            y_pred = pred_pos[pop_mask]

            TP = int((y_true & y_pred).sum())
            TN = int((~y_true & ~y_pred).sum())
            FP = int((~y_true & y_pred).sum())
            FN = int((y_true & ~y_pred).sum())
            N  = TP + TN + FP + FN

            if N == 0:
                st.info("No evaluable rows in current MTD population for accuracy.")
            else:
                acc = (TP + TN) / N * 100.0

                # KPI + Confusion matrix
                st.markdown(
                    """
                    <style>
                      .kpi-card { border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#ffffff; }
                      .kpi-title { font-size:0.85rem; color:#6b7280; margin-bottom:6px; }
                      .kpi-value { font-size:1.6rem; font-weight:700; }
                      .kpi-sub { font-size:0.8rem; color:#6b7280; margin-top:4px; }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                cA, cB = st.columns([1, 1.2])
                with cA:
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-title'>Accuracy%</div>"
                        f"<div class='kpi-value'>{acc:.1f}%</div>"
                        f"<div class='kpi-sub'>Population: Created {mstart} → {today_d} • Actuals: Paid in MTD</div></div>",
                        unsafe_allow_html=True
                    )
                with cB:
                    cm = pd.DataFrame(
                        {
                            "Predicted Positive": [TP, FP],
                            "Predicted Negative": [FN, TN],
                        },
                        index=["Actual Positive", "Actual Negative"],
                    )
                    st.markdown("Confusion matrix")
                    st.dataframe(cm, use_container_width=True)

                # If probability mode, show quick thresholds helper
                if 'prob' in locals():
                    with st.expander("Threshold helper (sample points)"):
                        cuts = [0.3, 0.4, 0.5, 0.6, 0.7]
                        rows = []
                        for t in cuts:
                            yp = (prob >= t)
                            tp = int((actual_pos & yp & pop_mask).sum())
                            tn = int((~actual_pos & ~yp & pop_mask).sum())
                            fp = int((~actual_pos & yp & pop_mask).sum())
                            fn = int((actual_pos & ~yp & pop_mask).sum())
                            n  = tp + tn + fp + fn
                            acc_t = (tp + tn) / n * 100.0 if n else np.nan
                            rows.append({"Thresh": t, "Accuracy%": acc_t, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "N": n})
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================
# Referrals Tab (full)
# =========================
elif view == "Referrals":
    def _referrals_tab():
        st.subheader("Referrals — Holistic View")

        # ---------- Resolve columns ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col in df_f.columns)    else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])
        _src    = source_col if (source_col in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _ris    = find_col(df_f, ["Referral Intent Source","Referral intent source"])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Create/Payment columns are missing. Please map 'Create Date' and 'Payment Received Date' in your sidebar.", icon="⚠️")
            return
        if not _src or _src not in df_f.columns:
            st.warning("Deal Source column is missing (e.g., 'JetLearn Deal Source'). Referrals view needs it.", icon="⚠️")
            return

        # ---------- Controls ----------
        col_top1, col_top2, col_top3 = st.columns([1.2, 1.2, 1])
        with col_top1:
            mode = st.radio(
                "Mode",
                ["MTD", "Cohort"],
                index=1,
                horizontal=True,
                key="ref_mode",
                help=("MTD: enrolments counted only if the deal was also created in the same window. "
                      "Cohort: enrolments count by payment date regardless of create month.")
            )
        with col_top2:
            scope = st.radio(
                "Date scope",
                ["This month", "Last month", "Custom"],
                index=0,
                horizontal=True,
                key="ref_dscope"
            )
        with col_top3:
            mom_trailing = st.selectbox("MoM trailing (months)", [3, 6, 12], index=1, key="ref_momh")

        today_d = date.today()
        if scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            c1, c2 = st.columns(2)
            with c1:
                range_start = st.date_input("Start date", value=today_d.replace(day=1), key="ref_start")
            with c2:
                range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="ref_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                return
        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}**")

        # ---------- Normalize base series ----------
        C = coerce_datetime(df_f[_create]).dt.date
        P = coerce_datetime(df_f[_pay]).dt.date
        SRC = df_f[_src].fillna("Unknown").astype(str)

        # Referral deals by deal source (contains 'referr')
        is_referral_deal = SRC.str.contains("referr", case=False, na=False)

        # Referral Intent Source (Sales Generated presence)
        if _ris and _ris in df_f.columns:
            RIS = df_f[_ris].fillna("Unknown").astype(str).str.strip()
            has_sales_generated = RIS.str.len().gt(0) & (RIS.str.lower() != "unknown")
        else:
            RIS = pd.Series("Unknown", index=df_f.index)
            has_sales_generated = pd.Series(False, index=df_f.index)

        # ---------- Window masks ----------
        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        mask_created_in_win = between_date(C, range_start, range_end)
        mask_paid_in_win    = between_date(P, range_start, range_end)

        if mode == "MTD":
            enrol_ref_mask = mask_created_in_win & mask_paid_in_win & is_referral_deal
            sales_gen_mask = enrol_ref_mask & has_sales_generated
        else:
            enrol_ref_mask = mask_paid_in_win & is_referral_deal
            sales_gen_mask = enrol_ref_mask & has_sales_generated

        # ---------- KPI strip ----------
        k1, k2, k3, k4 = st.columns(4)
        referral_created_cnt = int((mask_created_in_win & is_referral_deal).sum())
        referral_enrol_cnt   = int(enrol_ref_mask.sum())
        referral_sales_gen   = int(sales_gen_mask.sum())
        total_enrol_cnt      = int(mask_paid_in_win.sum()) if mode == "Cohort" else int((mask_paid_in_win & mask_created_in_win).sum())
        pct_ref_of_total     = (referral_enrol_cnt / total_enrol_cnt * 100.0) if total_enrol_cnt > 0 else np.nan

        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
              .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.4rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        with k1:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Referral Created</div>"
                f"<div class='kpi-value'>{referral_created_cnt:,}</div>"
                f"<div class='kpi-sub'>{range_start} → {range_end}</div></div>",
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Referral Enrolments</div>"
                f"<div class='kpi-value'>{referral_enrol_cnt:,}</div>"
                f"<div class='kpi-sub'>Mode: {mode}</div></div>",
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Sales Generated (RIS)</div>"
                f"<div class='kpi-value'>{referral_sales_gen:,}</div>"
                f"<div class='kpi-sub'>Known Referral Intent Source</div></div>",
                unsafe_allow_html=True
            )
        with k4:
            pct_txt = "–" if np.isnan(pct_ref_of_total) else f"{pct_ref_of_total:.1f}%"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>% of Total Conversions</div>"
                f"<div class='kpi-value'>{pct_txt}</div>"
                f"<div class='kpi-sub'>Referral enrolments / All enrolments</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ---------- Day-wise Trends ----------
        st.markdown("### Day-wise Trends (in selected window)")
        chart_mode = st.radio("View as", ["Graph", "Table"], horizontal=True, key="ref_viewmode")

        day_created = (
            pd.DataFrame({"Date": C[mask_created_in_win & is_referral_deal]})
              .groupby("Date").size().rename("Referral Created").reset_index()
            if (mask_created_in_win & is_referral_deal).any()
            else pd.DataFrame(columns=["Date","Referral Created"])
        )
        day_enrol = (
            pd.DataFrame({"Date": P[enrol_ref_mask]})
              .groupby("Date").size().rename("Referral Enrolments").reset_index()
            if enrol_ref_mask.any()
            else pd.DataFrame(columns=["Date","Referral Enrolments"])
        )
        day_join = pd.merge(day_created, day_enrol, on="Date", how="outer").fillna(0)

        if chart_mode == "Graph" and not day_join.empty:
            melt_day = day_join.melt(
                id_vars=["Date"],
                value_vars=["Referral Created","Referral Enrolments"],
                var_name="Metric", value_name="Count"
            )
            ch_day = (
                alt.Chart(melt_day)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]
                )
                .properties(height=320, title="Day-wise Referral Created vs Enrolments")
            )
            st.altair_chart(ch_day, use_container_width=True)
        else:
            st.dataframe(day_join.sort_values("Date"), use_container_width=True)
            st.download_button(
                "Download CSV — Day-wise Referral Created & Enrolments",
                day_join.sort_values("Date").to_csv(index=False).encode("utf-8"),
                "referrals_daywise.csv",
                "text/csv",
                key="ref_dl_day"
            )

        st.markdown("---")

        # ---------- Referral Intent Source split ----------
        st.markdown("### Referral Intent Source — Split (on Referral Enrolments)")
        if _ris and _ris in df_f.columns:
            ris_now = RIS[enrol_ref_mask]
            if ris_now.any():
                ris_tbl = (
                    ris_now.value_counts(dropna=False)
                          .rename_axis("Referral Intent Source")
                          .rename("Count")
                          .reset_index()
                          .sort_values("Count", ascending=False)
                )
                view2 = st.radio("View as", ["Graph", "Table"], horizontal=True, key="ref_viewmode_ris")
                if view2 == "Graph" and not ris_tbl.empty:
                    ch_ris = (
                        alt.Chart(ris_tbl)
                        .mark_bar(opacity=0.9)
                        .encode(
                            x=alt.X("Referral Intent Source:N", sort=ris_tbl["Referral Intent Source"].tolist()),
                            y=alt.Y("Count:Q"),
                            tooltip=[alt.Tooltip("Referral Intent Source:N"), alt.Tooltip("Count:Q")]
                        )
                        .properties(height=360, title="Referral Intent Source split (Enrolments)")
                    )
                    st.altair_chart(ch_ris, use_container_width=True)
                else:
                    st.dataframe(ris_tbl, use_container_width=True)
                    st.download_button(
                        "Download CSV — RIS split (Enrolments)",
                        ris_tbl.to_csv(index=False).encode("utf-8"),
                        "referrals_ris_split.csv",
                        "text/csv",
                        key="ref_dl_ris"
                    )
            else:
                st.info("No referral enrolments in range to split by Referral Intent Source.")
        else:
            st.info("Referral Intent Source column not found.")

        st.markdown("---")

        # ---------- Month-on-Month (Created & Enrolments) ----------
        st.markdown("### Month-on-Month Progress")
        end_month = pd.Period(range_end.replace(day=1), freq="M")
        months = pd.period_range(end=end_month, periods=mom_trailing, freq="M")
        months_list = months.astype(str).tolist()

        C_month = coerce_datetime(df_f[_create]).dt.to_period("M")
        P_month = coerce_datetime(df_f[_pay]).dt.to_period("M")

        mom_created = (
            pd.Series(1, index=df_f.index)
            .where(is_referral_deal & C_month.isin(months), other=np.nan)
            .groupby(C_month).count()
            .reindex(months, fill_value=0)
            .rename("Referral Created")
            .rename_axis("_month").reset_index()
        )
        mom_created["Month"] = mom_created["_month"].astype(str)
        mom_created = mom_created[["Month","Referral Created"]]

        if mode == "MTD":
            mtd_mask_mom = is_referral_deal & P_month.isin(months) & C_month.isin(months) & (P_month == C_month)
            grp = pd.Series(1, index=df_f.index).where(mtd_mask_mom, other=np.nan).groupby(P_month).count()
        else:
            cohort_mask_mom = is_referral_deal & P_month.isin(months)
            grp = pd.Series(1, index=df_f.index).where(cohort_mask_mom, other=np.nan).groupby(P_month).count()

        mom_enrol = grp.reindex(months, fill_value=0).rename("Referral Enrolments").rename_axis("_month").reset_index()
        mom_enrol["Month"] = mom_enrol["_month"].astype(str)
        mom_enrol = mom_enrol[["Month","Referral Enrolments"]]

        mom = mom_created.merge(mom_enrol, on="Month", how="outer").fillna(0)

        choice_mom = st.radio("MoM view as", ["Graph", "Table"], horizontal=True, key="ref_viewmode_mom")
        if choice_mom == "Graph" and not mom.empty:
            melt_mom = mom.melt(id_vars=["Month"], value_vars=["Referral Created","Referral Enrolments"], var_name="Metric", value_name="Count")
            ch_mom = (
                alt.Chart(melt_mom)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Month:N", sort=months_list),
                    y=alt.Y("Count:Q"),
                    color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]
                )
                .properties(height=360, title=f"MoM — Referral Created & Enrolments • Mode: {mode}")
            )
            st.altair_chart(ch_mom, use_container_width=True)
        else:
            st.dataframe(mom.sort_values("Month"), use_container_width=True)
            st.download_button(
                "Download CSV — MoM Created & Enrolments",
                mom.sort_values("Month").to_csv(index=False).encode("utf-8"),
                "referrals_mom_created_enrol.csv",
                "text/csv",
                key="ref_dl_mom"
            )

        # ---------- MoM split by RIS (enrolments) ----------
        st.markdown("#### MoM — Referral Intent Source split (on Enrolments)")
        if _ris and _ris in df_f.columns:
            if mode == "MTD":
                ris_mask_mom = is_referral_deal & P_month.isin(months) & C_month.isin(months) & (C_month == P_month) & has_sales_generated
            else:
                ris_mask_mom = is_referral_deal & P_month.isin(months) & has_sales_generated

            if ris_mask_mom.any():
                ris_mom = pd.DataFrame({
                    "Month": P_month[ris_mask_mom].astype(str),
                    "Referral Intent Source": RIS[ris_mask_mom],
                })
                ris_mom = (ris_mom.groupby(["Month","Referral Intent Source"]).size().rename("Count").reset_index())
                ch_ris_mom = (
                    alt.Chart(ris_mom)
                    .mark_bar(opacity=0.9)
                    .encode(
                        x=alt.X("Month:N", sort=months_list),
                        y=alt.Y("Count:Q"),
                        color=alt.Color("Referral Intent Source:N", legend=alt.Legend(orient="bottom")),
                        tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Referral Intent Source:N"), alt.Tooltip("Count:Q")]
                    )
                    .properties(height=360, title=f"MoM — RIS split (Enrolments) • Mode: {mode}")
                )
                st.altair_chart(ch_ris_mom, use_container_width=True)

                with st.expander("Table — MoM RIS split"):
                    st.dataframe(ris_mom.sort_values(["Month","Count"], ascending=[True, False]), use_container_width=True)
                    st.download_button(
                        "Download CSV — MoM RIS split",
                        ris_mom.to_csv(index=False).encode("utf-8"),
                        "referrals_mom_ris_split.csv",
                        "text/csv",
                        key="ref_dl_mom_ris"
                    )
            else:
                st.info("No referral enrolments with known Referral Intent Source in the MoM window.")
        else:
            st.info("Referral Intent Source column not found.")

        # ---------- MoM Stacked — Sibling Deal ----------
        st.markdown("#### MoM — Sibling Deal split (on Referral Enrolments)")
        sibling_col = find_col(df_f, ["Sibling Deal", "Sibling deal", "Sibling"])
        if not sibling_col or sibling_col not in df_f.columns:
            st.info("‘Sibling Deal’ column not found.", icon="ℹ️")
        else:
            sib_raw = df_f[sibling_col]

            def _norm_sib(x):
                s = str(x).strip().lower()
                if s in {"true","yes","y","1"}:   return "Yes"
                if s in {"false","no","n","0"}:   return "No"
                if s in {"", "nan", "none"}:      return "Unknown"
                return str(x)

            sib_norm = sib_raw.map(_norm_sib).fillna("Unknown").astype(str)

            if mode == "MTD":
                sib_mask = is_referral_deal & P_month.isin(months) & C_month.isin(months) & (C_month == P_month)
            else:
                sib_mask = is_referral_deal & P_month.isin(months)

            if sib_mask.any():
                sib_mom = pd.DataFrame({
                    "Month": P_month[sib_mask].astype(str),
                    "Sibling Deal": sib_norm[sib_mask],
                })
                sib_mom = (
                    sib_mom.groupby(["Month","Sibling Deal"])
                           .size().rename("Count").reset_index()
                           .sort_values(["Month","Count"], ascending=[True, False])
                )
                ch_sib = (
                    alt.Chart(sib_mom)
                    .mark_bar(opacity=0.9)
                    .encode(
                        x=alt.X("Month:N", title="Month", sort=months_list),
                        y=alt.Y("Count:Q", title="Referral Enrolments"),
                        color=alt.Color("Sibling Deal:N", legend=alt.Legend(orient="bottom")),
                        tooltip=[
                            alt.Tooltip("Month:N", title="Month"),
                            alt.Tooltip("Sibling Deal:N"),
                            alt.Tooltip("Count:Q", title="Count"),
                        ],
                    )
                    .properties(height=320, title=f"MoM — Sibling Deal split (Referral Enrolments • Mode: {mode})")
                )
                st.altair_chart(ch_sib, use_container_width=True)

                with st.expander("Table — MoM Sibling Deal split"):
                    st.dataframe(sib_mom, use_container_width=True)
                    st.download_button(
                        "Download CSV — MoM Sibling split",
                        sib_mom.to_csv(index=False).encode("utf-8"),
                        "referrals_mom_sibling_split.csv",
                        "text/csv",
                        key="ref_dl_mom_sibling",
                    )
            else:
                st.info("No referral enrolments found in the selected MoM window to split by ‘Sibling Deal’.")
    # call the tab
    _referrals_tab()
# =========================
# Heatmap Tab (with dynamic Top % option + extra derived ratios)
# =========================
elif view == "Heatmap":
    def _heatmap_tab():
        st.subheader("Heatmap — Interactive Crosstab (MTD / Cohort)")

        # ---------- Resolve key columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col in df_f.columns)    else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])
        _src    = source_col if (source_col in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cty    = country_col if (country_col in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _cns    = counsellor_col if (counsellor_col in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])
        _ris    = find_col(df_f, ["Referral Intent Source","Referral intent source"])
        _sib    = find_col(df_f, ["Sibling Deal","Sibling deal","Sibling"])

        # Calibration columns
        _first_cal = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
        _resched   = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
        _done      = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Cal Done Date","Calibration Completed"])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Create/Payment columns are missing. Please map 'Create Date' and 'Payment Received Date' in your sidebar.", icon="⚠️")
            return

        # ---------- Mode + Date scope ----------
        col_top1, col_top2 = st.columns([1.1, 1.4])
        with col_top1:
            mode = st.radio("Mode", ["MTD", "Cohort"], index=1, horizontal=True, key="hm_mode",
                            help=("MTD: Enrolments / events counted only if the deal was also created in the window. "
                                  "Cohort: Enrolments / events counted by their own date regardless of create month."))
        with col_top2:
            scope = st.radio("Date scope", ["This month", "Last month", "Custom"], index=0, horizontal=True, key="hm_dscope")

        today_d = date.today()
        if scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            d1, d2 = st.columns(2)
            with d1: range_start = st.date_input("Start date", value=today_d.replace(day=1), key="hm_start")
            with d2: range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="hm_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                return
        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}**")

        # ---------- Dimension picker ----------
        dim_options = {}
        if _src: dim_options["JetLearn Deal Source"] = _src
        if _cty: dim_options["Country"] = _cty
        if _cns: dim_options["Academic Counsellor"] = _cns
        if _ris: dim_options["Referral Intent Source"] = _ris
        if _sib: dim_options["Sibling Deal"] = _sib

        if len(dim_options) < 2:
            st.info("Need at least two categorical columns (e.g., Deal Source and Country) to draw a heatmap.")
            return

        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        with c1:
            x_label = st.selectbox("X axis (categories)", list(dim_options.keys()), index=0, key="hm_x")
        with c2:
            y_keys = [k for k in dim_options.keys() if k != x_label]
            y_label = st.selectbox("Y axis (categories)", y_keys, index=0, key="hm_y")
        with c3:
            metric = st.selectbox(
                "Metric",
                [
                    "Deals Created",
                    "Enrolments",
                    "First Calibration Scheduled — Count",
                    "Calibration Rescheduled — Count",
                    "Calibration Done — Count",
                    "Enrolments / Created %",                 # ratio (existing)
                    "Enrolments / Calibration Done %",        # CHANGED (was Cal Done / Enrolments %)
                    "Calibration Done / First Scheduled %",
                    "First Scheduled / Created %",
                ],
                index=1,
                key="hm_metric",
                help="Counts or % ratios per cell, computed with the same MTD/Cohort logic."
            )

        x_col = dim_options[x_label]
        y_col = dim_options[y_label]

        # ---------- Normalize/prepare base series ----------
        C = coerce_datetime(df_f[_create]).dt.date
        P = coerce_datetime(df_f[_pay]).dt.date
        F = coerce_datetime(df_f[_first_cal]).dt.date if _first_cal and _first_cal in df_f.columns else None
        R = coerce_datetime(df_f[_resched]).dt.date   if _resched   and _resched   in df_f.columns else None
        D = coerce_datetime(df_f[_done]).dt.date      if _done      and _done      in df_f.columns else None

        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        mask_created = between_date(C, range_start, range_end)
        mask_paid    = between_date(P, range_start, range_end)

        # Mode-aware masks
        enrol_mask = (mask_created & mask_paid) if mode == "MTD" else mask_paid

        first_mask = None
        if F is not None:
            f_in = between_date(F, range_start, range_end)
            first_mask = (mask_created & f_in) if mode == "MTD" else f_in

        resched_mask = None
        if R is not None:
            r_in = between_date(R, range_start, range_end)
            resched_mask = (mask_created & r_in) if mode == "MTD" else r_in

        done_mask = None
        if D is not None:
            d_in = between_date(D, range_start, range_end)
            done_mask = (mask_created & d_in) if mode == "MTD" else d_in

        def norm_cat(series):
            return series.fillna("Unknown").astype(str).str.strip()

        X = norm_cat(df_f[x_col])
        Y = norm_cat(df_f[y_col])

        # ---------- Filters with "All" for JLS / Counsellor ----------
        x_vals_all = sorted(X.unique().tolist())
        y_vals_all = sorted(Y.unique().tolist())

        def add_all_option(label, values):
            if label in {"JetLearn Deal Source", "Academic Counsellor"}:
                opts = ["All"] + values
                default = ["All"]
                return opts, default
            else:
                return values, values  # others selected by default

        x_options, x_default = add_all_option(x_label, x_vals_all)
        y_options, y_default = add_all_option(y_label, y_vals_all)

        f1, f2, f3 = st.columns([1.4, 1.4, 0.8])
        with f1:
            x_vals_sel = st.multiselect(f"Filter {x_label}", options=x_options, default=x_default, key="hm_xvals")
        with f2:
            y_vals_sel = st.multiselect(f"Filter {y_label}", options=y_options, default=y_default, key="hm_yvals")
        with f3:
            top_n = st.number_input("Top N per axis (0 = all)", min_value=0, max_value=200, value=0, step=1, key="hm_topn",
                                    help="Apply after filters to keep the heatmap readable.")

        if "All" in x_vals_sel:
            x_vals_sel = x_vals_all
        if "All" in y_vals_sel:
            y_vals_sel = y_vals_all

        base_mask = X.isin(x_vals_sel) & Y.isin(y_vals_sel)

        # ---------- Build cell counts ----------
        def _group_count(active_mask, name):
            if active_mask is None or not active_mask.any():
                return pd.DataFrame(columns=["X","Y",name])
            df_tmp = pd.DataFrame({"X": X[base_mask & active_mask], "Y": Y[base_mask & active_mask]})
            if df_tmp.empty:
                return pd.DataFrame(columns=["X","Y",name])
            return (
                df_tmp.assign(_one=1)
                      .groupby(["X","Y"], dropna=False)["_one"].sum()
                      .rename(name)
                      .reset_index()
            )

        created_ct = _group_count(mask_created, "Created")
        enrol_ct   = _group_count(enrol_mask,   "Enrolments")
        first_ct   = _group_count(first_mask,   "First Calibration Scheduled — Count")
        resch_ct   = _group_count(resched_mask, "Calibration Rescheduled — Count")
        done_ct    = _group_count(done_mask,    "Calibration Done — Count")

        # Merge all metrics
        grid = created_ct.merge(enrol_ct, on=["X","Y"], how="outer")
        grid = grid.merge(first_ct, on=["X","Y"], how="outer")
        grid = grid.merge(resch_ct, on=["X","Y"], how="outer")
        grid = grid.merge(done_ct, on=["X","Y"], how="outer")

        # Fill zeros, ints
        for coln in ["Created","Enrolments",
                     "First Calibration Scheduled — Count",
                     "Calibration Rescheduled — Count",
                     "Calibration Done — Count"]:
            if coln not in grid.columns:
                grid[coln] = 0
        grid = grid.fillna(0)
        for coln in ["Created","Enrolments",
                     "First Calibration Scheduled — Count",
                     "Calibration Rescheduled — Count",
                     "Calibration Done — Count"]:
            grid[coln] = grid[coln].astype(int)

        # ----- Derived ratios -----
        # Enrolments / Created %
        grid["Enrolments / Created %"] = np.where(
            grid["Created"] > 0, grid["Enrolments"] / grid["Created"] * 100.0, np.nan
        )
        # CHANGED: Enrolments / Calibration Done %
        grid["Enrolments / Calibration Done %"] = np.where(
            grid["Calibration Done — Count"] > 0, grid["Enrolments"] / grid["Calibration Done — Count"] * 100.0, np.nan
        )
        # Calibration Done / First Scheduled %
        grid["Calibration Done / First Scheduled %"] = np.where(
            grid["First Calibration Scheduled — Count"] > 0,
            grid["Calibration Done — Count"] / grid["First Calibration Scheduled — Count"] * 100.0, np.nan
        )
        # First Scheduled / Created %
        grid["First Scheduled / Created %"] = np.where(
            grid["Created"] > 0,
            grid["First Calibration Scheduled — Count"] / grid["Created"] * 100.0, np.nan
        )

        # ---------- Top-N trimming (optional) ----------
        if top_n and top_n > 0 and not grid.empty:
            by_x = grid.groupby("X")["Enrolments"].sum().sort_values(ascending=False)
            if (by_x == 0).all():
                by_x = grid.groupby("X")["Created"].sum().sort_values(ascending=False)
            top_x = by_x.head(top_n).index.tolist()

            by_y = grid.groupby("Y")["Enrolments"].sum().sort_values(ascending=False)
            if (by_y == 0).all():
                by_y = grid.groupby("Y")["Created"].sum().sort_values(ascending=False)
            top_y = by_y.head(top_n).index.tolist()

            grid = grid[grid["X"].isin(top_x) & grid["Y"].isin(top_y)]

        # ---------- Metric selection ----------
        if metric == "Deals Created":
            val_field = "Created"
        elif metric == "Enrolments":
            val_field = "Enrolments"
        elif metric == "First Calibration Scheduled — Count":
            val_field = "First Calibration Scheduled — Count"
        elif metric == "Calibration Rescheduled — Count":
            val_field = "Calibration Rescheduled — Count"
        elif metric == "Calibration Done — Count":
            val_field = "Calibration Done — Count"
        elif metric == "Enrolments / Calibration Done %":
            val_field = "Enrolments / Calibration Done %"
        elif metric == "Calibration Done / First Scheduled %":
            val_field = "Calibration Done / First Scheduled %"
        elif metric == "First Scheduled / Created %":
            val_field = "First Scheduled / Created %"
        else:
            val_field = "Enrolments / Created %"

        # ---------- NEW: Dynamic Top % subset ----------
        t1, t2 = st.columns([1, 1.6])
        with t1:
            subset_mode = st.radio("Subset", ["All", "Top %"], index=0, horizontal=True, key="hm_subset_mode",
                                   help=("Counts: minimal set of cells reaching ≤ your % of total (cumulative contribution). "
                                         "Ratio: top N% rows by value."))
        with t2:
            top_pct = st.number_input("Enter % threshold", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key="hm_pct",
                                      help="Example: 7.5 = keep cells that make up ~7.5% of the total; for ratios keep top 7.5% rows by value.")

        def _apply_top_percent(df, field, pct):
            if df.empty or pct <= 0:
                return df
            if pct >= 100:
                return df
            if field.endswith("%"):
                # Ratio: keep top N% rows by value
                k = max(1, int(np.ceil((pct / 100.0) * len(df))))
                return df.sort_values(field, ascending=False).head(k)
            # Counts: contribution threshold (cumulative)
            total = df[field].sum()
            if total <= 0:
                return df
            tmp = df.sort_values(field, ascending=False).copy()
            tmp["_cum_share"] = tmp[field].cumsum() / total * 100.0
            out = tmp[tmp["_cum_share"] <= pct].drop(columns="_cum_share")
            if out.empty and not tmp.empty:
                out = tmp.head(1).drop(columns="_cum_share")
            return out

        grid_view = grid.copy()
        if subset_mode == "Top %":
            grid_view = _apply_top_percent(grid_view, val_field, top_pct)

        # ---------- Output view ----------
        view_mode = st.radio("View as", ["Graph", "Table"], horizontal=True, key="hm_viewmode")

        if grid_view.empty:
            st.info("No data for the chosen filters/date range.")
            return

        if view_mode == "Graph":
            if val_field.endswith("%"):
                color_scale = alt.Scale(scheme="blues")
                tooltip_fmt = ".1f"
            else:
                color_scale = alt.Scale(scheme="greens")
                tooltip_fmt = "d"

            ch = (
                alt.Chart(grid_view)
                .mark_rect()
                .encode(
                    x=alt.X("X:N", title=x_label, sort=sorted(grid_view["X"].unique().tolist())),
                    y=alt.Y("Y:N", title=y_label, sort=sorted(grid_view["Y"].unique().tolist())),
                    color=alt.Color(f"{val_field}:Q", scale=color_scale, title=val_field),
                    tooltip=[
                        alt.Tooltip("X:N", title=x_label),
                        alt.Tooltip("Y:N", title=y_label),
                        alt.Tooltip("Created:Q", title="Deals Created", format="d"),
                        alt.Tooltip("Enrolments:Q", title="Enrolments", format="d"),
                        alt.Tooltip("First Calibration Scheduled — Count:Q", title="First Cal Scheduled", format="d"),
                        alt.Tooltip("Calibration Rescheduled — Count:Q", title="Cal Rescheduled", format="d"),
                        alt.Tooltip("Calibration Done — Count:Q", title="Cal Done", format="d"),
                        alt.Tooltip("Enrolments / Created %:Q", title="Enrolments / Created %", format=".1f"),
                        alt.Tooltip("Enrolments / Calibration Done %:Q", title="Enrolments / Cal Done %", format=".1f"),
                        alt.Tooltip("Calibration Done / First Scheduled %:Q", title="Cal Done / First Scheduled %", format=".1f"),
                        alt.Tooltip("First Scheduled / Created %:Q", title="First Scheduled / Created %", format=".1f"),
                    ]
                )
                .properties(
                    height=420,
                    title=f"Heatmap — {x_label} × {y_label} • Metric: {val_field} • Mode: {mode} • Subset: {subset_mode} {'' if subset_mode=='All' else f'({top_pct:.1f}%)'}"
                )
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            show_tbl = grid_view.copy()
            # Round all ratio columns if present
            ratio_cols = [
                "Enrolments / Created %",
                "Enrolments / Calibration Done %",
                "Calibration Done / First Scheduled %",
                "First Scheduled / Created %",
            ]
            for rc in ratio_cols:
                if rc in show_tbl.columns:
                    show_tbl[rc] = show_tbl[rc].round(1)

            st.dataframe(show_tbl.sort_values(["Y","X"]), use_container_width=True)
            st.download_button(
                "Download CSV — Heatmap data",
                show_tbl.to_csv(index=False).encode("utf-8"),
                "heatmap_data.csv", "text/csv",
                key="hm_dl"
            )

        # ---------- Totals / rollups (over the current grid subset) ----------
        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 14px; background: #ffffff; }
              .kpi-title { font-size: 0.85rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.6rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("#### Totals (for displayed cells)")
        cta, ctb, ctc, ctd = st.columns(4)
        with cta:
            tot_created = int(grid_view["Created"].sum())
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Overall Deals Created</div>"
                f"<div class='kpi-value'>{tot_created:,}</div>"
                f"<div class='kpi-sub'>{range_start} → {range_end}</div></div>",
                unsafe_allow_html=True
            )
        with ctb:
            tot_enrol = int(grid_view["Enrolments"].sum())
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Overall Enrolments</div>"
                f"<div class='kpi-value'>{tot_enrol:,}</div>"
                f"<div class='kpi-sub'>Mode: {mode}</div></div>",
                unsafe_allow_html=True
            )
        with ctc:
            tot_first = int(grid_view["First Calibration Scheduled — Count"].sum())
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>First Cal Scheduled (Total)</div>"
                f"<div class='kpi-value'>{tot_first:,}</div>"
                f"<div class='kpi-sub'>{range_start} → {range_end}</div></div>",
                unsafe_allow_html=True
            )
        with ctd:
            tot_done = int(grid_view["Calibration Done — Count"].sum())
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Calibration Done (Total)</div>"
                f"<div class='kpi-value'>{tot_done:,}</div>"
                f"<div class='kpi-sub'>{range_start} → {range_end}</div></div>",
                unsafe_allow_html=True
            )

        # Missing column hint
        missing = []
        if _first_cal is None or _first_cal not in df_f.columns: missing.append("First Calibration Scheduled Date")
        if _resched   is None or _resched   not in df_f.columns: missing.append("Calibration Rescheduled Date")
        if _done      is None or _done      not in df_f.columns: missing.append("Calibration Done Date")
        if missing:
            st.info("Missing columns: " + ", ".join(missing) + ". These counts show as 0.", icon="ℹ️")

    # run the tab
    _heatmap_tab()
# =========================
# Bubble Explorer Tab
# =========================
elif view == "Bubble Explorer":
    def _bubble_explorer():
        st.subheader("Bubble Explorer — Country × Deal Source (MTD / Cohort)")

        # ---------- Resolve key columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col in df_f.columns)    else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])
        _src    = source_col if (source_col in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cty    = country_col if (country_col in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])

        # Calibration columns (optional)
        _first_cal = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
        _resched   = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
        _done      = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Cal Done Date","Calibration Completed"])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Create/Payment columns are missing. Please map 'Create Date' and 'Payment Received Date' in your sidebar.", icon="⚠️")
            st.stop()
        if not _src or _src not in df_f.columns or not _cty or _cty not in df_f.columns:
            st.warning("Need both Country and JetLearn Deal Source columns to build the bubble view.", icon="⚠️")
            st.stop()

        # ---------- Mode + Time window ("seek bar") ----------
        col_top1, col_top2, col_top3 = st.columns([1.0, 1.2, 1.3])
        with col_top1:
            mode = st.radio(
                "Mode",
                ["MTD", "Cohort"],
                index=1,
                horizontal=True,
                help=("MTD: Enrolments/events counted only if the deal was also created in the window. "
                      "Cohort: Enrolments/events counted by their own date regardless of create month.")
            )
        with col_top2:
            time_preset = st.selectbox(
                "Time window",
                ["Last month", "Last 3 months", "Last 12 months", "Custom"],
                index=1,
                help="Quick ranges (inclusive). Use Custom for any date span."
            )
        with col_top3:
            # metric lists now include derived metrics too
            metric_options = [
                "Enrolments",
                "Deals Created",
                "First Calibration Scheduled — Count",
                "Calibration Rescheduled — Count",
                "Calibration Done — Count",
                # ---- derived (NEW) ----
                "Enrolments / Deals Created %",
                "Enrolments / Calibration Done %",
                "Calibration Done / First Scheduled %",
                "First Scheduled / Deals Created %",
            ]
            size_metric = st.selectbox(
                "Bubble size by",
                metric_options,
                index=0,
                help="Determines relative bubble size."
            )

        today_d = date.today()
        if time_preset == "Last month":
            start_d, end_d = last_month_bounds(today_d)
        elif time_preset == "Last 3 months":
            mstart, _ = month_bounds(today_d)
            start_d = (mstart - pd.offsets.MonthBegin(2)).date()
            end_d   = month_bounds(today_d)[1]
        elif time_preset == "Last 12 months":
            mstart, _ = month_bounds(today_d)
            start_d = (mstart - pd.offsets.MonthBegin(11)).date()
            end_d   = month_bounds(today_d)[1]
        else:
            d1, d2 = st.columns(2)
            with d1: start_d = st.date_input("Start date", value=today_d.replace(day=1), key="bx_start")
            with d2: end_d   = st.date_input("End date",   value=month_bounds(today_d)[1], key="bx_end")
            if end_d < start_d:
                st.error("End date cannot be before start date.")
                st.stop()

        st.caption(f"Scope: **{start_d} → {end_d}** • Mode: **{mode}**")

        # ---------- Filters (multi-select with 'All') ----------
        def norm_cat(series):
            return series.fillna("Unknown").astype(str).str.strip()

        X_src = norm_cat(df_f[_src])
        Y_cty = norm_cat(df_f[_cty])

        src_all = sorted(X_src.unique().tolist())
        cty_all = sorted(Y_cty.unique().tolist())

        f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.2, 1.0])
        with f1:
            src_pick = st.multiselect(
                "Filter JetLearn Deal Source",
                options=(["All"] + src_all),
                default=["All"],
                help="Choose one or more. 'All' selects everything."
            )
        with f2:
            cty_pick = st.multiselect(
                "Filter Country",
                options=(["All"] + cty_all),
                default=["All"],
                help="Choose one or more. 'All' selects everything."
            )
        with f3:
            agg_cty = st.toggle(
                "Aggregate selected Countries",
                value=False,
                help="Combine selected countries into a single bubble group."
            )
        with f4:
            agg_src = st.toggle(
                "Aggregate selected Sources",
                value=False,
                help="Combine selected sources into a single bubble group."
            )
        if "All" in src_pick:
            src_pick = src_all
        if "All" in cty_pick:
            cty_pick = cty_all

        # ---------- Normalize dates & masks ----------
        C = coerce_datetime(df_f[_create]).dt.date
        P = coerce_datetime(df_f[_pay]).dt.date
        F = coerce_datetime(df_f[_first_cal]).dt.date if _first_cal and _first_cal in df_f.columns else None
        R = coerce_datetime(df_f[_resched]).dt.date   if _resched   and _resched   in df_f.columns else None
        D = coerce_datetime(df_f[_done]).dt.date      if _done      and _done      in df_f.columns else None

        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        mask_created = between_date(C, start_d, end_d)
        mask_paid    = between_date(P, start_d, end_d)
        enrol_mask   = (mask_created & mask_paid) if mode == "MTD" else mask_paid

        first_mask = None
        if F is not None:
            f_in = between_date(F, start_d, end_d)
            first_mask = (mask_created & f_in) if mode == "MTD" else f_in

        resched_mask = None
        if R is not None:
            r_in = between_date(R, start_d, end_d)
            resched_mask = (mask_created & r_in) if mode == "MTD" else r_in

        done_mask = None
        if D is not None:
            d_in = between_date(D, start_d, end_d)
            done_mask = (mask_created & d_in) if mode == "MTD" else d_in

        base_mask = X_src.isin(src_pick) & Y_cty.isin(cty_pick)

        # ---------- Group keys (with optional aggregation) ----------
        if agg_cty and agg_src:
            gx = pd.Series("Selected Sources", index=df_f.index)
            gy = pd.Series("Selected Countries", index=df_f.index)
        elif agg_cty:
            gx = X_src.copy()
            gy = pd.Series("Selected Countries", index=df_f.index)
        elif agg_src:
            gx = pd.Series("Selected Sources", index=df_f.index)
            gy = Y_cty.copy()
        else:
            gx = X_src.copy()
            gy = Y_cty.copy()

        # ---------- Build counts ----------
        def _group_sum(active_mask, name):
            if active_mask is None or not active_mask.any():
                return pd.DataFrame(columns=["Source","Country",name])
            df_tmp = pd.DataFrame({"Source": gx[base_mask & active_mask], "Country": gy[base_mask & active_mask]})
            if df_tmp.empty:
                return pd.DataFrame(columns=["Source","Country",name])
            return (
                df_tmp.assign(_one=1)
                      .groupby(["Source","Country"], dropna=False)["_one"].sum()
                      .rename(name)
                      .reset_index()
            )

        created_ct = _group_sum(mask_created, "Deals Created")
        enrol_ct   = _group_sum(enrol_mask,   "Enrolments")
        first_ct   = _group_sum(first_mask,   "First Calibration Scheduled — Count")
        resch_ct   = _group_sum(resched_mask, "Calibration Rescheduled — Count")
        done_ct    = _group_sum(done_mask,    "Calibration Done — Count")

        bub = created_ct.merge(enrol_ct, on=["Source","Country"], how="outer")
        bub = bub.merge(first_ct, on=["Source","Country"], how="outer")
        bub = bub.merge(resch_ct, on=["Source","Country"], how="outer")
        bub = bub.merge(done_ct, on=["Source","Country"], how="outer")

        for coln in ["Deals Created","Enrolments","First Calibration Scheduled — Count","Calibration Rescheduled — Count","Calibration Done — Count"]:
            if coln not in bub.columns: bub[coln] = 0
        bub = bub.fillna(0)
        for coln in ["Deals Created","Enrolments","First Calibration Scheduled — Count","Calibration Rescheduled — Count","Calibration Done — Count"]:
            bub[coln] = bub[coln].astype(int)

        # ---------- Derived metrics (NEW) ----------
        bub["Enrolments / Deals Created %"] = np.where(
            bub["Deals Created"] > 0, bub["Enrolments"] / bub["Deals Created"] * 100.0, np.nan
        )
        bub["Enrolments / Calibration Done %"] = np.where(
            bub["Calibration Done — Count"] > 0, bub["Enrolments"] / bub["Calibration Done — Count"] * 100.0, np.nan
        )
        bub["Calibration Done / First Scheduled %"] = np.where(
            bub["First Calibration Scheduled — Count"] > 0, bub["Calibration Done — Count"] / bub["First Calibration Scheduled — Count"] * 100.0, np.nan
        )
        bub["First Scheduled / Deals Created %"] = np.where(
            bub["Deals Created"] > 0, bub["First Calibration Scheduled — Count"] / bub["Deals Created"] * 100.0, np.nan
        )

        # ---------- Chart configuration ----------
        c1, c2, c3 = st.columns([1.0, 1.0, 0.9])
        with c1:
            x_axis = st.selectbox("X axis", ["Country","JetLearn Deal Source"], index=0, help="Pick which dimension goes on X.")
        with c2:
            y_axis = st.selectbox("Y axis", ["JetLearn Deal Source","Country"], index=0 if x_axis=="JetLearn Deal Source" else 1, help="Pick which dimension goes on Y.")
        with c3:
            color_metric = st.selectbox(
                "Bubble color by",
                [
                    "Enrolments",
                    "Deals Created",
                    "First Calibration Scheduled — Count",
                    "Calibration Rescheduled — Count",
                    "Calibration Done — Count",
                    # ---- derived (NEW) ----
                    "Enrolments / Deals Created %",
                    "Enrolments / Calibration Done %",
                    "Calibration Done / First Scheduled %",
                    "First Scheduled / Deals Created %",
                ],
                index=1,
                help="Color encodes another metric for the same bubble."
            )

        # Ensure x/y refer to right columns in `bub`
        if x_axis == "Country":
            bub["X"] = bub["Country"]
            bub["Y"] = bub["Source"]
            x_title, y_title = "Country", "JetLearn Deal Source"
        else:
            bub["X"] = bub["Source"]
            bub["Y"] = bub["Country"]
            x_title, y_title = "JetLearn Deal Source", "Country"

        # ---------- Optional Top N limiter ----------
        tcol1, tcol2 = st.columns([1.0, 0.9])
        with tcol1:
            top_n = st.number_input("Top N bubbles by size metric (0 = all)", min_value=0, max_value=500, value=0, step=1)
        with tcol2:
            view_mode = st.radio("View as", ["Graph", "Table"], index=0, horizontal=True)

        size_field = size_metric
        color_field = color_metric

        # Sort and trim by chosen size metric
        bub_view = bub.copy()
        if top_n and top_n > 0:
            # Sort NaNs last to avoid losing real rows
            bub_view = bub_view.sort_values(size_field, ascending=False, na_position="last").head(top_n)

        if bub_view.empty:
            st.info("No data for the chosen filters/date range.")
            return

        # ---------- Render ----------
        if view_mode == "Graph":
            # scale bubble size for readability
            size_scale = alt.Scale(range=[30, 1500])  # visual range

            chart = (
                alt.Chart(bub_view)
                .mark_circle(opacity=0.7)
                .encode(
                    x=alt.X("X:N", title=x_title, sort=sorted(bub_view["X"].unique().tolist())),
                    y=alt.Y("Y:N", title=y_title, sort=sorted(bub_view["Y"].unique().tolist())),
                    size=alt.Size(f"{size_field}:Q", scale=size_scale, title=f"Size: {size_field}"),
                    color=alt.Color(f"{color_field}:Q", title=f"Color: {color_field}"),
                    tooltip=[
                        alt.Tooltip("Country:N"),
                        alt.Tooltip("Source:N", title="JetLearn Deal Source"),
                        alt.Tooltip("Deals Created:Q"),
                        alt.Tooltip("First Calibration Scheduled — Count:Q"),
                        alt.Tooltip("Calibration Rescheduled — Count:Q"),
                        alt.Tooltip("Calibration Done — Count:Q"),
                        alt.Tooltip("Enrolments:Q"),
                        # derived in tooltip (rounded)
                        alt.Tooltip("Enrolments / Deals Created %:Q", title="Enrolments / Created %", format=".1f"),
                        alt.Tooltip("Enrolments / Calibration Done %:Q", title="Enrolments / Cal Done %", format=".1f"),
                        alt.Tooltip("Calibration Done / First Scheduled %:Q", title="Cal Done / First Sched %", format=".1f"),
                        alt.Tooltip("First Scheduled / Deals Created %:Q", title="First Sched / Created %", format=".1f"),
                    ],
                )
                .properties(height=520, title=f"Bubble view • Size: {size_field} • Color: {color_field} • Mode: {mode}")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            tbl = bub_view.copy()
            for coln in [
                "Enrolments / Deals Created %",
                "Enrolments / Calibration Done %",
                "Calibration Done / First Scheduled %",
                "First Scheduled / Deals Created %",
            ]:
                tbl[coln] = tbl[coln].round(1)
            st.dataframe(tbl.sort_values([size_field, color_field], ascending=[False, False]), use_container_width=True)
            st.download_button(
                "Download CSV — Bubble Explorer data",
                tbl.to_csv(index=False).encode("utf-8"),
                "bubble_explorer_data.csv",
                "text/csv",
                key="bx_dl"
            )

        # ---------- Notes about missing columns ----------
        missing = []
        if _first_cal is None or _first_cal not in df_f.columns: missing.append("First Calibration Scheduled Date")
        if _resched   is None or _resched   not in df_f.columns: missing.append("Calibration Rescheduled Date")
        if _done      is None or _done      not in df_f.columns: missing.append("Calibration Done Date")
        if missing:
            st.info("Missing columns: " + ", ".join(missing) + ". These counts show as 0.", icon="ℹ️")

    # run the tab
    _bubble_explorer()
# =========================
# Deal Decay Tab (full)
# =========================
elif view == "Deal Decay":
    def _deal_decay_tab():
        st.subheader("Deal Decay — Time Between Key Stages (MTD / Cohort)")

        # ---------- Resolve columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, [
            "Create Date","Created Date","Deal Create Date","CreateDate"
        ])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, [
            "Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"
        ])
        _first  = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, [
            "First Calibration Scheduled Date","First Calibration","First Cal Scheduled"
        ])
        _resch  = cal_resched_col if (cal_resched_col in df_f.columns) else find_col(df_f, [
            "Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"
        ])
        _done   = cal_done_col if (cal_done_col in df_f.columns) else find_col(df_f, [
            "Calibration Done Date","Cal Done Date","Calibration Completed"
        ])

        # Optional dimensional filters
        _cty = country_col if (country_col in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _src = source_col  if (source_col  in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cns = counsellor_col if (counsellor_col in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])

        # Need at least Create + one other event to be meaningful
        core_missing = []
        if not _create or _create not in df_f.columns: core_missing.append("Create Date")
        if not _pay or _pay not in df_f.columns:       core_missing.append("Payment Received Date")
        if core_missing:
            st.warning("Missing columns: " + ", ".join(core_missing) + ". Please map them in the sidebar.", icon="⚠️")
            return

        # ---------- Controls ----------
        col_top1, col_top2, col_top3 = st.columns([1.2, 1.2, 1])
        with col_top1:
            mode = st.radio(
                "Mode",
                ["MTD", "Cohort"],
                index=1, horizontal=True, key="dd_mode",
                help=("MTD: keep pairs only if the DEAL WAS CREATED in the date window. "
                      "Cohort: keep pairs if the TO-event is in the window (create can be anywhere).")
            )
        with col_top2:
            scope = st.radio(
                "Date scope",
                ["This month", "Last month", "Custom"],
                index=0, horizontal=True, key="dd_dscope"
            )
        with col_top3:
            mom_trailing = st.selectbox("MoM trailing (months)", [3, 6, 12], index=1, key="dd_momh")

        today_d = date.today()
        if scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            c1, c2 = st.columns(2)
            with c1: range_start = st.date_input("Start date", value=today_d.replace(day=1), key="dd_start")
            with c2: range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="dd_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                return
        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}**")

        # Event picker (FROM → TO)
        events_map = {
            "Deal Created": _create,
            "First Calibration Scheduled": _first,
            "Calibration Rescheduled": _resch,
            "Calibration Done": _done,
            "Enrolment (Payment Received)": _pay,
        }
        # Only show available events
        avail_events = [k for k,v in events_map.items() if v and v in df_f.columns]
        if "Deal Created" not in avail_events:
            # we still allow picking pairs not involving create, but warn
            st.info("‘Deal Created’ not found; you can still compare other event pairs if both columns exist.")
        if len(avail_events) < 2:
            st.warning("Need at least two event columns to compute a decay.", icon="⚠️")
            return

        e1, e2, e3 = st.columns([1.2, 1.2, 1.0])
        with e1:
            from_ev = st.selectbox("From event", avail_events, index=max(0, avail_events.index("Deal Created")) if "Deal Created" in avail_events else 0, key="dd_from")
        with e2:
            to_choices = [e for e in avail_events if e != from_ev]
            to_ev = st.selectbox("To event", to_choices, index=min(1, len(to_choices)-1), key="dd_to")
        with e3:
            out_pref = st.radio("Output", ["Bell curve", "Table"], index=0, horizontal=True, key="dd_outpref")

        # Dimensional filters
        st.markdown("#### Filters")
        f1, f2, f3 = st.columns([1.4, 1.4, 1.4])
        def _opts(series):
            vals = sorted(series.fillna("Unknown").astype(str).unique().tolist())
            return ["All"] + vals, ["All"]

        if _cty:
            cty_series = df_f[_cty].fillna("Unknown").astype(str)
            cty_opts, cty_def = _opts(cty_series)
            with f1:
                pick_cty = st.multiselect("Country", options=cty_opts, default=cty_def, key="dd_cty")
        else:
            pick_cty = None

        if _src:
            src_series = df_f[_src].fillna("Unknown").astype(str)
            src_opts, src_def = _opts(src_series)
            with f2:
                pick_src = st.multiselect("JetLearn Deal Source", options=src_opts, default=src_def, key="dd_src")
        else:
            pick_src = None

        if _cns:
            cns_series = df_f[_cns].fillna("Unknown").astype(str)
            cns_opts, cns_def = _opts(cns_series)
            with f3:
                pick_cns = st.multiselect("Academic Counsellor", options=cns_opts, default=cns_def, key="dd_cns")
        else:
            pick_cns = None

        # ---------- Normalize timestamps safely ----------
        def _to_dt(s):
            # Always return datetime64[ns]; invalid -> NaT
            return coerce_datetime(s)

        df_use = df_f.copy()
        df_use["__Deal Created"]                 = _to_dt(df_use[events_map["Deal Created"]]) if "Deal Created" in events_map and events_map["Deal Created"] else pd.NaT
        if "First Calibration Scheduled" in events_map:
            df_use["__First Calibration Scheduled"] = _to_dt(df_use[events_map["First Calibration Scheduled"]])
        if "Calibration Rescheduled" in events_map:
            df_use["__Calibration Rescheduled"]     = _to_dt(df_use[events_map["Calibration Rescheduled"]])
        if "Calibration Done" in events_map:
            df_use["__Calibration Done"]            = _to_dt(df_use[events_map["Calibration Done"]])
        df_use["__Enrolment (Payment Received)"] = _to_dt(df_use[events_map["Enrolment (Payment Received)"]]) if "Enrolment (Payment Received)" in events_map else pd.NaT

        # Helper masks for date window
        def _between_dt(sdt, a, b):
            # sdt is datetime64[ns] series
            if sdt is None:
                return pd.Series(False, index=df_use.index)
            return sdt.notna() & (sdt.dt.date >= a) & (sdt.dt.date <= b)

        mask_created_in_win = _between_dt(df_use["__Deal Created"] if "__Deal Created" in df_use.columns else None, range_start, range_end)
        mask_to_in_win      = _between_dt(df_use[f"__{to_ev}"], range_start, range_end)

        # Base pair mask: both FROM and TO exist and TO >= FROM
        base_pair = df_use[f"__{from_ev}"].notna() & df_use[f"__{to_ev}"].notna()
        # keep only non-negative differences (TO on/after FROM)
        nonneg = base_pair & (df_use[f"__{to_ev}"] >= df_use[f"__{from_ev}"])

        # Mode logic
        if mode == "MTD":
            # only deals whose Create Date is in the window
            keep_mode = nonneg & mask_created_in_win
        else:
            # Cohort: pairs where the TO-event is in the window
            keep_mode = nonneg & mask_to_in_win

        # Dimensional filters
        if pick_cty is not None:
            if "All" not in pick_cty:
                keep_mode &= df_use[_cty].fillna("Unknown").astype(str).isin(pick_cty)
        if pick_src is not None:
            if "All" not in pick_src:
                keep_mode &= df_use[_src].fillna("Unknown").astype(str).isin(pick_src)
        if pick_cns is not None:
            if "All" not in pick_cns:
                keep_mode &= df_use[_cns].fillna("Unknown").astype(str).isin(pick_cns)

        d_use = df_use.loc[keep_mode, [f"__{from_ev}", f"__{to_ev}"]].copy()
        if d_use.empty:
            st.info("No matching pairs for the selected events/filters/date window.")
            return

        # Compute days (safe for older pandas): ensure datetime, then (to - from).dt.days
        # .dt accessor is valid because columns are datetime64[ns] from coerce_datetime
        d_use["__days"] = (d_use[f"__{to_ev}"] - d_use[f"__{from_ev}"]).dt.days
        # Drop negative/NaN just in case
        d_use = d_use[d_use["__days"].notna() & (d_use["__days"] >= 0)].copy()
        if d_use.empty:
            st.info("All matched rows had invalid or negative durations.")
            return

        # ---------- KPIs ----------
        mu   = float(d_use["__days"].mean())
        med  = float(d_use["__days"].median())
        std  = float(d_use["__days"].std(ddof=1)) if len(d_use) > 1 else 0.0
        p90  = float(d_use["__days"].quantile(0.90))
        mn   = int(d_use["__days"].min())
        mx   = int(d_use["__days"].max())
        nobs = int(len(d_use))

        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
              .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.4rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Mean (μ)</div><div class='kpi-value'>{mu:.1f} d</div><div class='kpi-sub'>n={nobs:,}</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Std Dev (σ)</div><div class='kpi-value'>{std:.1f} d</div><div class='kpi-sub'>min {mn} • max {mx}</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Median</div><div class='kpi-value'>{med:.1f} d</div><div class='kpi-sub'>p90 {p90:.1f} d</div></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Pairs kept</div><div class='kpi-value'>{nobs:,}</div><div class='kpi-sub'>{from_ev} → {to_ev}</div></div>", unsafe_allow_html=True)
        with c5: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Mode</div><div class='kpi-value'>{mode}</div><div class='kpi-sub'>{range_start} → {range_end}</div></div>", unsafe_allow_html=True)
        with c6: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Scope</div><div class='kpi-value'>{scope}</div><div class='kpi-sub'>Date window</div></div>", unsafe_allow_html=True)

        # ---------- Visualization / Table ----------
        if out_pref == "Bell curve":
            # Histogram + (optional) density estimate (Altair transform_density)
            hist = (
                alt.Chart(d_use.rename(columns={"__days":"Days"}))
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X("Days:Q", bin=alt.Bin(maxbins=40), title="Days between events"),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")]
                )
                .properties(height=320, title=f"Distribution of Days — {from_ev} → {to_ev}")
            )

            dens = (
                alt.Chart(d_use.rename(columns={"__days":"Days"}))
                .transform_density(
                    "Days", as_=["Days", "Density"], extent=[max(0, mn), mx], steps=200
                )
                .mark_line()
                .encode(
                    x="Days:Q",
                    y=alt.Y("Density:Q", title="Density"),
                    tooltip=[alt.Tooltip("Days:Q"), alt.Tooltip("Density:Q", format=".3f")]
                )
            )
            st.altair_chart(hist + dens, use_container_width=True)
        else:
            show = d_use.copy()
            show["Days"] = show["__days"].astype(int)
            show = show[["Days"]]
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Download CSV — Durations",
                show.to_csv(index=False).encode("utf-8"),
                "deal_decay_durations.csv", "text/csv",
                key="dd_dl_table"
            )

        # ---------- Month-on-Month trend of average days ----------
        st.markdown("### Month-on-Month — Average Days")
        # To-month series for the kept rows
        kept_to = df_use.loc[keep_mode, f"__{to_ev}"]
        to_month = kept_to.dt.to_period("M")

        end_month = pd.Period(pd.Timestamp(range_end).normalize(), freq="M")
        months = pd.period_range(end=end_month, periods=mom_trailing, freq="M")

        trend = (
            d_use.assign(_m=to_month.loc[d_use.index])  # align index
                .loc[lambda x: x["_m"].isin(months)]
                .groupby("_m")["__days"].mean()
                .reindex(months, fill_value=np.nan)
        )
        # Older pandas: avoid names=...; rename after reset
        trend = trend.reset_index()
        trend.columns = ["Month", "AvgDays"]
        trend["Month"] = trend["Month"].astype(str)
        months_order = trend["Month"].tolist()

        ch_trend = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("Month:N", sort=months_order),
                y=alt.Y("AvgDays:Q", title="Average Days"),
                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("AvgDays:Q", format=".1f")]
            )
            .properties(height=320, title=f"MoM Avg Days — {from_ev} → {to_ev}")
        )
        st.altair_chart(ch_trend, use_container_width=True)

    # run the tab
    _deal_decay_tab()
# =========================
# Sales Tracker (with Day-wise view added)
# =========================
# =========================
# Sales Tracker Tab (added "Monthly" granularity; everything else kept identical)
# =========================
elif view == "Sales Tracker":
    def _sales_tracker_tab():
        st.subheader("Sales Tracker — Counsellor / Source / Country (MTD & Cohort) + Day-wise")

        # ---------- Resolve columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col in df_f.columns)    else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])
        _cns    = counsellor_col if (counsellor_col in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])
        _src    = source_col if (source_col in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cty    = country_col if (country_col in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _ris    = find_col(df_f, ["Referral Intent Source","Referral intent source"])

        _first_cal = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
        _resched   = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
        _done      = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Cal Done Date","Calibration Completed"])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Create/Payment columns are missing. Please map 'Create Date' and 'Payment Received Date' in your sidebar.", icon="⚠️")
            st.stop()
        if not _cns or _cns not in df_f.columns:
            st.warning("Academic Counsellor column not found.", icon="⚠️")
            st.stop()

        # ---------- Controls ----------
        c0, c1, c2, c3 = st.columns([1.1, 1.1, 1.2, 1])
        with c0:
            mode = st.radio(
                "Mode",
                ["Cohort", "MTD"],
                index=0, horizontal=True, key="st_mode",
                help=("Cohort: payments/events counted by their own date regardless of Create. "
                      "MTD: payments/events counted only if also Created in window.")
            )
        with c1:
            scope = st.radio("Date scope", ["Today", "Yesterday", "This month", "Last month", "Custom"], index=2, horizontal=True, key="st_scope")
        with c2:
            # ADDED: "Monthly" keeps everything else intact
            gran = st.radio("Granularity", ["Summary", "Day-wise", "Monthly"], index=0, horizontal=True, key="st_gran")
        with c3:
            chart_type = st.selectbox("Chart", ["Bar", "Line"], index=1, key="st_chart")

        today_d = date.today()
        if scope == "Today":
            range_start, range_end = today_d, today_d
        elif scope == "Yesterday":
            yd = today_d - timedelta(days=1)
            range_start, range_end = yd, yd
        elif scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            d1, d2 = st.columns(2)
            with d1: range_start = st.date_input("Start date", value=today_d.replace(day=1), key="st_start")
            with d2: range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="st_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                st.stop()

        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}** • Granularity: **{gran}**")

        # ---------- Normalize base series ----------
        C   = coerce_datetime(df_f[_create]).dt.date
        P   = coerce_datetime(df_f[_pay]).dt.date
        CNS = df_f[_cns].fillna("Unknown").astype(str).str.strip()
        SRC = df_f[_src].fillna("Unknown").astype(str).str.strip() if _src else pd.Series("Unknown", index=df_f.index)
        CTY = df_f[_cty].fillna("Unknown").astype(str).str.strip() if _cty else pd.Series("Unknown", index=df_f.index)
        F   = coerce_datetime(df_f[_first_cal]).dt.date if _first_cal and _first_cal in df_f.columns else None
        R   = coerce_datetime(df_f[_resched]).dt.date   if _resched   and _resched   in df_f.columns else None
        D   = coerce_datetime(df_f[_done]).dt.date      if _done      and _done      in df_f.columns else None

        if _ris and _ris in df_f.columns:
            RIS = df_f[_ris].fillna("Unknown").astype(str).str.strip()
            has_sales_generated = RIS.str.len().gt(0) & (RIS.str.lower() != "unknown")
        else:
            RIS = pd.Series("Unknown", index=df_f.index)
            has_sales_generated = pd.Series(False, index=df_f.index)

        # ---------- Window masks ----------
        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        m_created = between_date(C, range_start, range_end)
        m_paid    = between_date(P, range_start, range_end)
        m_first   = between_date(F, range_start, range_end) if F is not None else None
        m_resch   = between_date(R, range_start, range_end) if R is not None else None
        m_done    = between_date(D, range_start, range_end) if D is not None else None

        # Mode logic
        m_enrol = (m_created & m_paid) if mode == "MTD" else m_paid
        m_first_eff = (m_created & m_first) if (mode == "MTD" and m_first is not None) else m_first
        m_resch_eff = (m_created & m_resch) if (mode == "MTD" and m_resch is not None) else m_resch
        m_done_eff  = (m_created & m_done)  if (mode == "MTD" and m_done  is not None) else m_done
        m_sales_gen = (m_enrol & has_sales_generated)  # Sales-generated enrolments

        # ---------- Filters ----------
        fc1, fc2, fc3 = st.columns([1.3, 1.3, 1.2])
        with fc1:
            cns_opts = ["All"] + sorted(CNS.unique().tolist())
            pick_cns = st.multiselect("Academic Counsellor", options=cns_opts, default=["All"], key="st_cns")
        with fc2:
            src_opts = ["All"] + (sorted(SRC.unique().tolist()) if _src else [])
            pick_src = st.multiselect("JetLearn Deal Source", options=src_opts, default=["All"], key="st_src")
        with fc3:
            cty_opts = ["All"] + (sorted(CTY.unique().tolist()) if _cty else [])
            pick_cty = st.multiselect("Country", options=cty_opts, default=["All"], key="st_cty")

        # Resolve "All"
        def _resolve(vals, all_vals):
            return all_vals if ("All" in vals or not vals) else vals

        cns_sel = _resolve(pick_cns, sorted(CNS.unique().tolist()))
        src_sel = _resolve(pick_src, sorted(SRC.unique().tolist())) if _src else ["Unknown"]
        cty_sel = _resolve(pick_cty, sorted(CTY.unique().tolist())) if _cty else ["Unknown"]

        base_mask = CNS.isin(cns_sel) & SRC.isin(src_sel) & CTY.isin(cty_sel)

        # ---------- Metrics builder helpers ----------
        def _count(mask):
            return int((base_mask & mask).sum()) if mask is not None else 0

        # Summary KPIs (unchanged style)
        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 14px; background: #ffffff; }
              .kpi-title { font-size: 0.85rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.6rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            val = _count(m_created)
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div><div class='kpi-value'>{val:,}</div><div class='kpi-sub'>{range_start} → {range_end}</div></div>", unsafe_allow_html=True)
        with k2:
            val = _count(m_enrol)
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrolments</div><div class='kpi-value'>{val:,}</div><div class='kpi-sub'>Mode: {mode}</div></div>", unsafe_allow_html=True)
        with k3:
            val = _count(m_first_eff)
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>First Cal Scheduled</div><div class='kpi-value'>{val:,}</div><div class='kpi-sub'>Window</div></div>", unsafe_allow_html=True)
        with k4:
            val = _count(m_done_eff)
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Calibration Done</div><div class='kpi-value'>{val:,}</div><div class='kpi-sub'>Window</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ============
        # Summary view
        # ============
        if gran == "Summary":
            # Group by Counsellor with Source & Country breakdowns
            metrics = []
            gmask = base_mask
            df_view = df_f.loc[gmask].copy()

            # Build per-counsellor counts
            def _series_count(by, mask, name):
                if mask is None:
                    return pd.DataFrame(columns=[by, name])
                d = df_view.loc[mask.loc[gmask].values, [by]].copy()
                if d.empty:
                    return pd.DataFrame(columns=[by, name])
                return d.assign(_one=1).groupby(by, dropna=False)["_one"].sum().rename(name).reset_index()

            created_by = _series_count(_cns, m_created, "Deals Created")
            enrol_by   = _series_count(_cns, m_enrol, "Enrolments")
            first_by   = _series_count(_cns, m_first_eff, "First Cal Scheduled")
            resch_by   = _series_count(_cns, m_resch_eff, "Cal Rescheduled")
            done_by    = _series_count(_cns, m_done_eff, "Cal Done")
            sales_by   = _series_count(_cns, m_sales_gen, "Sales Generated (RIS)")

            out = created_by.merge(enrol_by, on=_cns, how="outer") \
                            .merge(first_by, on=_cns, how="outer") \
                            .merge(resch_by, on=_cns, how="outer") \
                            .merge(done_by, on=_cns, how="outer") \
                            .merge(sales_by, on=_cns, how="outer")
            for c in ["Deals Created","Enrolments","First Cal Scheduled","Cal Rescheduled","Cal Done","Sales Generated (RIS)"]:
                if c not in out.columns: out[c] = 0
            out = out.fillna(0).sort_values("Enrolments", ascending=False)

            st.markdown("### Counsellor Summary")
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download CSV — Counsellor Summary",
                out.to_csv(index=False).encode("utf-8"),
                "sales_tracker_counsellor_summary.csv",
                "text/csv",
                key="st_dl_summary"
            )

        # ==========
        # Day-wise
        # ==========
        elif gran == "Day-wise":
            st.markdown("### Day-wise — by Academic Counsellor")

            # Build a day-wise frame for each metric with the correct date basis
            def _daily(by_cns=True):
                frames = []

                # Deals Created by Create Date
                d1 = pd.DataFrame({
                    "Date": C[base_mask & m_created],
                    "Counsellor": CNS[base_mask & m_created],
                    "Metric": "Deals Created",
                })
                frames.append(d1)

                # Enrolments by Payment Date (per mode already handled in m_enrol)
                d2 = pd.DataFrame({
                    "Date": P[base_mask & m_enrol],
                    "Counsellor": CNS[base_mask & m_enrol],
                    "Metric": "Enrolments",
                })
                frames.append(d2)

                # First Cal
                if m_first_eff is not None:
                    d3 = pd.DataFrame({
                        "Date": F[base_mask & m_first_eff],
                        "Counsellor": CNS[base_mask & m_first_eff],
                        "Metric": "First Cal Scheduled",
                    })
                    frames.append(d3)

                # Rescheduled
                if m_resch_eff is not None:
                    d4 = pd.DataFrame({
                        "Date": R[base_mask & m_resch_eff],
                        "Counsellor": CNS[base_mask & m_resch_eff],
                        "Metric": "Cal Rescheduled",
                    })
                    frames.append(d4)

                # Done
                if m_done_eff is not None:
                    d5 = pd.DataFrame({
                        "Date": D[base_mask & m_done_eff],
                        "Counsellor": CNS[base_mask & m_done_eff],
                        "Metric": "Cal Done",
                    })
                    frames.append(d5)

                # Sales Generated (RIS) — counted off payment date where RIS is present
                d6 = pd.DataFrame({
                    "Date": P[base_mask & m_sales_gen],
                    "Counsellor": CNS[base_mask & m_sales_gen],
                    "Metric": "Sales Generated (RIS)",
                })
                frames.append(d6)

                df_all = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=["Date","Counsellor","Metric"])
                if df_all.empty:
                    return df_all

                df_all["Date"] = pd.to_datetime(df_all["Date"])
                g = df_all.groupby(["Counsellor","Date","Metric"], observed=True).size().rename("Count").reset_index()
                return g

            day = _daily()
            if day.empty:
                st.info("No day-wise data for the selected filters/date range.")
                st.stop()

            # Pick metric(s) to plot
            all_metrics = day["Metric"].unique().tolist()
            msel = st.multiselect("Metrics", options=all_metrics, default=all_metrics[:2], key="st_day_metrics")

            day_show = day[day["Metric"].isin(msel)].copy()
            day_show["Date"] = pd.to_datetime(day_show["Date"])

            # Chart
            if chart_type == "Bar":
                ch = (
                    alt.Chart(day_show)
                    .mark_bar(opacity=0.9)
                    .encode(
                        x=alt.X("yearmonthdate(Date):T", title="Date"),
                        y=alt.Y("Count:Q", title="Count"),
                        color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                        column=alt.Column("Counsellor:N", title="Academic Counsellor"),
                        tooltip=[alt.Tooltip("yearmonthdate(Date):T", title="Date"),
                                 alt.Tooltip("Counsellor:N"),
                                 alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Count:Q")]
                    )
                    .properties(height=220)
                )
            else:
                ch = (
                    alt.Chart(day_show)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("yearmonthdate(Date):T", title="Date"),
                        y=alt.Y("Count:Q", title="Count"),
                        color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                        facet=alt.Facet("Counsellor:N", title="Academic Counsellor", columns=2),
                        tooltip=[alt.Tooltip("yearmonthdate(Date):T", title="Date"),
                                 alt.Tooltip("Counsellor:N"),
                                 alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Count:Q")]
                    )
                    .properties(height=220)
                )

            st.altair_chart(ch, use_container_width=True)

            # Table + Download
            st.dataframe(
                day_show.sort_values(["Counsellor","Date","Metric"]),
                use_container_width=True
            )
            st.download_button(
                "Download CSV — Day-wise Counsellor Metrics",
                day_show.sort_values(["Counsellor","Date","Metric"]).to_csv(index=False).encode("utf-8"),
                "sales_tracker_daywise_counsellor.csv",
                "text/csv",
                key="st_dl_daywise"
            )

        # ==========
        # Monthly (ADDED)
        # ==========
        else:
            st.markdown("### Monthly — by Academic Counsellor")

            # Build a month-wise frame for each metric with the correct date basis
            def _monthly():
                frames = []

                # Helper to convert date series to month-start timestamps
                def to_month_start(s):
                    s2 = pd.to_datetime(s, errors="coerce")
                    return s2.dt.to_period("M").dt.to_timestamp()

                # Deals Created (Create date month)
                idx1 = base_mask & m_created
                d1 = pd.DataFrame({
                    "Month": to_month_start(C[idx1]),
                    "Counsellor": CNS[idx1],
                    "Metric": "Deals Created",
                })
                frames.append(d1)

                # Enrolments (Payment date month, mode already in m_enrol)
                idx2 = base_mask & m_enrol
                d2 = pd.DataFrame({
                    "Month": to_month_start(P[idx2]),
                    "Counsellor": CNS[idx2],
                    "Metric": "Enrolments",
                })
                frames.append(d2)

                # First Cal
                if m_first_eff is not None:
                    idx3 = base_mask & m_first_eff
                    d3 = pd.DataFrame({
                        "Month": to_month_start(F[idx3]),
                        "Counsellor": CNS[idx3],
                        "Metric": "First Cal Scheduled",
                    })
                    frames.append(d3)

                # Rescheduled
                if m_resch_eff is not None:
                    idx4 = base_mask & m_resch_eff
                    d4 = pd.DataFrame({
                        "Month": to_month_start(R[idx4]),
                        "Counsellor": CNS[idx4],
                        "Metric": "Cal Rescheduled",
                    })
                    frames.append(d4)

                # Done
                if m_done_eff is not None:
                    idx5 = base_mask & m_done_eff
                    d5 = pd.DataFrame({
                        "Month": to_month_start(D[idx5]),
                        "Counsellor": CNS[idx5],
                        "Metric": "Cal Done",
                    })
                    frames.append(d5)

                # Sales Generated (RIS)
                idx6 = base_mask & m_sales_gen
                d6 = pd.DataFrame({
                    "Month": to_month_start(P[idx6]),
                    "Counsellor": CNS[idx6],
                    "Metric": "Sales Generated (RIS)",
                })
                frames.append(d6)

                df_all = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=["Month","Counsellor","Metric"])
                if df_all.empty:
                    return df_all

                df_all["Month"] = pd.to_datetime(df_all["Month"])
                g = df_all.groupby(["Counsellor","Month","Metric"], observed=True).size().rename("Count").reset_index()
                return g

            mon = _monthly()
            if mon.empty:
                st.info("No monthly data for the selected filters/date range.")
                st.stop()

            # Pick metric(s) to plot
            all_metrics_m = mon["Metric"].unique().tolist()
            msel_m = st.multiselect("Metrics", options=all_metrics_m, default=all_metrics_m[:2], key="st_month_metrics")

            mon_show = mon[mon["Metric"].isin(msel_m)].copy()
            mon_show["Month"] = pd.to_datetime(mon_show["Month"])

            # Chart
            if chart_type == "Bar":
                chm = (
                    alt.Chart(mon_show)
                    .mark_bar(opacity=0.9)
                    .encode(
                        x=alt.X("yearmonth(Month):T", title="Month"),
                        y=alt.Y("Count:Q", title="Count"),
                        color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                        column=alt.Column("Counsellor:N", title="Academic Counsellor"),
                        tooltip=[alt.Tooltip("yearmonth(Month):T", title="Month"),
                                 alt.Tooltip("Counsellor:N"),
                                 alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Count:Q")]
                    )
                    .properties(height=220)
                )
            else:
                chm = (
                    alt.Chart(mon_show)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("yearmonth(Month):T", title="Month"),
                        y=alt.Y("Count:Q", title="Count"),
                        color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                        facet=alt.Facet("Counsellor:N", title="Academic Counsellor", columns=2),
                        tooltip=[alt.Tooltip("yearmonth(Month):T", title="Month"),
                                 alt.Tooltip("Counsellor:N"),
                                 alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Count:Q")]
                    )
                    .properties(height=220)
                )

            st.altair_chart(chm, use_container_width=True)

            # Table + Download
            st.dataframe(
                mon_show.sort_values(["Counsellor","Month","Metric"]),
                use_container_width=True
            )
            st.download_button(
                "Download CSV — Monthly Counsellor Metrics",
                mon_show.sort_values(["Counsellor","Month","Metric"]).to_csv(index=False).encode("utf-8"),
                "sales_tracker_monthly_counsellor.csv",
                "text/csv",
                key="st_dl_monthly"
            )

    # run it
    _sales_tracker_tab()

# =========================
# Deal Velocity Tab (full)
# =========================
elif view == "Deal Velocity":
    def _deal_velocity_tab():
        st.subheader("Deal Velocity — Volume & Velocity (MTD / Cohort)")

        # ---------- Resolve key columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])
        _first  = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
        _resch  = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
        _done   = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, ["Calibration Done Date","Cal Done Date","Calibration Completed"])

        _cty    = country_col    if (country_col    in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _src    = source_col     if (source_col     in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cns    = counsellor_col if (counsellor_col in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])

        if not _create or _create not in df_f.columns:
            st.warning("Create Date column is required for this view.", icon="⚠️"); st.stop()

        # ---------- Controls ----------
        col_top1, col_top2, col_top3 = st.columns([1.2, 1.2, 1.4])
        with col_top1:
            mode = st.radio(
                "Mode",
                ["MTD", "Cohort"],
                index=1,
                horizontal=True,
                key="stage_mode",
                help=("MTD: count a stage only if its own date is in-range AND the deal was created in-range. "
                      "Cohort: count by the stage's own date, regardless of create month.")
            )
        with col_top2:
            scope = st.radio(
                "Date scope",
                ["This month", "Last month", "Custom"],
                index=0,
                horizontal=True,
                key="stage_dscope"
            )
        with col_top3:
            agg_view = st.radio(
                "Time grain",
                ["Month-on-Month", "Day-wise"],
                index=0,
                horizontal=True,
                key="stage_grain"
            )

        today_d = date.today()
        if scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            c1, c2 = st.columns(2)
            with c1: range_start = st.date_input("Start date", value=today_d.replace(day=1), key="stage_start")
            with c2: range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="stage_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                st.stop()
        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}** • Grain: **{agg_view}**")

        # ---------- Optional filters (+ 'All') ----------
        def norm_cat(s):
            return s.fillna("Unknown").astype(str).str.strip()

        if _cty and _cty in df_f.columns:
            cty_vals_all = sorted(norm_cat(df_f[_cty]).unique().tolist())
            cty_opts = ["All"] + cty_vals_all
            cty_sel = st.multiselect("Filter Country", options=cty_opts, default=["All"], key="stage_cty")
        else:
            cty_vals_all, cty_sel = [], []

        if _src and _src in df_f.columns:
            src_vals_all = sorted(norm_cat(df_f[_src]).unique().tolist())
            src_opts = ["All"] + src_vals_all
            src_sel = st.multiselect("Filter JetLearn Deal Source", options=src_opts, default=["All"], key="stage_src")
        else:
            src_vals_all, src_sel = [], []

        if _cns and _cns in df_f.columns:
            cns_vals_all = sorted(norm_cat(df_f[_cns]).unique().tolist())
            cns_opts = ["All"] + cns_vals_all
            cns_sel = st.multiselect("Filter Academic Counsellor", options=cns_opts, default=["All"], key="stage_cns")
        else:
            cns_vals_all, cns_sel = [], []

        # Apply 'All' behavior
        def _apply_multi_all(series, sel, all_vals):
            if not sel or "All" in sel:  # no filter
                return pd.Series(True, index=series.index)
            return norm_cat(series).isin(sel)

        mask_cty = _apply_multi_all(df_f[_cty] if _cty else pd.Series("Unknown", index=df_f.index), cty_sel, cty_vals_all)
        mask_src = _apply_multi_all(df_f[_src] if _src else pd.Series("Unknown", index=df_f.index), src_sel, src_vals_all)
        mask_cns = _apply_multi_all(df_f[_cns] if _cns else pd.Series("Unknown", index=df_f.index), cns_sel, cns_vals_all)

        filt_mask = mask_cty & mask_src & mask_cns

        # ---------- Normalize datetime series ----------
        C = coerce_datetime(df_f[_create]).dt.date
        F = coerce_datetime(df_f[_first]).dt.date if _first else None
        R = coerce_datetime(df_f[_resch]).dt.date if _resch else None
        D = coerce_datetime(df_f[_done]).dt.date  if _done  else None
        P = coerce_datetime(df_f[_pay]).dt.date   if _pay   else None

        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        mask_created_in = between_date(C, range_start, range_end)
        # Mode-aware per-stage inclusion
        def stage_mask(stage_series):
            if stage_series is None:
                return pd.Series(False, index=df_f.index)
            in_range = between_date(stage_series, range_start, range_end)
            return (in_range & mask_created_in) if (mode == "MTD") else in_range

        m_create = stage_mask(C)                    # Created
        m_first  = stage_mask(F) if F is not None else pd.Series(False, index=df_f.index)
        m_resch  = stage_mask(R) if R is not None else pd.Series(False, index=df_f.index)
        m_done   = stage_mask(D) if D is not None else pd.Series(False, index=df_f.index)
        m_enrol  = stage_mask(P) if P is not None else pd.Series(False, index=df_f.index)

        # Apply global filters
        m_create &= filt_mask
        if F is not None: m_first &= filt_mask
        if R is not None: m_resch &= filt_mask
        if D is not None: m_done  &= filt_mask
        if P is not None: m_enrol &= filt_mask

        # ---------- KPI strip ----------
        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
              .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.4rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div><div class='kpi-value'>{int(m_create.sum()):,}</div><div class='kpi-sub'>{range_start} → {range_end}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>First Cal Scheduled</div><div class='kpi-value'>{int(m_first.sum()) if F is not None else 0:,}</div><div class='kpi-sub'>{'—' if not _first else _first}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Cal Rescheduled</div><div class='kpi-value'>{int(m_resch.sum()) if R is not None else 0:,}</div><div class='kpi-sub'>{'—' if not _resch else _resch}</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Cal Done</div><div class='kpi-value'>{int(m_done.sum()) if D is not None else 0:,}</div><div class='kpi-sub'>{'—' if not _done else _done}</div></div>", unsafe_allow_html=True)
        with k5:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrolments</div><div class='kpi-value'>{int(m_enrol.sum()) if P is not None else 0:,}</div><div class='kpi-sub'>Mode: {mode}</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------- Build long event frame for charts ----------
        long_rows = []
        def _append(stage_name, series, mask):
            if series is None: return
            if mask.any():
                tmp = pd.DataFrame({"Date": series[mask]})
                tmp["Stage"] = stage_name
                long_rows.append(tmp)

        _append("Deal Created", C, m_create)
        if F is not None: _append("First Calibration Scheduled", F, m_first)
        if R is not None: _append("Calibration Rescheduled", R, m_resch)
        if D is not None: _append("Calibration Done", D, m_done)
        if P is not None: _append("Enrolment", P, m_enrol)

        long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(columns=["Date","Stage"])
        if not long_df.empty:
            long_df["Date"] = pd.to_datetime(long_df["Date"])

        # ---------- Charts: MoM or Day-wise ----------
        if long_df.empty:
            st.info("No stage events match the selected filters/date range.")
        else:
            view_mode = st.radio("View as", ["Graph", "Table"], horizontal=True, key="stage_viewmode")

            if agg_view == "Month-on-Month":
                long_df["_m"] = long_df["Date"].dt.to_period("M")
                agg_tbl = (long_df.groupby(["_m","Stage"]).size()
                                   .rename("Count").reset_index())
                agg_tbl["Month"] = agg_tbl["_m"].astype(str)
                agg_tbl = agg_tbl[["Month","Stage","Count"]]

                if view_mode == "Graph":
                    ch = (
                        alt.Chart(agg_tbl)
                        .mark_bar(opacity=0.9)
                        .encode(
                            x=alt.X("Month:N", sort=sorted(agg_tbl["Month"].unique().tolist())),
                            y=alt.Y("Count:Q", title="Count"),
                            color=alt.Color("Stage:N", legend=alt.Legend(orient="bottom")),
                            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Stage:N"), alt.Tooltip("Count:Q")]
                        )
                        .properties(height=360, title="MoM — Stage Volumes (stacked)")
                    )
                    st.altair_chart(ch, use_container_width=True)
                else:
                    pivot = agg_tbl.pivot(index="Month", columns="Stage", values="Count").fillna(0).astype(int).reset_index()
                    st.dataframe(pivot, use_container_width=True)
                    st.download_button(
                        "Download CSV — MoM Stage Volumes",
                        pivot.to_csv(index=False).encode("utf-8"),
                        "deal_velocity_mom.csv", "text/csv", key="stage_dl_mom"
                    )
            else:
                # Day-wise
                long_df["Day"] = long_df["Date"].dt.date
                agg_tbl = (long_df.groupby(["Day","Stage"]).size()
                                   .rename("Count").reset_index())

                if view_mode == "Graph":
                    ch = (
                        alt.Chart(agg_tbl)
                        .mark_bar(opacity=0.9)
                        .encode(
                            x=alt.X("Day:T", title="Day"),
                            y=alt.Y("Count:Q", title="Count"),
                            color=alt.Color("Stage:N", legend=alt.Legend(orient="bottom")),
                            tooltip=[alt.Tooltip("Day:T"), alt.Tooltip("Stage:N"), alt.Tooltip("Count:Q")]
                        )
                        .properties(height=360, title="Day-wise — Stage Volumes (stacked)")
                    )
                    st.altair_chart(ch, use_container_width=True)
                else:
                    pivot = agg_tbl.pivot(index="Day", columns="Stage", values="Count").fillna(0).astype(int).reset_index()
                    st.dataframe(pivot, use_container_width=True)
                    st.download_button(
                        "Download CSV — Day-wise Stage Volumes",
                        pivot.to_csv(index=False).encode("utf-8"),
                        "deal_velocity_daywise.csv", "text/csv", key="stage_dl_day"
                    )

        st.markdown("---")

        # ---------- Velocity (time between stages) ----------
        st.markdown("### Velocity — Time Between Stages")
        trans_pairs = [
            ("Deal Created", "First Calibration Scheduled", _create, _first),
            ("First Calibration Scheduled", "Calibration Done", _first, _done),
            ("Deal Created", "Enrolment", _create, _pay),
            ("First Calibration Scheduled", "Calibration Rescheduled", _first, _resch),
            ("Calibration Done", "Enrolment", _done, _pay),  # added
        ]
        # Allow user to pick a pair
        valid_pairs = [(a,b) for (a,b,fc,tc) in trans_pairs if fc and tc and fc in df_f.columns and tc in df_f.columns]
        pair_labels = [f"{a} → {b}" for (a,b) in valid_pairs]
        if not pair_labels:
            st.info("Not enough stage date columns to compute velocity (need at least a valid from/to pair).")
            st.stop()
        pick = st.selectbox("Pick a transition", pair_labels, index=0, key="stage_pair")

        from_label, to_label = valid_pairs[pair_labels.index(pick)]
        # Get actual column names for the chosen pair
        col_map = {
            "Deal Created": _create,
            "First Calibration Scheduled": _first,
            "Calibration Rescheduled": _resch,
            "Calibration Done": _done,
            "Enrolment": _pay,
        }
        from_col = col_map[from_label]
        to_col   = col_map[to_label]

        # Build masks per mode for the TO event (what belongs to this window),
        # then compute deltas only for rows that pass global filters + this window.
        from_dt = coerce_datetime(df_f[from_col])
        to_dt   = coerce_datetime(df_f[to_col])

        # Mode window for "to" event
        to_in = between_date(to_dt.dt.date, range_start, range_end)
        mask_created_in = between_date(C, range_start, range_end)  # reuse created window
        window_mask = (to_in & mask_created_in) if (mode == "MTD") else to_in

        # Apply global filters too
        window_mask &= filt_mask

        d_use = df_f.loc[window_mask, [from_col, to_col]].copy()
        # Ensure datetime
        d_use["__from"] = coerce_datetime(d_use[from_col])
        d_use["__to"]   = coerce_datetime(d_use[to_col])

        # Keep only rows where both sides exist and to >= from
        good = d_use["__from"].notna() & d_use["__to"].notna()
        d_use = d_use.loc[good].copy()
        if not d_use.empty:
            d_use["__days"] = (d_use["__to"] - d_use["__from"]).dt.days
            d_use = d_use[d_use["__days"] >= 0]

        if d_use.empty:
            st.info("No valid transitions in the selected window/filters to compute velocity.")
        else:
            mu = float(np.mean(d_use["__days"]))
            sigma = float(np.std(d_use["__days"], ddof=0))
            med = float(np.median(d_use["__days"]))
            p95 = float(np.percentile(d_use["__days"], 95))

            st.markdown(
                """
                <style>
                  .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
                  .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
                  .kpi-value { font-size: 1.4rem; font-weight: 700; }
                  .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
                </style>
                """,
                unsafe_allow_html=True
            )
            kpa, kpb, kpc, kpd = st.columns(4)
            with kpa:
                st.markdown(f"<div class='kpi-card'><div class='kpi-title'>μ (Average days)</div><div class='kpi-value'>{mu:.1f}</div><div class='kpi-sub'>{from_label} → {to_label}</div></div>", unsafe_allow_html=True)
            with kpb:
                st.markdown(f"<div class='kpi-card'><div class='kpi-title'>σ (Std dev)</div><div class='kpi-value'>{sigma:.1f}</div><div class='kpi-sub'>Population σ</div></div>", unsafe_allow_html=True)
            with kpc:
                st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Median</div><div class='kpi-value'>{med:.1f}</div><div class='kpi-sub'>Days</div></div>", unsafe_allow_html=True)
            with kpd:
                st.markdown(f"<div class='kpi-card'><div class='kpi-title'>p95</div><div class='kpi-value'>{p95:.1f}</div><div class='kpi-sub'>Days</div></div>", unsafe_allow_html=True)

            # Bell curve toggle
            vmode = st.radio("Velocity view", ["Histogram + Bell curve", "Table"], index=0, horizontal=True, key="stage_vel_view")
            if vmode == "Histogram + Bell curve":
                hist_df = d_use[["__days"]].rename(columns={"__days":"Days"}).copy()
                x_max = max(hist_df["Days"].max(), p95) * 1.2
                x_vals = np.linspace(0, max(1, x_max), 200)
                binw = max(1, round(hist_df["Days"].max() / 20))
                sigma_safe = (sigma if sigma > 0 else 1.0)
                pdf = (1.0/sigma_safe/np.sqrt(2*np.pi)) * np.exp(-(x_vals - mu)**2/(2*sigma_safe**2))
                scale = len(hist_df) * binw
                curve_df = pd.DataFrame({"Days": x_vals, "ScaledPDF": pdf * scale})

                ch_hist = (
                    alt.Chart(hist_df)
                    .mark_bar(opacity=0.85)
                    .encode(
                        x=alt.X("Days:Q", bin=alt.Bin(maxbins=30), title="Days"),
                        y=alt.Y("count():Q", title="Count"),
                        tooltip=[alt.Tooltip("count():Q", title="Count")]
                    )
                    .properties(height=320, title=f"Velocity: {from_label} → {to_label}")
                )
                ch_curve = (
                    alt.Chart(curve_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Days:Q"),
                        y=alt.Y("ScaledPDF:Q", title="Count"),
                        tooltip=[alt.Tooltip("Days:Q"), alt.Tooltip("ScaledPDF:Q", title="Scaled PDF", format=".1f")]
                    )
                )
                st.altair_chart(ch_hist + ch_curve, use_container_width=True)
            else:
                out_tbl = d_use["__days"].describe(percentiles=[0.5, 0.95]).to_frame(name="Days").reset_index()
                st.dataframe(out_tbl, use_container_width=True)
                st.download_button(
                    "Download CSV — Velocity samples",
                    d_use[["__days"]].rename(columns={"__days":"Days"}).to_csv(index=False).encode("utf-8"),
                    "deal_velocity_samples.csv","text/csv", key="stage_dl_vel"
                )

    # run tab
    _deal_velocity_tab()
# =========================
# Carry Forward Tab (Cohort Contributions) — Enrolments only
# =========================
elif view == "Carry Forward":
    def _carry_forward_tab():
        st.subheader("Carry Forward — Cohort Contribution of Created → Enrolments")

        # ---------- Resolve core columns ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, [
            "Create Date","Created Date","Deal Create Date","CreateDate"
        ])
        _pay    = pay_col if (pay_col in df_f.columns) else find_col(df_f, [
            "Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"
        ])

        # Optional filters
        _cty = country_col if (country_col in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _src = source_col if (source_col in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cns = counsellor_col if (counsellor_col in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])

        if not _create or not _pay or _create not in df_f.columns or _pay not in df_f.columns:
            st.warning("Create/Payment columns are required. Please map them in the sidebar.", icon="⚠️")
            st.stop()

        # ---------- Controls ----------
        col_top1, col_top2 = st.columns([1.2, 1.2])
        with col_top1:
            trailing = st.selectbox("Payment MoM horizon", [3, 6, 9, 12, 18, 24], index=2, key="cf_trailing")
        with col_top2:
            show_mode = st.radio("View", ["Graph", "Table"], index=0, horizontal=True, key="cf_view")

        # Date scope (payment-month range end)
        today_d = date.today()
        end_m_default = today_d.replace(day=1)
        end_month = st.date_input("End month (use 1st of month)", value=end_m_default, key="cf_end_month")
        if isinstance(end_month, tuple):
            end_month = end_month[0]
        end_period = pd.Period(end_month.replace(day=1), freq="M")
        months = pd.period_range(end=end_period, periods=int(trailing), freq="M")
        months_list = months.astype(str).tolist()

        # Filters with “All”
        def norm_cat(s):
            return s.fillna("Unknown").astype(str).str.strip()

        fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1.2])
        if _cty:
            all_cty = sorted(norm_cat(df_f[_cty]).unique().tolist())
            opt_cty = ["All"] + all_cty
            sel_cty = fcol1.multiselect("Filter Country", options=opt_cty, default=["All"], key="cf_cty")
        else:
            all_cty, sel_cty = [], ["All"]

        if _src:
            all_src = sorted(norm_cat(df_f[_src]).unique().tolist())
            opt_src = ["All"] + all_src
            sel_src = fcol2.multiselect("Filter JetLearn Deal Source", options=opt_src, default=["All"], key="cf_src")
        else:
            all_src, sel_src = [], ["All"]

        if _cns:
            all_cns = sorted(norm_cat(df_f[_cns]).unique().tolist())
            opt_cns = ["All"] + all_cns
            sel_cns = fcol3.multiselect("Filter Academic Counsellor", options=opt_cns, default=["All"], key="cf_cns")
        else:
            all_cns, sel_cns = [], ["All"]

        def _apply_multi_all(series, selected):
            if series is None or "All" in selected:
                return pd.Series(True, index=df_f.index)
            return norm_cat(series).isin(selected)

        mask_cty = _apply_multi_all(df_f[_cty] if _cty else None, sel_cty)
        mask_src = _apply_multi_all(df_f[_src] if _src else None, sel_src)
        mask_cns = _apply_multi_all(df_f[_cns] if _cns else None, sel_cns)
        filt_mask = mask_cty & mask_src & mask_cns

        # ---------- Build cohort frame ----------
        C = coerce_datetime(df_f[_create]).dt.to_period("M")
        P = coerce_datetime(df_f[_pay]).dt.to_period("M")

        # Only keep rows where payment month is inside the horizon window
        in_pay_window = P.isin(months)
        base_mask = in_pay_window & filt_mask

        # Enrolments metric (counts)
        val_name = "Enrolments"
        metric_series = pd.Series(1, index=df_f.index, dtype=float)

        # Cohort matrix: Payment Month × Create Month
        cohort_df = pd.DataFrame({
            "PayMonth": P[base_mask].astype(str),
            "CreateMonth": C[base_mask].astype(str),
            val_name: metric_series[base_mask]
        })

        if cohort_df.empty:
            st.info("No rows in selected horizon/filters.")
            st.stop()

        # Pivot: rows=PayMonth, cols=CreateMonth
        matrix = (cohort_df
                  .groupby(["PayMonth","CreateMonth"], as_index=False)[val_name]
                  .sum())
        pivot = (matrix.pivot(index="PayMonth", columns="CreateMonth", values=val_name)
                        .reindex(index=months_list, columns=sorted(matrix["CreateMonth"].unique()))
                        .fillna(0.0))

        # ---------- Same-month vs carry-forward (lag buckets) ----------
        m_long = pivot.stack().rename(val_name).reset_index()

        # Robust month-index lag computation (handles missing safely)
        def _ym_index(period_str_series: pd.Series) -> pd.Series:
            """Convert 'YYYY-MM' (Period 'M') strings to monotonic month index (year*12+month)."""
            # Coerce to Period; invalid -> NaT
            p = pd.PeriodIndex(period_str_series.astype(str), freq="M")
            # Build month index; will be Int64 with <NA> if any NaT
            return (pd.Series(p.year, index=period_str_series.index).astype("Int64") * 12 +
                    pd.Series(p.month, index=period_str_series.index).astype("Int64"))

        pay_idx = _ym_index(m_long["PayMonth"])
        cre_idx = _ym_index(m_long["CreateMonth"])
        # Difference in months; keep as Int64 so <NA> is allowed
        m_long["Lag"] = (pay_idx - cre_idx)

        # Negative lags (future-created vs pay month) → set to NA
        m_long.loc[m_long["Lag"].notna() & (m_long["Lag"] < 0), "Lag"] = pd.NA

        def lag_bucket(n):
            if pd.isna(n): return np.nan
            n = int(n)
            if n <= 0: return "Lag 0 (Same Month)"
            if n == 1: return "Lag 1"
            if n == 2: return "Lag 2"
            if n == 3: return "Lag 3"
            if n == 4: return "Lag 4"
            if n == 5: return "Lag 5"
            return "Lag 6+"

        m_long["LagBucket"] = m_long["Lag"].map(lag_bucket)

        lag_tbl = (m_long.dropna(subset=["LagBucket"])
                          .groupby(["PayMonth","LagBucket"], as_index=False)[val_name]
                          .sum())
        lag_tbl = lag_tbl[lag_tbl["PayMonth"].isin(months_list)]
        bucket_order = ["Lag 0 (Same Month)","Lag 1","Lag 2","Lag 3","Lag 4","Lag 5","Lag 6+"]
        lag_tbl["LagBucket"] = pd.Categorical(lag_tbl["LagBucket"], categories=bucket_order, ordered=True)

        # KPIs for latest month
        latest = months_list[-1]
        latest_tot = int(lag_tbl.loc[lag_tbl["PayMonth"] == latest, val_name].sum())
        latest_same = int(lag_tbl.loc[(lag_tbl["PayMonth"] == latest) & (lag_tbl["LagBucket"]=="Lag 0 (Same Month)") , val_name].sum())
        latest_pct_same = (latest_same / latest_tot * 100.0) if latest_tot > 0 else np.nan

        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
              .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.4rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Latest Month Total ({val_name})</div>"
                f"<div class='kpi-value'>{latest_tot:,}</div>"
                f"<div class='kpi-sub'>{latest}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Same-Month ({val_name})</div>"
                f"<div class='kpi-value'>{latest_same:,}</div>"
                f"<div class='kpi-sub'>{latest}</div></div>", unsafe_allow_html=True)
        with k3:
            pct_txt = "–" if np.isnan(latest_pct_same) else f"{latest_pct_same:.1f}%"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Same-Month % of Total</div>"
                f"<div class='kpi-value'>{pct_txt}</div>"
                f"<div class='kpi-sub'>Share of {val_name} from Lag 0</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------- Graphs / Tables ----------
        tabs = st.tabs(["Cohort Heatmap", "Lag Breakdown (stacked)", "Downloads"])
        with tabs[0]:
            view_heat = st.radio("View as", ["Graph", "Table"], index=0, horizontal=True, key="cf_heat_view")
            if view_heat == "Graph":
                heat_df = matrix.copy()
                heat_df["PayMonthCat"] = pd.Categorical(heat_df["PayMonth"], categories=months_list, ordered=True)
                ch = (
                    alt.Chart(heat_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("CreateMonth:N", title="Create Month", sort=sorted(heat_df["CreateMonth"].unique())),
                        y=alt.Y("PayMonthCat:N", title="Payment Month", sort=months_list),
                        color=alt.Color(f"{val_name}:Q", title=val_name, scale=alt.Scale(scheme="greens")),
                        tooltip=[
                            alt.Tooltip("PayMonth:N", title="Payment Month"),
                            alt.Tooltip("CreateMonth:N", title="Create Month"),
                            alt.Tooltip(f"{val_name}:Q", title=val_name, format="d"),
                        ],
                    )
                    .properties(height=420, title=f"Cohort Heatmap — {val_name}")
                )
                st.altair_chart(ch, use_container_width=True)
            else:
                st.dataframe(pivot.reset_index().rename(columns={"index":"PayMonth"}), use_container_width=True)

        with tabs[1]:
            view_lag = st.radio("Lag view as", ["Graph", "Table"], index=0, horizontal=True, key="cf_lag_view")
            if view_lag == "Graph":
                ch2 = (
                    alt.Chart(lag_tbl)
                    .mark_bar()
                    .encode(
                        x=alt.X("PayMonth:N", sort=months_list, title="Payment Month"),
                        y=alt.Y(f"{val_name}:Q", title=val_name),
                        color=alt.Color("LagBucket:N", legend=alt.Legend(orient="bottom")),
                        tooltip=[alt.Tooltip("PayMonth:N"),
                                 alt.Tooltip("LagBucket:N"),
                                 alt.Tooltip(f"{val_name}:Q", format="d")]
                    )
                    .properties(height=360, title=f"Lag Breakdown — {val_name}")
                )
                st.altair_chart(ch2, use_container_width=True)

                pct_toggle = st.checkbox("Show % share per month", value=False, key="cf_pct_stack")
                if pct_toggle:
                    pct_tbl = lag_tbl.copy()
                    month_tot = pct_tbl.groupby("PayMonth")[val_name].transform("sum")
                    pct_tbl["Pct"] = np.where(month_tot>0, pct_tbl[val_name]/month_tot*100.0, 0.0)
                    ch3 = (
                        alt.Chart(pct_tbl)
                        .mark_bar()
                        .encode(
                            x=alt.X("PayMonth:N", sort=months_list, title="Payment Month"),
                            y=alt.Y("Pct:Q", title="% of month", scale=alt.Scale(domain=[0,100])),
                            color=alt.Color("LagBucket:N", legend=alt.Legend(orient="bottom")),
                            tooltip=[alt.Tooltip("PayMonth:N"),
                                     alt.Tooltip("LagBucket:N"),
                                     alt.Tooltip("Pct:Q", format=".1f")]
                        )
                        .properties(height=320, title="Lag Breakdown — % Share")
                    )
                    st.altair_chart(ch3, use_container_width=True)
            else:
                wide = (lag_tbl
                        .pivot(index="PayMonth", columns="LagBucket", values=val_name)
                        .reindex(index=months_list, columns=bucket_order)
                        .fillna(0.0))
                st.dataframe(wide.reset_index(), use_container_width=True)

        with tabs[2]:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Download CSV — Cohort Heatmap Data",
                    pivot.reset_index().to_csv(index=False).encode("utf-8"),
                    "carry_forward_cohort_matrix.csv", "text/csv", key="cf_dl_matrix"
                )
            with col_d2:
                wide = (lag_tbl
                        .pivot(index="PayMonth", columns="LagBucket", values=val_name)
                        .reindex(index=months_list, columns=bucket_order)
                        .fillna(0.0))
                st.download_button(
                    "Download CSV — Lag Breakdown",
                    wide.reset_index().to_csv(index=False).encode("utf-8"),
                    "carry_forward_lag_breakdown.csv", "text/csv", key="cf_dl_lag"
                )

    # run the tab
    _carry_forward_tab()
# =========================
# =========================
# Buying Propensity Tab (Sales Subscription uses Installment Terms fallback = 1)
# + NEW: Installment Type Dynamics (kept everything else identical)
# =========================
elif view == "Buying Propensity":
    def _buying_propensity_tab():
        st.subheader("Buying Propensity — Payment Term & Payment Type")

        # ---------- Resolve columns (defensive) ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, ["Create Date","Created Date","Deal Create Date","CreateDate"])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate"])

        # Core variables
        _term  = find_col(df_f, ["Payment Term","PaymentTerm","Term","Installments","Tenure"])
        _ptype = find_col(df_f, ["Payment Type","PaymentType","Payment Mode","PaymentMode","Mode of Payment","Mode"])

        # For derived metric (Sales Subscription)
        _inst  = find_col(df_f, ["Installment Terms","Installments Terms","Installment Count","No. of Installments","EMI Count","Installments"])

        # NEW: Installment Type variable
        _itype = find_col(df_f, ["Installment Type","InstallmentType","EMI Type","Installment Plan Type","Plan Type"])

        # Optional filters
        _cty = country_col     if (country_col     in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _src = source_col      if (source_col      in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])
        _cns = counsellor_col  if (counsellor_col  in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])

        if not _create or not _pay or _create not in df_f.columns or _pay not in df_f.columns:
            st.warning("Create/Payment columns are required for Buying Propensity. Please map them in the sidebar.", icon="⚠️")
            st.stop()
        if not _term or _term not in df_f.columns:
            st.warning("‘Payment Term’ column not found. Please ensure a column like ‘Payment Term’/‘PaymentTerm’ exists.", icon="⚠️")
            st.stop()
        if not _ptype or _ptype not in df_f.columns:
            st.warning("‘Payment Type’ column not found. Please ensure a column like ‘Payment Type’/‘Payment Mode’ exists.", icon="⚠️")
            st.stop()

        # ---------- Controls ----------
        col_top1, col_top2, col_top3 = st.columns([1.0, 1.2, 1.2])
        with col_top1:
            mode = st.radio(
                "Mode",
                ["MTD", "Cohort"],
                index=1,
                horizontal=True,
                key="bp_mode",
                help=("MTD: enrolments/events counted only if the deal was also created in the same window/month. "
                      "Cohort: enrolments/events counted by payment date regardless of create month.")
            )
        with col_top2:
            scope = st.radio("Date scope (window KPIs)", ["This month", "Last month", "Custom"], index=0, horizontal=True, key="bp_dscope")
        with col_top3:
            mom_trailing = st.selectbox("MoM trailing (months)", [3, 6, 9, 12, 18, 24], index=3, key="bp_momh")

        today_d = date.today()
        if scope == "This month":
            range_start, range_end = month_bounds(today_d)
        elif scope == "Last month":
            range_start, range_end = last_month_bounds(today_d)
        else:
            c1, c2 = st.columns(2)
            with c1:
                range_start = st.date_input("Start date", value=today_d.replace(day=1), key="bp_start")
            with c2:
                range_end   = st.date_input("End date", value=month_bounds(today_d)[1], key="bp_end")
            if range_end < range_start:
                st.error("End date cannot be before start date.")
                st.stop()
        st.caption(f"Scope: **{scope}** ({range_start} → {range_end}) • Mode: **{mode}**")

        # MoM anchor month
        end_m_default = today_d.replace(day=1)
        end_month = st.date_input("Trend anchor month (use 1st of month for MoM)", value=end_m_default, key="bp_end_month")
        if isinstance(end_month, tuple):
            end_month = end_month[0]
        end_period = pd.Period(end_month.replace(day=1), freq="M")
        months = pd.period_range(end=end_period, periods=int(mom_trailing), freq="M")
        months_list = months.astype(str).tolist()

        # ---------- Filters (All defaults) ----------
        def norm_cat(s):
            return s.fillna("Unknown").astype(str).str.strip()

        f1, f2, f3 = st.columns([1.2, 1.2, 1.2])
        if _cty:
            all_cty = sorted(norm_cat(df_f[_cty]).unique().tolist())
            opt_cty = ["All"] + all_cty
            sel_cty = f1.multiselect("Filter Country", options=opt_cty, default=["All"], key="bp_cty")
        else:
            all_cty, sel_cty = [], ["All"]

        if _src:
            all_src = sorted(norm_cat(df_f[_src]).unique().tolist())
            opt_src = ["All"] + all_src
            sel_src = f2.multiselect("Filter JetLearn Deal Source", options=opt_src, default=["All"], key="bp_src")
        else:
            all_src, sel_src = [], ["All"]

        if _cns:
            all_cns = sorted(norm_cat(df_f[_cns]).unique().tolist())
            opt_cns = ["All"] + all_cns
            sel_cns = f3.multiselect("Filter Academic Counsellor", options=opt_cns, default=["All"], key="bp_cns")
        else:
            all_cns, sel_cns = [], ["All"]

        def _apply_multi_all(series, selected):
            if series is None or "All" in selected:
                return pd.Series(True, index=df_f.index)
            return norm_cat(series).isin(selected)

        mask_cty = _apply_multi_all(df_f[_cty] if _cty else None, sel_cty)
        mask_src = _apply_multi_all(df_f[_src] if _src else None, sel_src)
        mask_cns = _apply_multi_all(df_f[_cns] if _cns else None, sel_cns)
        filt_mask = mask_cty & mask_src & mask_cns

        # ---------- Normalize base series ----------
        C = coerce_datetime(df_f[_create]).dt.date
        P = coerce_datetime(df_f[_pay]).dt.date
        PT = norm_cat(df_f[_ptype])
        TERM = pd.to_numeric(df_f[_term], errors="coerce")  # numeric
        INST_raw = pd.to_numeric(df_f[_inst], errors="coerce") if _inst and _inst in df_f.columns else pd.Series(np.nan, index=df_f.index)
        ITYPE = norm_cat(df_f[_itype]) if _itype and _itype in df_f.columns else pd.Series("Unknown", index=df_f.index)

        def between_date(s, a, b):
            return s.notna() & (s >= a) & (s <= b)

        mask_created = between_date(C, range_start, range_end)
        mask_paid    = between_date(P, range_start, range_end)

        # Mode-aware inclusion mask for the window
        if mode == "MTD":
            win_mask = filt_mask & mask_created & mask_paid
        else:
            win_mask = filt_mask & mask_paid

        # ---------- Window DF + Sales Subscription with fallback (=1 when blank/0) ----------
        df_win = pd.DataFrame({
            "_create": C, "_pay": P,
            "Payment Type": PT,
            "Payment Term": TERM,
            "Installment Terms": INST_raw,
            "Installment Type": ITYPE,   # NEW
        }).loc[win_mask].copy()

        # If Installment Terms is NaN or <= 0, treat as 1
        inst_eff = np.where(
            (~df_win["Installment Terms"].isna()) & (df_win["Installment Terms"] > 0),
            df_win["Installment Terms"],
            1.0
        )
        df_win["Sales Subscription"] = np.where(
            (df_win["Payment Term"].notna()) & (df_win["Payment Term"] > 0),
            df_win["Payment Term"] / inst_eff,
            np.nan
        )

        # ---------- KPI strip ----------
        avg_term = float(df_win["Payment Term"].mean()) if df_win["Payment Term"].notna().any() else np.nan
        med_term = float(df_win["Payment Term"].median()) if df_win["Payment Term"].notna().any() else np.nan
        n_payments = int(len(df_win))
        top_type = (df_win["Payment Type"].value_counts().idxmax()
                    if not df_win.empty and df_win["Payment Type"].notna().any() else "—")

        st.markdown(
            """
            <style>
              .kpi-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 12px; background: #ffffff; }
              .kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }
              .kpi-value { font-size: 1.4rem; font-weight: 700; }
              .kpi-sub { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            a_txt = "–" if np.isnan(avg_term) else f"{avg_term:.2f}"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Avg Payment Term</div>"
                f"<div class='kpi-value'>{a_txt}</div>"
                f"<div class='kpi-sub'>{range_start} → {range_end}</div></div>", unsafe_allow_html=True)
        with c2:
            m_txt = "–" if np.isnan(med_term) else f"{med_term:.1f}"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Median Payment Term</div>"
                f"<div class='kpi-value'>{m_txt}</div>"
                f"<div class='kpi-sub'>Window (Mode: {mode})</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Payments in Window</div>"
                f"<div class='kpi-value'>{n_payments:,}</div>"
                f"<div class='kpi-sub'>Mode: {mode}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Top Payment Type</div>"
                f"<div class='kpi-value'>{top_type}</div>"
                f"<div class='kpi-sub'>By count in window</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------- MoM Helpers ----------
        C_m = coerce_datetime(df_f[_create]).dt.to_period("M")
        P_m = coerce_datetime(df_f[_pay]).dt.to_period("M")

        def month_mask(period_m):
            if period_m is pd.NaT:
                return pd.Series(False, index=df_f.index)
            if mode == "MTD":
                return (P_m == period_m) & (C_m == period_m) & filt_mask
            else:
                return (P_m == period_m) & filt_mask

        # =============================
        # Tabs
        # =============================
        tabs = st.tabs(["Payment Term Dynamics", "Payment Type Dynamics", "Installment Type Dynamics", "Term × Type (Correlation)"])

        # ---- 1) Payment Term Dynamics (includes Sales Subscription) ----
        with tabs[0]:
            sub_view = st.radio("View as", ["Graph", "Table"], horizontal=True, key="bp_term_view")

            # MoM Mean Term
            rows = []
            for pm in months:
                msk = month_mask(pm)
                term_mean = pd.to_numeric(df_f.loc[msk, _term], errors="coerce").mean()
                rows.append({"Month": str(pm), "AvgTerm": float(term_mean) if not np.isnan(term_mean) else np.nan,
                             "Count": int(msk.sum())})
            term_mom = pd.DataFrame(rows)

            # Window distributions
            dist = df_win[["Payment Term"]].copy()
            dist = dist[dist["Payment Term"].notna()]

            if sub_view == "Graph":
                if not term_mom.empty:
                    ch_line = (
                        alt.Chart(term_mom)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Month:N", sort=months_list, title="Month"),
                            y=alt.Y("AvgTerm:Q", title="Avg Payment Term"),
                            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("AvgTerm:Q", format=".2f"), alt.Tooltip("Count:Q")]
                        )
                        .properties(height=300, title="MoM — Average Payment Term")
                    )
                    st.altair_chart(ch_line, use_container_width=True)
                else:
                    st.info("No data for MoM Average Payment Term in the selected horizon.")

                if not dist.empty:
                    ch_hist = (
                        alt.Chart(dist)
                        .mark_bar(opacity=0.9)
                        .encode(
                            x=alt.X("Payment Term:Q", bin=alt.Bin(maxbins=30), title="Payment Term"),
                            y=alt.Y("count():Q", title="Count"),
                            tooltip=[alt.Tooltip("count():Q", title="Count")]
                        )
                        .properties(height=260, title="Window Distribution — Payment Term")
                    )
                    st.altair_chart(ch_hist, use_container_width=True)
                else:
                    st.info("No Payment Term values in the current window to plot a distribution.")

                # ===== Sales Subscription dynamics (MoM + histogram) with fallback (=1) =====
                st.markdown("#### Sales Subscription (Payment Term ÷ Installment Terms)")
                has_inst_col = _inst and _inst in df_f.columns
                if has_inst_col:
                    sales_rows = []
                    term_series_all = pd.to_numeric(df_f[_term], errors="coerce")
                    inst_series_all = pd.to_numeric(df_f[_inst], errors="coerce")
                    for pm in months:
                        msk = month_mask(pm)
                        term_m = term_series_all[msk]
                        inst_m = inst_series_all[msk]
                        # fallback: if inst is NaN or <= 0, use 1
                        inst_eff_m = np.where((~inst_m.isna()) & (inst_m > 0), inst_m, 1.0)
                        valid = (~term_m.isna()) & (term_m > 0)
                        ss = (term_m[valid] / inst_eff_m[valid]).mean() if valid.any() else np.nan
                        sales_rows.append({"Month": str(pm), "AvgSalesSubscription": float(ss) if not np.isnan(ss) else np.nan,
                                           "Count": int(valid.sum())})
                    ss_mom = pd.DataFrame(sales_rows)
                    if not ss_mom.empty:
                        ch_ss = (
                            alt.Chart(ss_mom)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Month:N", sort=months_list, title="Month"),
                                y=alt.Y("AvgSalesSubscription:Q", title="Avg Sales Subscription"),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("AvgSalesSubscription:Q", format=".2f"), alt.Tooltip("Count:Q")]
                            )
                            .properties(height=300, title="MoM — Average Sales Subscription (fallback: Installment Terms=1 if blank/0)")
                        )
                        st.altair_chart(ch_ss, use_container_width=True)
                    # Window histogram
                    win_ss = df_win["Sales Subscription"].dropna()
                    if not win_ss.empty:
                        ch_ss_hist = (
                            alt.Chart(pd.DataFrame({"Sales Subscription": win_ss}))
                            .mark_bar(opacity=0.9)
                            .encode(
                                x=alt.X("Sales Subscription:Q", bin=alt.Bin(maxbins=30)),
                                y=alt.Y("count():Q"),
                                tooltip=[alt.Tooltip("count():Q", title="Count")]
                            )
                            .properties(height=260, title="Window Distribution — Sales Subscription")
                        )
                        st.altair_chart(ch_ss_hist, use_container_width=True)
                else:
                    st.info("‘Installment Terms’ column not found — Sales Subscription is unavailable for dynamics.", icon="ℹ️")

            else:
                st.dataframe(term_mom, use_container_width=True)
                st.download_button(
                    "Download CSV — MoM Average Payment Term",
                    term_mom.to_csv(index=False).encode("utf-8"),
                    "buying_propensity_term_mom.csv", "text/csv",
                    key="bp_dl_term_mom"
                )
                if not dist.empty:
                    st.dataframe(dist.rename(columns={"Payment Term":"Payment Term (window)"}).head(1000), use_container_width=True)

                # Sales Subscription table (MoM) with fallback (=1)
                has_inst_col = _inst and _inst in df_f.columns
                if has_inst_col:
                    sales_rows = []
                    term_series_all = pd.to_numeric(df_f[_term], errors="coerce")
                    inst_series_all = pd.to_numeric(df_f[_inst], errors="coerce")
                    for pm in months:
                        msk = month_mask(pm)
                        term_m = term_series_all[msk]
                        inst_m = inst_series_all[msk]
                        inst_eff_m = np.where((~inst_m.isna()) & (inst_m > 0), inst_m, 1.0)
                        valid = (~term_m.isna()) & (term_m > 0)
                        ss = (term_m[valid] / inst_eff_m[valid]).mean() if valid.any() else np.nan
                        sales_rows.append({"Month": str(pm), "AvgSalesSubscription": float(ss) if not np.isnan(ss) else np.nan,
                                           "ValidRows": int(valid.sum())})
                    ss_mom = pd.DataFrame(sales_rows)
                    st.dataframe(ss_mom, use_container_width=True)
                    st.download_button(
                        "Download CSV — MoM Sales Subscription",
                        ss_mom.to_csv(index=False).encode("utf-8"),
                        "buying_propensity_sales_subscription_mom.csv", "text/csv",
                        key="bp_dl_ss_mom"
                    )

        # ---- 2) Payment Type Dynamics ----
        with tabs[1]:
            type_view = st.radio("View as", ["Graph", "Table"], horizontal=True, key="bp_type_view")
            pct_mode  = st.checkbox("Show % share per month", value=False, key="bp_type_pct")

            type_rows = []
            for pm in months:
                msk = month_mask(pm)
                if msk.any():
                    tmp = norm_cat(df_f.loc[msk, _ptype]).value_counts(dropna=False).rename_axis("Payment Type").rename("Count").reset_index()
                    tmp["Month"] = str(pm)
                    type_rows.append(tmp)
            if type_rows:
                type_mom = pd.concat(type_rows, ignore_index=True)
            else:
                type_mom = pd.DataFrame(columns=["Payment Type","Count","Month"])

            if type_view == "Graph":
                if type_mom.empty:
                    st.info("No data for Payment Type MoM dynamics.")
                else:
                    if pct_mode:
                        pct_df = type_mom.copy()
                        month_tot = pct_df.groupby("Month")["Count"].transform("sum")
                        pct_df["Pct"] = np.where(month_tot > 0, pct_df["Count"] / month_tot * 100.0, 0.0)
                        ch_type = (
                            alt.Chart(pct_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Month:N", sort=months_list),
                                y=alt.Y("Pct:Q", title="% of month", scale=alt.Scale(domain=[0,100])),
                                color=alt.Color("Payment Type:N", legend=alt.Legend(orient="bottom")),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Payment Type:N"), alt.Tooltip("Pct:Q", format=".1f")]
                            )
                            .properties(height=340, title="MoM — Payment Type % Share")
                        )
                    else:
                        ch_type = (
                            alt.Chart(type_mom)
                            .mark_bar()
                            .encode(
                                x=alt.X("Month:N", sort=months_list),
                                y=alt.Y("Count:Q", title="Count"),
                                color=alt.Color("Payment Type:N", legend=alt.Legend(orient="bottom")),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Payment Type:N"), alt.Tooltip("Count:Q")]
                            )
                            .properties(height=340, title="MoM — Payment Type Counts")
                        )
                    st.altair_chart(ch_type, use_container_width=True)
            else:
                if type_mom.empty:
                    st.info("No data for Payment Type MoM dynamics.")
                else:
                    wide = (type_mom
                            .pivot(index="Month", columns="Payment Type", values="Count")
                            .reindex(index=months_list)
                            .fillna(0.0)
                            .reset_index())
                    st.dataframe(wide, use_container_width=True)
                    st.download_button(
                        "Download CSV — MoM Payment Type",
                        wide.to_csv(index=False).encode("utf-8"),
                        "buying_propensity_type_mom.csv", "text/csv",
                        key="bp_dl_type_mom"
                    )

        # ---- 3) Installment Type Dynamics (NEW; mirrors Payment Type Dynamics) ----
        with tabs[2]:
            if not (_itype and _itype in df_f.columns):
                st.info("‘Installment Type’ column not found — this section is unavailable.", icon="ℹ️")
            else:
                itype_view = st.radio("View as", ["Graph", "Table"], horizontal=True, key="bp_itype_view")
                itype_pct  = st.checkbox("Show % share per month", value=False, key="bp_itype_pct")

                itype_rows = []
                for pm in months:
                    msk = month_mask(pm)
                    if msk.any():
                        tmp = norm_cat(df_f.loc[msk, _itype]).value_counts(dropna=False).rename_axis("Installment Type").rename("Count").reset_index()
                        tmp["Month"] = str(pm)
                        itype_rows.append(tmp)
                if itype_rows:
                    itype_mom = pd.concat(itype_rows, ignore_index=True)
                else:
                    itype_mom = pd.DataFrame(columns=["Installment Type","Count","Month"])

                if itype_view == "Graph":
                    if itype_mom.empty:
                        st.info("No data for Installment Type MoM dynamics.")
                    else:
                        if itype_pct:
                            pct_df = itype_mom.copy()
                            month_tot = pct_df.groupby("Month")["Count"].transform("sum")
                            pct_df["Pct"] = np.where(month_tot > 0, pct_df["Count"] / month_tot * 100.0, 0.0)
                            ch_itype = (
                                alt.Chart(pct_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Month:N", sort=months_list),
                                    y=alt.Y("Pct:Q", title="% of month", scale=alt.Scale(domain=[0,100])),
                                    color=alt.Color("Installment Type:N", legend=alt.Legend(orient="bottom")),
                                    tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Installment Type:N"), alt.Tooltip("Pct:Q", format=".1f")]
                                )
                                .properties(height=340, title="MoM — Installment Type % Share")
                            )
                        else:
                            ch_itype = (
                                alt.Chart(itype_mom)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Month:N", sort=months_list),
                                    y=alt.Y("Count:Q", title="Count"),
                                    color=alt.Color("Installment Type:N", legend=alt.Legend(orient="bottom")),
                                    tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Installment Type:N"), alt.Tooltip("Count:Q")]
                                )
                                .properties(height=340, title="MoM — Installment Type Counts")
                            )
                        st.altair_chart(ch_itype, use_container_width=True)
                else:
                    if itype_mom.empty:
                        st.info("No data for Installment Type MoM dynamics.")
                    else:
                        wide_i = (itype_mom
                                  .pivot(index="Month", columns="Installment Type", values="Count")
                                  .reindex(index=months_list)
                                  .fillna(0.0)
                                  .reset_index())
                        st.dataframe(wide_i, use_container_width=True)
                        st.download_button(
                            "Download CSV — MoM Installment Type",
                            wide_i.to_csv(index=False).encode("utf-8"),
                            "buying_propensity_installment_type_mom.csv", "text/csv",
                            key="bp_dl_itype_mom"
                        )

        # ---- 4) Term × Type (Correlation-ish view) ----
        with tabs[3]:
            metric_pick = st.radio(
                "Metric",
                ["Payment Term (mean ± std)", "Sales Subscription (mean ± std)"],
                index=0, horizontal=True, key="bp_corr_metric",
                help="Switch to ‘Sales Subscription’ (fallback: Installment Terms=1 if blank/0) to analyze Term ÷ Installment Terms by group."
            )
            # NEW: choose grouping dimension (Payment Type OR Installment Type)
            group_dim = st.radio(
                "Group by",
                ["Payment Type", "Installment Type"],
                index=0,
                horizontal=True,
                key="bp_corr_groupdim"
            )
            corr_view = st.radio("View as", ["Graph", "Table"], horizontal=True, key="bp_corr_view")

            # Build corr_df with selected group dimension
            gcol = "Payment Type" if group_dim == "Payment Type" else "Installment Type"
            corr_df = df_win[[gcol,"Payment Term","Sales Subscription"]].copy()
            if metric_pick.startswith("Payment Term"):
                corr_df = corr_df.dropna(subset=[gcol,"Payment Term"])
                value_col = "Payment Term"
                ttl = f"Window — {value_col} by {gcol} (mean ± std)"
            else:
                corr_df = corr_df.dropna(subset=[gcol,"Sales Subscription"])
                value_col = "Sales Subscription"
                ttl = f"Window — {value_col} by {gcol} (mean ± std)"

            if corr_df.empty:
                st.info("No rows in the current window for the selected metric/group.")
            else:
                summary = (
                    corr_df.groupby(gcol, as_index=False)
                           .agg(Count=(value_col,"size"),
                                Mean=(value_col,"mean"),
                                Median=(value_col,"median"),
                                Std=(value_col,"std"))
                )
                summary["Mean"]   = summary["Mean"].round(2)
                summary["Median"] = summary["Median"].round(2)
                summary["Std"]    = summary["Std"].fillna(0.0).round(2)

                if corr_view == "Graph":
                    mean_err = summary.copy()
                    mean_err["Low"]  = (mean_err["Mean"] - mean_err["Std"]).clip(lower=0)
                    mean_err["High"] =  mean_err["Mean"] + mean_err["Std"]

                    base = alt.Chart(mean_err).encode(
                        x=alt.X(f"{gcol}:N", sort=summary.sort_values("Mean")[gcol].tolist())
                    )
                    error = base.mark_errorbar().encode(y=alt.Y("Low:Q", title=value_col), y2="High:Q")
                    bars  = base.mark_bar().encode(
                        y="Mean:Q",
                        tooltip=[
                            alt.Tooltip(f"{gcol}:N"),
                            alt.Tooltip("Count:Q"),
                            alt.Tooltip("Mean:Q", format=".2f"),
                            alt.Tooltip("Median:Q", format=".2f"),
                            alt.Tooltip("Std:Q", format=".2f"),
                        ],
                    ).properties(height=360, title=ttl)
                    st.altair_chart(error + bars, use_container_width=True)

                    show_scatter = st.checkbox("Show raw points (jitter)", value=False, key="bp_corr_scatter")
                    if show_scatter:
                        pts = (
                            alt.Chart(corr_df)
                            .mark_circle(opacity=0.35, size=40)
                            .encode(
                                x=alt.X(f"{gcol}:N"),
                                y=alt.Y(f"{value_col}:Q"),
                                tooltip=[alt.Tooltip(f"{gcol}:N"), alt.Tooltip(f"{value_col}:Q")]
                            )
                            .properties(height=320, title=f"Points — {value_col} by {gcol}")
                        )
                        st.altair_chart(pts, use_container_width=True)
                else:
                    st.dataframe(summary.sort_values(["Mean","Count"], ascending=[False, False]), use_container_width=True)
                    st.download_button(
                        "Download CSV — Summary",
                        summary.to_csv(index=False).encode("utf-8"),
                        "buying_propensity_correlation_summary.csv", "text/csv",
                        key="bp_dl_corr_summary"
                    )

    # run the tab
    _buying_propensity_tab()

# =========================
# Cash-in Tab (live Google Sheet range A2:D13, fixed header & footer)
# =========================
# =========================
# Cash-in Tab (Google Sheet A2:D13, header from first row in range, last 'Team' row fixed)
# =========================
# =========================
# Cash-in (GS -> A2:C12 table + separate A13:D13 total row)
# =========================
elif view == "Cash-in":
    # --- Right-side drawer: Cash-in resources (only on Cash-in) ---
    st.markdown(
        """
        <style>
        #cashin-drawer-toggle { display:none; }
        .cashin-drawer-button {
            position: fixed; right: 14px; top: 120px; z-index: 9998;
            background: #1D4ED8; color:#fff; border:1px solid #1E40AF;
            padding: 8px 10px; border-radius: 999px; font-weight:600; font-size:12.5px;
            box-shadow: 0 6px 18px rgba(29,78,216,0.35); cursor:pointer;
        }
        .cashin-drawer { 
            position: fixed; right: 14px; top: 164px; width: 340px; max-width: 85vw;
            background:#fff; border:1px solid #e7e8ea; border-radius: 14px; 
            box-shadow: 0 14px 38px rgba(2,6,23,0.20); padding: 12px; z-index: 9999; display:none;
        }
        #cashin-drawer-toggle:checked ~ .cashin-drawer { display:block; }
        .cashin-close { position:absolute; top:8px; right:10px; font-weight:700; color:#334155; cursor:pointer; }
        .cashin-title { font-weight:700; color:#0f172a; margin:0 0 6px 0; }
        </style>
        <input type="checkbox" id="cashin-drawer-toggle"/>
        <label for="cashin-drawer-toggle" class="cashin-drawer-button">Cash-in Docs</label>
        <div class="cashin-drawer">
          <label for="cashin-drawer-toggle" class="cashin-close" title="Close">×</label>
          <div class="cashin-title">Cash-in — Google Sheet snapshot</div>
          <a href="https://docs.google.com/spreadsheets/d/1tw6gTaUEycAD5DJjw5ASSdF-WwYEt2TqcMb2lTKtKps/edit?gid=0#gid=0" target="_blank" rel="noopener">Open the Google Sheet</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    import re
    import pandas as pd
    import streamlit as st

    st.subheader("Cash-in — Google Sheet snapshot")

    # --- Your Google Sheet URL (you can change it in the UI) ---
    default_gsheet_url = "https://docs.google.com/spreadsheets/d/1tw6gTaUEycAD5DJjw5ASSdF-WwYEt2TqcMb2lTKtKps/edit?gid=0#gid=0"
    sheet_url = st.text_input(
        "Google Sheet URL",
        value=default_gsheet_url,
        help="Reads live CSV export of the given sheet (keeps the gid).",
        key="cashin_sheet_url",
    )

    # --- Helper: convert a normal GSheet URL to a CSV export URL (preserves gid) ---
    def gsheet_to_csv_url(url: str) -> str | None:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        gid_match = re.search(r"[?#]gid=(\d+)", url)
        if not m:
            return None
        sheet_id = m.group(1)
        gid = gid_match.group(1) if gid_match else "0"
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    csv_url = gsheet_to_csv_url(sheet_url)
    if not csv_url:
        st.error("Invalid Google Sheet URL. Please paste a standard Sheets link (that contains `/spreadsheets/d/<id>`).")
        st.stop()

    @st.cache_data(ttl=90, show_spinner=False)
    def load_gsheet_csv(url: str) -> pd.DataFrame:
        df = pd.read_csv(url, header=0)  # A1 header row
        # Keep only the first 4 columns (A..D) if sheet has extras
        if df.shape[1] > 4:
            df = df.iloc[:, :4]
        # Drop fully empty rows
        df = df.dropna(how="all")
        # Strip header names
        df.columns = [str(c).strip() for c in df.columns]
        return df

    try:
        df_raw = load_gsheet_csv(csv_url)
    except Exception as e:
        st.error(f"Could not load the Google Sheet CSV. Details: {e}")
        st.stop()

    if df_raw.empty:
        st.info("The sheet appears to be empty.")
        st.stop()

    # We expect:
    #  - A1..D1  = headers
    #  - A2..C12 = main table (11 rows max after header)
    #  - A13..D13 = TOTAL row (separate, not part of the table)
    #
    # Robust: Prefer a row whose first non-empty cell equals 'Team' (case-insensitive) as TOTAL.
    # If not found, use the last non-empty row as TOTAL.
    df_no_header = df_raw.iloc[1:].copy()          # rows starting at A2
    df_no_header = df_no_header.reset_index(drop=True)

    # Identify a 'Team' total row if present
    total_row_idx = None
    if not df_no_header.empty:
        first_col_name = df_raw.columns[0]
        first_col = df_no_header[first_col_name].astype(str).str.strip().str.lower()
        team_hits = first_col[first_col.eq("team")]
        if not team_hits.empty:
            total_row_idx = team_hits.index[-1]     # use the last 'Team' as total

    # Fallback to the very last row (as total)
    if total_row_idx is None and not df_no_header.empty:
        total_row_idx = df_no_header.index[-1]

    # Main table: A2..C12 (i.e., up to 11 rows max) — but ensure we do NOT include total_row_idx
    # Build candidate table rows up to row 11 (0-based slice :11)
    table_slice = df_no_header.iloc[:11].copy() if len(df_no_header) > 0 else df_no_header.copy()

    # If the chosen total row is inside that slice, drop it from the table
    if total_row_idx is not None and total_row_idx < len(table_slice):
        table_slice = table_slice.drop(index=total_row_idx).copy()

    # Use only A..C for the table (first 3 columns)
    if table_slice.shape[1] >= 3:
        table_df = table_slice.iloc[:, :3].copy()
    else:
        # Pad if sheet has fewer than 3 cols (rare)
        table_df = table_slice.copy()

    # Separate total row (A13..D13): take identified total_row_idx from df_no_header, use A..D (first 4 columns)
    total_df = None
    if total_row_idx is not None and total_row_idx < len(df_no_header):
        total_row_series = df_no_header.iloc[total_row_idx, :4] if df_no_header.shape[1] >= 4 else df_no_header.iloc[total_row_idx, :]
        total_df = pd.DataFrame([total_row_series.values], columns=df_raw.columns[:len(total_row_series)])

    # ---- Show the table (A2..C12) ----
    st.markdown("#### Table (A2:C12)")
    if table_df.empty:
        st.info("No data rows found in A2:C12 (excluding any total row).")
    else:
        st.dataframe(table_df, use_container_width=True)
        st.download_button(
            "Download CSV (table A2:C12)",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name="cashin_table_A2_C12.csv",
            mime="text/csv",
            key="cashin_tbl_dl"
        )

    # ---- Show the separate Total row (A13:D13) below the table ----
    st.markdown("#### Total (A13:D13)")
    if total_df is None or total_df.empty:
        st.info("Total row not found. If your sheet has a 'Team' row, it will be shown here; otherwise the last non-empty row is used.")
    else:
        st.dataframe(total_df, use_container_width=True)
        st.download_button(
            "Download CSV (total A13:D13)",
            data=total_df.to_csv(index=False).encode("utf-8"),
            file_name="cashin_total_A13_D13.csv",
            mime="text/csv",
            key="cashin_total_dl"
        )

    # Optional: Quick link back to the source sheet
    with st.expander("Open the source Google Sheet"):
        st.link_button("Open Google Sheet", sheet_url)

# ---- Professional Font & Sizing (Global) ----
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        /* Import Inter with sensible fallbacks */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
          --app-font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                      Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, sans-serif;
          --text-xxs: 11.5px;
          --text-xs:  12.5px;
          --text-sm:  13.5px;
          --text-md:  14.5px;
          --text-lg:  16px;
          --text-xl:  18px;
          --text-2xl: 22px;
          --text-3xl: 26px;
        }

        html, body, [class^="css"], .block-container {
          font-family: var(--app-font) !important;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          font-size: var(--text-md) !important;
          line-height: 1.45;
          letter-spacing: 0.1px;
        }

        /* Headings */
        .stMarkdown h1, h1, .stTitle {
          font-family: var(--app-font) !important;
          font-weight: 700 !important;
          font-size: var(--text-3xl) !important;
          letter-spacing: 0.1px;
        }
        .stMarkdown h2, h2 {
          font-weight: 600 !important;
          font-size: var(--text-2xl) !important;
        }
        .stMarkdown h3, h3 {
          font-weight: 600 !important;
          font-size: var(--text-xl) !important;
        }
        .stMarkdown h4, h4 {
          font-weight: 600 !important;
          font-size: var(--text-lg) !important;
        }

        /* Body, captions, small text */
        .stMarkdown p, .stText, .stDataFrame, .stTable, .stCaption, .stCheckbox, .stRadio, .stSelectbox, .stTextInput, .stDateInput {
          font-size: var(--text-md) !important;
        }
        .stCaption, .stMarkdown small, .markdown-text-container p small {
          font-size: var(--text-xs) !important;
          opacity: .85;
        }
        .stMarkdown code, code, pre {
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
          font-size: var(--text-sm) !important;
        }

        /* Buttons & pill chips */
        .stButton > button, .stDownloadButton > button {
          font-family: var(--app-font) !important;
          font-weight: 600 !important;
          font-size: var(--text-md) !important;
        }
        /* Right-side active pill (rendered via Markdown) already styled; ensure text uses Inter */
        .stMarkdown div[style*="border-radius:999px"] { font-family: var(--app-font) !important; }

        /* Inputs */
        .stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox > div, [data-baseweb="select"] * {
          font-family: var(--app-font) !important;
          font-size: var(--text-md) !important;
        }

        /* Sidebar radios & labels */
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div[role="radiogroup"] {
          font-family: var(--app-font) !important;
          font-size: var(--text-md) !important;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
          font-family: var(--app-font) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
          font-size: var(--text-2xl) !important;
          font-weight: 700 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"], 
        div[data-testid="stMetric"] label {
          font-size: var(--text-xs) !important;
        }

        /* DataFrame grid cells */
        .stDataFrame [role="grid"] * {
          font-family: var(--app-font) !important;
          font-size: var(--text-sm) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception:
    pass




# ---- Brand Blue Accent (Selected states) ----
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        :root {
          --brand-blue: #1D4ED8; /* blue-600 */
          --brand-blue-dark: #1E40AF; /* blue-700 */
        }
        /* Selected chips/pills rendered as buttons (fallback outline -> filled) */
        .stButton > button[aria-pressed="true"] {
            background: var(--brand-blue) !important;
            color: #fff !important;
            border-color: var(--brand-blue) !important;
        }
        /* Any selected tab buttons */
        button[role="tab"][aria-selected="true"] {
            background: var(--brand-blue) !important;
            border-color: var(--brand-blue) !important;
            color: #fff !important;
        }
        /* Accents for active controls in sidebar */
        section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
            outline: 2px solid var(--brand-blue) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
except Exception:
    pass




# ---- Floating "Back to Top" control (CSS only) ----

def _render_back_to_top():
    import streamlit as st
    st.markdown(
        """
        <style>
        html { scroll-behavior: smooth; }
        #back-to-top {
            position: fixed;
            right: 14px;
            bottom: 14px;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            border-radius: 999px;
            background: #ffffff;             /* subtle, clean */
            color: #1D4ED8;                  /* brand blue icon */
            border: 1px solid #CBD5E1;       /* slate-300 */
            box-shadow: 0 6px 18px rgba(2, 6, 23, 0.10); /* soft shadow */
            font-weight: 700;
            font-size: 18px;
            z-index: 9999;
            transition: transform .08s ease, box-shadow .12s ease, opacity .15s ease, background .12s ease, color .12s ease, border-color .12s ease;
            opacity: 0.85;
        }
        #back-to-top:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.16);
            background: #1D4ED8;
            color: #ffffff;
            border-color: #1E40AF;
            opacity: 1.0;
        }
        #back-to-top:active {
            transform: translateY(0);
            box-shadow: 0 4px 12px rgba(2, 6, 23, 0.20);
        }
        @media (max-width: 640px) {
            #back-to-top {
                right: 10px;
                bottom: 10px;
                width: 32px;
                height: 32px;
                font-size: 16px;
            }
        }
        </style>
        <a href="#" id="back-to-top" aria-label="Back to top" title="Back to top">↑</a>
        """,
        unsafe_allow_html=True
    )


# ===== Pro UI & Layout Polish (logic unchanged) =====
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        .block-container { max-width: 1440px !important; padding-top: 1.0rem !important; padding-bottom: 2.0rem !important; }
        .ui-gap-xs { margin-top: 4px;  margin-bottom: 4px; }
        .ui-gap-sm { margin-top: 8px;  margin-bottom: 8px; }
        .ui-gap-md { margin-top: 14px; margin-bottom: 14px; }
        .ui-gap-lg { margin-top: 20px; margin-bottom: 20px; }

        .ui-card { border: 1px solid #e7e8ea; border-radius: 14px; background: #ffffff; box-shadow: 0 1px 6px rgba(16,24,40,.06); padding: 12px; }

        .stVegaLiteChart, .stAltairChart, .stPlotlyChart, .stDataFrame, .stTable, .element-container [data-baseweb="table"] {
            border: 1px solid #e7e8ea; border-radius: 14px; background: #fff; box-shadow: 0 1px 6px rgba(16,24,40,.06); padding: 10px;
        }
        .stDataFrame [role="grid"] { border-radius: 12px; overflow: hidden; border: 1px solid #eef0f2; }

        div[data-testid="stMetric"] { background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%); border: 1px solid #eef0f2; border-radius: 16px; padding: 12px 14px; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700; }

        button[role="tab"] { border-radius: 999px !important; padding: 8px 14px !important; margin-right: 6px !important; border: 1px solid #e7e8ea !important; }
        button[role="tab"][aria-selected="true"] { background: #1D4ED8 !important; color: #fff !important; border-color: #1E40AF !important; }

        .stTextInput > div, .stNumberInput > div, .stDateInput > div, div[data-baseweb="select"] { border-radius: 12px !important; box-shadow: 0 1px 4px rgba(16,24,40,.04); }

        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0f172a; letter-spacing: 0.1px; }
        .stMarkdown hr { margin: 14px 0; border: none; border-top: 1px dashed #e5e7eb; }
        .stCaption, .stMarkdown p small, .stMarkdown small { color:#64748B !important; }

        section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stMultiselect, section[data-testid="stSidebar"] details[data-testid="stExpander"] {
            margin-bottom: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass



# ---- Professional pill styling for sub-view buttons ----
try:
    import streamlit as st
    st.markdown(
        """
        <style>
        /* Make default buttons look like subtle chips */
        .stButton > button { }
        .stButton > button:hover {
            border-color: #1E40AF !important;       /* blue-700 */
            color: #1D4ED8 !important;              /* blue-600 */
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        }
        /* Active pill is rendered via Markdown; ensure consistent spacing/line-height */
        .stMarkdown div[style*="border-radius:999px"] {
            line-height: 1.1;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass




# ---- Live active pill (subtle motion) ----
try:
    import streamlit as st
    st.markdown(
        """
<style>
.pill-live {
    box-shadow: 0 6px 20px rgba(29,78,216,0.18);
    transform: translateZ(0);
}
.pill-live .pill-dot {
    position: absolute;
    left: 10px;
    top: 50%;
    width: 6px;
    height: 6px;
    margin-top: -3px;
    border-radius: 999px;
    background: #93C5FD;            /* blue-300 */
    box-shadow: 0 0 0 0 rgba(29,78,216,0.55);
    animation: pill-pulse 2.2s ease-out infinite;
}
.pill-live .pill-sheen {
    position: absolute;
    top: 0; left: 0;
    height: 100%; width: 40%;
    pointer-events: none;
    background: linear-gradient(120deg, rgba(255,255,255,0) 0%, rgba(255,255,255,.18) 48%, rgba(255,255,255,0) 100%);
    transform: translateX(-120%);
    animation: pill-sheen 3.0s linear infinite;
    opacity: .75;
}
@keyframes pill-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(29,78,216,0.55); }
    60%  { box-shadow: 0 0 0 10px rgba(29,78,216,0); }
    100% { box-shadow: 0 0 0 0 rgba(29,78,216,0); }
}
@keyframes pill-sheen {
    0%   { transform: translateX(-130%); }
    100% { transform: translateX(130%); }
}
.pill-live:hover {
    box-shadow: 0 10px 28px rgba(29,78,216,0.26);
}
</style>
        """,
        unsafe_allow_html=True
    )
except Exception:
    pass






# ---- Refresh button styles (scoped, circular, enforced) ----
try:
    import streamlit as st
    st.markdown("""
<style>
/* Scope only inside #refresh-ctl */
#refresh-ctl { display:flex; justify-content:flex-end; margin-top:-6px; margin-bottom:6px; }
#refresh-ctl .stButton > button { }
#refresh-ctl .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 14px 30px rgba(2,132,199,0.34) !important; }
#refresh-ctl .stButton > button:active { transform: translateY(0); box-shadow: 0 8px 16px rgba(2,132,199,0.28) !important; }
</style>
""", unsafe_allow_html=True)
except Exception:
    pass




# ===== Super Cool UI (visual-only; NO logic changes) =====
try:
    import streamlit as st
    st.markdown(
        """
<style>
/* ---------- Global page polish ---------- */
html { scroll-behavior: smooth; }
:root {
  --bg: #f7f8fb;
  --card: #ffffff;
  --ink-900: #0f172a;
  --ink-700: #334155;
  --ink-500: #64748B;
  --line: #e7e8ea;
  --shadow-sm: 0 1px 6px rgba(16,24,40,.06);
  --shadow-md: 0 8px 24px rgba(15,23,42,.08);
  --brand: #1D4ED8;
  --brand-600: #2563EB;
  --brand-700: #1E40AF;
  --accent: #0ea5e9;
  --accent-600: #0284c7;
}
/* Page container */
.block-container {
  max-width: 1440px !important;
  padding-top: 12px !important;
  padding-bottom: 28px !important;
}
/* Subtle radial wash behind content */
.block-container::before {
  content: "";
  position: fixed;
  inset: -20% -20% auto -20%;
  height: 38vh;
  background: radial-gradient(42% 60% at 20% 10%, rgba(37,99,235,0.12) 0%, rgba(37,99,235,0) 60%),
              radial-gradient(32% 40% at 80% 0%, rgba(14,165,233,0.14) 0%, rgba(14,165,233,0) 60%);
  z-index: -1;
  pointer-events: none;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4, h5 {
  color: var(--ink-900) !important;
  letter-spacing: .2px;
}
p, li, span, label, .stMarkdown, .stText, .stCaption {
  color: var(--ink-700);
}

/* ---------- Cards for charts/tables ---------- */
.stVegaLiteChart, .stAltairChart, .stPlotlyChart,
.stDataFrame, .stTable, .element-container [data-baseweb="table"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  box-shadow: var(--shadow-sm);
  padding: 10px;
}

/* ---------- DataFrames ---------- */
.stDataFrame [role="grid"] {
  border-radius: 12px;
  border: 1px solid #eef0f2;
  overflow: hidden;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
  border: 1px solid #eef0f2;
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: var(--shadow-sm);
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: #f9fbff;
  border-right: 1px solid #eef0f2;
}
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stMultiselect,
section[data-testid="stSidebar"] details[data-testid="stExpander"] {
  margin-bottom: 10px !important;
}

/* ---------- Pills (sub-view chips) ---------- */
button[role="tab"] {
  border-radius: 999px !important;
  padding: 8px 14px !important;
  margin-right: 6px !important;
  border: 1px solid var(--line) !important;
  background: #fff !important;
  color: var(--ink-900) !important;
  box-shadow: 0 1px 3px rgba(15,23,42,.04);
}
button[role="tab"][aria-selected="true"] {
  background: var(--brand) !important;
  color: #fff !important;
  border-color: var(--brand-700) !important;
  box-shadow: 0 8px 24px rgba(29,78,216,.22);
}

/* ---------- Status line ---------- */
.status-line, .stMarkdown p.small {
  color: var(--ink-500) !important;
}

/* ---------- Right-side Cash-in drawer polish ---------- */
.cashin-drawer-button {
  backdrop-filter: blur(4px);
}
.cashin-drawer {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow-md);
}
.cashin-title { font-weight: 800; color: var(--ink-900); }
.cashin-caption { color: var(--ink-500); }

/* ---------- Circular refresh (already scoped via #refresh-ctl) ---------- */
#refresh-ctl .stButton > button {
  width: 42px !important; height: 42px !important;
  min-width: 42px !important; max-width: 42px !important;
  min-height: 42px !important; max-height: 42px !important;
  border-radius: 999px !important;
  background: var(--accent) !important; color:#fff !important; border:1px solid var(--accent-600) !important;
  font-weight: 800 !important; font-size: 18px !important; line-height: 42px !important;
  text-align: center !important;
  box-shadow: 0 10px 22px rgba(2,132,199,0.28) !important;
}
#refresh-ctl .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 14px 30px rgba(2,132,199,0.34) !important; }
#refresh-ctl .stButton > button:active { transform: translateY(0); box-shadow: 0 8px 16px rgba(2,132,199,0.28) !important; }

/* ---------- Tiny utilities ---------- */
.ui-gap-xs { margin: 4px 0; } .ui-gap-sm { margin: 8px 0; }
.ui-gap-md { margin: 14px 0; } .ui-gap-lg { margin: 20px 0; }
</style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass



# ---- Brand accent for radios/checkboxes (UI only) ----
try:
    import streamlit as st
    st.markdown("""
<style>
:root { --primary-color: #1D4ED8 !important; }
input[type="radio"], input[type="checkbox"] { accent-color: #1D4ED8 !important; }
</style>
""", unsafe_allow_html=True)
except Exception:
    pass



# ===== Force-attach Daily Business view (UI-only; logic unchanged) =====
try:
    view  # ensure name exists
except NameError:
    view = None

if view == "Daily Business":
    import pandas as pd, numpy as np, altair as alt
    from datetime import date, timedelta

    st.subheader("Daily Business — Time & Mix Explorer (MTD / Cohort)")

    # ---------------------------
    # Local helpers relying on global dataset/funcs
    # ---------------------------
    def _pick(df, preferred, cands):
        if preferred and preferred in df.columns: return preferred
        for c in cands:
            if c in df.columns: return c
        return None

    _create = _pick(df_f, globals().get("create_col"),
                    ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _pick(df_f, globals().get("pay_col"),
                    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _first  = _pick(df_f, globals().get("first_cal_sched_col"),
                    ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
    _resch  = _pick(df_f, globals().get("cal_resched_col"),
                    ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
    _done   = _pick(df_f, globals().get("cal_done_col"),
                    ["Calibration Done Date","Cal Done Date","Calibration Completed"])

    _cns    = _pick(df_f, globals().get("counsellor_col"),
                    ["Academic Counsellor","Counsellor","Advisor"])
    _cty    = _pick(df_f, globals().get("country_col"),
                    ["Country","Student Country","Deal Country"])
    _src    = _pick(df_f, globals().get("source_col"),
                    ["JetLearn Deal Source","Deal Source","Source","Lead Source"])

    if not _create or not _pay:
        st.warning("This tab needs 'Create Date' and 'Payment Received Date' columns mapped.", icon="⚠️")
        st.stop()

    # ---------------------------
    # Controls
    # ---------------------------
    c0, c1, c2 = st.columns([1.0, 1.2, 1.2])
    with c0:
        mode = st.radio("Mode", ["MTD", "Cohort"], index=0, horizontal=True,
                        help=("MTD: count an event only if the deal was also created in the window. "
                              "Cohort: count by the event date regardless of create month."))
    with c1:
        scope = st.radio("Date scope (Create-date based window)", ["Today", "Yesterday", "This month", "Custom"],
                         index=2, horizontal=True)
    with c2:
        gran = st.radio("Time granularity (x-axis)", ["Day", "Week", "Month", "Year"], index=0, horizontal=True)

    today_d = date.today()
    if scope == "Today":
        range_start, range_end = today_d, today_d
    elif scope == "Yesterday":
        yd = today_d - timedelta(days=1)
        range_start, range_end = yd, yd
    elif scope == "This month":
        range_start, range_end = month_bounds(today_d)
    else:
        d1, d2 = st.columns(2)
        with d1:
            range_start = st.date_input("Start (inclusive)", value=today_d.replace(day=1))
        with d2:
            range_end   = st.date_input("End (inclusive)", value=month_bounds(today_d)[1])
        if range_end < range_start:
            st.error("End date cannot be before start date.")
            st.stop()

    st.caption(f"Window: **{range_start} → {range_end}** • Mode: **{mode}** • Granularity: **{gran}**")

    # Group-by & mapping (as before)
    label_to_col = {
        "None": None,
        "Academic Counsellor": "Counsellor",
        "Country": "Country",
        "JetLearn Deal Source": "JetLearn Deal Source",
    }
    gp1, gp2, gp3 = st.columns([1.2, 1.2, 1.2])
    with gp1:
        group_by_label = st.selectbox("Group by (color/series)",
                                      list(label_to_col.keys()), index=0)
        group_by_col = label_to_col[group_by_label]
    with gp2:
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area (stack)", "Bar + Line (overlay)"], index=3,
                                  help="Bar + Line overlays the 1st metric as bars and 2nd metric as a line (only when Group by is None).")
    with gp3:
        stack_opt = st.checkbox("Stack series (for Bar/Area)", value=True)

    METRIC_LABELS = [
        "Deals Created",
        "Enrolments",
        "First Cal Scheduled — Count",
        "Calibration Rescheduled — Count",
        "Calibration Done — Count",
        "Enrolments / Created %",
        "Enrolments / Cal Done %",
        "Cal Done / First Cal %",
        "First Cal / Created %",
    ]
    default_metrics = ["Deals Created", "Enrolments"]
    metrics_sel = st.multiselect("Metrics to plot", options=METRIC_LABELS,
                                 default=default_metrics, help="Pick 1–4. For Bar+Line, pick up to 2 for best clarity.")

    # Filters with "All"
    def _norm_cat(s):
        return s.fillna("Unknown").astype(str).str.strip()

    f1, f2, f3 = st.columns([1.2, 1.2, 1.2])
    if _cns:
        cns_all = sorted(_norm_cat(df_f[_cns]).unique().tolist())
        pick_cns = f1.multiselect("Filter Academic Counsellor", options=["All"] + cns_all, default=["All"])
    else:
        pick_cns = ["All"]
    if _cty:
        cty_all = sorted(_norm_cat(df_f[_cty]).unique().tolist())
        pick_cty = f2.multiselect("Filter Country", options=["All"] + cty_all, default=["All"])
    else:
        pick_cty = ["All"]
    if _src:
        src_all = sorted(_norm_cat(df_f[_src]).unique().tolist())
        pick_src = f3.multiselect("Filter JetLearn Deal Source", options=["All"] + src_all, default=["All"])
    else:
        pick_src = ["All"]

    # ---------------------------
    # Normalize base series
    # ---------------------------
    C = coerce_datetime(df_f[_create]).dt.date
    P = coerce_datetime(df_f[_pay]).dt.date
    F = coerce_datetime(df_f[_first]).dt.date if _first else None
    R = coerce_datetime(df_f[_resch]).dt.date if _resch else None
    D = coerce_datetime(df_f[_done]).dt.date  if _done  else None

    CNS = _norm_cat(df_f[_cns]) if _cns else pd.Series("Unknown", index=df_f.index)
    CTY = _norm_cat(df_f[_cty]) if _cty else pd.Series("Unknown", index=df_f.index)
    SRC = _norm_cat(df_f[_src]) if _src else pd.Series("Unknown", index=df_f.index)

    # Apply filters
    def _apply_all(series, picks):
        if (series is None) or ("All" in picks): return pd.Series(True, index=df_f.index)
        return _norm_cat(series).isin(picks)

    fmask = _apply_all(CNS, pick_cns) & _apply_all(CTY, pick_cty) & _apply_all(SRC, pick_src)

    # Window mask by Create-date (denominator window)
    def _between(s, a, b):
        return s.notna() & (s >= a) & (s <= b)

    m_created_win = _between(C, range_start, range_end)

    # Event-in-window masks (by their “own” dates)
    m_pay_win   = _between(P, range_start, range_end)
    m_first_win = _between(F, range_start, range_end) if F is not None else pd.Series(False, index=df_f.index)
    m_resc_win  = _between(R, range_start, range_end) if R is not None else pd.Series(False, index=df_f.index)
    m_done_win  = _between(D, range_start, range_end) if D is not None else pd.Series(False, index=df_f.index)

    # Mode logic for events
    if mode == "MTD":
        m_enrol_eff = m_pay_win   & m_created_win
        m_first_eff = m_first_win & m_created_win
        m_resc_eff  = m_resc_win  & m_created_win
        m_done_eff  = m_done_win  & m_created_win
    else:
        m_enrol_eff = m_pay_win
        m_first_eff = m_first_win
        m_resc_eff  = m_resc_win
        m_done_eff  = m_done_win

    # Focused dataset
    base = pd.DataFrame({
        "_C": C, "_P": P,
        "_F": F if F is not None else pd.Series(pd.NaT, index=df_f.index),
        "_R": R if R is not None else pd.Series(pd.NaT, index=df_f.index),
        "_D": D if D is not None else pd.Series(pd.NaT, index=df_f.index),
        "Counsellor": CNS, "Country": CTY, "JetLearn Deal Source": SRC,
    })
    base = base.loc[fmask].copy()

    # ---------------------------
    # Pretty bucketing (Key + Label)
    # ---------------------------
    def _bucket_key_label(series_date):
        s = pd.to_datetime(series_date, errors="coerce")
        if gran == "Day":
            key   = s.dt.date
            label = (s.dt.strftime("%b ") + s.dt.day.astype(str))
        elif gran == "Week":
            per   = s.dt.to_period("W")
            key   = per.apply(lambda p: p.start_time.date() if pd.notna(p) else pd.NaT)
            wkno  = s.dt.isocalendar().week.astype("Int64")
            label = "Wk " + wkno.astype(str)
        elif gran == "Month":
            per   = s.dt.to_period("M")
            key   = per.dt.to_timestamp().dt.date
            label = per.strftime("%b")
        else:
            per   = s.dt.to_period("Y")
            key   = per.dt.to_timestamp().dt.date
            label = per.strftime("%Y")
        return key, label

    # ---------------------------
    # Build per-event frames
    # ---------------------------
    def _frame(date_series, mask, name):
        if mask is None or not mask.any():
            return pd.DataFrame(columns=["BucketKey","BucketLabel","Group","Metric","Count"])
        df = base.loc[mask.loc[base.index]].copy()
        if df.empty:
            return pd.DataFrame(columns=["BucketKey","BucketLabel","Group","Metric","Count"])

        bk, bl = _bucket_key_label(date_series.loc[df.index])
        df["BucketKey"]   = bk
        df["BucketLabel"] = bl

        if group_by_col:
            if group_by_col not in df.columns:
                return pd.DataFrame(columns=["BucketKey","BucketLabel","Group","Metric","Count"])
            g = (df.groupby(["BucketKey","BucketLabel", group_by_col], dropna=False)
                   .size().rename("Count").reset_index())
            g["Group"] = g[group_by_col].astype(str)
        else:
            g = (df.groupby(["BucketKey","BucketLabel"], dropna=False)
                   .size().rename("Count").reset_index())
            g["Group"] = "All"

        g["Metric"] = name
        return g[["BucketKey","BucketLabel","Group","Metric","Count"]]

    created_df = _frame(base["_C"], m_created_win.loc[base.index], "Deals Created")
    enrol_df   = _frame(base["_P"], m_enrol_eff.loc[base.index],  "Enrolments")
    first_df   = _frame(base["_F"], m_first_eff.loc[base.index],  "First Cal Scheduled — Count")
    resc_df    = _frame(base["_R"], m_resc_eff.loc[base.index],   "Calibration Rescheduled — Count")
    done_df    = _frame(base["_D"], m_done_eff.loc[base.index],   "Calibration Done — Count")

    # Merge to compute derived ratios per (Bucket, Group)
    def _merge_counts(dfs):
        if not dfs: 
            return pd.DataFrame(columns=["BucketKey","BucketLabel","Group"])
        out = None
        for dfi in dfs:
            if dfi is None or dfi.empty: 
                continue
            piv = (dfi.pivot_table(index=["BucketKey","BucketLabel","Group"],
                                   columns="Metric", values="Count", aggfunc="sum")
                      .reset_index())
            out = piv if out is None else out.merge(piv, on=["BucketKey","BucketLabel","Group"], how="outer")
        if out is None:
            return pd.DataFrame(columns=["BucketKey","BucketLabel","Group"])
        out = out.fillna(0)
        # Ensure base columns exist
        for col in ["Deals Created","Enrolments","Calibration Done — Count","First Cal Scheduled — Count","Calibration Rescheduled — Count"]:
            if col not in out.columns: out[col] = 0
        # Derived %
        out["Enrolments / Created %"]  = np.where(out["Deals Created"] > 0, out["Enrolments"] / out["Deals Created"] * 100.0, np.nan)
        out["Enrolments / Cal Done %"] = np.where(out["Calibration Done — Count"] > 0, out["Enrolments"] / out["Calibration Done — Count"] * 100.0, np.nan)
        out["Cal Done / First Cal %"]  = np.where(out["First Cal Scheduled — Count"] > 0, out["Calibration Done — Count"] / out["First Cal Scheduled — Count"] * 100.0, np.nan)
        out["First Cal / Created %"]   = np.where(out["Deals Created"] > 0, out["First Cal Scheduled — Count"] / out["Deals Created"] * 100.0, np.nan)
        return out

    wide = _merge_counts([created_df, enrol_df, first_df, resc_df, done_df])
    if wide.empty:
        st.info("No data in the selected window/filters.")
        st.stop()

    wide = wide.sort_values("BucketKey")
    ordered_labels = wide.drop_duplicates(["BucketKey","BucketLabel"])["BucketLabel"].tolist()

    keep_cols = ["BucketKey","BucketLabel","Group"] + metrics_sel
    for k in metrics_sel:
        if k not in wide.columns:
            wide[k] = np.nan
    plot_df = wide[keep_cols].melt(id_vars=["BucketKey","BucketLabel","Group"], var_name="Metric", value_name="Value")

    base_enc = alt.Chart(plot_df)
    def _is_ratio(m): return m.endswith("%")

    overlay_possible = (chart_type == "Bar + Line (overlay)") and (group_by_col is None) and (len(metrics_sel) >= 1)
    if overlay_possible:
        m1 = metrics_sel[0]
        bars = (
            base_enc.transform_filter(alt.datum.Metric == m1)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("BucketLabel:N", title="Period", sort=ordered_labels),
                y=alt.Y("Value:Q", title=m1, axis=alt.Axis(format=".1f" if _is_ratio(m1) else "")),
                tooltip=[alt.Tooltip("BucketLabel:N", title="Period"),
                         alt.Tooltip("Metric:N"),
                         alt.Tooltip("Value:Q", format=".1f" if _is_ratio(m1) else "d")],
                color=alt.value("#A8C5FD")
            )
        )
        if len(metrics_sel) >= 2:
            m2 = metrics_sel[1]
            line = (
                base_enc.transform_filter(alt.datum.Metric == m2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("BucketLabel:N", title="Period", sort=ordered_labels),
                    y=alt.Y("Value:Q", title=m1, axis=alt.Axis(format=".1f" if _is_ratio(m1) else "")),
                    color=alt.value("#333333"),
                    tooltip=[alt.Tooltip("BucketLabel:N", title="Period"),
                             alt.Tooltip("Metric:N"),
                             alt.Tooltip("Value:Q", format=".1f" if _is_ratio(m2) else "d")],
                )
            )
            ch = bars + line
            ttl = f"{m1} (bar) + {m2} (line)"
        else:
            ch = bars
            ttl = f"{m1} (bar)"
        st.altair_chart(ch.properties(height=360, title=ttl), use_container_width=True)

    else:
        color_field = "Metric:N" if group_by_col is None else "Group:N"
        tooltip_common = [
            alt.Tooltip("BucketLabel:N", title="Period"),
            alt.Tooltip("Group:N", title=("Group" if group_by_col is None else group_by_label)),
            alt.Tooltip("Metric:N"),
        ]
        if chart_type == "Line":
            mark = base_enc.mark_line(point=True)
        elif chart_type == "Area (stack)":
            mark = base_enc.mark_area(opacity=0.85)
        else:
            mark = base_enc.mark_bar(opacity=0.85)

        ch = mark.encode(
            x=alt.X("BucketLabel:N", title="Period", sort=ordered_labels),
            y=alt.Y(
                "Value:Q",
                title="Value",
                stack=("normalize" if (chart_type!="Line" and stack_opt and group_by_col is not None and all(not _is_ratio(m) for m in metrics_sel)) else None)
            ),
            color=alt.Color(color_field, legend=alt.Legend(orient="bottom")),
            tooltip=tooltip_common + [alt.Tooltip("Value:Q", format=".1f")]
        ).properties(height=360, title=f"{' / '.join(metrics_sel)}")
        st.altair_chart(ch, use_container_width=True)

    st.markdown("---")

    # ---------------------------
    # Table + download
    # ---------------------------
    show = wide.copy()
    for k in ["Enrolments / Created %","Enrolments / Cal Done %","Cal Done / First Cal %","First Cal / Created %"]:
        if k in show.columns: show[k] = show[k].round(1)
    out_cols = ["BucketLabel","Group"] + [c for c in show.columns if c not in {"BucketKey","BucketLabel","Group"}]
    st.dataframe(show[out_cols].rename(columns={"BucketLabel":"Period"}), use_container_width=True)
    st.download_button(
        "Download CSV — Daily Business",
        data=show[out_cols].rename(columns={"BucketLabel":"Period"}).to_csv(index=False).encode("utf-8"),
        file_name="daily_business.csv",
        mime="text/csv",
        key="db_dl"
    )




# =========================
# Funnel Tab (fixed Cohort filtering for current month)
# =========================
if view == "Funnel":
    def _funnel_tab():
        st.subheader("Funnel — Leads → Trials → Enrolments (MTD / Cohort)")

        # ---------- Resolve columns defensively ----------
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, [
            "Create Date","Created Date","Deal Create Date","CreateDate","Created On"
        ])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, [
            "Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"
        ])
        _first  = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, [
            "First Calibration Scheduled Date","First Calibration","First Cal Scheduled"
        ])
        _resch  = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, [
            "Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"
        ])
        _done   = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, [
            "Calibration Done Date","Cal Done Date","Calibration Completed"
        ])
        _slot   = find_col(df_f, ["Calibration Slot (Deal)","Calibration Slot","Cal Slot (Deal)"])

        _cty    = country_col     if (country_col     in df_f.columns) else find_col(df_f, ["Country","Student Country","Deal Country"])
        _cns    = counsellor_col  if (counsellor_col  in df_f.columns) else find_col(df_f, ["Academic Counsellor","Counsellor","Advisor"])
        _src    = source_col      if (source_col      in df_f.columns) else find_col(df_f, ["JetLearn Deal Source","Deal Source","Source"])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Funnel needs 'Create Date' and 'Payment Received Date'. Please map them in the sidebar.", icon="⚠️")
            st.stop()

        # ---------- Controls ----------
        c0, c1, c2 = st.columns([1.0, 1.1, 1.1])
        with c0:
            mode = st.radio(
                "Mode", ["MTD", "Cohort"], index=0, horizontal=True, key="fn_mode",
                help=("MTD: count events only when the deal was also created in that same period.\n"
                      "Cohort: count events by their own date (creation can be anywhere).")
            )
        with c1:
            gran = st.radio("Granularity", ["Day", "Week", "Month", "Year"], index=2, horizontal=True, key="fn_gran")
        with c2:
            x_dim = st.selectbox("X-axis", ["Time","Country","Academic Counsellor","JetLearn Deal Source"], index=0, key="fn_xdim")

        # Create-date window (requested)
        from datetime import date
        today_d = date.today()
        d1, d2 = st.columns(2)
        with d1:
            create_start = st.date_input("Create Date — Start", value=today_d.replace(day=1), key="fn_cstart")
        with d2:
            create_end   = st.date_input("Create Date — End",   value=month_bounds(today_d)[1], key="fn_cend")
        if create_end < create_start:
            st.error("End date cannot be before start date.")
            st.stop()

        # Booking type options
        col_b1, col_b2, col_b3 = st.columns([1.0, 1.0, 1.2])
        with col_b1:
            booking_filter = st.selectbox("Booking filter", ["All", "Pre-Book only", "Sales-Book only"], index=0, key="fn_bkf",
                                          help="Pre-Book = 'Calibration Slot (Deal)' has a value; Sales-Book = it is blank.")
        with col_b2:
            stack_booking = st.checkbox("Stack by Booking Type in chart", value=False, key="fn_stack_bk")
        with col_b3:
            view_mode = st.radio("View as", ["Graph", "Table"], index=0, horizontal=True, key="fn_view")

        # Chart options
        col_ch1, col_ch2 = st.columns([1.0, 1.2])
        with col_ch1:
            chart_type = st.selectbox("Chart", ["Bar","Line","Area","Stacked Bar","Combo (Leads bar + Enrolments line)"], index=0, key="fn_chart")
        with col_ch2:
            metrics_pick = st.multiselect(
                "Metrics to show",
                ["Deals Created","Enrolments","First Cal Scheduled","Cal Rescheduled","Cal Done",
                 "Enrolments / Leads %","Enrolments / Cal Done %","Cal Done / First Cal %","First Cal / Leads %"],
                default=["Deals Created","Enrolments","First Cal Scheduled","Cal Done"],
                key="fn_metrics"
            )

        # ---------- Normalize / derived fields ----------
        import pandas as pd, numpy as np, altair as alt
        C = coerce_datetime(df_f[_create])
        P = coerce_datetime(df_f[_pay])
        F = coerce_datetime(df_f[_first]) if (_first and _first in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        R = coerce_datetime(df_f[_resch]) if (_resch and _resch in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        D = coerce_datetime(df_f[_done])  if (_done  and _done  in df_f.columns)  else pd.Series(pd.NaT, index=df_f.index)

        C_date = C.dt.date
        P_date = P.dt.date
        F_date = F.dt.date if F is not None else pd.Series(pd.NaT, index=df_f.index)
        R_date = R.dt.date if R is not None else pd.Series(pd.NaT, index=df_f.index)
        D_date = D.dt.date if D is not None else pd.Series(pd.NaT, index=df_f.index)

        # Booking type
        if _slot and _slot in df_f.columns:
            pre_book = df_f[_slot].astype(str).str.strip().replace({"nan":""}).ne("")
        else:
            pre_book = pd.Series(False, index=df_f.index)
        booking_type = np.where(pre_book, "Pre-Book", "Sales-Book")

        # Dimension columns
        def norm_cat(s): 
            return s.fillna("Unknown").astype(str).str.strip()
        X_country = norm_cat(df_f[_cty]) if _cty else pd.Series("Unknown", index=df_f.index)
        X_cns     = norm_cat(df_f[_cns]) if _cns else pd.Series("Unknown", index=df_f.index)
        X_src     = norm_cat(df_f[_src]) if _src else pd.Series("Unknown", index=df_f.index)

        # Helper: between for dates
        def _between(dates, a, b):
            return dates.notna() & (dates >= a) & (dates <= b)

        # Population by Create Date range (for MTD universe and for "Leads" metric)
        in_create_window = _between(C_date, create_start, create_end)

        # ---------- Period key based on granularity ----------
        def period_key(dt_series):
            dts = pd.to_datetime(dt_series, errors="coerce")
            if gran == "Day":
                return dts.dt.floor("D")
            if gran == "Week":
                # ISO week start (Mon)
                return (dts - pd.to_timedelta(dts.dt.weekday, unit="D")).dt.floor("D")
            if gran == "Month":
                return dts.dt.to_period("M").dt.to_timestamp()
            if gran == "Year":
                return dts.dt.to_period("Y").dt.to_timestamp()
            return dts.dt.to_period("M").dt.to_timestamp()

        # ---------- Build masks (FIX: Cohort uses each event's own date window) ----------
        per_create = period_key(C)
        per_pay    = period_key(P)
        per_first  = period_key(F)
        per_resch  = period_key(R)
        per_done   = period_key(D)

        same_period_pay   = (per_create == per_pay)
        same_period_first = (per_create == per_first)
        same_period_resch = (per_create == per_resch)
        same_period_done  = (per_create == per_done)

        # Masks for each event series
        m_created = in_create_window & C.notna()

        if mode == "MTD":
            # Event must exist, creation within window, and event in the SAME period as creation
            m_enrol = in_create_window & P.notna() & same_period_pay
            m_first = in_create_window & F.notna() & same_period_first
            m_resch = in_create_window & R.notna() & same_period_resch
            m_done  = in_create_window & D.notna() & same_period_done
        else:
            # COHORT FIX: count events by THEIR OWN DATE between start/end, regardless of create month
            m_enrol = _between(P_date, create_start, create_end) & P.notna()
            m_first = _between(F_date, create_start, create_end) & F.notna()
            m_resch = _between(R_date, create_start, create_end) & R.notna()
            m_done  = _between(D_date, create_start, create_end) & D.notna()

        # Apply booking filter
        if booking_filter == "Pre-Book only":
            mask_booking = pre_book
        elif booking_filter == "Sales-Book only":
            mask_booking = ~pre_book
        else:
            mask_booking = pd.Series(True, index=df_f.index)

        m_created &= mask_booking
        m_enrol   &= mask_booking
        m_first   &= mask_booking
        m_resch   &= mask_booking
        m_done    &= mask_booking

        # ---------- Select X group ----------
        if x_dim == "Time":
            X_label = "Period"
            X_series_created = per_create
            X_series_enrol   = per_pay
            X_series_first   = per_first
            X_series_resch   = per_resch
            X_series_done    = per_done
            group_fields = ["Period"]
        elif x_dim == "Country":
            X_label = "Country"
            X_base = X_country
            X_series_created = X_base
            X_series_enrol   = X_base
            X_series_first   = X_base
            X_series_resch   = X_base
            X_series_done    = X_base
            group_fields = ["Country"]
        elif x_dim == "Academic Counsellor":
            X_label = "Academic Counsellor"
            X_base = X_cns
            X_series_created = X_base
            X_series_enrol   = X_base
            X_series_first   = X_base
            X_series_resch   = X_base
            X_series_done    = X_base
            group_fields = ["Academic Counsellor"]
        else:
            X_label = "JetLearn Deal Source"
            X_base = X_src
            X_series_created = X_base
            X_series_enrol   = X_base
            X_series_first   = X_base
            X_series_resch   = X_base
            X_series_done    = X_base
            group_fields = ["JetLearn Deal Source"]

        # Optional color for booking type (if stacked by booking)
        color_field = "Booking Type" if stack_booking else None

        def _frame(mask, x_series, metric_name):
            import pandas as pd
            if not mask.any():
                cols = group_fields + ([color_field] if color_field else []) + [metric_name]
                return pd.DataFrame(columns=cols)
            df_tmp = pd.DataFrame({
                group_fields[0]: x_series[mask],
                "Booking Type": pd.Series(booking_type, index=df_f.index)[mask]
            })
            use_fields = group_fields + ([color_field] if color_field else [])
            out = (df_tmp.assign(_one=1)
                         .groupby(use_fields, dropna=False)["_one"]
                         .sum().rename(metric_name).reset_index())
            return out

        g_created = _frame(m_created, X_series_created, "Deals Created")
        g_enrol   = _frame(m_enrol,   X_series_enrol,   "Enrolments")
        g_first   = _frame(m_first,   X_series_first,   "First Cal Scheduled")
        g_resch   = _frame(m_resch,   X_series_resch,   "Cal Rescheduled")
        g_done    = _frame(m_done,    X_series_done,    "Cal Done")

        # Merge all
        def _merge(a, b):
            return a.merge(b, on=(group_fields + ([color_field] if color_field else [])), how="outer")
        grid = _merge(g_created, g_enrol)
        grid = _merge(grid, g_first)
        grid = _merge(grid, g_resch)
        grid = _merge(grid, g_done)

        for c in ["Deals Created","Enrolments","First Cal Scheduled","Cal Rescheduled","Cal Done"]:
            if c not in grid.columns: grid[c] = 0
        grid = grid.fillna(0)

        # Derived ratios
        grid["Enrolments / Leads %"]      = np.where(grid["Deals Created"]>0, grid["Enrolments"]/grid["Deals Created"]*100.0, np.nan)
        grid["Enrolments / Cal Done %"]   = np.where(grid["Cal Done"]>0,     grid["Enrolments"]/grid["Cal Done"]*100.0,     np.nan)
        grid["Cal Done / First Cal %"]    = np.where(grid["First Cal Scheduled"]>0, grid["Cal Done"]/grid["First Cal Scheduled"]*100.0, np.nan)
        grid["First Cal / Leads %"]       = np.where(grid["Deals Created"]>0, grid["First Cal Scheduled"]/grid["Deals Created"]*100.0, np.nan)

        # Sort keys
        import pandas as pd, altair as alt
        if x_dim == "Time":
            grid["__sort"] = pd.to_datetime(grid["Period"])
            grid = grid.sort_values("__sort").drop(columns="__sort")
        else:
            grid = grid.sort_values(group_fields)

        # ---------- Output ----------
        if view_mode == "Table":
            show_cols = group_fields + ([color_field] if color_field else []) + metrics_pick
            st.dataframe(grid[show_cols], use_container_width=True)
            st.download_button(
                "Download CSV — Funnel",
                grid[show_cols].to_csv(index=False).encode("utf-8"),
                "funnel_output.csv","text/csv",
                key="fn_dl"
            )
            return

        # ---- Graph mode ----
        if chart_type.startswith("Combo"):
            if x_dim != "Time":
                st.info("Combo chart is available only for Time on X-axis. Showing standard chart instead.")
            elif not set(["Deals Created","Enrolments"]).issubset(set(grid.columns)):
                st.info("Combo chart needs Deals Created and Enrolments. Showing standard chart instead.")
            else:
                g = grid.copy()
                base_enc = [
                    alt.X("Period:T", title="Period"),
                    alt.Tooltip("yearmonthdate(Period):T", title="Period")
                ]
                if color_field:
                    created_bar = (
                        alt.Chart(g)
                        .mark_bar(opacity=0.85)
                        .encode(*base_enc,
                                y=alt.Y("Deals Created:Q", title="Deals Created"),
                                column=alt.Column(f"{color_field}:N", title=color_field),
                                tooltip=base_enc + [alt.Tooltip("Deals Created:Q", format="d")])
                    )
                    enrol_line = (
                        alt.Chart(g)
                        .mark_line(point=True)
                        .encode(*base_enc,
                                y=alt.Y("Enrolments:Q", title="Enrolments"),
                                column=alt.Column(f"{color_field}:N"),
                                tooltip=base_enc + [alt.Tooltip("Enrolments:Q", format="d")])
                    )
                    st.altair_chart(created_bar & enrol_line, use_container_width=True)
                else:
                    created_bar = (
                        alt.Chart(g).mark_bar(opacity=0.85)
                        .encode(alt.X("Period:T", title="Period"),
                                alt.Y("Deals Created:Q", title="Deals Created"),
                                tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                         alt.Tooltip("Deals Created:Q", format="d")])
                    )
                    enrol_line = (
                        alt.Chart(g).mark_line(point=True)
                        .encode(alt.X("Period:T", title="Period"),
                                alt.Y("Enrolments:Q", title="Enrolments"),
                                tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                         alt.Tooltip("Enrolments:Q", format="d")])
                    )
                    st.altair_chart(created_bar + enrol_line, use_container_width=True)
                st.download_button(
                    "Download CSV — Combo data",
                    grid.to_csv(index=False).encode("utf-8"),
                    "funnel_combo_data.csv","text/csv",
                    key="fn_dl_combo"
                )
                return

        # General chart
        g = grid.copy()
        tidy = g.melt(
            id_vars=group_fields + ([color_field] if color_field else []),
            value_vars=metrics_pick,
            var_name="Metric",
            value_name="Value"
        )

        if x_dim == "Time":
            x_enc = alt.X("Period:T", title="Period")
            tooltip_x = alt.Tooltip("yearmonthdate(Period):T", title="Period")
        else:
            x_enc = alt.X(f"{group_fields[0]}:N", title=group_fields[0], sort=sorted(tidy[group_fields[0]].dropna().unique()))
            tooltip_x = alt.Tooltip(f"{group_fields[0]}:N", title=group_fields[0])

        any_pct = any(s.endswith("%") for s in metrics_pick)
        y_title = "Value (count)" if not any_pct else "Value"
        y_enc = alt.Y("Value:Q", title=y_title)

        import altair as alt
        if chart_type == "Stacked Bar":
            if color_field:
                ch = (
                    alt.Chart(tidy)
                    .mark_bar(opacity=0.9)
                    .encode(
                        x=x_enc, y=y_enc,
                        color=alt.Color(f"{color_field}:N", title="Booking Type"),
                        column=alt.Column("Metric:N", title="Metric"),
                        tooltip=[tooltip_x,
                                 alt.Tooltip(f"{(color_field or 'Metric')}:N"),
                                 alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Value:Q", format=".1f" if any_pct else "d")]
                    ).properties(height=320)
                )
            else:
                ch = (
                    alt.Chart(tidy).mark_bar(opacity=0.9)
                    .encode(
                        x=x_enc, y=y_enc,
                        color=alt.Color("Metric:N", title="Metric", legend=alt.Legend(orient="bottom")),
                        tooltip=[tooltip_x, alt.Tooltip("Metric:N"),
                                 alt.Tooltip("Value:Q", format=".1f" if any_pct else "d")]
                    ).properties(height=360)
                )
        else:
            mark = {"Bar":"bar","Line":"line","Area":"area"}.get(chart_type, "bar")
            if color_field:
                ch = (
                    alt.Chart(tidy)
                    .mark_line(point=True) if mark=="line" else
                    alt.Chart(tidy).mark_area(opacity=0.5) if mark=="area" else
                    alt.Chart(tidy).mark_bar(opacity=0.9)
                ).encode(
                    x=x_enc, y=y_enc,
                    color=alt.Color(f"{color_field}:N", title="Booking Type", legend=alt.Legend(orient="bottom")),
                    column=alt.Column("Metric:N", title="Metric"),
                    tooltip=[tooltip_x, alt.Tooltip(f"{color_field}:N"), alt.Tooltip("Metric:N"),
                             alt.Tooltip("Value:Q", format=".1f" if any_pct else "d")]
                ).properties(height=320)
            else:
                ch = (
                    alt.Chart(tidy)
                    .mark_line(point=True) if mark=="line" else
                    alt.Chart(tidy).mark_area(opacity=0.5) if mark=="area" else
                    alt.Chart(tidy).mark_bar(opacity=0.9)
                ).encode(
                    x=x_enc, y=y_enc,
                    color=alt.Color("Metric:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[tooltip_x, alt.Tooltip("Metric:N"),
                             alt.Tooltip("Value:Q", format=".1f" if any_pct else "d")]
                ).properties(height=360)

        st.altair_chart(ch, use_container_width=True)

        # Download underlying data
        st.download_button(
            "Download CSV — Funnel (chart data)",
            tidy.to_csv(index=False).encode("utf-8"),
            "funnel_chart_data.csv","text/csv",
            key="fn_dl_chartdata"
        )

    # run it
    _funnel_tab()




# =========================
# Business Projection (no sklearn; excludes current month from training; adds MTD vs Projected chart)
# =========================
if view == "Business Projection":
    def _business_projection_tab():
        import pandas as pd, numpy as np
        from datetime import date
        from calendar import monthrange
        import altair as alt

        st.subheader("Business Projection — Monthly Enrolment Forecast (model selection & Accuracy %, no sklearn)")

        # ---------- Resolve columns ----------
        def _pick(df, preferred, cands):
            if preferred and preferred in df.columns: return preferred
            for c in cands:
                if c in df.columns: return c
            return None

        _pay = _pick(df_f, globals().get("pay_col"),
                     ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
        _cty = _pick(df_f, globals().get("country_col"),
                     ["Country","Student Country","Deal Country"])
        _src = _pick(df_f, globals().get("source_col"),
                     ["JetLearn Deal Source","Deal Source","Source","Lead Source","_src_raw"])

        if not _pay:
            st.warning("This tile needs a ‘Payment Received Date’ column to count enrolments. Please map it.", icon="⚠️")
            st.stop()

        # ---------- Controls ----------
        c1, c2, c3 = st.columns([1.1, 1.1, 1.5])
        with c1:
            lookback = st.selectbox("Backtest window (months, exclude current)", [6, 9, 12, 18, 24], index=2)
        with c2:
            max_lag = st.selectbox("Max lag (months)", [3, 6, 9, 12], index=2)
        with c3:
            model_name = st.selectbox(
                "Model",
                [
                    "Ridge (NumPy, lags + seasonality)",
                    "Holt-Winters Additive (s=12)",
                    "Naive Seasonal (month mean)"
                ],
                index=0
            )

        # Target month to predict
        t1, t2 = st.columns([1, 2])
        with t1:
            target_month = st.date_input("Forecast month (use 1st of month)", value=date.today().replace(day=1), key="bp_target")
            if isinstance(target_month, tuple): target_month = target_month[0]
        with t2:
            show_components = st.multiselect(
                "Show on chart",
                ["History", "Backtest (pred vs. actual)", "Forecast", "Current Month (Actual MTD)"],
                default=["History","Forecast","Current Month (Actual MTD)"]
            )

        # ---------- Build monthly target series (counts) ----------
        dfp = df_f.copy()
        dfp["_P"] = pd.to_datetime(dfp[_pay], errors="coerce", dayfirst=True)
        dfp = dfp[dfp["_P"].notna()].copy()
        dfp["_PM"] = dfp["_P"].dt.to_period("M")

        y_full = dfp.groupby("_PM").size().rename("Enrolments").sort_index()
        if y_full.empty:
            st.info("No payments found to build a monthly series.")
            st.stop()

        # Ensure continuous monthly index
        full_idx = pd.period_range(start=y_full.index.min(), end=max(y_full.index.max(), pd.Period(date.today(), freq="M")), freq="M")
        y_full = y_full.reindex(full_idx, fill_value=0).sort_index()

        # Split history vs current
        cur_per   = pd.Period(date.today(), freq="M")
        last_comp = cur_per - 1  # last complete month
        y_hist    = y_full.loc[:last_comp].copy()  # strictly excludes current month
        y_current_mtd = int((dfp["_P"].dt.to_period("M") == cur_per).sum())  # actual MTD count

        # ---------- Helpers (design, ridge, holt-winters, naive) ----------
        def build_design(y_ser: pd.Series, max_lag: int):
            """Return X (lags + month dummies) and y (aligned), dropping initial NaNs."""
            y_ser = y_ser.sort_index()
            dfX = pd.DataFrame({"y": y_ser.astype(float)})
            for L in range(1, max_lag+1):
                dfX[f"lag_{L}"] = dfX["y"].shift(L)
            months = pd.Series([p.month for p in dfX.index], index=dfX.index, name="month")
            dummies = pd.get_dummies(months.astype("category"), prefix="m", drop_first=True)
            dfX = pd.concat([dfX, dummies], axis=1)
            dfX = dfX.dropna()
            X = dfX.drop(columns=["y"])
            y_out = dfX["y"]
            return X, y_out

        def ridge_fit_predict(X_tr, y_tr, X_te, alpha=2.0):
            """Closed-form ridge: (X'X + aI)^-1 X'y"""
            X = X_tr.to_numpy(dtype=float)
            yv = y_tr.to_numpy(dtype=float)
            XtX = X.T @ X
            n_feat = XtX.shape[0]
            A = XtX + alpha * np.eye(n_feat)
            beta = np.linalg.solve(A, X.T @ yv)
            yhat = (X_te.to_numpy(dtype=float) @ beta).astype(float)
            return yhat

        def ridge_forecast_iterative(y_series: pd.Series, tgt: pd.Period, max_lag: int, alpha=2.0):
            """
            Iteratively forecast month-by-month from the month after y_series.index.max() up to `tgt`.
            For each step:
              • create the step in the series with a temporary 0 (so design has a row),
              • train only on rows < step,
              • predict the step and overwrite that exact row (no concat).
            This keeps the PeriodIndex unique and avoids Series-to-float errors.
            """
            y_work = y_series.sort_index().astype(float).copy()
            step = y_work.index.max() + 1
            while step <= tgt:
                # Ensure 'step' exists exactly once with a placeholder 0.0
                if step not in y_work.index:
                    new_idx = pd.period_range(y_work.index.min(), step, freq="M")
                    y_work = y_work.reindex(new_idx)
                    y_work.loc[step] = 0.0  # placeholder (will be excluded from training but used as feature row)
                y_work = y_work.sort_index()

                # Build design on the whole (placeholder keeps row present for features)
                X_all, y_all = build_design(y_work, max_lag=max_lag)

                # Exclude the step from training; use the step row only for prediction features
                if step not in X_all.index:
                    # Not enough lags yet; fallback to recent mean
                    next_val = float(y_work.iloc[-min(12, len(y_work)):].mean())
                else:
                    train_mask = X_all.index < step
                    if train_mask.sum() < max(8, max_lag + 4):
                        next_val = float(y_work.iloc[-min(12, len(y_work)):].mean())
                    else:
                        yhat = ridge_fit_predict(X_all.loc[train_mask], y_all.loc[train_mask], X_all.loc[[step]], alpha=alpha)[0]
                        next_val = float(max(0.0, yhat))

                # Overwrite the placeholder with the forecast (no concat → no duplicates)
                y_work.loc[step] = next_val
                step += 1

            return float(y_work.loc[tgt])

        def holt_winters_additive(y_series: pd.Series, season_len=12, alphas=(0.2,0.4,0.6,0.8), betas=(0.1,0.2), gammas=(0.1,0.2)):
            yv = y_series.astype(float).values
            n = len(yv)
            if n < season_len + 5:
                return np.full(n, yv.mean()), (np.nan, np.nan, np.nan)
            best_mse = np.inf
            best_fit = None
            best_params = None
            season_mean = yv[:season_len].mean()
            season_init = np.array([yv[i] - season_mean for i in range(season_len)], dtype=float)
            for a in alphas:
                for b in betas:
                    for g in gammas:
                        L = season_mean
                        T = (yv[season_len:2*season_len].mean() - season_mean) / season_len
                        S = season_init.copy()
                        fit = np.zeros(n, dtype=float)
                        for t in range(n):
                            s_idx = t % season_len
                            prev_L = L
                            prev_T = T
                            L = a * (yv[t] - S[s_idx]) + (1 - a) * (prev_L + prev_T)
                            T = b * (L - prev_L) + (1 - b) * prev_T
                            S[s_idx] = g * (yv[t] - L) + (1 - g) * S[s_idx]
                            fit[t] = L + T + S[s_idx]
                        mse = np.mean((fit[season_len:] - yv[season_len:])**2)
                        if mse < best_mse:
                            best_mse = mse
                            best_fit = fit
                            best_params = (a, b, g)
            return best_fit, best_params

        def holt_winters_forecast_next(y_series: pd.Series, season_len=12, params=(0.4,0.2,0.1)):
            yv = y_series.astype(float).values
            n = len(yv)
            a, b, g = params
            if n < season_len + 5 or any(np.isnan([a,b,g])):
                return float(yv.mean())
            season_mean = yv[:season_len].mean()
            L = season_mean
            T = (yv[season_len:2*season_len].mean() - season_mean) / season_len
            S = np.array([yv[i] - season_mean for i in range(season_len)], dtype=float)
            for t in range(n):
                s_idx = t % season_len
                prev_L = L
                prev_T = T
                L = a * (yv[t] - S[s_idx]) + (1 - a) * (prev_L + prev_T)
                T = b * (L - prev_L) + (1 - b) * prev_T
                S[s_idx] = g * (yv[t] - L) + (1 - g) * S[s_idx]
            s_idx_next = n % season_len
            return float(L + T + S[s_idx_next])

        def holt_winters_iterative(y_series: pd.Series, tgt: pd.Period, season_len=12):
            y_work = y_series.sort_index().copy()
            while y_work.index.max() < tgt:
                fit, params = holt_winters_additive(y_work, season_len=season_len)
                nxt = holt_winters_forecast_next(y_work, season_len=season_len, params=params)
                y_work.loc[y_work.index.max() + 1] = max(0.0, float(nxt))
                y_work = y_work.sort_index()
            return float(y_work.loc[tgt])

        def naive_seasonal_forecast(y_series: pd.Series, target_per):
            month = target_per.month
            idx = y_series.index
            same_month_vals = [y_series[p] for p in idx if p.month == month]
            if len(same_month_vals) == 0:
                return float(y_series.mean())
            sm_mean = float(np.mean(same_month_vals))
            recent_mean = float(y_series.iloc[-min(12, len(y_series)):].mean())
            return 0.7 * sm_mean + 0.3 * recent_mean

        def naive_iterative(y_series: pd.Series, tgt: pd.Period):
            y_work = y_series.sort_index().copy()
            while y_work.index.max() < tgt:
                nxt = naive_seasonal_forecast(y_work, y_work.index.max()+1)
                y_work.loc[y_work.index.max() + 1] = max(0.0, float(nxt))
                y_work = y_work.sort_index()
            return float(y_work.loc[tgt])

        # ---------- Backtest (rolling-origin) to compute Accuracy % (uses y_hist) ----------
        hist_end = y_hist.index.max()
        hist_start = max(y_hist.index.min(), hist_end - (lookback - 1))
        y_bt = y_hist.loc[hist_start:hist_end].copy()

        preds_bt, actual_bt, idx_bt = [], [], []

        if model_name.startswith("Ridge"):
            # backtest: 1-step ahead, training up to t-1 (all strictly <= last complete month)
            for t in y_bt.index:
                y_tr = y_hist.loc[:(t - 1)].copy()
                if len(y_tr) < max(8, max_lag + 4):
                    continue
                yhat = ridge_forecast_iterative(y_tr, t, max_lag=max_lag, alpha=2.0)
                preds_bt.append(max(0.0, float(yhat)))
                actual_bt.append(float(y_hist.loc[t]))
                idx_bt.append(t)

        elif model_name.startswith("Holt-Winters"):
            for t in y_bt.index:
                y_tr = y_hist.loc[:(t - 1)].copy()
                if len(y_tr) < 18:
                    continue
                yhat = holt_winters_iterative(y_tr, t, season_len=12)
                preds_bt.append(max(0.0, float(yhat)))
                actual_bt.append(float(y_hist.loc[t]))
                idx_bt.append(t)

        else:  # Naive Seasonal
            for t in y_bt.index:
                y_tr = y_hist.loc[:(t - 1)].copy()
                if len(y_tr) < 6:
                    continue
                yhat = naive_iterative(y_tr, t)
                preds_bt.append(max(0.0, float(yhat)))
                actual_bt.append(float(y_hist.loc[t]))
                idx_bt.append(t)

        if preds_bt and actual_bt:
            actual_arr = np.array(actual_bt, dtype=float)
            pred_arr   = np.array(preds_bt, dtype=float)
            denom = np.where(actual_arr == 0, np.nan, actual_arr)
            ape = np.abs(pred_arr - actual_arr) / denom
            mape = np.nanmean(ape)
            acc_pct = max(0.0, min(100.0, 100.0 * (1.0 - (mape if np.isfinite(mape) else 1.0))))
        else:
            acc_pct = np.nan

        # ---------- Final forecast (train on y_hist only; never on current month) ----------
        tgt_per = pd.Period(target_month, freq="M")
        forecast_val = None

        if model_name.startswith("Ridge"):
            if len(y_hist) >= max(8, max_lag + 4):
                forecast_val = ridge_forecast_iterative(y_hist, tgt_per, max_lag=max_lag, alpha=2.0)
        elif model_name.startswith("Holt-Winters"):
            if len(y_hist) >= 18:
                forecast_val = holt_winters_iterative(y_hist, tgt_per, season_len=12)
        else:
            if len(y_hist) >= 6:
                forecast_val = naive_iterative(y_hist, tgt_per)

        # ---------- Chart (history + backtest + forecast + current MTD) ----------
        chart_rows = []
        if "History" in show_components:
            for per, v in y_hist.items():
                chart_rows.append({"Month": str(per), "Component": "History", "Count": float(v)})
        if "Backtest (pred vs. actual)" in show_components and preds_bt:
            for per, yhat, yact in zip(idx_bt, preds_bt, actual_bt):
                chart_rows.append({"Month": str(per), "Component": "Backtest Pred",    "Count": float(yhat)})
                chart_rows.append({"Month": str(per), "Component": "Backtest Actual",  "Count": float(yact)})
        if "Current Month (Actual MTD)" in show_components and y_current_mtd > 0:
            chart_rows.append({"Month": str(cur_per), "Component": "Current MTD", "Count": float(y_current_mtd)})
        if "Forecast" in show_components and forecast_val is not None:
            chart_rows.append({"Month": str(tgt_per), "Component": "Forecast", "Count": float(forecast_val)})

        if chart_rows:
            ch_df = pd.DataFrame(chart_rows)
            ch = (
                alt.Chart(ch_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Month:N", sort=sorted(ch_df["Month"].unique().tolist())),
                    y=alt.Y("Count:Q", title="Enrolments (count)"),
                    color=alt.Color("Component:N", legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Component:N"), alt.Tooltip("Count:Q")]
                )
                .properties(height=340, title=f"Monthly Enrolments — {model_name} (training excludes current month)")
            )
            st.altair_chart(ch, use_container_width=True)

        # ---------- Running month: Actual vs Projected ----------
        st.markdown("### Running Month — Actual vs Projected")
        proj_cur = None
        if model_name.startswith("Ridge"):
            if len(y_hist) >= max(8, max_lag + 4):
                proj_cur = ridge_forecast_iterative(y_hist, cur_per, max_lag=max_lag, alpha=2.0)
        elif model_name.startswith("Holt-Winters"):
            if len(y_hist) >= 18:
                proj_cur = holt_winters_iterative(y_hist, cur_per, season_len=12)
        else:
            if len(y_hist) >= 6:
                proj_cur = naive_iterative(y_hist, cur_per)

        if proj_cur is not None:
            remain = max(0.0, float(proj_cur) - float(y_current_mtd))
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Actual so far (MTD)", f"{y_current_mtd:,}")
            with k2:
                st.metric("Projected Month-End", f"{proj_cur:.1f}")
            with k3:
                st.metric("Projected Remaining", f"{remain:.1f}")

            small_df = pd.DataFrame({
                "Metric": ["Actual MTD", "Projected Month-End"],
                "Count":  [float(y_current_mtd), float(proj_cur)]
            })
            ch2 = (
                alt.Chart(small_df)
                .mark_bar()
                .encode(
                    x=alt.X("Metric:N", title=""),
                    y=alt.Y("Count:Q", title="Enrolments"),
                    tooltip=["Metric","Count"]
                )
                .properties(height=220, title=f"{str(cur_per)} — Actual MTD vs Projected Month-End")
            )
            st.altair_chart(ch2, use_container_width=True)
        else:
            st.info("Not enough historical data to project the current month with the selected model.")

        st.markdown("---")

        # ---------- KPI strip (model & accuracy) ----------
        st.markdown(
            """
            <style>
              .kpi-card { border:1px solid #e5e7eb; border-radius:14px; padding:10px 12px; background:#ffffff; }
              .kpi-title { font-size:0.9rem; color:#6b7280; margin-bottom:6px; }
              .kpi-value { font-size:1.4rem; font-weight:700; }
              .kpi-sub { font-size:0.8rem; color:#6b7280; margin-top:4px; }
            </style>
            """, unsafe_allow_html=True
        )
        kA, kB, kC = st.columns(3)
        with kA:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Model</div>"
                f"<div class='kpi-value'>{model_name}</div>"
                f"<div class='kpi-sub'>lags={max_lag if model_name.startswith('Ridge') else '—'}, lookback={lookback}m</div></div>",
                unsafe_allow_html=True
            )
        with kB:
            acc_txt = "–" if np.isnan(acc_pct) else f"{acc_pct:.1f}%"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Backtest Accuracy</div>"
                f"<div class='kpi-value'>{acc_txt}</div>"
                f"<div class='kpi-sub'>1-step ahead, last {lookback}m (excludes current month)</div></div>",
                unsafe_allow_html=True
            )
        with kC:
            f_txt = "–" if forecast_val is None else f"{forecast_val:.1f}"
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>Forecast for {str(tgt_per)}</div>"
                f"<div class='kpi-value'>{f_txt}</div>"
                f"<div class='kpi-sub'>Monthly enrolments</div></div>",
                unsafe_allow_html=True
            )

        # ---------- Per-Country & Per-Source allocation of the forecast ----------
        st.markdown("#### Forecast Allocation (Country / Source)")
        alloc_cols = st.columns(2)
        with alloc_cols[0]:
            do_cty = st.checkbox("Allocate by Country", value=bool(_cty), help="Uses last lookback months’ composition.", key="bp_alloc_cty")
        with alloc_cols[1]:
            do_src = st.checkbox("Allocate by JetLearn Deal Source", value=bool(_src), help="Uses last lookback months’ composition.", key="bp_alloc_src")

        def _alloc_series(group_col):
            if (group_col is None) or (group_col not in df_f.columns) or (forecast_val is None):
                return pd.DataFrame(columns=["Group","Projected"])
            sub = dfp.copy()
            sub["_G"] = df_f[group_col].fillna("Unknown").astype(str).str.strip()
            lb_end = y_hist.index.max()
            lb_start = max(y_hist.index.min(), lb_end - (lookback - 1))
            mask_lb = sub["_PM"].between(lb_start, lb_end)
            if not mask_lb.any():
                return pd.DataFrame(columns=["Group","Projected"])
            comp = sub.loc[mask_lb].groupby("_G").size().rename("cnt").reset_index()
            if comp["cnt"].sum() == 0:
                return pd.DataFrame(columns=["Group","Projected"])
            comp["w"] = comp["cnt"] / comp["cnt"].sum()
            comp["Projected"] = comp["w"] * float(forecast_val)
            comp = comp.rename(columns={"_G":"Group"})
            return comp[["Group","Projected"]].sort_values("Projected", ascending=False)

        if do_cty:
            out_cty = _alloc_series(_cty)
            if out_cty.empty:
                st.info("Not enough data to allocate by Country.")
            else:
                st.dataframe(out_cty, use_container_width=True)
                st.download_button("Download CSV — Country Allocation",
                                   out_cty.to_csv(index=False).encode("utf-8"),
                                   "business_projection_country_allocation.csv", "text/csv")
        if do_src:
            out_src = _alloc_series(_src)
            if out_src.empty:
                st.info("Not enough data to allocate by Deal Source.")
            else:
                st.dataframe(out_src, use_container_width=True)
                st.download_button("Download CSV — Source Allocation",
                                   out_src.to_csv(index=False).encode("utf-8"),
                                   "business_projection_source_allocation.csv", "text/csv")

        # ---------- Backtest details (optional) ----------
        with st.expander("Backtest details"):
            if preds_bt and actual_bt:
                bt_df = pd.DataFrame({
                    "Month": [str(p) for p in idx_bt],
                    "Actual": actual_bt,
                    "Predicted": preds_bt
                }).sort_values("Month")
                bt_df["Abs % Error"] = np.where(
                    bt_df["Actual"]>0,
                    np.abs(bt_df["Predicted"]-bt_df["Actual"])/bt_df["Actual"]*100.0,
                    np.nan
                )
                st.dataframe(bt_df, use_container_width=True)
                st.download_button("Download CSV — Backtest",
                                   bt_df.to_csv(index=False).encode("utf-8"),
                                   "business_projection_backtest.csv", "text/csv")
            else:
                st.info("Backtest window too short to compute accuracy. Add more months of data.")

    # run it
    _business_projection_tab()




# ========================= 
# Marketing Lead Performance & Requirement (FULL — per-source Top N country scope)
# =========================
if view == "Marketing Lead Performance & Requirement":
    def _mlpr_tab():
        st.subheader("Marketing Lead Performance & Requirement")

        # ---------- Resolve columns ----------
        def _pick(df, preferred, cands):
            if preferred and preferred in df.columns: return preferred
            for c in cands:
                if c in df.columns: return c
            return None

        _create = _pick(df_f, globals().get("create_col"),
                        ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
        _pay    = _pick(df_f, globals().get("pay_col"),
                        ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
        _src    = _pick(df_f, globals().get("source_col"),
                        ["JetLearn Deal Source","Deal Source","Source","_src_raw","Lead Source"])
        _cty    = _pick(df_f, globals().get("country_col"),
                        ["Country","Student Country","Deal Country","Lead Country"])

        if not _create or not _pay or not _src:
            st.warning("Please map Create Date, Payment Received Date and JetLearn Deal Source in the sidebar.", icon="⚠️")
            return

        # ---------- Controls ----------
        from datetime import date
        import pandas as pd, numpy as np, altair as alt
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            scope = st.selectbox("Date scope", ["This month","Last month","Custom"], index=0)
        with c2:
            lookback = st.selectbox("Lookback (full months, excl. current)", [3, 6, 12], index=1)
        with c3:
            mode = st.radio("Mode for 'Actual so far'", ["Cohort","MTD"], index=0, horizontal=True,
                            help="Cohort: payments in window; MTD: created & paid in window.")

        today_d = date.today()
        if scope == "This month":
            mstart, mend = month_bounds(today_d)
        elif scope == "Last month":
            mstart, mend = last_month_bounds(today_d)
        else:
            d1, d2 = st.columns(2)
            with d1: mstart = st.date_input("Start", value=today_d.replace(day=1), key="mlpr_start")
            with d2: mend   = st.date_input("End", value=month_bounds(today_d)[1], key="mlpr_end")
            if mend < mstart:
                st.error("End date cannot be before start date.")
                return

        st.caption(f"Scope: **{mstart} → {mend}** • Lookback: **{lookback}m** • Mode: **{mode}**")

        # ---------- Prep dataframe ----------
        d = df_f.copy()
        d["__C"] = coerce_datetime(d[_create])
        d["__P"] = coerce_datetime(d[_pay])
        d["__SRC"] = d[_src].fillna("Unknown").astype(str).str.strip()
        if _cty:
            d["__CTY"] = d[_cty].fillna("Unknown").astype(str).str.strip()
        else:
            d["__CTY"] = "All"

        # ---------- Build lookback months ----------
        cur_per = pd.Period(mstart, freq="M")
        from calendar import monthrange
        lb_months = [cur_per - i for i in range(1, lookback+1)]  # exclude current
        if not lb_months:
            st.info("No lookback months selected.")
            return

        C_per = d["__C"].dt.to_period("M")
        P_per = d["__P"].dt.to_period("M")

        # ---------- Historical per-month aggregates across lookback ----------
        rows = []
        for per in lb_months:
            ms = date(per.year, per.month, 1)
            ml = monthrange(per.year, per.month)[1]
            me = date(per.year, per.month, ml)

            c_mask = d["__C"].dt.date.between(ms, me)
            p_mask = d["__P"].dt.date.between(ms, me)

            # SAME (M0): created in month & paid in same month
            same_mask = p_mask & (P_per == per) & (C_per == per)
            # PREV (carry-in): paid in month but created before month
            prev_mask = p_mask & (C_per < per)

            grp_cols = ["__SRC","__CTY"]
            creates = d.loc[c_mask, grp_cols].assign(_one=1).groupby(grp_cols)["_one"].sum().rename("Creates").reset_index()
            same    = d.loc[same_mask, grp_cols].assign(_one=1).groupby(grp_cols)["_one"].sum().rename("SamePaid").reset_index()
            prev    = d.loc[prev_mask, grp_cols].assign(_one=1).groupby(grp_cols)["_one"].sum().rename("PrevPaid").reset_index()

            g = creates.merge(same, on=grp_cols, how="outer").merge(prev, on=grp_cols, how="outer").fillna(0)
            g["per"] = str(per)
            rows.append(g)

        hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["__SRC","__CTY","Creates","SamePaid","PrevPaid","per"])
        if hist.empty:
            st.info("No lookback history. Increase lookback or check data.")
            return

        # Aggregate over lookback months
        agg = (hist.groupby(["__SRC","__CTY"], dropna=False)[["Creates","SamePaid","PrevPaid"]]
                    .sum().reset_index())
        agg["M0_Rate"] = np.where(agg["Creates"] > 0, agg["SamePaid"] / agg["Creates"], 0.0)
        agg["MN_Avg"]  = agg["PrevPaid"] / float(lookback)

        # ---------- Build per-source Top lists (Top 5 / Top 10 / All) ----------
        # Rank countries by lookback enrolments (SamePaid + PrevPaid). Fallback to Creates.
        rank_base = agg.copy()
        rank_base["EnrollLike"] = rank_base["SamePaid"] + rank_base["PrevPaid"]
        if (rank_base["EnrollLike"].sum() == 0) and (rank_base["Creates"].sum() > 0):
            rank_base["EnrollLike"] = rank_base["Creates"]

        top_lists = {}
        for src, g in rank_base.groupby("__SRC"):
            g = g.sort_values("EnrollLike", ascending=False)
            all_c = g["__CTY"].tolist()
            top5  = g.head(5)["__CTY"].tolist()
            top10 = g.head(10)["__CTY"].tolist()
            top_lists[src] = {"All": all_c, "Top 5": top5 or all_c, "Top 10": top10 or all_c}

        # ---------- Planner inputs (per-source) ----------
        st.markdown("### Planner — Enter Planned Creates and Country Scope per Source")
        sources_all = sorted(agg["__SRC"].unique().tolist())
        pick_src = st.multiselect("Pick Source(s)", options=["All"] + sources_all, default=["All"])
        if "All" in pick_src or not pick_src:
            pick_src = sources_all

        planned_by_src = {}
        scope_by_src   = {}
        cols = st.columns(3)
        for i, src in enumerate(pick_src):
            with cols[i % 3]:
                planned_by_src[src] = st.number_input(f"Planned Creates — {src}", min_value=0, value=0, step=1, key=f"plan_{src}")
                scope_by_src[src]   = st.selectbox(f"Country scope — {src}", ["Top 5","Top 10","All"], index=1, key=f"scope_{src}",
                                                   help="Filters line items for this source to the chosen country subset.")

        # ---------- Construct line items (Source × Country) & apply per-source scope ----------
        lines = agg[agg["__SRC"].isin(pick_src)].copy()

        if not lines.empty:
            def _row_keep(r):
                src = r["__SRC"]
                chosen = scope_by_src.get(src, "Top 10")
                allowed = top_lists.get(src, {}).get(chosen, [r["__CTY"]])
                return r["__CTY"] in allowed
            lines = lines[lines.apply(_row_keep, axis=1)].copy()

        # Historical share within source (based on lookback Creates; fallback to EnrollLike)
        share = agg[agg["__SRC"].isin(pick_src)][["__SRC","__CTY","Creates"]].rename(columns={"Creates":"C"}).copy()
        if share["C"].sum() == 0:
            share = rank_base[rank_base["__SRC"].isin(pick_src)][["__SRC","__CTY","EnrollLike"]].rename(columns={"EnrollLike":"C"}).copy()

        lines = lines.merge(share, on=["__SRC","__CTY"], how="left")
        lines["C"] = lines["C"].fillna(0.0)
        src_tot = lines.groupby("__SRC")["C"].transform(lambda s: s.sum() if s.sum() > 0 else np.nan)
        lines["Share"] = np.where(
            src_tot.notna(),
            np.where(src_tot > 0, lines["C"] / src_tot, 0.0),
            1.0 / lines.groupby("__SRC")["__CTY"].transform("count")
        )

        # Allocation of planned creates to lines
        lines["PlannedDeals_Line"] = lines.apply(lambda r: planned_by_src.get(r["__SRC"], 0) * r["Share"], axis=1)

        # Expected Enrolments per line (M0 + carry-in)
        lines["Expected_Enrolments_Line"] = lines["PlannedDeals_Line"] * lines["M0_Rate"] + lines["MN_Avg"]

        # Keep helper for baseline mix later
        lines["HistCreates"] = lines["C"].astype(float)

        # ---------- Actual so far (in scope) ----------
        st.markdown("### Actual so far (for context)")
        c_mask_win = d["__C"].dt.date.between(mstart, mend)
        p_mask_win = d["__P"].dt.date.between(mstart, mend)
        if mode == "MTD":
            paid_now = int((c_mask_win & p_mask_win).sum())
        else:
            paid_now = int(p_mask_win.sum())
        created_now = int(c_mask_win.sum())
        conv_now = (paid_now / created_now * 100.0) if created_now > 0 else np.nan

        k1, k2, k3 = st.columns(3)
        k1.metric("Creates (scope)", f"{created_now:,}")
        k2.metric("Enrolments (scope)", f"{paid_now:,}")
        k3.metric("Conv% (scope)", "–" if np.isnan(conv_now) else f"{conv_now:.1f}%")

        # ---------- Totals for plan ----------
        st.markdown("### Totals (Plan)")
        tA, tB = st.columns(2)
        with tA:
            plan_creates_total = float(sum(planned_by_src.values()))
            st.metric("Planned Creates (total)", f"{plan_creates_total:,.0f}")
        with tB:
            exp_total = float(lines["Expected_Enrolments_Line"].sum()) if not lines.empty else 0.0
            st.metric("Expected Enrolments (M0 + carry-in)", f"{exp_total:,.1f}")

        # ---------- Line-item Planner table & chart ----------
        st.markdown("### Line-item Planner — by Source × Country")
        show_n = st.number_input("Show top N lines", min_value=5, max_value=500, value=50, step=5)
        line_rows = lines.sort_values("Expected_Enrolments_Line", ascending=False).head(int(show_n))

        nice = line_rows.rename(columns={
            "__SRC":"Source","__CTY":"Country",
            "M0_Rate":"M0 Rate",
            "MN_Avg":"Avg carry-in (per month)",
            "PlannedDeals_Line":"Planned Deals",
            "Expected_Enrolments_Line":"Expected Enrolments"
        }).copy()
        for c in ["M0 Rate"]:
            nice[c] = (nice[c].astype(float) * 100).round(1)
        for c in ["Avg carry-in (per month)","Planned Deals","Expected Enrolments"]:
            nice[c] = nice[c].astype(float).round(2)

        st.dataframe(nice[["Source","Country","Planned Deals","M0 Rate","Avg carry-in (per month)","Expected Enrolments"]],
                     use_container_width=True)

        if not line_rows.empty:
            ch = (
                alt.Chart(line_rows)
                .mark_circle(size=200, opacity=0.7)
                .encode(
                    x=alt.X("M0_Rate:Q", title="M0 Rate", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0, max(0.01, float(line_rows["M0_Rate"].max())*1.05)])),
                    y=alt.Y("Expected_Enrolments_Line:Q", title="Expected Enrolments"),
                    size=alt.Size("PlannedDeals_Line:Q", title="Planned Deals"),
                    color=alt.Color("__SRC:N", title="Source", legend=alt.Legend(orient="bottom")),
                    tooltip=[
                        alt.Tooltip("__SRC:N", title="Source"),
                        alt.Tooltip("__CTY:N", title="Country"),
                        alt.Tooltip("PlannedDeals_Line:Q", title="Planned Deals", format=".1f"),
                        alt.Tooltip("M0_Rate:Q", title="M0 Rate", format=".1%"),
                        alt.Tooltip("MN_Avg:Q", title="Avg carry-in", format=".2f"),
                        alt.Tooltip("Expected_Enrolments_Line:Q", title="Expected Enrolments", format=".2f"),
                    ]
                )
                .properties(height=420, title="Bubble view — Expected Enrolments vs M0 Rate (size = Planned Deals)")
            )
            st.altair_chart(ch, use_container_width=True)

        # ---------- Mix Effect — Plan vs Baseline (MTD create mix) ----------
        if not lines.empty:
            st.markdown("---")
            st.markdown("### Mix Effect — Impact of Lead Mix on Expected Enrolments (Plan vs Baseline)")

            total_planned_deals = float(lines["PlannedDeals_Line"].sum())

            # MTD create mix (Source × Country) in current month
            mtd_create_mask = d["__C"].dt.date.between(mstart, today_d)
            grp_cols = ["__SRC","__CTY"]
            mtd_mix = (
                d.loc[mtd_create_mask, grp_cols]
                 .assign(_ones=1)
                 .groupby(grp_cols, dropna=False)["_ones"].sum()
                 .reset_index()
            )

            lines["__key"] = lines["__SRC"].astype(str).str.strip() + "||" + lines["__CTY"].astype(str).str.strip()
            if not mtd_mix.empty:
                mtd_mix["__key"] = mtd_mix["__SRC"].astype(str).str.strip() + "||" + mtd_mix["__CTY"].astype(str).str.strip()

            # Baseline weights
            if mtd_mix.empty or mtd_mix["_ones"].sum() == 0:
                st.info("No creates so far this month; baseline mix falls back to historical mix used above.")
                denom = float(lines["HistCreates"].sum())
                if denom > 0:
                    base_weights = (lines["HistCreates"] / denom).fillna(0.0)
                else:
                    base_weights = pd.Series(1.0 / max(len(lines), 1), index=lines.index)
            else:
                w = mtd_mix.set_index("__key")["_ones"].reindex(lines["__key"]).fillna(0.0)
                if w.sum() > 0:
                    base_weights = w / w.sum()
                else:
                    denom = float(lines["HistCreates"].sum())
                    base_weights = (lines["HistCreates"] / denom).fillna(0.0) if denom > 0 else pd.Series(1.0 / max(len(lines),1), index=lines.index)

            # Baseline allocation & expected
            lines["Baseline_PlannedDeals"] = total_planned_deals * base_weights.values
            lines["Baseline_Expected"] = lines["Baseline_PlannedDeals"] * lines["M0_Rate"] + lines["MN_Avg"]

            # Deltas
            lines["Delta_Deals"] = lines["PlannedDeals_Line"] - lines["Baseline_PlannedDeals"]
            lines["Delta_Expected_Enrol"] = lines["Expected_Enrolments_Line"] - lines["Baseline_Expected"]

            plan_total = float(lines["Expected_Enrolments_Line"].sum())
            base_total = float(lines["Baseline_Expected"].sum())
            delta_total = plan_total - base_total

            st.metric("Δ Expected Enrolments (Plan – Baseline mix)", f"{delta_total:+.1f}")

            with st.expander("Detail — Plan vs Baseline mix (per line)"):
                det = lines[[
                    "__SRC","__CTY",
                    "PlannedDeals_Line","Baseline_PlannedDeals","Delta_Deals",
                    "Expected_Enrolments_Line","Baseline_Expected","Delta_Expected_Enrol"
                ]].rename(columns={
                    "__SRC":"Source","__CTY":"Country",
                    "PlannedDeals_Line":"Planned Deals",
                    "Baseline_PlannedDeals":"Baseline Deals",
                    "Delta_Deals":"Δ Deals",
                    "Expected_Enrolments_Line":"Planned Expected Enrolments",
                    "Baseline_Expected":"Baseline Expected Enrolments",
                    "Delta_Expected_Enrol":"Δ Expected Enrolments"
                }).copy()

                for c in ["Planned Deals","Baseline Deals","Δ Deals",
                          "Planned Expected Enrolments","Baseline Expected Enrolments","Δ Expected Enrolments"]:
                    det[c] = det[c].astype(float).round(2)

                st.dataframe(det.sort_values("Δ Expected Enrolments", ascending=False), use_container_width=True)
                st.download_button("Download CSV — Mix Effect detail",
                                   det.to_csv(index=False).encode("utf-8"),
                                   "mix_effect_plan_vs_baseline.csv", "text/csv",
                                   key="mix_eff_dl")

            chart_on = st.radio("Visualize line-item mix impact?", ["No","Yes"], index=0, horizontal=True, key="mix_eff_chart_toggle")
            if chart_on == "Yes":
                top_lines = lines.sort_values("Delta_Expected_Enrol", ascending=False)
                top_lines = pd.concat([top_lines.head(15), top_lines.tail(15)]) if len(top_lines) > 30 else top_lines
                ch2 = (
                    alt.Chart(top_lines)
                    .mark_bar()
                    .encode(
                        x=alt.X("Delta_Expected_Enrol:Q", title="Δ Expected Enrolments (Plan – Baseline)"),
                        y=alt.Y("__CTY:N", sort="-x", title="Country"),
                        color=alt.Color("__SRC:N", title="Source", legend=alt.Legend(orient="bottom")),
                        tooltip=[
                            alt.Tooltip("__SRC:N", title="Source"),
                            alt.Tooltip("__CTY:N", title="Country"),
                            alt.Tooltip("Delta_Expected_Enrol:Q", title="Δ Expected Enrolments", format=".2f"),
                            alt.Tooltip("PlannedDeals_Line:Q", title="Planned Deals", format=".1f"),
                            alt.Tooltip("Baseline_PlannedDeals:Q", title="Baseline Deals", format=".1f"),
                        ]
                    )
                    .properties(height=420, title="Mix Effect — Which lines drive the difference?")
                )
                st.altair_chart(ch2, use_container_width=True)

    _mlpr_tab()




# =========================
# Master Graph – flexible chart builder (single / combined / ratio)
# =========================
if view == "Master Graph":
    def _master_graph_tab():
        import pandas as pd, numpy as np, altair as alt
        from datetime import date
        st.subheader("Master Graph — Flexible Visuals (MTD / Cohort)")

        # ---------- Resolve columns (defensive)
        _create = create_col if (create_col in df_f.columns) else find_col(df_f, [
            "Create Date","Created Date","Deal Create Date","CreateDate","Created On"
        ])
        _pay    = pay_col    if (pay_col    in df_f.columns) else find_col(df_f, [
            "Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"
        ])
        _first  = first_cal_sched_col if (first_cal_sched_col in df_f.columns) else find_col(df_f, [
            "First Calibration Scheduled Date","First Calibration","First Cal Scheduled"
        ])
        _resch  = cal_resched_col     if (cal_resched_col     in df_f.columns) else find_col(df_f, [
            "Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"
        ])
        _done   = cal_done_col        if (cal_done_col        in df_f.columns) else find_col(df_f, [
            "Calibration Done Date","Cal Done Date","Calibration Completed"
        ])

        if not _create or _create not in df_f.columns or not _pay or _pay not in df_f.columns:
            st.warning("Master Graph needs 'Create Date' and 'Payment Received Date'. Please map them in the sidebar.", icon="⚠️")
            st.stop()

        # ---------- Controls: mode, scope, granularity
        c0, c1, c2, c3 = st.columns([1.0, 1.0, 1.1, 1.1])
        with c0:
            mode = st.radio("Mode", ["MTD","Cohort"], index=0, horizontal=True, key="mg_mode",
                            help=("MTD: count events only when the deal was also created in the same period;"
                                  " Cohort: count events by their own date (create can be anywhere)."))
        with c1:
            gran = st.radio("Granularity", ["Day","Week","Month"], index=2, horizontal=True, key="mg_gran")
        today_d = date.today()
        with c2:
            c_start = st.date_input("Create start", value=today_d.replace(day=1), key="mg_cstart")
        with c3:
            c_end   = st.date_input("Create end",   value=month_bounds(today_d)[1], key="mg_cend")
        if c_end < c_start:
            st.error("End date cannot be before start date.")
            st.stop()

        # ---------- Choose build type & chart type
        c4, c5 = st.columns([1.2, 1.2])
        with c4:
            build_type = st.radio("Build", ["Single metric","Combined (dual-axis)","Derived ratio"], index=0, horizontal=True, key="mg_build")
        with c5:
            chart_type = st.selectbox(
                "Chart type",
                ["Line","Bar","Area","Stacked Bar","Histogram","Bell Curve"],
                index=0,
                key="mg_chart",
                help="Histogram/Bell Curve apply to daily/period counts of a single metric."
            )

        # ---------- Normalize event timestamps
        C = coerce_datetime(df_f[_create])
        P = coerce_datetime(df_f[_pay])
        F = coerce_datetime(df_f[_first]) if (_first and _first in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        R = coerce_datetime(df_f[_resch]) if (_resch and _resch in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
        D = coerce_datetime(df_f[_done])  if (_done  and _done  in df_f.columns)  else pd.Series(pd.NaT, index=df_f.index)

        # Period keys by granularity
        def _per(s):
            ds = pd.to_datetime(s, errors="coerce")
            if gran == "Day":
                return ds.dt.floor("D")
            if gran == "Week":
                # ISO week start Monday
                return (ds - pd.to_timedelta(ds.dt.weekday, unit="D")).dt.floor("D")
            return ds.dt.to_period("M").dt.to_timestamp()

        perC, perP, perF, perR, perD = _per(C), _per(P), _per(F), _per(R), _per(D)

        # Universe = created within [c_start, c_end]
        C_date = C.dt.date
        in_window = C_date.notna() & (C_date >= c_start) & (C_date <= c_end)

        # MTD requires event period == create period; Cohort uses event’s own period
        sameP = perC == perP
        sameF = perC == perF
        sameR = perC == perR
        sameD = perC == perD

        if mode == "MTD":
            m_created = in_window & C.notna()
            m_enrol   = in_window & P.notna() & sameP
            m_first   = in_window & F.notna() & sameF
            m_resch   = in_window & R.notna() & sameR
            m_done    = in_window & D.notna() & sameD
        else:
            m_created = in_window & C.notna()
            m_enrol   = in_window & P.notna()
            m_first   = in_window & F.notna()
            m_resch   = in_window & R.notna()
            m_done    = in_window & D.notna()

        # Metric → (period_series, mask)
        metric_defs = {
            "Deals Created":              (perC, m_created),
            "Enrolments":                 (perP, m_enrol),
            "First Cal Scheduled":        (perF, m_first),
            "Cal Rescheduled":            (perR, m_resch),
            "Cal Done":                   (perD, m_done),
        }
        metric_names = list(metric_defs.keys())

        # ---------- Metric pickers (depend on build type)
        if build_type == "Single metric":
            m1 = st.selectbox("Metric", metric_names, index=0, key="mg_m1")
        elif build_type == "Combined (dual-axis)":
            cA, cB = st.columns(2)
            with cA:
                m1 = st.selectbox("Left Y", metric_names, index=0, key="mg_m1l")
            with cB:
                m2 = st.selectbox("Right Y", [m for m in metric_names if m != m1], index=0, key="mg_m2r")
        else:  # Derived ratio
            cA, cB = st.columns(2)
            with cA:
                num_m = st.selectbox("Numerator", metric_names, index=1, key="mg_num")
            with cB:
                den_m = st.selectbox("Denominator", [m for m in metric_names if m != num_m], index=0, key="mg_den")
            as_pct = st.checkbox("Show as % (×100)", value=True, key="mg_ratio_pct")

        # ---------- Helpers to aggregate counts by period
        def _count_series(per_s, mask, label):
            if mask is None or not mask.any():
                return pd.DataFrame(columns=["Period", label])
            df = pd.DataFrame({"Period": per_s[mask]})
            if df.empty:
                return pd.DataFrame(columns=["Period", label])
            return df.assign(_one=1).groupby("Period")["_one"].sum().rename(label).reset_index()

        # ---------- Build outputs
        if build_type == "Single metric":
            per_s, msk = metric_defs[m1]
            counts = _count_series(per_s, msk, m1)

            # Graph / Histogram / Bell
            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_single")

            if view == "Table":
                st.dataframe(counts.sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", counts.sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_single.csv","text/csv", key="mg_dl_single")
            else:
                if chart_type in {"Histogram","Bell Curve"}:
                    # Build distribution of period counts (e.g., daily counts)
                    vals = counts[m1].astype(float)
                    if vals.empty:
                        st.info("No data to plot a distribution.")
                    else:
                        hist = alt.Chart(counts).mark_bar(opacity=0.9).encode(
                            x=alt.X(f"{m1}:Q", bin=alt.Bin(maxbins=30), title=f"{m1} per {gran}"),
                            y=alt.Y("count():Q", title="Frequency"),
                            tooltip=[alt.Tooltip("count():Q", title="Freq")]
                        ).properties(height=320, title=f"Histogram — {m1} per {gran}")

                        if chart_type == "Bell Curve":
                            mu  = float(vals.mean())
                            sig = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                            # synthetic normal
                            xs = np.linspace(max(0, vals.min()), vals.max() if vals.max()>0 else 1.0, 200)
                            # scale PDF to same area ~ total count of bars
                            pdf = (1.0/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu)/max(sig,1e-9))**2)) if sig>0 else np.zeros_like(xs)
                            pdf = pdf / pdf.max() * (counts["count()"].max() if "count()" in counts.columns else 1.0) if sig>0 else pdf
                            bell_df = pd.DataFrame({"x": xs, "pdf": pdf})
                            bell = alt.Chart(bell_df).mark_line().encode(
                                x=alt.X("x:Q", title=f"{m1} per {gran}"),
                                y=alt.Y("pdf:Q", title="Density (scaled)")
                            )
                            st.altair_chart(hist + bell, use_container_width=True)
                            st.caption(f"μ = {mu:.2f}, σ = {sig:.2f}")
                        else:
                            st.altair_chart(hist, use_container_width=True)
                else:
                    mark = {"Line":"line","Bar":"bar","Area":"area","Stacked Bar":"bar"}.get(chart_type, "line")
                    base = alt.Chart(counts)
                    ch = (
                        base.mark_line(point=True) if mark=="line" else
                        base.mark_area(opacity=0.5) if mark=="area" else
                        base.mark_bar(opacity=0.9)
                    ).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    ).properties(height=360, title=f"{m1} by {gran}")
                    st.altair_chart(ch, use_container_width=True)

        elif build_type == "Combined (dual-axis)":
            per1, m1_mask = metric_defs[m1]
            per2, m2_mask = metric_defs[m2]
            s1 = _count_series(per1, m1_mask, m1)
            s2 = _count_series(per2, m2_mask, m2)
            combined = s1.merge(s2, on="Period", how="outer").fillna(0)

            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_combo")
            if view == "Table":
                st.dataframe(combined.sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", combined.sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_combined.csv","text/csv", key="mg_dl_combined")
            else:
                # Dual-axis with layering (left = bars/line, right = line)
                if chart_type == "Bar":
                    left = alt.Chart(combined).mark_bar(opacity=0.85).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )
                elif chart_type == "Area":
                    left = alt.Chart(combined).mark_area(opacity=0.5).encode(
                        x=alt.X("Period:T"), y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )
                else:
                    left = alt.Chart(combined).mark_line(point=True).encode(
                        x=alt.X("Period:T"), y=alt.Y(f"{m1}:Q", title=m1),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{m1}:Q", format="d")]
                    )

                right = alt.Chart(combined).mark_line(point=True).encode(
                    x=alt.X("Period:T"),
                    y=alt.Y(f"{m2}:Q", title=m2, axis=alt.Axis(orient="right")),
                    tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                             alt.Tooltip(f"{m2}:Q", format="d")]
                )

                st.altair_chart(alt.layer(left, right).resolve_scale(y='independent').properties(height=360,
                                  title=f"{m1} (left) + {m2} (right) by {gran}"), use_container_width=True)

        else:
            # Derived ratio
            perN, maskN = metric_defs[num_m]
            perD, maskD = metric_defs[den_m]
            sN = _count_series(perN, maskN, "Num")
            sD = _count_series(perD, maskD, "Den")
            ratio = sN.merge(sD, on="Period", how="outer").fillna(0.0)
            ratio["Value"] = np.where(ratio["Den"]>0, ratio["Num"]/ratio["Den"], np.nan)
            if as_pct:
                ratio["Value"] = ratio["Value"] * 100.0

            label = f"{num_m} / {den_m}" + (" (%)" if as_pct else "")
            ratio = ratio.rename(columns={"Value": label})

            view = st.radio("View as", ["Graph","Table"], index=0, horizontal=True, key="mg_view_ratio")
            if view == "Table":
                st.dataframe(ratio[["Period", label]].sort_values("Period"), use_container_width=True)
                st.download_button("Download CSV", ratio[["Period", label]].sort_values("Period").to_csv(index=False).encode("utf-8"),
                                   "master_graph_ratio.csv","text/csv", key="mg_dl_ratio")
            else:
                if chart_type in {"Histogram","Bell Curve"}:
                    vals = ratio[label].dropna().astype(float)
                    if vals.empty:
                        st.info("No data to plot a distribution.")
                    else:
                        hist = alt.Chart(ratio.dropna()).mark_bar(opacity=0.9).encode(
                            x=alt.X(f"{label}:Q", bin=alt.Bin(maxbins=30), title=label),
                            y=alt.Y("count():Q", title="Frequency"),
                            tooltip=[alt.Tooltip("count():Q", title="Freq")]
                        ).properties(height=320, title=f"Histogram — {label}")
                        if chart_type == "Bell Curve":
                            mu  = float(vals.mean())
                            sig = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
                            xs = np.linspace(vals.min(), vals.max() if vals.max()!=vals.min() else vals.min()+1.0, 200)
                            pdf = (1.0/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu)/max(sig,1e-9))**2)) if sig>0 else np.zeros_like(xs)
                            pdf = pdf / pdf.max() *  (ratio["count()"].max() if "count()" in ratio.columns else 1.0)
                            bell_df = pd.DataFrame({"x": xs, "pdf": pdf})
                            bell = alt.Chart(bell_df).mark_line().encode(x="x:Q", y="pdf:Q")
                            st.altair_chart(hist + bell, use_container_width=True)
                            st.caption(f"μ = {mu:.3f}, σ = {sig:.3f}")
                        else:
                            st.altair_chart(hist, use_container_width=True)
                else:
                    mark = {"Line":"line","Bar":"bar","Area":"area","Stacked Bar":"bar"}.get(chart_type, "line")
                    base = alt.Chart(ratio)
                    ch = (
                        base.mark_line(point=True) if mark=="line" else
                        base.mark_area(opacity=0.5) if mark=="area" else
                        base.mark_bar(opacity=0.9)
                    ).encode(
                        x=alt.X("Period:T", title="Period"),
                        y=alt.Y(f"{label}:Q", title=label),
                        tooltip=[alt.Tooltip("yearmonthdate(Period):T", title="Period"),
                                 alt.Tooltip(f"{label}:Q", format=".2f" if as_pct else ".3f")]
                    ).properties(height=360, title=f"{label} by {gran}")
                    st.altair_chart(ch, use_container_width=True)

    # run it
    _master_graph_tab()




# =========================================
# HubSpot Deal Score tracker (fresh build)
# =========================================
if view == "HubSpot Deal Score tracker":
    import pandas as pd, numpy as np
    import altair as alt
    from datetime import date, timedelta
    from calendar import monthrange

    st.subheader("HubSpot Deal Score tracker — Score Calibration & Month Prediction")

    # ---------- Helpers ----------
    def _pick(df, preferred, cands):
        if preferred and preferred in df.columns: return preferred
        for c in cands:
            if c in df.columns: return c
        return None

    def month_bounds(d: date):
        return date(d.year, d.month, 1), date(d.year, d.month, monthrange(d.year, d.month)[1])

    # ---------- Resolve columns ----------
    _create = _pick(df_f, globals().get("create_col"),
                    ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _pick(df_f, globals().get("pay_col"),
                    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _score  = _pick(df_f, None,
                    ["HubSpot Deal Score","HubSpot DLSCore","HubSpot DLS Score","Deal Score","HubSpot Score","DLSCore"])

    if not _create or not _pay or not _score:
        st.warning("Need columns: Create Date, Payment Received Date, and HubSpot Deal Score. Please map them.", icon="⚠️")
        st.stop()

    dfm = df_f.copy()
    dfm["__C"] = pd.to_datetime(dfm[_create], errors="coerce", dayfirst=True)
    dfm["__P"] = pd.to_datetime(dfm[_pay],    errors="coerce", dayfirst=True)
    dfm["__S"] = pd.to_numeric(dfm[_score], errors="coerce")  # score as float

    has_score = dfm["__S"].notna()

    # ---------- Controls ----------
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
    with c1:
        lookback = st.selectbox("Lookback (months, exclude current)", [3, 6, 9, 12], index=1)
    with c2:
        n_bins = st.selectbox("# Score bins", [6, 8, 10, 12, 15], index=2)
    with c3:
        ref_age_days = st.number_input("Normalization age (days)", min_value=7, max_value=120, value=30, step=1,
                                       help="Young deals have lower scores; normalize each score up to this age (cap at 1×).")

    # Current month scope
    today_d = date.today()
    mstart_cur, mend_cur = month_bounds(today_d)
    c4, c5 = st.columns(2)
    with c4:
        cur_start = st.date_input("Prediction window start", value=mstart_cur, key="hsdls_cur_start")
    with c5:
        cur_end   = st.date_input("Prediction window end",   value=mend_cur,   key="hsdls_cur_end")
    if cur_end < cur_start:
        st.error("Prediction window end cannot be before start.")
        st.stop()

    # ---------- Build Training (historical) ----------
    cur_per = pd.Period(today_d, freq="M")
    hist_months = [cur_per - i for i in range(1, lookback+1)]
    if not hist_months:
        st.info("No historical months selected.")
        st.stop()

    # A deal is in a historical month by its Create month
    dfm["__Cper"] = dfm["__C"].dt.to_period("M")
    hist_mask = dfm["__Cper"].isin(hist_months) & has_score
    # Label = converted (ever got a payment date)
    dfm["__converted"] = dfm["__P"].notna()

    hist_df = dfm.loc[hist_mask, ["__C","__P","__S","__converted"]].copy()

    if hist_df.empty:
        st.info("No historical rows with HubSpot Deal Score found in the selected lookback.")
        st.stop()

    # ---------- Normalization for "young" deals ----------
    # adjusted_score = score * min(ref_age_days / max(age_days,1), 1.0)
    age_days_hist = (pd.Timestamp(today_d) - hist_df["__C"]).dt.days.clip(lower=1)
    hist_df["__S_adj"] = hist_df["__S"] * np.minimum(ref_age_days / age_days_hist, 1.0)

    # ---------- Learn probability by score range ----------
    # Quantile-based bins for even coverage; fallback to linear if ties dominate
    try:
        q = np.linspace(0, 1, n_bins+1)
        edges = np.unique(np.nanquantile(hist_df["__S_adj"], q))
        if len(edges) < 3:
            raise ValueError
    except Exception:
        smin, smax = float(hist_df["__S_adj"].min()), float(hist_df["__S_adj"].max())
        if smax <= smin:
            smax = smin + 1e-6
        edges = np.linspace(smin, smax, n_bins+1)

    hist_df["__bin"] = pd.cut(hist_df["__S_adj"], bins=edges, include_lowest=True, right=True)

    # Laplace smoothing to avoid 0/100%
    grp = (hist_df.groupby("__bin", observed=True)["__converted"]
                 .agg(Total="count", Conversions="sum"))
    grp["Prob%"] = (grp["Conversions"] + 1) / (grp["Total"] + 2) * 100.0
    grp = grp.reset_index()
    grp["Range"] = grp["__bin"].astype(str)

    # ---------- Show Calibration (bins) ----------
    st.markdown("### Calibration: HubSpot Deal Score → Conversion Probability (historical)")
    left, right = st.columns([2, 1])
    with left:
        if not grp.empty:
            base = alt.Chart(grp).encode(x=alt.X("Range:N", sort=list(grp["Range"])))
            bars = base.mark_bar(opacity=0.9).encode(
                y=alt.Y("Total:Q", title="Count"),
                tooltip=["Range:N","Total:Q","Conversions:Q","Prob%:Q"]
            )
            line = base.mark_line(point=True).encode(
                y=alt.Y("Prob%:Q", title="Conversion Rate (%)", axis=alt.Axis(titleColor="#1f77b4")),
                color=alt.value("#1f77b4")
            )
            st.altair_chart(
                alt.layer(bars, line).resolve_scale(y='independent').properties(
                    height=360, title=f"Learned bins (lookback={lookback} mo, ref age={ref_age_days}d)"
                ),
                use_container_width=True
            )
        else:
            st.info("Not enough data to learn a calibration curve.")
    with right:
        st.dataframe(grp[["Range","Total","Conversions","Prob%"]].sort_values("Range"), use_container_width=True)
        st.download_button(
            "Download bins CSV",
            grp[["Range","Total","Conversions","Prob%"]].to_csv(index=False).encode("utf-8"),
            "hubspot_deal_score_bins.csv","text/csv", key="dl_hs_bins"
        )

    st.markdown("---")

    # ---------- Predict current-month likelihoods ----------
    st.markdown("### Running-month: normalized score → probability & expected conversions")

    cur_mask = dfm["__C"].dt.date.between(cur_start, cur_end) & has_score
    cur_df = dfm.loc[cur_mask, ["__C","__S"]].copy()
    if cur_df.empty:
        st.info("No deals created in the selected prediction window with a HubSpot Deal Score.")
        st.stop()

    cur_age = (pd.Timestamp(today_d) - cur_df["__C"]).dt.days.clip(lower=1)
    cur_df["__S_adj"] = cur_df["__S"] * np.minimum(ref_age_days / cur_age, 1.0)
    cur_df["__bin"] = pd.cut(cur_df["__S_adj"], bins=edges, include_lowest=True, right=True)

    cur_df = cur_df.merge(grp[["__bin","Prob%"]], on="__bin", how="left")
    cur_df["Prob%"] = cur_df["Prob%"].fillna(method="ffill").fillna(method="bfill")
    cur_df["Prob%"] = cur_df["Prob%"].fillna(float(grp["Prob%"].mean() if not grp["Prob%"].empty else 0.0))
    cur_df["Prob"] = cur_df["Prob%"] / 100.0

    expected_conversions = float(cur_df["Prob"].sum())
    total_deals = int(len(cur_df))

    k1, k2, k3 = st.columns(3)
    k1.metric("Deals in window", f"{total_deals:,}", help=f"{cur_start} → {cur_end}")
    k2.metric("Expected conversions (E[∑p])", f"{expected_conversions:.1f}")
    k3.metric("Avg probability", f"{(cur_df['Prob'].mean()*100.0):.1f}%")

    present = (cur_df.groupby("__bin").size().rename("Count").reset_index()
                      .merge(grp[["__bin","Prob%"]], on="__bin", how="left"))
    present["Range"] = present["__bin"].astype(str)
    st.altair_chart(
        alt.Chart(present).mark_bar(opacity=0.9).encode(
            x=alt.X("Range:N", sort=list(grp["Range"]), title="Score range (normalized)"),
            y=alt.Y("Count:Q"),
            tooltip=["Range:N","Count:Q","Prob%:Q"]
        ).properties(height=320, title="Current window — deal count by normalized score bin"),
        use_container_width=True
    )

    with st.expander("Download current-window probabilities"):
        out = cur_df[["__C","__S","__S_adj","Prob%"]].rename(columns={
            "__C":"Create Date", "__S":"HubSpot Deal Score", "__S_adj":f"Score (normalized to {ref_age_days}d)", "Prob%":"Estimated Conversion %"
        })
        st.dataframe(out.head(1000), use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                           "hubspot_deal_score_current_window_probs.csv","text/csv", key="dl_hs_cur")

    st.caption(
        "Notes: Normalization multiplies young-deal scores by min(ref_age / age, 1). "
        "Calibration uses historical lookback (excluding current month) with Laplace smoothing."
    )

