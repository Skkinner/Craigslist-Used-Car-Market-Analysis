"""
Craigslist Used Car Market Analysis — Interactive Streamlit Dashboard
=====================================================================

Continuation of the notebook `Craigslist_Used_Car_Market_Analysis.ipynb`.
The dashboard reads the cleaned dataset produced by that notebook
(`clean_data.csv`, 374,762 rows x 9 columns) and computes every statistic,
ranking and insight from the data that is currently selected in the sidebar.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Craigslist Used Car Market Analysis",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "clean_data.csv"

NUMERIC_COLS = ["price", "year", "odometer"]
CATEGORICAL_COLS = ["manufacturer", "condition", "cylinders", "fuel", "transmission", "type"]
EXPECTED_COLS = ["price", "year", "manufacturer", "condition", "cylinders",
                 "fuel", "odometer", "transmission", "type"]

ORIGINAL_RECORDS = 426_880          # raw Kaggle file, documented in the notebook
SCATTER_SEED = 42                   # same random_state the notebook used for sampling

CONDITION_ORDER = ["new", "like new", "excellent", "good", "fair", "salvage", "unknown"]

COLUMN_LABELS = {
    "price": "Price (USD)",
    "year": "Manufacturing Year",
    "odometer": "Mileage (mi)",
    "manufacturer": "Manufacturer",
    "condition": "Condition",
    "cylinders": "Cylinders",
    "fuel": "Fuel",
    "transmission": "Transmission",
    "type": "Vehicle Type",
}

# Colour system — one palette used everywhere so the dashboard reads as one system.
BLUE = "#2a78d6"        # primary / single-series marks
TEAL = "#1baf7a"
ORANGE = "#eb6834"
NAVY = "#0f1b33"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b4b2aa"     # "other" / "unknown" buckets
NEUTRAL_2 = "#d6d4cc"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# Fixed colour per entity (colour follows the category, never its rank).
FUEL_COLORS = {"gas": CATEGORICAL[0], "diesel": CATEGORICAL[1], "hybrid": CATEGORICAL[2],
               "electric": CATEGORICAL[3], "other": NEUTRAL, "unknown": NEUTRAL_2}
TRANSMISSION_COLORS = {"automatic": CATEGORICAL[0], "manual": CATEGORICAL[1],
                       "other": NEUTRAL, "unknown": NEUTRAL_2}
# Scatter plots: at most three hues + neutral "other/unknown" (readable for colour-blind users).
CONDITION_GROUPS = {"new": "Excellent or better", "like new": "Excellent or better",
                    "excellent": "Excellent or better", "good": "Good",
                    "fair": "Fair / Salvage", "salvage": "Fair / Salvage",
                    "unknown": "Not specified"}
CONDITION_GROUP_COLORS = {"Excellent or better": CATEGORICAL[0], "Good": CATEGORICAL[1],
                          "Fair / Salvage": CATEGORICAL[2], "Not specified": NEUTRAL}

ACRONYMS = {"gmc": "GMC", "bmw": "BMW", "ram": "RAM", "mini": "MINI", "suv": "SUV"}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pretty(value: object) -> str:
    """Human-friendly label for a raw category value (data itself is untouched)."""
    text = str(value)
    if text.lower() in ACRONYMS:
        return ACRONYMS[text.lower()]
    if text == "SUV":
        return text
    return " ".join(part.capitalize() for part in text.replace("-", " ").split())


def fmt_usd(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"${value:,.0f}"


def fmt_int(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f}"


def fmt_miles(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f} mi"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def n_listings(n: float) -> str:
    n = int(n)
    return f"{n:,} listing{'s' if n != 1 else ''}"


def fmt_r(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def corr_strength(r: float | None) -> str:
    """Wording consistent with the notebook (0.353 and -0.517 are both 'moderate')."""
    if r is None:
        return "not computable"
    a = abs(r)
    if a < 0.1:
        label = "negligible"
    elif a < 0.3:
        label = "weak"
    elif a < 0.6:
        label = "moderate"
    else:
        label = "strong"
    if label == "negligible":
        return label
    return f"{label} {'positive' if r > 0 else 'negative'}"


# ---------------------------------------------------------------------------
# Data loading (cached — the CSV is read once, not on every filter change)
# ---------------------------------------------------------------------------
class DataLoadError(Exception):
    """Raised when the CSV is missing or does not match the notebook's output."""


@st.cache_data(show_spinner="Loading the cleaned Craigslist dataset…")
def load_data(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise DataLoadError(
            f"`{file.name}` was not found next to app.py (looked in `{file.parent}`). "
            "Export it from the notebook (section 4: `df_clean.to_csv('clean_data.csv', index=False)`) "
            "and place it in the project folder."
        )
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise DataLoadError(f"`{file.name}` could not be read as CSV: {exc}") from exc

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"`{file.name}` is missing expected column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(EXPECTED_COLS)}."
        )
    if df.empty:
        raise DataLoadError(f"`{file.name}` contains no rows.")

    df = df[EXPECTED_COLS].copy()
    # Same dtypes as the notebook: integer price/year, float odometer.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Categoricals: identical values, much faster filtering and smaller memory.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").fillna("unknown").astype("category")
    return df


def numeric_bounds(series: pd.Series, step: int) -> tuple[int, int]:
    """Slider bounds rounded outward to the step so no record is excluded by default."""
    lo = int(np.floor(series.min() / step) * step)
    hi = int(np.ceil(series.max() / step) * step)
    return lo, hi


# ---------------------------------------------------------------------------
# Filtering (vectorised boolean masks; the cached source frame is never mutated)
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col in ("manufacturer", "condition", "fuel", "transmission", "type"):
        selected = f.get(col) or []
        if selected:                                 # empty selection = "All"
            mask &= df[col].isin(selected).to_numpy()
    for col in ("year", "price", "odometer"):
        lo, hi = f[col]
        values = df[col].to_numpy()
        mask &= (values >= lo) & (values <= hi)
    return df.loc[mask]


def search_frame(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive search across text columns (and exact model year for numbers)."""
    query = query.strip()
    if not query:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for col in CATEGORICAL_COLS:
        cats = df[col].cat.categories.astype(str)
        hits = cats[cats.str.contains(query, case=False, regex=False)]
        if len(hits):
            mask |= df[col].isin(hits).to_numpy()
    if query.isdigit():
        mask |= (df["year"] == int(query)).to_numpy()
    return df.loc[mask]


def filters_key(f: dict) -> tuple:
    """Hashable representation of the filter state (used for cached exports)."""
    return tuple((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in sorted(f.items()))


@st.cache_data(show_spinner="Preparing CSV…", max_entries=8)
def export_csv(key: tuple, query: str, sort_col: str, ascending: bool) -> bytes:
    df = load_data(str(DATA_PATH))
    f = {k: list(v) if isinstance(v, tuple) else v for k, v in key}
    out = search_frame(apply_filters(df, f), query)
    out = out.sort_values(sort_col, ascending=ascending, kind="stable")
    return out.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Statistics helpers (always computed on the full filtered data, never a sample)
# ---------------------------------------------------------------------------
def safe_corr(df: pd.DataFrame, a: str, b: str) -> float | None:
    sub = df[[a, b]].dropna()
    if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
        return None
    r = sub[a].corr(sub[b])
    return None if pd.isna(r) else float(r)


def manufacturer_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (df.groupby("manufacturer", observed=True)["price"]
               .agg(avg="mean", median="median", count="size")
               .reset_index())
    return stats[stats["count"] > 0]


def category_counts(df: pd.DataFrame, col: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = df[col].value_counts()
    counts = counts[counts > 0]
    out = counts.rename_axis(col).reset_index(name="count")
    out[col] = out[col].astype(str)
    out["share"] = out["count"] / out["count"].sum()
    if order:
        rank = {v: i for i, v in enumerate(order)}
        out = out.sort_values(col, key=lambda s: s.map(lambda v: rank.get(v, len(rank))))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, title: str, height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height,
        margin=dict(l=8, r=16, t=56, b=8),
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    title_text="", font=dict(size=12, color=INK_2)),
        bargap=0.15,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# Styling (custom CSS only where Streamlit has no native option)
# ---------------------------------------------------------------------------
CSS = f"""
<style>
:root {{ --navy: {NAVY}; --blue: {BLUE}; --teal: {TEAL}; }}
.stApp {{ background: #f5f6f8; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Sidebar — dark navy */
section[data-testid="stSidebar"] {{ background: var(--navy); }}
section[data-testid="stSidebar"] * {{ color: #e6ebf5; }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #ffffff; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background: #17264a; border-color: #2b3d68;
}}
section[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: var(--blue) !important; }}
section[data-testid="stSidebar"] [data-baseweb="tag"] span {{ color: #fff !important; }}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color: #9fb0d0; }}
section[data-testid="stSidebar"] .stButton button {{
    background: transparent; border: 1px solid #4a5f8f; color: #ffffff; width: 100%;
}}
section[data-testid="stSidebar"] .stButton button:hover {{ border-color: var(--teal); color: var(--teal); }}
section[data-testid="stSidebar"] hr {{ border-color: #24365e; }}
.sb-desc {{ font-size: 0.86rem; line-height: 1.5; color: #c6d0e4 !important; }}
.sb-count {{ background: #17264a; border: 1px solid #2b3d68; border-radius: 12px;
            padding: 12px 14px; margin: 6px 0 4px 0; }}
.sb-count .n {{ font-size: 1.45rem; font-weight: 700; color: #ffffff !important; }}
.sb-count .s {{ font-size: 0.8rem; color: #9fb0d0 !important; }}
.sb-bar {{ height: 6px; border-radius: 3px; background: #24365e; margin-top: 8px; overflow: hidden; }}
.sb-bar > div {{ height: 100%; background: var(--teal); border-radius: 3px; }}

/* Header */
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #1d3a75 100%); border-radius: 16px;
         padding: 22px 26px; color: #fff; margin-bottom: 18px; }}
.hero h1 {{ color: #ffffff; font-size: 1.75rem; margin: 0 0 4px 0; padding: 0; }}
.hero p {{ color: #c6d0e4; margin: 0; font-size: 0.95rem; }}
.hero .chips {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }}
.hero .chip {{ background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
               border-radius: 999px; padding: 3px 12px; font-size: 0.8rem; color: #e6ebf5; }}

/* KPI cards */
.kpi {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 14px;
        padding: 16px 18px; height: 100%; box-shadow: 0 1px 2px rgba(15,27,51,0.05);
        border-top: 3px solid var(--blue); }}
.kpi.teal {{ border-top-color: var(--teal); }}
.kpi .label {{ font-size: 0.78rem; font-weight: 600; letter-spacing: .03em; text-transform: uppercase;
               color: {INK_2}; }}
.kpi .value {{ font-size: 1.6rem; font-weight: 700; color: {INK}; margin-top: 4px; white-space: nowrap; }}
.kpi .sub {{ font-size: 0.78rem; color: {MUTED}; margin-top: 2px; }}

/* Section headings & notes */
.section-title {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 6px 0 2px 0; }}
.section-sub {{ font-size: 0.9rem; color: {INK_2}; margin-bottom: 10px; }}
.note {{ font-size: 0.86rem; color: {INK_2}; line-height: 1.5; }}

/* Correlation stat */
.stat {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px;
         padding: 12px 14px; margin-bottom: 10px; }}
.stat .label {{ font-size: 0.76rem; font-weight: 600; text-transform: uppercase; color: {INK_2}; }}
.stat .value {{ font-size: 1.5rem; font-weight: 700; color: {INK}; }}
.stat .sub {{ font-size: 0.8rem; color: {MUTED}; }}

/* Insight cards */
.insight {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-left: 4px solid var(--blue);
            border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.insight.teal {{ border-left-color: var(--teal); }}
.insight h4 {{ margin: 0 0 4px 0; font-size: 0.98rem; color: {INK}; }}
.insight p {{ margin: 0; font-size: 0.9rem; color: {INK_2}; line-height: 1.55; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08);
        border-radius: 10px 10px 0 0; padding: 8px 14px; }}
.stTabs [aria-selected="true"] {{ color: var(--blue) !important; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #ffffff; border-radius: 14px; }}

@media (max-width: 640px) {{
  .hero h1 {{ font-size: 1.3rem; }}
  .kpi .value {{ font-size: 1.3rem; }}
}}
</style>
"""


def kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> str:
    return (f'<div class="kpi {accent}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def stat_box(label: str, value: str, sub: str) -> None:
    st.markdown(f'<div class="stat"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>'
                + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def init_state(defaults: dict) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_filters(defaults: dict) -> None:
    for key, value in defaults.items():
        st.session_state[key] = value


def sidebar(df: pd.DataFrame) -> dict:
    year_bounds = (int(df["year"].min()), int(df["year"].max()))
    price_bounds = numeric_bounds(df["price"], 500)
    miles_bounds = numeric_bounds(df["odometer"], 1000)

    options = {
        "manufacturer": sorted(df["manufacturer"].cat.categories.astype(str)),
        "condition": [c for c in CONDITION_ORDER if c in set(df["condition"].cat.categories)]
                     + sorted(set(df["condition"].cat.categories) - set(CONDITION_ORDER)),
        "fuel": sorted(df["fuel"].cat.categories.astype(str)),
        "transmission": sorted(df["transmission"].cat.categories.astype(str)),
        "type": sorted(df["type"].cat.categories.astype(str), key=str.lower),
    }
    defaults = {
        "f_manufacturer": [], "f_condition": [], "f_fuel": [], "f_transmission": [], "f_type": [],
        "f_year": year_bounds, "f_price": price_bounds, "f_odometer": miles_bounds,
    }
    init_state(defaults)

    with st.sidebar:
        st.markdown("## 🚗 Used Car Market")
        st.markdown(
            '<div class="sb-desc">Listings scraped from <b>Craigslist</b> across the United States '
            "(Kaggle: <i>Craigslist Cars and Trucks Data</i>). The notebook cleaned "
            f"{ORIGINAL_RECORDS:,} raw listings down to <b>{len(df):,}</b> records with 9 features "
            "(price, year, manufacturer, condition, cylinders, fuel, odometer, transmission, type).<br><br>"
            "Use the filters below to explore how <b>price</b> relates to model year, mileage, "
            "manufacturer and vehicle characteristics.</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Filters")
        st.button("↺  Reset filters", on_click=reset_filters, args=(defaults,), use_container_width=True)

        st.multiselect("Manufacturer", options["manufacturer"], key="f_manufacturer",
                       format_func=pretty, placeholder="All manufacturers")
        st.slider("Manufacturing year", year_bounds[0], year_bounds[1], key="f_year", step=1)
        st.slider("Price (USD)", price_bounds[0], price_bounds[1], key="f_price", step=500,
                  format="$%d")
        st.slider("Mileage (odometer)", miles_bounds[0], miles_bounds[1], key="f_odometer",
                  step=1000, format="%d mi")
        st.multiselect("Vehicle condition", options["condition"], key="f_condition",
                       format_func=pretty, placeholder="All conditions")
        st.multiselect("Fuel type", options["fuel"], key="f_fuel",
                       format_func=pretty, placeholder="All fuel types")
        st.multiselect("Transmission", options["transmission"], key="f_transmission",
                       format_func=pretty, placeholder="All transmissions")
        st.multiselect("Vehicle type", options["type"], key="f_type",
                       format_func=pretty, placeholder="All vehicle types")

        count_slot = st.empty()
        st.divider()
        st.caption("Source: [Kaggle — austinreese/craigslist-carstrucks-data]"
                   "(https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)")

    filters = {
        "manufacturer": st.session_state["f_manufacturer"],
        "condition": st.session_state["f_condition"],
        "fuel": st.session_state["f_fuel"],
        "transmission": st.session_state["f_transmission"],
        "type": st.session_state["f_type"],
        "year": tuple(st.session_state["f_year"]),
        "price": tuple(st.session_state["f_price"]),
        "odometer": tuple(st.session_state["f_odometer"]),
    }
    filters["_count_slot"] = count_slot
    filters["_active"] = sum(
        bool(st.session_state[k]) if isinstance(defaults[k], list) else tuple(st.session_state[k]) != tuple(defaults[k])
        for k in defaults
    )
    return filters


def render_count(slot, n: int, total: int, active: int) -> None:
    share = n / total if total else 0
    slot.markdown(
        f'<div class="sb-count"><div class="s">Matching listings</div>'
        f'<div class="n">{n:,}</div>'
        f'<div class="s">{fmt_pct(share)} of {total:,} · {active} filter{"s" if active != 1 else ""} active</div>'
        f'<div class="sb-bar"><div style="width:{share * 100:.2f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_header(df: pd.DataFrame) -> None:
    st.markdown(
        f"""<div class="hero"><h1>Craigslist Used Car Market Analysis</h1>
        <p>Interactive exploration of used-vehicle listing prices, vehicle characteristics and
        manufacturer differences — built on the cleaned dataset from the analysis notebook.</p>
        <div class="chips"><span class="chip">{ORIGINAL_RECORDS:,} raw listings</span>
        <span class="chip">{len(df):,} cleaned records</span>
        <span class="chip">{df.shape[1]} features</span>
        <span class="chip">{int(df.isna().sum().sum())} missing values</span>
        <span class="chip">Model years {int(df['year'].min())}–{int(df['year'].max())}</span></div></div>""",
        unsafe_allow_html=True,
    )


def render_kpis(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    section("Market Overview", "Key figures for the current filter selection; the second line shows the full cleaned dataset for reference.")
    cols = st.columns(5)
    cards = [
        ("Total listings", fmt_int(len(fdf)), f"{fmt_pct(len(fdf) / len(df))} of {len(df):,}", ""),
        ("Average price", fmt_usd(fdf["price"].mean()), f"All data: {fmt_usd(df['price'].mean())}", ""),
        ("Median price", fmt_usd(fdf["price"].median()), f"All data: {fmt_usd(df['price'].median())}", "teal"),
        ("Average mileage", fmt_miles(fdf["odometer"].mean()), f"All data: {fmt_miles(df['odometer'].mean())}", ""),
        ("Average model year", "—" if fdf.empty else f"{fdf['year'].mean():.1f}",
         f"All data: {df['year'].mean():.1f}", "teal"),
    ]
    for col, (label, value, sub, accent) in zip(cols, cards):
        col.markdown(kpi_card(label, value, sub, accent), unsafe_allow_html=True)
    st.write("")


def render_overview(fdf: pd.DataFrame) -> None:
    section("Data Preview", "First rows of the filtered dataset (unmodified values from clean_data.csv).")
    st.dataframe(fdf.head(10), use_container_width=True, hide_index=True,
                 column_config=table_column_config())

    left, right = st.columns([1.15, 1])
    with left:
        section("Summary Statistics — numeric features")
        desc = fdf[NUMERIC_COLS].describe().T
        fmt = {"price": fmt_usd, "year": lambda v: f"{v:,.1f}" if pd.notna(v) else "—", "odometer": fmt_miles}
        table = pd.DataFrame({
            "Feature": [COLUMN_LABELS[c] for c in desc.index],
            "Count": [fmt_int(v) for v in desc["count"]],
            "Mean": [fmt[c](desc.loc[c, "mean"]) for c in desc.index],
            "Std. dev.": [fmt[c](desc.loc[c, "std"]) for c in desc.index],
            "Min": [fmt[c](desc.loc[c, "min"]) if c != "year" else fmt_int(desc.loc[c, "min"]).replace(",", "") for c in desc.index],
            "Median": [fmt[c](desc.loc[c, "50%"]) if c != "year" else fmt_int(desc.loc[c, "50%"]).replace(",", "") for c in desc.index],
            "Max": [fmt[c](desc.loc[c, "max"]) if c != "year" else fmt_int(desc.loc[c, "max"]).replace(",", "") for c in desc.index],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
    with right:
        section("Summary Statistics — categorical features")
        rows = []
        for col in CATEGORICAL_COLS:
            vc = fdf[col].value_counts()
            vc = vc[vc > 0]
            rows.append({
                "Feature": COLUMN_LABELS[col],
                "Unique values": len(vc),
                "Most frequent": pretty(vc.index[0]) if len(vc) else "—",
                "Share": fmt_pct(vc.iloc[0] / vc.sum()) if len(vc) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_price_distribution(fdf: pd.DataFrame) -> None:
    section("Price Distribution",
            "How listing prices are spread across the selected vehicles. Each bar counts the listings whose "
            "price falls in that price band; the dashed lines mark the average and median price.")
    c1, c2 = st.columns([3, 1])
    with c2:
        bins = st.slider("Number of bins", 10, 200, 60, step=5, key="hist_bins")
        log_y = st.checkbox("Log scale for counts", value=False, key="hist_log",
                            help="Makes the long tail of expensive listings easier to see.")
    prices = fdf["price"].to_numpy(dtype=float)
    mean_p, median_p = float(np.mean(prices)), float(np.median(prices))

    # Binning is done server-side so only `bins` bars are sent to the browser.
    lo, hi = prices.min(), prices.max()
    if lo == hi:
        lo, hi = lo - 500, hi + 500
    counts, edges = np.histogram(prices, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    hover = [f"${a:,.0f} – ${b:,.0f}" for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure(go.Bar(
        x=centers, y=counts, width=widths, marker=dict(color=BLUE, line=dict(color="#ffffff", width=1)),
        customdata=np.column_stack([hover, counts / counts.sum() * 100]),
        hovertemplate="<b>%{customdata[0]}</b><br>Listings: %{y:,}<br>Share: %{customdata[1]:.2f}%<extra></extra>",
        name="Listings",
    ))
    fig.add_vline(x=mean_p, line=dict(color=ORANGE, width=2, dash="dash"))
    fig.add_vline(x=median_p, line=dict(color=NAVY, width=2, dash="dot"))
    fig.add_annotation(x=mean_p, y=1, yref="paper", text=f"Average {fmt_usd(mean_p)}", showarrow=False,
                       xanchor="left" if mean_p >= median_p else "right", xshift=6 if mean_p >= median_p else -6,
                       yanchor="top", font=dict(color=ORANGE, size=12), bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x=median_p, y=0.9, yref="paper", text=f"Median {fmt_usd(median_p)}", showarrow=False,
                       xanchor="right" if mean_p >= median_p else "left", xshift=-6 if mean_p >= median_p else 6,
                       yanchor="top", font=dict(color=NAVY, size=12), bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title_text="Listing price (USD)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title_text="Number of listings", tickformat=",", type="log" if log_y else "linear")
    style_fig(fig, "Distribution of listing prices", height=430, legend=False)
    with c1:
        show_chart(fig)
    with c2:
        q1, q3 = np.percentile(prices, [25, 75])
        stat_box("Middle 50% of prices", f"{fmt_usd(q1)} – {fmt_usd(q3)}", "Interquartile range")
        skew_note = ("right-skewed: a tail of expensive listings pulls the average above the median"
                     if mean_p > median_p * 1.02 else
                     "left-skewed: cheap listings pull the average below the median"
                     if mean_p < median_p * 0.98 else "roughly symmetric around the median")
        st.markdown(f'<div class="note">The selection is <b>{skew_note}</b>.</div>', unsafe_allow_html=True)


def scatter_color(sample: pd.DataFrame, fdf: pd.DataFrame, mode: str):
    """Returns (series, colour map). Max three hues + neutral to stay colour-blind readable."""
    if mode == "Condition":
        groups = sample["condition"].astype(str).map(lambda c: CONDITION_GROUPS.get(c, "Not specified"))
        return groups, CONDITION_GROUP_COLORS, list(CONDITION_GROUP_COLORS)
    if mode == "Manufacturer (top 3)":
        top = fdf["manufacturer"].value_counts().head(3).index.astype(str).tolist()
        labels = sample["manufacturer"].astype(str).map(lambda m: pretty(m) if m in top else "Other manufacturers")
        cmap = {pretty(m): CATEGORICAL[i] for i, m in enumerate(top)}
        cmap["Other manufacturers"] = NEUTRAL
        return labels, cmap, [pretty(m) for m in top] + ["Other manufacturers"]
    return None, None, None


def render_relationship(fdf: pd.DataFrame, df: pd.DataFrame, x: str, title: str, color_default: str,
                        key: str, explain: str) -> None:
    r = safe_corr(fdf, x, "price")
    r_all = safe_corr(df, x, "price")
    ctrl, chart = st.columns([1, 3])
    with ctrl:
        stat_box("Pearson correlation (r)", fmt_r(r),
                 f"{corr_strength(r).capitalize()} · {n_listings(len(fdf))}")
        st.markdown(f'<div class="note">Full cleaned dataset: <b>{fmt_r(r_all)}</b> '
                    f'({corr_strength(r_all)}). The value above is recalculated for your filters.</div>',
                    unsafe_allow_html=True)
        st.write("")
        color_mode = st.radio("Colour points by", ["Condition", "Manufacturer (top 3)", "None"],
                              index=["Condition", "Manufacturer (top 3)", "None"].index(color_default),
                              key=f"{key}_color")
        n_max = len(fdf)
        default_n = min(10_000, n_max)
        if n_max > 1_000:
            sample_n = st.slider("Points plotted (sample)", 1_000, min(30_000, n_max), default_n,
                                 step=1_000 if n_max >= 2_000 else 1, key=f"{key}_n")
        else:
            sample_n = n_max
        trend = st.checkbox("Show linear trend line", value=True, key=f"{key}_trend")
        st.markdown(f'<div class="note">{explain}</div>', unsafe_allow_html=True)

    sample = fdf.sample(n=sample_n, random_state=SCATTER_SEED) if sample_n < len(fdf) else fdf
    plot_df = pd.DataFrame({
        x: sample[x].to_numpy(),
        "price": sample["price"].to_numpy(),
        "Manufacturer": sample["manufacturer"].astype(str).map(pretty).to_numpy(),
        "Year": sample["year"].to_numpy(),
        "Mileage": sample["odometer"].to_numpy(),
        "Condition": sample["condition"].astype(str).map(pretty).to_numpy(),
        "Type": sample["type"].astype(str).map(pretty).to_numpy(),
    })
    labels, cmap, order = scatter_color(sample, fdf, color_mode)
    if labels is not None:
        plot_df["Group"] = labels.to_numpy()

    fig = px.scatter(
        plot_df, x=x, y="price",
        color="Group" if labels is not None else None,
        color_discrete_map=cmap, category_orders={"Group": order} if order else None,
        custom_data=["Manufacturer", "Year", "Mileage", "Condition", "Type"],
        render_mode="webgl",
    )
    marker = dict(size=6, opacity=0.45)
    if labels is None:
        marker["color"] = BLUE
    fig.update_traces(
        marker=marker,
        hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                       "Price: $%{y:,.0f}<br>Mileage: %{customdata[2]:,.0f} mi<br>"
                       "Condition: %{customdata[3]}<br>Type: %{customdata[4]}<extra></extra>"),
    )
    # Trend line fitted on the FULL filtered data (not just the plotted sample).
    if trend and r is not None:
        xs = fdf[x].to_numpy(dtype=float)
        slope, intercept = np.polyfit(xs, fdf["price"].to_numpy(dtype=float), 1)
        line_x = np.array([xs.min(), xs.max()])
        fig.add_trace(go.Scatter(
            x=line_x, y=slope * line_x + intercept, mode="lines", name="Linear trend",
            line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip",
        ))
    if x == "year":
        fig.update_xaxes(title_text="Manufacturing year", tickformat="d")
    else:
        fig.update_xaxes(title_text="Mileage (odometer, miles)", tickformat=",.0f", ticksuffix=" mi")
    fig.update_yaxes(title_text="Listing price (USD)", tickprefix="$", tickformat=",.0f")
    style_fig(fig, title, height=480, legend=True)
    with chart:
        show_chart(fig)
        if sample_n < len(fdf):
            st.caption(f"Plotting a reproducible random sample of {sample_n:,} of {n_listings(len(fdf))} "
                       f"(random_state={SCATTER_SEED}). Correlation and trend line use all {n_listings(len(fdf))}.")


def render_manufacturers(fdf: pd.DataFrame) -> None:
    section("Manufacturer Comparison",
            "Ranking of manufacturers for the current selection. Hover a bar to see the number of listings "
            "behind each figure — averages based on few listings are less reliable.")
    stats = manufacturer_stats(fdf)
    c1, c2, c3 = st.columns([1.4, 1, 1])
    metric = c1.radio("Rank by", ["Average price", "Median price", "Listing count"], horizontal=True,
                      key="mfr_metric")
    top_n = c2.slider("Manufacturers shown", 5, 25, 10, key="mfr_top")
    max_count = int(stats["count"].max()) if len(stats) else 1
    min_n = c3.number_input("Minimum listings per manufacturer", min_value=1, max_value=max(1, max_count),
                            value=1, step=10, key="mfr_min",
                            help="Raise this to exclude manufacturers with very few listings (e.g. exotic brands).")
    stats = stats[stats["count"] >= min_n]
    if stats.empty:
        st.info("No manufacturer has that many listings in the current selection — lower the minimum.")
        return

    col = {"Average price": "avg", "Median price": "median", "Listing count": "count"}[metric]
    top = stats.sort_values(col, ascending=False).head(top_n).iloc[::-1]
    names = top["manufacturer"].astype(str).map(pretty)
    text = top[col].map(fmt_int) if col == "count" else top[col].map(fmt_usd)
    fig = go.Figure(go.Bar(
        x=top[col], y=names, orientation="h", text=text, textposition="outside",
        cliponaxis=False, marker=dict(color=BLUE if col != "count" else TEAL),
        customdata=np.column_stack([top["avg"], top["median"], top["count"]]),
        hovertemplate=("<b>%{y}</b><br>Average price: $%{customdata[0]:,.0f}<br>"
                       "Median price: $%{customdata[1]:,.0f}<br>Listings: %{customdata[2]:,}<extra></extra>"),
    ))
    fig.update_xaxes(title_text=metric if col == "count" else f"{metric} (USD)",
                     tickprefix="" if col == "count" else "$", tickformat=",.0f",
                     range=[0, float(top[col].max()) * 1.18])
    fig.update_yaxes(title_text="", tickfont=dict(color=INK_2, size=12), showgrid=False)
    style_fig(fig, f"Top {len(top)} manufacturers by {metric.lower()}", height=max(360, 34 * len(top) + 110),
              legend=False)
    show_chart(fig)

    leader = top.iloc[-1]
    st.markdown(
        f'<div class="note">Leader by {metric.lower()}: <b>{pretty(leader["manufacturer"])}</b> — '
        f'average {fmt_usd(leader["avg"])}, median {fmt_usd(leader["median"])}, '
        f'{n_listings(leader["count"])}. {len(stats):,} manufacturers meet the minimum of {min_n:,} listing(s).</div>',
        unsafe_allow_html=True)
    with st.expander("Show manufacturer table"):
        table = stats.sort_values(col, ascending=False).assign(manufacturer=lambda d: d["manufacturer"].astype(str).map(pretty))
        st.dataframe(table.rename(columns={"manufacturer": "Manufacturer", "avg": "Average price",
                                           "median": "Median price", "count": "Listings"}),
                     use_container_width=True, hide_index=True,
                     column_config={"Average price": st.column_config.NumberColumn(format="$%d"),
                                    "Median price": st.column_config.NumberColumn(format="$%d")})


def bar_chart(counts: pd.DataFrame, col: str, title: str, horizontal: bool = False) -> go.Figure:
    names = counts[col].map(pretty)
    colors = [NEUTRAL if v in ("unknown", "other") else BLUE for v in counts[col]]
    hover = "<b>%{customdata[0]}</b><br>Listings: %{customdata[1]:,}<br>Share: %{customdata[2]:.1%}<extra></extra>"
    custom = np.column_stack([names, counts["count"], counts["share"]])
    if horizontal:
        fig = go.Figure(go.Bar(x=counts["count"][::-1], y=names[::-1], orientation="h",
                               marker=dict(color=colors[::-1]), customdata=custom[::-1],
                               text=counts["share"][::-1].map(lambda s: f"{s:.1%}"), textposition="outside",
                               cliponaxis=False, hovertemplate=hover))
        fig.update_xaxes(title_text="Listings", tickformat=",", range=[0, counts["count"].max() * 1.18])
        fig.update_yaxes(showgrid=False)
    else:
        fig = go.Figure(go.Bar(x=names, y=counts["count"], marker=dict(color=colors), customdata=custom,
                               text=counts["share"].map(lambda s: f"{s:.1%}"), textposition="outside",
                               cliponaxis=False, hovertemplate=hover))
        fig.update_yaxes(title_text="Listings", tickformat=",", range=[0, counts["count"].max() * 1.18])
        fig.update_xaxes(showgrid=False)
    return style_fig(fig, title, height=380 if not horizontal else 470, legend=False)


def donut_chart(counts: pd.DataFrame, col: str, title: str, color_map: dict) -> go.Figure:
    names = counts[col].map(pretty)
    colors = [color_map.get(v, NEUTRAL) for v in counts[col]]
    fig = go.Figure(go.Pie(
        labels=names, values=counts["count"], hole=0.55, sort=False, direction="clockwise",
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent", textposition="outside", insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>Listings: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    total = int(counts["count"].sum())
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:11px'>listings</span>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=15, color=INK))
    style_fig(fig, title, height=380, legend=False)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=30))
    return fig


def render_characteristics(fdf: pd.DataFrame) -> None:
    section("Vehicle Characteristics",
            "Composition of the selected listings. Grey marks the 'unknown' / 'other' values that the notebook "
            "kept (missing categories were filled with 'unknown' rather than dropped).")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            show_chart(bar_chart(category_counts(fdf, "condition", CONDITION_ORDER), "condition",
                                 "Vehicle condition distribution"))
    with r1c2:
        with st.container(border=True):
            fuel = category_counts(fdf, "fuel", ["gas", "diesel", "hybrid", "electric", "other", "unknown"])
            show_chart(donut_chart(fuel, "fuel", "Fuel type distribution", FUEL_COLORS))
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with st.container(border=True):
            trans = category_counts(fdf, "transmission", ["automatic", "manual", "other", "unknown"])
            show_chart(donut_chart(trans, "transmission", "Transmission type distribution", TRANSMISSION_COLORS))
    with r2c2:
        with st.container(border=True):
            types = category_counts(fdf, "type")
            show_chart(bar_chart(types, "type", "Vehicle type distribution", horizontal=True))


def render_correlation(fdf: pd.DataFrame) -> None:
    section("Correlation Analysis",
            "Pearson correlation between the three numeric features of the cleaned dataset, "
            "calculated on all listings in the current selection.")
    if len(fdf) < 3:
        st.info("At least 3 listings are needed to compute correlations. Broaden the filters.")
        return
    labels = ["Price", "Manufacturing Year", "Odometer"]
    matrix = np.full((3, 3), np.nan)
    for i, a in enumerate(NUMERIC_COLS):
        for j, b in enumerate(NUMERIC_COLS):
            if a == b:
                matrix[i, j] = 1.0 if fdf[a].nunique() > 1 else np.nan
            else:
                r = safe_corr(fdf, a, b)
                matrix[i, j] = np.nan if r is None else r
    text = [[("n/a" if np.isnan(v) else f"{v:+.3f}") for v in row] for row in matrix]
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure(go.Heatmap(
            z=matrix, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING,
            text=text, texttemplate="%{text}", textfont=dict(size=15, color=INK),
            xgap=3, ygap=3, colorbar=dict(title="r", thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
            hovertemplate="%{y} vs %{x}<br>r = %{text}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, side="bottom")
        style_fig(fig, "Correlation matrix (Pearson r)", height=440, legend=False)
        show_chart(fig)
        if np.isnan(matrix).any():
            st.caption("'n/a' means the correlation cannot be computed because a feature has no variation "
                       "in the current selection (e.g. a single model year).")
    with right:
        r_year, r_miles, r_ym = matrix[0, 1], matrix[0, 2], matrix[1, 2]
        st.markdown(
            f"""<div class="insight"><h4>How to read it</h4><p>
            <b>Positive</b> values (blue) mean two features rise together; <b>negative</b> values (red) mean one
            falls as the other rises. Values near <b>0</b> mean no linear relationship. Correlation describes
            association, not causation, and only captures straight-line patterns.</p></div>
            <div class="insight teal"><h4>Price & manufacturing year: {fmt_r(None if np.isnan(r_year) else r_year)}</h4><p>
            {corr_strength(None if np.isnan(r_year) else r_year).capitalize()} — {'newer vehicles tend to be listed at higher prices.' if not np.isnan(r_year) and r_year > 0.1 else 'model year shows little linear link with price here.' if not np.isnan(r_year) else 'not computable for this selection.'}</p></div>
            <div class="insight"><h4>Price & odometer: {fmt_r(None if np.isnan(r_miles) else r_miles)}</h4><p>
            {corr_strength(None if np.isnan(r_miles) else r_miles).capitalize()} — {'higher-mileage vehicles tend to be listed at lower prices.' if not np.isnan(r_miles) and r_miles < -0.1 else 'mileage shows little linear link with price here.' if not np.isnan(r_miles) else 'not computable for this selection.'}</p></div>
            <div class="insight teal"><h4>Manufacturing year & odometer: {fmt_r(None if np.isnan(r_ym) else r_ym)}</h4><p>
            {corr_strength(None if np.isnan(r_ym) else r_ym).capitalize()} — older vehicles usually have accumulated more miles, so year and
            mileage partly overlap as price drivers.</p></div>""",
            unsafe_allow_html=True,
        )


def build_insights(fdf: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Every sentence is computed from the filtered data. Returns (title, body, accent)."""
    out: list[tuple[str, str, str]] = []
    n = len(fdf)

    # 1. Model year vs price
    r_year = safe_corr(fdf, "year", "price")
    if r_year is not None:
        y_lo, y_hi = fdf["year"].quantile([0.25, 0.75])
        old = fdf.loc[fdf["year"] <= y_lo, "price"].median()
        new = fdf.loc[fdf["year"] >= y_hi, "price"].median()
        ratio = (new / old) if old else np.nan
        body = (f"Correlation r = <b>{fmt_r(r_year)}</b> ({corr_strength(r_year)}). Listings from model year "
                f"<b>{int(y_hi)} or newer</b> have a median price of <b>{fmt_usd(new)}</b>, versus "
                f"<b>{fmt_usd(old)}</b> for model year <b>{int(y_lo)} or older</b>")
        body += (f" — about <b>{ratio:.1f}×</b> as high." if pd.notna(ratio) and y_lo != y_hi and ratio >= 1.05
                 else ".")
        out.append(("Vehicle age and price", body, ""))
    else:
        out.append(("Vehicle age and price", "Not enough variation in model year or price to measure a relationship.", ""))

    # 2. Mileage vs price
    r_miles = safe_corr(fdf, "odometer", "price")
    if r_miles is not None:
        m_lo, m_hi = fdf["odometer"].quantile([0.25, 0.75])
        low = fdf.loc[fdf["odometer"] <= m_lo, "price"].median()
        high = fdf.loc[fdf["odometer"] >= m_hi, "price"].median()
        body = (f"Correlation r = <b>{fmt_r(r_miles)}</b> ({corr_strength(r_miles)}). The lowest-mileage quarter "
                f"(≤ {fmt_miles(m_lo)}) has a median price of <b>{fmt_usd(low)}</b>; the highest-mileage quarter "
                f"(≥ {fmt_miles(m_hi)}) has <b>{fmt_usd(high)}</b>.")
        if r_year is not None:
            stronger = "mileage" if abs(r_miles) > abs(r_year) else "model year"
            body += f" In this selection, <b>{stronger}</b> has the stronger linear relationship with price."
        out.append(("Mileage and price", body, "teal"))
    else:
        out.append(("Mileage and price", "Not enough variation in mileage or price to measure a relationship.", "teal"))

    # 3. Manufacturers
    stats = manufacturer_stats(fdf)
    if len(stats):
        top = stats.sort_values("avg", ascending=False).iloc[0]
        body = (f"<b>{pretty(top['manufacturer'])}</b> has the highest average listing price at "
                f"<b>{fmt_usd(top['avg'])}</b> (median {fmt_usd(top['median'])}, {n_listings(top['count'])}).")
        robust = stats[stats["count"] >= 100]
        if len(robust) and robust.sort_values("avg", ascending=False).iloc[0]["manufacturer"] != top["manufacturer"]:
            rt = robust.sort_values("avg", ascending=False).iloc[0]
            body += (f" Among manufacturers with at least 100 listings, <b>{pretty(rt['manufacturer'])}</b> leads at "
                     f"{fmt_usd(rt['avg'])} ({n_listings(rt['count'])}).")
        most = stats.sort_values("count", ascending=False).iloc[0]
        body += (f" The most-listed manufacturer is <b>{pretty(most['manufacturer'])}</b> "
                 f"({n_listings(most['count'])}, {fmt_pct(most['count'] / n)} of the selection).")
        out.append(("Manufacturers", body, ""))

    # 4. Condition
    cond = fdf["condition"].value_counts()
    cond = cond[cond > 0]
    if len(cond):
        known = cond.drop(labels=["unknown"], errors="ignore")
        unknown_share = cond.get("unknown", 0) / cond.sum()
        if len(known):
            body = (f"The most common reported condition is <b>{pretty(known.index[0])}</b> "
                    f"({n_listings(known.iloc[0])}, {fmt_pct(known.iloc[0] / cond.sum())}).")
        else:
            body = "No listing in this selection reports a condition."
        if unknown_share > 0:
            body += f" Condition was not specified for <b>{fmt_pct(unknown_share)}</b> of listings."
        out.append(("Vehicle condition", body, "teal"))

    # 5. Price distribution
    mean_p, median_p = fdf["price"].mean(), fdf["price"].median()
    q1, q3 = fdf["price"].quantile([0.25, 0.75])
    shape = ("right-skewed — a smaller number of expensive listings pulls the average above the median"
             if mean_p > median_p * 1.02 else "close to symmetric")
    out.append(("Price distribution",
                f"The average price is <b>{fmt_usd(mean_p)}</b> and the median is <b>{fmt_usd(median_p)}</b>; "
                f"half of the listings fall between <b>{fmt_usd(q1)}</b> and <b>{fmt_usd(q3)}</b>. "
                f"The distribution is {shape}.", ""))

    # 6. Popular configuration
    typ = fdf["type"].value_counts().drop(labels=["unknown"], errors="ignore")
    fuel = fdf["fuel"].value_counts()
    trans = fdf["transmission"].value_counts()
    if len(typ) and typ.iloc[0] > 0:
        out.append(("Typical listing",
                    f"The most common known body type is <b>{pretty(typ.index[0])}</b> "
                    f"({fmt_pct(typ.iloc[0] / n)}); <b>{pretty(fuel.index[0])}</b> is the dominant fuel "
                    f"({fmt_pct(fuel.iloc[0] / n)}) and <b>{pretty(trans.index[0])}</b> the most common "
                    f"transmission ({fmt_pct(trans.iloc[0] / n)}).", "teal"))
    return out


def render_insights(fdf: pd.DataFrame, df: pd.DataFrame) -> None:
    section("Key Insights & Findings",
            f"Generated from the <b>{len(fdf):,}</b> listings matching the current filters — they change as you "
            "filter, so they describe the selection, not the whole market.")
    insights = build_insights(fdf)
    left, right = st.columns(2)
    for i, (title, body, accent) in enumerate(insights):
        (left if i % 2 == 0 else right).markdown(
            f'<div class="insight {accent}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with st.expander("Reference: findings on the full cleaned dataset (notebook results, recomputed)"):
        stats = manufacturer_stats(df).sort_values("avg", ascending=False).head(3)
        st.markdown(
            f"- Cleaning reduced **{ORIGINAL_RECORDS:,}** raw listings to **{len(df):,}** records "
            f"({ORIGINAL_RECORDS - len(df):,} removed) with 9 features and no missing values.\n"
            f"- Price vs manufacturing year: r = **{fmt_r(safe_corr(df, 'year', 'price'))}** "
            f"({corr_strength(safe_corr(df, 'year', 'price'))}).\n"
            f"- Price vs mileage: r = **{fmt_r(safe_corr(df, 'odometer', 'price'))}** "
            f"({corr_strength(safe_corr(df, 'odometer', 'price'))}) — the stronger linear relationship.\n"
            f"- Highest average prices: "
            + ", ".join(f"**{pretty(r.manufacturer)}** ({fmt_usd(r.avg)}, {n_listings(r['count'])})"
                        for _, r in stats.iterrows())
            + ".\n- Prices were filtered to $1,000–$200,000, mileage to ≤ 500,000, and extreme log-price "
              "z-scores (|z| ≥ 3) removed; remaining IQR outliers were kept as plausible luxury/specialty listings."
        )


def table_column_config() -> dict:
    return {
        "price": st.column_config.NumberColumn("Price", format="$%d"),
        "year": st.column_config.NumberColumn("Year", format="%d"),
        "odometer": st.column_config.NumberColumn("Mileage", format="%d mi"),
        "manufacturer": st.column_config.TextColumn("Manufacturer"),
        "condition": st.column_config.TextColumn("Condition"),
        "cylinders": st.column_config.TextColumn("Cylinders"),
        "fuel": st.column_config.TextColumn("Fuel"),
        "transmission": st.column_config.TextColumn("Transmission"),
        "type": st.column_config.TextColumn("Type"),
    }


def render_explorer(fdf: pd.DataFrame, filters: dict) -> None:
    section("Data Explorer",
            "Search, sort and download the filtered records. Click a column header to sort the visible rows.")
    c1, c2, c3, c4 = st.columns([2.2, 1.3, 1, 1])
    query = c1.text_input("Search", placeholder="e.g. toyota, pickup, diesel, 2015…", key="exp_query",
                          help="Case-insensitive match on manufacturer, condition, cylinders, fuel, "
                               "transmission or type; whole numbers also match the model year.")
    sort_col = c2.selectbox("Sort by", EXPECTED_COLS, index=0, format_func=lambda c: COLUMN_LABELS[c],
                            key="exp_sort")
    order = c3.radio("Order", ["Descending", "Ascending"], key="exp_order", horizontal=False)
    limit = c4.selectbox("Rows to display", [100, 1_000, 5_000, 10_000], index=1, key="exp_limit")

    result = search_frame(fdf, query)
    ascending = order == "Ascending"
    shown = result.sort_values(sort_col, ascending=ascending, kind="stable").head(limit)

    st.markdown(f'<div class="note">Displaying <b>{len(shown):,}</b> of <b>{len(result):,}</b> matching records'
                + (f' (search “{query}”)' if query.strip() else "") + ".</div>", unsafe_allow_html=True)
    if result.empty:
        st.info("No records match this search. Try a different term.")
        return
    st.dataframe(shown, use_container_width=True, hide_index=True, height=460,
                 column_config=table_column_config())

    key = filters_key({k: v for k, v in filters.items() if not k.startswith("_")})
    st.download_button(
        f"⬇  Download {len(result):,} records as CSV",
        data=export_csv(key, query, sort_col, ascending),
        file_name="craigslist_used_cars_filtered.csv",
        mime="text/csv",
        type="primary",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        df = load_data(str(DATA_PATH))
    except DataLoadError as exc:
        st.error(f"**Could not load the dataset.** {exc}")
        st.stop()
        return

    invalid = int(df[NUMERIC_COLS].isna().any(axis=1).sum())

    filters = sidebar(df)
    fdf = apply_filters(df, filters)
    render_count(filters["_count_slot"], len(fdf), len(df), filters["_active"])

    render_header(df)
    if invalid:
        st.warning(f"{invalid:,} records have non-numeric price, year or odometer values and cannot match the "
                   "range filters. They are kept in the file but excluded from the charts.")

    if fdf.empty:
        st.warning("**No listings match the current filters.** Widen a range or remove a selection, "
                   "or use **Reset filters** in the sidebar.")
        st.stop()
        return

    render_kpis(fdf, df)

    tabs = st.tabs(["📋 Overview", "💲 Price Distribution", "📈 Price Drivers", "🏭 Manufacturers",
                    "🚙 Vehicle Characteristics", "🔗 Correlation", "💡 Key Insights", "🗂️ Data Explorer"])
    with tabs[0]:
        render_overview(fdf)
    with tabs[1]:
        render_price_distribution(fdf)
    with tabs[2]:
        section("Price vs Manufacturing Year",
                "Each point is a listing. A rising pattern means newer vehicles are listed at higher prices.")
        render_relationship(
            fdf, df, "year", "Listing price by manufacturing year", "Condition", "yr",
            "Classic and collectible vehicles appear as high-priced points on the far left, which weakens "
            "the straight-line relationship.")
        st.divider()
        section("Price vs Mileage",
                "Each point is a listing. A falling pattern means higher-mileage vehicles are listed for less.")
        render_relationship(
            fdf, df, "odometer", "Listing price by mileage", "Condition", "mi",
            "Listings with 0 or very low mileage include new vehicles as well as listings where the "
            "odometer was not entered accurately.")
    with tabs[3]:
        render_manufacturers(fdf)
    with tabs[4]:
        render_characteristics(fdf)
    with tabs[5]:
        render_correlation(fdf)
    with tabs[6]:
        render_insights(fdf, df)
    with tabs[7]:
        render_explorer(fdf, filters)

    st.markdown(
        f'<div class="note" style="text-align:center;margin-top:28px;color:{MUTED}">'
        "Craigslist Used Car Market Analysis · data: Kaggle (austinreese/craigslist-carstrucks-data) · "
        "listing prices are asking prices, not transaction prices.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
