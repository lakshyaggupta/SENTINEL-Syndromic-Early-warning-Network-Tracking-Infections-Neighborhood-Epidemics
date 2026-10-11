"""
SENTINEL: Syndromic Early-warning Network Tracking Infections & Neighborhood Epidemics  (v3: live water signals + DengAI)
==================================================================================================================
HOW THE PIECES FIT TOGETHER
  1. MAP EVENTS   - st_folium() renders a Folium map and returns the user's last
                    click / drawn marker or rectangle. read_map_event() turns that
                    into (lat, lon) and stores it in st.session_state.
  2. DATA FETCH   - fetch_environment(lat, lon) calls the free Open-Meteo APIs (no key):
                    weather (14-day mean temperature/humidity, 30-day rain and ET0, soil
                    moisture), air quality (AQI, PM2.5) and river discharge. fetch_water_mgmt()
                    adds country water/sanitation access from the World Bank WDI, and
                    reverse_geocode() names the place via OpenStreetMap Nominatim.
                    Every source is cached and fails independently, with a deterministic
                    offline fallback or a labelled proxy.
  3. RISK ENGINE  - compute_risk() builds three sub-indices (Vector Breeding, Health System
                    Load, Environmental Vulnerability) from live weather/AQI/water signals plus
                    sanitation and healthcare inputs, then merges them non-linearly into the
                    0-100 Outbreak Hazard Score.
  4. DENGAI MODEL - optional: if dengai_model.py and its trained bundle are present, San Juan and
                    Iquitos become ML model cities. Clicking near one prefills recent cases
                    from a climate-only estimate and unlocks a model-details panel.
  5. RERENDER     - A new click updates session_state and calls st.rerun(), so the
                    whole dashboard (map overlay, KPIs, charts, alerts) recalculates
                    instantly without a browser page refresh.

The scoring model is a transparent rule-based SIMULATION, not a validated
epidemiological model.  Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import datetime as dt
import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Draw, HeatMap
from streamlit_folium import st_folium

try:   # optional ML model: DengAI features are simply hidden if it (or its trained bundle) is unavailable
    import dengai_model as dengai
except Exception:
    dengai = None

st.set_page_config(page_title="SENTINEL", page_icon="🦟", layout="wide")

# ----------------------------------------------------------------------------
# 1. THEME
# ----------------------------------------------------------------------------
BG, PANEL, EDGE, TEXT, MUTED, ACCENT = "#0E1A24", "#15262F", "#243A47", "#E8EEF2", "#8FA6B4", "#4FD1C5"
GLASS = "rgba(12, 24, 33, 0.62)"        # dark translucent card fill
GLASS_EDGE = "rgba(79, 209, 197, 0.30)"  # subtle cyan border
LEVEL_COLORS = {"Low": "#4FD1A5", "Moderate": "#F2C14E", "High": "#F28B3C", "Critical": "#E5484D"}

# Looks for background.png (or .jpg/.jpeg/.webp) in the same folder as app.py.
BACKGROUND_CANDIDATES = ("background.png", "background.jpg", "background.jpeg", "background.webp")
MODELS_DIR = Path(__file__).resolve().parent / "models"   # DengAI bundle + history/test-prediction CSVs


@lru_cache(maxsize=1)
def load_background_css() -> str:
    """Return a CSS `url(data:...)` value for the background image, or "" if none is found.
    Embedding the image as base64 avoids broken-URL problems, because Streamlit does
    not serve arbitrary local files. Cached so the file is only read and encoded once."""
    folder = Path(__file__).resolve().parent
    for name in BACKGROUND_CANDIDATES:
        path = folder / name
        if path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'url("data:{mime};base64,{data}")'
    return ""


# Dark overlay sits on top of the image; with no image the same overlay gives a plain dark gradient.
_bg_image = load_background_css()
BACKGROUND_CSS = ("linear-gradient(rgba(10, 19, 26, 0.85), rgba(10, 19, 26, 0.95)), " + _bg_image
                  if _bg_image else "linear-gradient(#122330, #0A131A)")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'IBM Plex Sans',sans-serif; color:{TEXT}; }}
.stApp {{ background:radial-gradient(ellipse at 20% 0%, rgba(79,209,197,0.10), rgba(10,19,26,0) 55%), {BACKGROUND_CSS};
         background-size:cover; background-position:center; background-repeat:no-repeat; background-attachment:fixed;
         isolation:isolate; }}  /* own stacking context, so the z-index:-1 sky layer stays above the background */
section[data-testid="stSidebar"] {{ background:rgba(10,19,26,0.72); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
                                   border-right:1px solid {GLASS_EDGE}; box-shadow:4px 0 24px rgba(0,0,0,0.45); }}
section[data-testid="stSidebar"] > div {{ background:transparent; }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color:#FFFFFF; font-family:'Fraunces',serif; }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ color:#E8EEF2 !important; }}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"], section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {{ color:{MUTED} !important; }}
.block-container {{ padding-top:1.6rem; max-width:1320px; }}
.ow-title {{ font-family:'Fraunces',serif; font-size:2.6rem; font-weight:700; margin:0; color:#E6FFFB;
            text-shadow:0 0 15px rgba(79,209,197,0.5), 0 0 38px rgba(79,209,197,0.22); letter-spacing:-0.01em; }}
.ow-title .ow-title-sub {{ display:block; font-size:1.35rem; font-weight:600; letter-spacing:0; margin-top:.25rem; color:#BFF5EE;
                          text-shadow:0 0 12px rgba(79,209,197,0.30); text-wrap:balance; }}
.ow-sub {{ color:#A9C0CC; font-size:1.05rem; margin:.25rem 0 .8rem 0; max-width:64ch; }}
.ow-card {{ background:{GLASS}; border:1px solid {GLASS_EDGE}; border-radius:12px; padding:16px 18px; height:100%;
            backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); box-shadow:0 4px 24px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(79,209,197,0.05); }}
.ow-card .label {{ color:{MUTED}; font-size:.85rem; margin-bottom:4px; }}
.ow-card .value {{ font-family:'Fraunces',serif; font-size:2.1rem; font-weight:700; line-height:1.1; }}
.ow-card .note {{ color:{MUTED}; font-size:.82rem; margin-top:6px; }}
.ow-card .bar {{ height:5px; border-radius:3px; background:rgba(143,166,180,0.22); margin-top:10px; overflow:hidden; }}
.ow-card .bar > span {{ display:block; height:100%; border-radius:3px; }}
.ow-badge {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; color:#0E1A24; }}
.ow-section {{ font-family:'Fraunces',serif; font-size:1.35rem; font-weight:600; margin:1.5rem 0 .6rem 0; text-shadow:0 0 12px rgba(79,209,197,0.30); }}
.ow-loc {{ background:{GLASS}; border:1px solid {GLASS_EDGE}; border-radius:10px; padding:10px 14px; margin-bottom:12px;
            backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); box-shadow:0 4px 24px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(79,209,197,0.05); }}
.ow-loc small {{ color:{MUTED}; }}
.ow-alert {{ background:{GLASS}; border:1px solid {GLASS_EDGE}; border-left:5px solid var(--c); border-radius:10px; padding:12px 16px; margin-bottom:10px;
             backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); box-shadow:0 4px 24px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(79,209,197,0.05); }}
.ow-alert b {{ color:var(--c); }} .ow-alert small {{ color:{MUTED}; }}
.ow-foot {{ color:{MUTED}; font-size:.8rem; margin-top:2rem; }}
/* Interactive hover / tap effects */
.ow-card, .ow-loc, .ow-alert, .ow-badge, .stButton > button {{ transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1); }}
.ow-loc:hover {{ transform:translateY(-3px); border-color:#4FD1C5; box-shadow:0 6px 24px rgba(79, 209, 197, 0.22); }}
.ow-alert:hover {{ transform:translateY(-3px); border-top-color:#4FD1C5; border-right-color:#4FD1C5; border-bottom-color:#4FD1C5;
                   box-shadow:0 6px 24px rgba(79, 209, 197, 0.22); }}
.ow-loc:active, .ow-alert:active {{ transform:translateY(0px) scale(0.99); }}
/* Focus & dim: the hovered KPI card grows and glows while the other cards in its row shrink and fade */
.ow-card {{ position:relative; transition:transform .28s cubic-bezier(0.4, 0, 0.2, 1), opacity .28s ease, box-shadow .28s ease, border-color .28s ease; }}
.ow-card:hover {{ transform:scale(1.05); z-index:2; border-color:#4FD1C5;
                  box-shadow:0 0 0 1px rgba(79,209,197,0.55), 0 0 26px rgba(79,209,197,0.45), 0 12px 32px rgba(0,0,0,0.5); }}
.block-container:has(.ow-card:hover) .ow-card:not(:hover) {{ transform:scale(0.95); opacity:0.4; }}
/* KPI grid: one 4-column grid with equal-height rows, so every card is exactly the same size */
.ow-grid {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); grid-auto-rows:1fr; gap:14px; margin:.25rem 0 .5rem 0; }}
@media (max-width: 900px) {{ .ow-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 520px) {{ .ow-grid {{ grid-template-columns:minmax(0, 1fr); }} }}
.ow-card:active {{ transform:scale(1.03); }}
/* Shooting stars: a fixed sky layer under all content; co-prime durations make the streaks look randomly timed */
.sentinel-sky {{ position:fixed; inset:0; z-index:-1; pointer-events:none; overflow:hidden; }}
.sentinel-sky i {{ position:absolute; left:var(--x); top:var(--y); width:var(--len, 160px); height:2px; border-radius:999px; opacity:0;
                  background:linear-gradient(90deg, rgba(232,255,251,0.95), rgba(79,209,197,0.45) 30%, rgba(79,209,197,0));
                  transform:rotate(-45deg); transform-origin:left center; animation:shootingStar var(--d) linear var(--delay) infinite; }}
.sentinel-sky i::before {{ content:""; position:absolute; left:-2px; top:50%; width:5px; height:5px; margin-top:-2.5px; border-radius:50%;
                          background:#E8FFFB; box-shadow:0 0 8px 2px rgba(79,209,197,0.8), 0 0 18px 4px rgba(79,209,197,0.35); }}
.sentinel-sky i:nth-child(1) {{ --x:78%;  --y:-4%; --d:9s;  --delay:0.8s; }}
.sentinel-sky i:nth-child(2) {{ --x:104%; --y:12%; --d:13s; --delay:3.5s;  --len:200px; }}
.sentinel-sky i:nth-child(3) {{ --x:55%;  --y:-8%; --d:11s; --delay:6.2s;  --len:120px; }}
.sentinel-sky i:nth-child(4) {{ --x:98%;  --y:38%; --d:17s; --delay:1.9s; }}
.sentinel-sky i:nth-child(5) {{ --x:38%;  --y:-6%; --d:19s; --delay:9.4s;  --len:140px; }}
.sentinel-sky i:nth-child(6) {{ --x:90%;  --y:60%; --d:23s; --delay:12.7s; --len:180px; }}
@keyframes shootingStar {{ 0% {{ transform:rotate(-45deg) translateX(0); opacity:0; }} 1% {{ opacity:1; }}
                           7% {{ transform:rotate(-45deg) translateX(-75vmax); opacity:0; }}
                           100% {{ transform:rotate(-45deg) translateX(-75vmax); opacity:0; }} }}
/* The cursor-trail injector is a zero-height component iframe: take its container out of the layout flow */
[data-testid="stElementContainer"]:has(> iframe[title="st.iframe"]) {{
    position:absolute; width:0; height:0; overflow:hidden; margin:0; }}
.ow-badge:hover {{ transform:translateY(-2px); box-shadow:0 4px 14px rgba(79, 209, 197, 0.25); filter:brightness(1.08); }}
.ow-badge:active {{ transform:scale(0.97); }}
.stButton > button, .stDownloadButton > button {{ background:{GLASS}; color:{TEXT}; border:1px solid {GLASS_EDGE}; border-radius:8px; backdrop-filter:blur(12px); }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:#4FD1C5; color:#4FD1C5; transform:translateY(-2px);
                   box-shadow:0 6px 24px rgba(79, 209, 197, 0.22); }}
.stButton > button:active, .stDownloadButton > button:active {{ transform:translateY(0px) scale(0.98); }}
/* Remove Streamlit's top bar and wasted space */
header[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem !important; }}
/* Entry animations */
@keyframes fadeSlideUp {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes pulseGlow {{ 0% {{ box-shadow: 0 0 0 0 rgba(229, 72, 77, 0.4); }} 70% {{ box-shadow: 0 0 0 10px rgba(229, 72, 77, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(229, 72, 77, 0); }} }}
/* "backwards" (not "forwards") so the finished animation doesn't pin transform and block the hover lift */
.ow-card, .ow-alert, .ow-loc {{ animation: fadeSlideUp 0.5s ease-out backwards; animation-delay: 0.1s; }}
[data-testid="stColumn"]:nth-child(2) .ow-alert {{ animation-delay: 0.18s; }}
.ow-grid .ow-card:nth-child(4n+2) {{ animation-delay: 0.18s; }}
.ow-grid .ow-card:nth-child(4n+3) {{ animation-delay: 0.26s; }}
.ow-grid .ow-card:nth-child(4n+4) {{ animation-delay: 0.34s; }}
.ow-badge.crit {{ animation: pulseGlow 1.8s ease-out infinite; }}
@media (prefers-reduced-motion: reduce) {{ .ow-card, .ow-alert, .ow-loc, .ow-badge.crit {{ animation: none; }}
                                           .sentinel-sky {{ display:none; }} }}
</style>
<div class="sentinel-sky" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div>""", unsafe_allow_html=True)

# Cursor trail. The component iframe hands a script to the parent page, so the listener lives in the app's own
# window: it sees mouse moves over the whole dashboard and survives reruns without being added twice.
CURSOR_TRAIL_HTML = """<script>
(() => {
  const doc = window.parent.document;
  if (doc.getElementById("sentinel-trail-js")) return;
  const s = doc.createElement("script");
  s.id = "sentinel-trail-js";
  s.textContent = "(" + function () {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const LIFE_MS = 500, STEP_PX = 10;
    const layer = document.createElement("div");
    layer.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden";
    document.body.appendChild(layer);
    let lastX = -1e3, lastY = -1e3;
    addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse" || Math.hypot(e.clientX - lastX, e.clientY - lastY) < STEP_PX) return;
      lastX = e.clientX; lastY = e.clientY;
      const dot = document.createElement("span");
      dot.style.cssText = "position:absolute;width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;background:#4FD1C5;"
        + "box-shadow:0 0 10px 3px rgba(79,209,197,.6);left:" + lastX + "px;top:" + lastY + "px";
      layer.appendChild(dot);
      dot.animate([{ opacity: 0.9, transform: "scale(1)" }, { opacity: 0, transform: "scale(0.25)" }],
                  { duration: LIFE_MS, easing: "ease-out" }).onfinish = () => dot.remove();
    }, { passive: true });
  } + ")();";
  doc.head.appendChild(s);
})();
</script>"""

# ----------------------------------------------------------------------------
# 2. DATA STRUCTURES
# ----------------------------------------------------------------------------
@dataclass
class Inputs:
    lat: float
    lon: float
    rainfall: float          # mm over the last 30 days
    temperature: float       # deg C, 14-day mean
    humidity: float          # %, 14-day mean
    aqi: float               # 0-500
    med_availability: float  # %
    med_demand: float        # %
    past_cases: int
    season: str
    area_type: str
    horizon: int
    stagnation: float        # % standing water / blocked drains (higher = worse)
    drainage: float          # % drainage infrastructure quality (higher = better)
    contamination: float     # % water-supply contamination / unsafe storage (higher = worse)
    density: float           # people per sq km
    water_condition: float   # 0-100 live water condition (surplus, soil wetness, river flow; higher = worse)
    water_mgmt: float        # 0-100 water management gap (lack of safe water / sanitation; higher = worse)


@dataclass
class RiskResult:
    vector: float        # Vector Breeding Risk, 0-100
    health: float        # Health System Load Risk, 0-100
    env: float           # Environmental Vulnerability Risk, 0-100
    supply: float        # Supply Shortage Risk, 0-100
    climate: float       # Climate Impact Factor, 0-100
    overall: float       # Outbreak Hazard Score, 0-100
    level: str
    factors: dict        # individual factor severities for the breakdown chart


SEASON_FACTOR = {"Wet season": 1.15, "Transition": 1.05, "Dry season": 0.85, "Cool season": 0.7}
AREA_FACTOR = {"Dense urban": 1.15, "Semi-urban": 1.0, "Rural": 0.85}
US_TERRITORIES = {"PR", "GU", "VI", "AS", "MP"}


def clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(x, lo, hi))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mean_valid(xs) -> float | None:
    """Mean of the non-null values, or None if there are none."""
    v = [float(x) for x in xs if x is not None and x == x]
    return float(np.mean(v)) if v else None


def auto_season(temp14: float, rain30: float) -> str:
    """Season label from local weather: cool if 14-day mean < 18 C, else wet / dry / transition by 30-day rain."""
    return "Cool season" if temp14 < 18 else "Wet season" if rain30 >= 150 else "Dry season" if rain30 < 40 else "Transition"


# ----------------------------------------------------------------------------
# 3. DATA FETCHING (Open-Meteo + World Bank + Nominatim + DengAI, each source fails independently)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_environment(lat: float, lon: float) -> dict:
    """Live signals for one ~1 km cell (cached 10 min). Weather: current temp, 14-day mean temp/humidity,
    30-day rain and ET0, 7-day soil moisture. Air: US AQI + PM2.5. River: latest discharge / 1-year median.
    `sources` records which of the three succeeded; weather falls back to a deterministic simulation."""
    env = {"temp_now": None, "temperature": None, "humidity": None, "rainfall": None, "et0": None, "soil": None,
           "aqi": None, "pm25": None, "river_ratio": None, "river_none": False,
           "sources": {"Weather": False, "Air quality": False, "River flow": False}}
    try:
        w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_mean,relative_humidity_2m_mean",
            "hourly": "soil_moisture_3_to_9cm", "past_days": 30, "forecast_days": 1,
        })
        w.raise_for_status()
        j = w.json()
        d, now = j["daily"], j["current"]       # daily: 30 past days + today (index 30)
        temp_now, hum_now = float(now["temperature_2m"]), float(now["relative_humidity_2m"])
        et0 = (d.get("et0_fao_evapotranspiration") or [])[:30]
        t14, h14 = mean_valid(d.get("temperature_2m_mean", [])[16:30]), mean_valid(d.get("relative_humidity_2m_mean", [])[16:30])
        env.update(temp_now=temp_now, temperature=temp_now if t14 is None else t14, humidity=hum_now if h14 is None else h14,
                   rainfall=float(sum(x or 0 for x in d["precipitation_sum"][:30])),
                   et0=float(sum(x or 0 for x in et0)) if mean_valid(et0) is not None else None,
                   soil=mean_valid(j.get("hourly", {}).get("soil_moisture_3_to_9cm", [])[-192:-24]))
        env["sources"]["Weather"] = True
    except Exception:
        # Deterministic fallback so the app still works offline.
        t = round(33 - 0.35 * abs(lat), 1)
        env.update(temp_now=t, temperature=t, humidity=72.0, rainfall=round(180 + 90 * np.sin(np.radians(lon * 3)), 0))
    try:
        a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=8,
                         params={"latitude": lat, "longitude": lon, "current": "us_aqi,pm2_5"})
        a.raise_for_status()
        cur = a.json()["current"]
        env.update(aqi=cur.get("us_aqi"), pm25=cur.get("pm2_5"))
        env["sources"]["Air quality"] = cur.get("us_aqi") is not None
    except Exception:
        pass
    try:
        today = dt.date.today()
        f = requests.get("https://flood-api.open-meteo.com/v1/flood", timeout=10, params={
            "latitude": lat, "longitude": lon, "daily": "river_discharge",
            "start_date": (today - dt.timedelta(days=365)).isoformat(), "end_date": today.isoformat()})
        f.raise_for_status()
        q = [x for x in f.json()["daily"]["river_discharge"] if x is not None]
        if q:
            med = float(np.median(q))
            env.update(river_none=med < 1.0, river_ratio=None if med < 1.0 else float(q[-1]) / med)   # <1 m3/s: no real river
            env["sources"]["River flow"] = True
    except Exception:
        pass
    return env


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> tuple[str, str]:
    """(human-readable place name, ISO alpha-2 country code); falls back to (coordinates, "")."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", timeout=6,
                         params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
                         headers={"User-Agent": "SENTINEL/3.0 (dengue early-warning demo)"})
        j = r.json()
        name, addr = j.get("display_name", ""), j.get("address", {})
        code = addr.get("country_code", "").upper()
        # Nominatim files US territories under "us"; World Bank reports them separately (e.g. Puerto Rico = PR).
        region = addr.get("ISO3166-2-lvl4", "")
        if code == "US" and region[3:] in US_TERRITORIES:
            code = region[3:]
        return (", ".join(name.split(", ")[:3]) if name else f"{lat:.3f}, {lon:.3f}"), code
    except Exception:
        return f"{lat:.3f}, {lon:.3f}", ""


@st.cache_data(ttl=86400, show_spinner=False)
def forward_geocode(query: str) -> tuple[float, float, str] | None:
    """Search text (city, neighbourhood, address) -> (lat, lon, display name), or None."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", timeout=6,
                         params={"q": query, "format": "jsonv2", "limit": 1},
                         headers={"User-Agent": "SENTINEL/3.0 (dengue early-warning demo)"})
        r.raise_for_status()
        hits = r.json()
        if not hits:
            return None
        return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("display_name", query)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_water_mgmt(iso2: str) -> dict | None:
    """Country water & sanitation access, World Bank WDI (WHO/UNICEF JMP): % safely managed drinking water and
    sanitation (basic service as fallback), latest year. None if the country has no data."""
    def latest(codes: tuple[str, ...]) -> tuple[float, int] | None:
        for code in codes:
            try:
                r = requests.get(f"https://api.worldbank.org/v2/country/{iso2}/indicator/{code}", timeout=8,
                                 params={"format": "json", "mrnev": 1})
                r.raise_for_status()
                rec = r.json()[1][0]
                if rec["value"] is not None:
                    return float(rec["value"]), int(rec["date"])
            except Exception:
                continue
        return None
    water, san = latest(("SH.H2O.SMDW.ZS", "SH.H2O.BASW.ZS")), latest(("SH.STA.SMSS.ZS", "SH.STA.BASS.ZS"))
    if not (water or san):
        return None
    return {"water": water[0] if water else None, "sanitation": san[0] if san else None,
            "year": max(x[1] for x in (water, san) if x)}


@st.cache_resource(ttl=600, show_spinner=False)
def get_bundle() -> dict | None:
    """Trained DengAI bundle, or None when dengai_model.py / its trained models are unavailable."""
    try:
        return dengai.load_bundle() if dengai else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dengai_live(city: str) -> dict:
    """Live ML estimate for a DengAI city: recent Open-Meteo weather -> features -> weekly cases (cached 1 h).
    Raises on failure, so errors are not cached."""
    c = dengai.CITIES[city]
    daily = dengai.fetch_daily(c["lat"], c["lon"])
    if daily is None:
        raise RuntimeError("DengAI weather fetch failed")
    return dengai.predict_live(get_bundle(), city, daily)


def dengai_live(city: str | None) -> dict | None:
    try:
        return fetch_dengai_live(city) if city else None
    except Exception:
        return None


def dengai_city(lat: float, lon: float, bundle: dict | None) -> str | None:
    """Code of the DengAI model city within NEAR_KM of the point (needs a loaded bundle), else None."""
    try:
        code = dengai.nearest_city(lat, lon) if bundle else None
    except Exception:
        return None
    return code if code in (bundle or {}).get("cities", {}) else None


@st.cache_data(ttl=3600, show_spinner=False)
def load_dengai_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """models/dengai_history.csv and models/dengai_test_predictions.csv (None for a missing file)."""
    def read(name: str) -> pd.DataFrame | None:
        path = MODELS_DIR / name
        return pd.read_csv(path, parse_dates=["week_start_date"]) if path.is_file() else None
    return read("dengai_history.csv"), read("dengai_test_predictions.csv")


def water_condition(wx: dict, rain: float) -> tuple[float | None, float | None]:
    """(Water Condition Index 0-100, 30-day surplus mm). Index = 0.5 surplus (rain - ET0) + 0.3 soil wetness
    + 0.2 river-flow anomaly, re-weighted over the available parts; None if no live part is available."""
    surplus = rain - wx["et0"] if wx["et0"] is not None else None
    parts = []
    if surplus is not None:
        parts.append((0.5, clip((surplus + 50) / 250 * 100)))
    if wx["soil"] is not None:
        parts.append((0.3, clip((wx["soil"] - 0.15) / 0.25 * 100)))
    if wx["river_ratio"] is not None:
        parts.append((0.2, clip((wx["river_ratio"] - 1) / 2 * 100)))
    elif wx["river_none"]:
        parts.append((0.2, 0.0))           # no significant river nearby: no river-driven flood pools
    if not parts:
        return None, surplus
    return sum(w * s for w, s in parts) / sum(w for w, _ in parts), surplus


def water_gap(wm: dict | None) -> float | None:
    """Water Management Gap 0-100 = mean shortfall of safe-water and sanitation coverage; None if no data."""
    gaps = [100 - v for v in ((wm or {}).get("water"), (wm or {}).get("sanitation")) if v is not None]
    return clip(float(np.mean(gaps))) if gaps else None


# ----------------------------------------------------------------------------
# 4. RISK ENGINE
# ----------------------------------------------------------------------------
def risk_level(score: float) -> str:
    return "Low" if score < 25 else "Moderate" if score < 50 else "High" if score < 75 else "Critical"


def compute_risk(p: Inputs) -> RiskResult:
    """Multi-factor scoring. Every factor is first scaled to a 0-100 severity (higher = worse),
    then grouped into three sub-indices, then merged non-linearly.

    1. Vector Breeding Risk   - rainfall (30-day total, Aedes favours ~150-400 mm), temperature
                                (optimum ~29 C), humidity, water stagnation, unsafe water storage and the
                                live water condition (surplus, soil wetness, river flow), x season.
                                = (0.24 Rain + 0.20 Temp + 0.14 Humidity + 0.16 Stagnation
                                   + 0.10 Storage/contamination + 0.16 Water condition) x season
    2. Health System Load     - AQI (poor air quality raises respiratory/fever caseload and can
                                mask fever symptoms), recent cases, medicine demand, stock gap.
    3. Environmental Vulnerability - population density, poor drainage, water contamination and the
                                country water-management gap (safe water + sanitation), x area type.
                                = (0.35 Density + 0.30 Poor drainage + 0.10 Storage/contamination
                                   + 0.25 Water management gap) x area

    Overall = 80% power mean (p=1.5, so one very bad sub-index is not averaged away) +
              20% geometric-mean synergy (risk compounds when all three are elevated together)."""
    f = {
        "Rainfall": 100 * np.exp(-((p.rainfall - 280) ** 2) / (2 * 140 ** 2)),
        "Temperature": 100 * np.exp(-((p.temperature - 29) ** 2) / (2 * 4.0 ** 2)),
        "Humidity": clip((p.humidity - 40) / 50 * 100),
        "Water stagnation": clip(p.stagnation),
        "Water storage / contamination": clip(p.contamination),
        "Water condition (live)": clip(p.water_condition),
        "Air quality (AQI)": clip(p.aqi / 300 * 100),
        "Recent cases": clip(100 * (1 - np.exp(-p.past_cases / 250))),
        "Medicine demand": clip(p.med_demand),
        "Stock gap": clip(100 - p.med_availability),
        "Poor drainage": clip(100 - p.drainage),
        "Water management gap": clip(p.water_mgmt),
        "Population density": clip(100 * (1 - np.exp(-p.density / 9000))),
    }
    climate = clip((0.40 * f["Rainfall"] + 0.35 * f["Temperature"] + 0.25 * f["Humidity"]) * SEASON_FACTOR[p.season])
    vector = clip((0.24 * f["Rainfall"] + 0.20 * f["Temperature"] + 0.14 * f["Humidity"]
                   + 0.16 * f["Water stagnation"] + 0.10 * f["Water storage / contamination"]
                   + 0.16 * f["Water condition (live)"]) * SEASON_FACTOR[p.season])
    health = clip(0.20 * f["Air quality (AQI)"] + 0.28 * f["Recent cases"] + 0.26 * f["Medicine demand"]
                  + 0.26 * f["Stock gap"])
    env = clip((0.35 * f["Population density"] + 0.30 * f["Poor drainage"]
                + 0.10 * f["Water storage / contamination"] + 0.25 * f["Water management gap"]) * AREA_FACTOR[p.area_type])
    pressure = max(0.0, p.med_demand - p.med_availability)
    supply = clip(0.55 * f["Stock gap"] + 0.45 * pressure + 0.15 * p.med_demand * f["Stock gap"] / 100)

    v, h, e = vector / 100, health / 100, env / 100
    power = 1.5
    base = (0.45 * v ** power + 0.30 * h ** power + 0.25 * e ** power) ** (1 / power)
    synergy = (v * h * e) ** (1 / 3)
    overall = clip(100 * (0.8 * base + 0.2 * synergy))

    factors = {k: clip(val) for k, val in f.items()}
    factors.update({"◆ Vector breeding": vector, "◆ Health system load": health, "◆ Environmental vulnerability": env})
    return RiskResult(vector, health, env, supply, climate, overall, risk_level(overall), factors)


def forecast_outbreak(p: Inputs, r: RiskResult) -> pd.DataFrame:
    weeks = np.arange(1, p.horizon + 1)
    current = max(p.past_cases / 4, 1.0)
    rain_push = clip((p.rainfall - 150) / 300 * 100) / 100
    demand_push = clip(p.med_demand - 50, 0, 50) / 50
    stag_push = clip(p.stagnation) / 100
    cases = []
    for w in weeks:
        lag = 1.0 if w >= 3 else 0.4          # rainfall acts after ~2 weeks (breeding cycle)
        current *= np.exp(-0.06 + 0.30 * r.overall / 100 + 0.08 * rain_push * lag + 0.05 * demand_push
                          + 0.04 * stag_push)
        cases.append(current)
    cases, spread = np.array(cases), 0.08 + 0.03 * weeks
    return pd.DataFrame({"week": [f"Week {w}" for w in weeks], "cases": cases.round(1),
                         "low": (cases * (1 - spread)).round(1), "high": (cases * (1 + spread)).round(1)})


# ----------------------------------------------------------------------------
# 5. SIMULATED NEIGHBOURHOOD ZONES around the selected point
# ----------------------------------------------------------------------------
ZONES = [("Zone 1 - Riverside", 0.020, -0.015, 1.25), ("Zone 2 - Old Market", 0.008, 0.010, 1.10),
         ("Zone 3 - Eastern Wards", 0.004, 0.032, 1.30), ("Zone 4 - Lakeside", -0.018, 0.018, 1.15),
         ("Zone 5 - Industrial Belt", -0.026, -0.012, 0.95), ("Zone 6 - Hillview", 0.030, 0.020, 0.70),
         ("Zone 7 - Central Heights", 0.000, -0.010, 0.60), ("Zone 8 - Northern Fringe", 0.040, -0.030, 0.80)]


def build_zone_data(p: Inputs, r: RiskResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(r.overall * 100) + p.past_cases)
    rows, pts = [], []
    for name, dlat, dlon, vuln in ZONES:
        risk = clip(r.overall * vuln + rng.normal(0, 3))
        lat, lon = p.lat + dlat, p.lon + dlon
        rows.append({"zone": name, "lat": lat, "lon": lon, "risk": round(risk, 1), "level": risk_level(risk),
                     "est_cases_wk": int(max(p.past_cases / 4, 1) * risk / max(r.overall, 1) / len(ZONES) * 2)})
        n = int(15 + risk * 0.9)
        pts.append(pd.DataFrame({"lat": rng.normal(lat, 0.007, n), "lon": rng.normal(lon, 0.007, n), "weight": risk / 100}))
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)


# ----------------------------------------------------------------------------
# 6. MAP: build overlays + read click/draw events
# ----------------------------------------------------------------------------
def build_map(p: Inputs, r: RiskResult, zones: pd.DataFrame, pts: pd.DataFrame, place: str,
              ml_note: str | None, cities: dict) -> folium.Map:
    """Colour-coded map: heatmap, risk circle on the chosen area, DengAI city markers and a summary popup."""
    color = LEVEL_COLORS[r.level]
    m = folium.Map(location=[p.lat, p.lon], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    add_selection_tools(m, cities)
    HeatMap(pts[["lat", "lon", "weight"]].values.tolist(), radius=32, blur=26, min_opacity=0.25,
            gradient={0.2: "#4FD1A5", 0.5: "#F2C14E", 0.75: "#F28B3C", 1.0: "#E5484D"}).add_to(m)
    folium.Circle([p.lat, p.lon], radius=5000, color=color, weight=2, fill=True, fill_color=color,
                  fill_opacity=0.18).add_to(m)
    for z in zones.itertuples():
        folium.CircleMarker([z.lat, z.lon], radius=7, color="#fff", weight=1, fill=True,
                            fill_color=LEVEL_COLORS[z.level], fill_opacity=0.9,
                            tooltip=f"{z.zone}: {z.risk}% ({z.level})").add_to(m)
    popup =(f"<div style='font-family:sans-serif;min-width:200px'><b>{place}</b><br>"
             f"{p.lat:.4f}, {p.lon:.4f}<hr style='margin:4px 0'>"
             f"Temperature: <b>{p.temperature:.1f} °C</b><br>Rainfall (30 d): <b>{p.rainfall:.0f} mm</b><br>"
             f"AQI: <b>{p.aqi:.0f}</b><br>Water condition: <b>{p.water_condition:.0f}%</b><br>"
             + (f"<i>{ml_note}</i><br>" if ml_note else "") +
             f"Outbreak risk: <b style='color:{color}'>{r.overall:.0f}% ({r.level})</b></div>")
    folium.CircleMarker([p.lat, p.lon], radius=13, color="#fff", weight=3, fill=True, fill_color=color,
                        fill_opacity=1, popup=folium.Popup(popup, max_width=280, show=True),
                        tooltip=f"{r.overall:.0f}% - {r.level}").add_to(m)
    return m


def build_start_map(cities: dict) -> folium.Map:
    """World view shown before any location is chosen: no default place, nothing analysed yet."""
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap", control_scale=True)
    add_selection_tools(m, cities)
    return m


def add_selection_tools(m: folium.Map, cities: dict) -> None:
    """Marker / rectangle draw tools (a rectangle's centre is used) and the clickable DengAI city markers."""
    Draw(draw_options={"rectangle": True, "marker": True, "polyline": False, "polygon": False,
                       "circle": False, "circlemarker": False}, edit_options={"edit": False}).add_to(m)
    for c in cities.values():              # DengAI model cities: click one to analyse it with the ML estimate
        folium.Marker([c["lat"], c["lon"]], icon=folium.Icon(color="cadetblue", icon="info-sign"),
                      tooltip=f"DengAI model city: {c['name']} — click to analyse").add_to(m)


def geometry_to_point(g: dict) -> tuple[float, float] | None:
    """Drawn marker -> its point; drawn rectangle -> centre of its bounding box."""
    if g.get("type") == "Point":
        lon, lat = g["coordinates"][:2]
        return round(lat, 6), round(lon, 6)
    if g.get("type") == "Polygon":
        ring = g["coordinates"][0]
        return round(float(np.mean([c[1] for c in ring])), 6), round(float(np.mean([c[0] for c in ring])), 6)
    return None


def read_map_event(out: dict | None, cities: dict) -> tuple[float, float] | None:
    """Return a NEW (lat, lon) from the latest click, drawing or DengAI-city marker click, else None.
    st_folium keeps returning the last event on every rerun, so we remember the
    last one we handled to avoid an endless rerun loop."""
    if not out:
        return None
    click, draw, obj = out.get("last_clicked"), out.get("last_active_drawing"), out.get("last_object_clicked")
    if draw:
        pt = geometry_to_point(draw.get("geometry", {}))
        if pt and pt != st.session_state.seen_draw:
            st.session_state.seen_draw = pt
            return pt
    if obj:   # markers don't fire a map click, so a DengAI-city marker click arrives as an "object click"
        pt = (round(obj["lat"], 6), round(obj["lng"], 6))
        if pt != st.session_state.seen_obj:
            st.session_state.seen_obj = pt
            hit = next((c for c in cities.values() if abs(c["lat"] - pt[0]) < 1e-3 and abs(c["lon"] - pt[1]) < 1e-3), None)
            if hit:
                return round(hit["lat"], 6), round(hit["lon"], 6)
    if click:
        pt = (round(click["lat"], 6), round(click["lng"], 6))
        if pt != st.session_state.seen_click:
            st.session_state.seen_click = pt
            return pt
    return None


# ----------------------------------------------------------------------------
# 7. CHARTS & CARDS
# ----------------------------------------------------------------------------
def kpi_card(label: str, value: float, note: str, suffix: str = "%", color: str | None = None, decimals: int = 0) -> str:
    """Risk card: colour and bar follow the 0-100 risk level. Pass `color` for counts: fixed colour, no bar."""
    c = color or LEVEL_COLORS[risk_level(value)]
    bar = "" if color else f'<div class="bar"><span style="width:{min(value, 100):.0f}%;background:{c}"></span></div>'
    return (f'<div class="ow-card"><div class="label">{label}</div>'
            f'<div class="value" style="color:{c}">{value:.{decimals}f}{suffix}</div>'
            f'{bar}<div class="note">{note}</div></div>')


def kpi_grid(cards: list[str]) -> None:
    """Render cards in one equal-size CSS grid (st.columns rows size each row to its own content)."""
    st.markdown(f'<div class="ow-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def gauge(score: float, level: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 44, "color": TEXT}, "valueformat": ".0f"},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": LEVEL_COLORS[level], "thickness": 0.28},
               "bgcolor": PANEL, "borderwidth": 0,
               "steps": [{"range": [0, 25], "color": "#1B3A38"}, {"range": [25, 50], "color": "#3A3822"},
                         {"range": [50, 75], "color": "#3F2D20"}, {"range": [75, 100], "color": "#43232A"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


def breakdown_chart(factors: dict) -> go.Figure:
    """Horizontal bars of every factor's severity. Sub-indices (prefixed with a diamond) get a white outline."""
    items = sorted(factors.items(), key=lambda kv: kv[1])
    is_sub = [k.startswith("◆") for k, _ in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=[LEVEL_COLORS[risk_level(v)] for _, v in items],
                                       line=dict(color="#FFFFFF", width=[2 if s else 0 for s in is_sub])),
                           hovertemplate="%{y}: %{x:.0f}<extra></extra>"))
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT},
                      xaxis=dict(range=[0, 100], title="Severity (0-100, higher = worse)", gridcolor=EDGE))
    return fig


def forecast_chart(df: pd.DataFrame, level: str) -> go.Figure:
    c = LEVEL_COLORS[level]
    rgb = hex_to_rgb(c)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["week"], y=df["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["week"], y=df["low"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)", name="Uncertainty band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["week"], y=df["cases"], mode="lines+markers", name="Predicted cases",
                             line=dict(color=c, width=3), marker=dict(size=8)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT}, legend=dict(orientation="h", y=1.14),
                      yaxis=dict(title="Cases per week", gridcolor=EDGE, zeroline=False), xaxis=dict(gridcolor=EDGE))
    return fig


def dengai_style(fig: go.Figure, **layout) -> go.Figure:
    """Shared transparent styling for the four DengAI charts (same look as forecast_chart)."""
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT}, legend=dict(orientation="h", y=1.14), **layout)
    fig.update_xaxes(gridcolor=EDGE)
    fig.update_yaxes(gridcolor=EDGE, zeroline=False)
    return fig


def dengai_history_chart(h: pd.DataFrame) -> go.Figure:
    """Actual vs model-predicted weekly cases; the held-out (never trained on) weeks are shaded."""
    fig = go.Figure()
    ho = h[h["split"] == "holdout"]
    if len(ho):
        fig.add_vrect(x0=ho["week_start_date"].min(), x1=ho["week_start_date"].max(), fillcolor="rgba(79,209,197,0.10)",
                      line_width=0, annotation_text="Holdout", annotation_position="top left", annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(x=h["week_start_date"], y=h["actual"], mode="lines", name="Actual", line=dict(color=MUTED, width=2)))
    fig.add_trace(go.Scatter(x=h["week_start_date"], y=h["predicted"], mode="lines", name="Model", line=dict(color=ACCENT, width=2)))
    return dengai_style(fig, yaxis_title="Cases per week")


def dengai_test_chart(t: pd.DataFrame) -> go.Figure:
    """Model-predicted weekly cases over the DengAI test period (no actuals are published)."""
    fig = go.Figure(go.Scatter(x=t["week_start_date"], y=t["predicted"], mode="lines", name="Predicted cases",
                               line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(79,209,197,0.15)"))
    return dengai_style(fig, yaxis_title="Predicted cases per week", showlegend=False)


def dengai_importance_chart(fi: dict, top: int = 12) -> go.Figure:
    """Horizontal bars of the model's top feature-importance shares."""
    items = sorted(fi.items(), key=lambda kv: kv[1])[-top:]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker=dict(color=ACCENT),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig = dengai_style(fig, showlegend=False, xaxis=dict(title="Importance share", tickformat=".0%", gridcolor=EDGE))
    fig.update_yaxes(automargin=True)
    return fig


def dengai_week_of_year(dates: pd.Series) -> pd.Series:
    """ISO week capped at 52, the same convention the model uses."""
    return dates.dt.isocalendar().week.astype(int).clip(upper=52)


def dengai_season_chart(h: pd.DataFrame, week: int, estimate: float) -> go.Figure:
    """Mean actual cases by week of year, with this week and the live estimate marked."""
    prof = h.assign(week=dengai_week_of_year(h["week_start_date"])).groupby("week")["actual"].mean()
    fig = go.Figure(go.Scatter(x=prof.index, y=prof.values.round(1), mode="lines", name="Mean actual cases",
                               line=dict(color=MUTED, width=3)))
    fig.add_vline(x=week, line=dict(color=ACCENT, width=1, dash="dash"), annotation_text="This week",
                  annotation_position="top", annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(x=[week], y=[estimate], mode="markers", name="Live estimate",
                             marker=dict(size=13, color=ACCENT, line=dict(color="#FFFFFF", width=2))))
    return dengai_style(fig, xaxis_title="Week of year", yaxis_title="Cases per week")


# ----------------------------------------------------------------------------
# 8. RECOMMENDATIONS
# ----------------------------------------------------------------------------
WATERSIDE_ZONES = ("Zone 1 - Riverside", "Zone 4 - Lakeside")   # zones next to standing/open water


def build_recommendations(p: Inputs, r: RiskResult, zones: pd.DataFrame, place: str, river_ratio: float | None) -> list[dict]:
    ranked = zones.sort_values("risk", ascending=False)
    top, second = ranked.iloc[0], ranked.iloc[1]
    water_zone = ranked[ranked["zone"].isin(WATERSIDE_ZONES)].iloc[0]["zone"]   # riskiest waterside zone
    recs: list[dict] = []

    def add(who: str, title: str, detail: str, sev: str) -> None:
        recs.append({"who": who, "title": title, "detail": detail, "sev": sev})

    # --- Municipal & Sanitation teams ---
    if p.stagnation >= 50 or r.vector >= 60:
        add("Municipal & Sanitation", f"Deploy larvicide in stagnant water bodies near {water_zone}",
            f"Water stagnation index is {p.stagnation:.0f}% and vector breeding risk {r.vector:.0f}%. Treat ponds, "
            f"construction pits and roadside pools; repeat every 7-10 days.",
            "Critical" if r.vector >= 75 else "High" if p.stagnation >= 60 else "Moderate")
    if (100 - p.drainage) >= 50 and (p.stagnation >= 50 or p.rainfall > 200):
        add("Municipal & Sanitation", "Drainage flushing required due to high water stagnation score",
            f"Drainage quality is {p.drainage:.0f}% with {p.rainfall:.0f} mm rain in 30 days. Clear blocked drains and "
            f"culverts, starting in {top['zone']}.", "High" if p.stagnation >= 60 else "Moderate")
    if p.contamination >= 50:
        add("Municipal & Sanitation", "Fix water storage practices and check supply contamination",
            f"Contamination/storage index is {p.contamination:.0f}%. Run a cover-and-empty-weekly campaign for household "
            f"containers and test supply chlorination near {second['zone']}.", "High" if p.contamination >= 70 else "Moderate")
    if p.water_mgmt >= 40:
        add("Municipal & Sanitation", "Invest in safe-water and sanitation infrastructure",
            f"Water management gap is {p.water_mgmt:.0f}%: safely managed water and sanitation coverage is limited. "
            f"Prioritise piped supply, covered storage and sewer connections, starting in {second['zone']}.",
            "High" if p.water_mgmt >= 60 else "Moderate")
    if p.density >= 15000 and r.env >= 50:
        add("Municipal & Sanitation", "Prioritise door-to-door container checks in dense wards",
            f"About {p.density:,.0f} people per sq km with environmental vulnerability at {r.env:.0f}%. Focus crews on "
            f"{top['zone']} first.", "High")
    if river_ratio is not None and river_ratio >= 2:
        add("Municipal & Sanitation", f"Run larval surveys around receding flood pools near {water_zone}",
            f"River flow is {river_ratio:.1f}x its yearly median. Receding floodwater leaves shallow pools that breed "
            f"Aedes within days; survey banks and low-lying areas and treat positive sites.",
            "High" if river_ratio >= 3 else "Moderate")

    # --- Supply chain ---
    if r.supply >= 50:
        add("Supply chain", f"Refill Paracetamol & IV fluids in {top['zone']}",
            f"Stock is {p.med_availability:.0f}% vs demand {p.med_demand:.0f}% near {place}. Prioritise "
            f"{top['zone']} and {second['zone']}; add ORS and platelet-support supplies.",
            "Critical" if r.supply >= 75 else "High")
    elif r.supply >= 25:
        add("Supply chain", "Raise reorder levels for fever medicines",
            "Demand is outpacing stock. Increase safety stock by 15-20% before next week.", "Moderate")

    # --- Health authority ---
    if r.climate >= 50 or p.rainfall > 250:
        add("Health authority", "Start fogging and larval source reduction",
            f"{p.rainfall:.0f} mm rain in 30 days, {p.humidity:.0f}% humidity and {p.temperature:.1f} °C favour "
            f"breeding. Coordinate fogging with the sanitation drive in {top['zone']}.",
            "High" if r.climate >= 60 else "Moderate")
    if r.health >= 50:
        add("Health authority", "Open fever clinics and daily case reporting",
            f"Health system load is {r.health:.0f}%. Extend clinic hours, stock NS1/IgM rapid tests and reserve beds "
            f"for warning-sign cases.", "Critical" if r.health >= 75 else "High")
    if p.past_cases > 150:
        add("Health authority", "Trace clusters from recent cases",
            f"{p.past_cases} cases in 4 weeks. Map household clusters and run door-to-door checks.", "High")
    if p.aqi > 150:
        add("Health authority", "Issue a combined air-quality and fever advisory",
            f"AQI {p.aqi:.0f} adds respiratory load and can mask fever symptoms. Advise early testing for persistent fever.",
            "High" if p.aqi > 200 else "Moderate")
    if not recs:
        add("All teams", "Maintain routine surveillance",
            "Signals are within normal ranges. Keep weekly reporting and watch rainfall forecasts.", "Low")
    return sorted(recs, key=lambda x: ["Critical", "High", "Moderate", "Low"].index(x["sev"]))


# ----------------------------------------------------------------------------
# 9. SIDEBAR (healthcare, sanitation + optional weather/AQI overrides)
# ----------------------------------------------------------------------------
def sidebar_inputs(wx: dict, wm: dict | None, dg: dict | None, lat: float, lon: float) -> Inputs:
    k = f"{lat:.3f}_{lon:.3f}"   # new key per location so defaults refresh after a map click
    st.sidebar.markdown("### Local healthcare metrics")
    availability = st.sidebar.slider("Medicine Availability Index (%)", 0, 100, 55)
    demand = st.sidebar.slider("Medicine Demand / Pharmacy Search Volume (%)", 0, 100, 70)
    if dg:   # DengAI model city: prefill recent cases from the ML estimate
        past_cases = st.sidebar.number_input("Historical outbreak cases (last 4 weeks)", 0, 5000,
                                             int(np.clip(round(dg["last4_cases"]), 0, 5000)), 10, key=f"c{k}",
                                             help="Prefilled by the DengAI model")
    else:
        past_cases = st.sidebar.number_input("Historical outbreak cases (last 4 weeks)", 0, 5000, 120, 10)
    area_type = st.sidebar.selectbox("Area type", list(AREA_FACTOR), index=0)
    seasons = list(SEASON_FACTOR)
    season = st.sidebar.selectbox("Season (auto-detected from local weather)", seasons, key=f"s{k}",
                                  index=seasons.index(auto_season(wx["temperature"], wx["rainfall"])))
    horizon = st.sidebar.slider("Forecast horizon (weeks)", 4, 8, 6)

    st.sidebar.markdown("### Sanitation & infrastructure")
    stagnation = st.sidebar.slider("Stagnant Water Index (%) - higher is worse", 0, 100, 45,
                                   help="Standing water, open drains, blocked gutters, construction pits.")
    drainage = st.sidebar.slider("Drainage Quality Index (%) - higher is better", 0, 100, 50,
                                 help="Condition and coverage of drains, culverts and sewers.")
    contamination = st.sidebar.slider("Water Supply Contamination / Storage Index (%) - higher is worse", 0, 100, 40,
                                      help="Contaminated supply and open or uncovered household water storage.")
    density = st.sidebar.slider("Local Population Density (people per sq km)", 100, 50000, 6000, 100)

    st.sidebar.markdown("### Environment")
    live_aqi = wx["aqi"] is not None
    use_live_aqi = st.sidebar.checkbox("Use live AQI for this location", value=live_aqi, disabled=not live_aqi)
    aqi = float(wx["aqi"]) if (use_live_aqi and live_aqi) else float(st.sidebar.slider("Air Quality Index (AQI)", 0, 500, 110))

    temp, hum, rain = wx["temperature"], wx["humidity"], wx["rainfall"]
    if st.sidebar.checkbox("Override fetched weather"):
        temp = st.sidebar.slider("Temperature, 14-day mean (°C)", 10.0, 45.0, float(np.clip(temp, 10, 45)), 0.5, key=f"t{k}")
        hum = st.sidebar.slider("Humidity, 14-day mean (%)", 10, 100, int(np.clip(hum, 10, 100)), key=f"h{k}")
        rain = st.sidebar.slider("Rainfall, 30 days (mm)", 0, 800, int(np.clip(rain, 0, 800)), key=f"r{k}")
    # Live water signals; if unavailable, proxy from the sliders above
    wcond, gap = water_condition(wx, float(rain))[0], water_gap(wm)
    return Inputs(lat=lat, lon=lon, rainfall=float(rain), temperature=float(temp), humidity=float(hum), aqi=aqi,
                  med_availability=availability, med_demand=demand, past_cases=int(past_cases), season=season,
                  area_type=area_type, horizon=horizon, stagnation=float(stagnation), drainage=float(drainage),
                  contamination=float(contamination), density=float(density),
                  water_condition=wcond if wcond is not None else float(stagnation),
                  water_mgmt=gap if gap is not None else float(np.mean([100 - drainage, contamination])))


def water_notes(p: Inputs, wx: dict, wm: dict | None) -> tuple[str, str]:
    """KPI-card notes for the water-condition and water-management cards ('proxy' when slider-based)."""
    wcond, surplus = water_condition(wx, p.rainfall)
    if wcond is None:
        cond = f"Proxy: Stagnant Water slider ({p.stagnation:.0f}%)"
    else:
        parts = ([f"Surplus {surplus:+.0f} mm"] if surplus is not None else []) + \
                ([f"soil {wx['soil']:.2f} m³/m³"] if wx["soil"] is not None else []) + \
                ([f"river {wx['river_ratio']:.1f}× normal"] if wx["river_ratio"] is not None
                 else ["no significant river"] if wx["river_none"] else [])
        cond = " · ".join(parts)
        cond = cond[:1].upper() + cond[1:]
    if water_gap(wm) is None:
        mgmt = "Proxy: mean of drainage gap and contamination"
    else:
        pct = lambda v: f"{v:.0f}%" if v is not None else "n/a"
        mgmt = f"Safe water {pct(wm['water'])} · sanitation {pct(wm['sanitation'])} (World Bank, {wm['year']})"
    return cond, mgmt


def source_line(sources: dict) -> str:
    """Which live data sources answered for this location."""
    live, off = [k for k, ok in sources.items() if ok], [k for k, ok in sources.items() if not ok]
    return f"Live: {', '.join(live) or 'none'}" + (f" · Unavailable: {', '.join(off)}" if off else "")


# ----------------------------------------------------------------------------
# 10. DENGAI MODEL PANEL (only for the model cities, behind a toggle)
# ----------------------------------------------------------------------------
def dengai_panel(code: str, dg: dict, bundle: dict) -> None:
    """Full-width model details: KPI row, 2x2 charts (history, test period, importances, seasonality) and caption."""
    city, info = dengai.CITIES[code], bundle["cities"][code]
    m = info.get("metrics", {})
    hist, test = load_dengai_tables()
    h = hist[hist["city"] == code].sort_values("week_start_date") if hist is not None else None
    t = test[test["city"] == code].sort_values("week_start_date") if test is not None else None
    est, base = dg["weekly_cases"], m.get("mae_baseline")
    mae = m.get("mae_model_live_inputs", m.get("mae_model"))   # honest accuracy: error with Open-Meteo (live) inputs
    low_skill = bool(mae and base and mae >= base * 0.95)

    st.markdown(f'<div class="ow-section">DengAI model details: {city["name"]}</div>', unsafe_allow_html=True)
    kpi_grid([
        kpi_card("Live estimate", est, f"Cases/week · ~{dg['last4_cases']:.0f} over the last 4 weeks", suffix="", color=ACCENT),
        kpi_card("Holdout MAE", mae or 0, f"Cases/week with live-input weather, weeks since {str(m.get('holdout_start', 'n/a'))[:10]}",
                 suffix="", color=ACCENT, decimals=1),
        kpi_card("Seasonal-baseline MAE", base or 0,
                 f"{100 * (1 - mae / base):.0f}% lower error with the model" if mae and base and mae < base
                 else "Mean cases by week of year", suffix="", color=ACCENT, decimals=1),
        kpi_card("Training weeks", m.get("n_train_weeks", 0), f"{city['name']} · {city['period']}", suffix="", color=ACCENT),
    ])

    cfg = {"displayModeBar": False}
    top = st.columns(2, gap="large")
    with top[0]:
        st.markdown('<div class="ow-section">Actual vs predicted weekly cases</div>', unsafe_allow_html=True)
        if h is not None and len(h):
            st.plotly_chart(dengai_history_chart(h), width="stretch", config=cfg)
        else:
            st.caption("models/dengai_history.csv not found.")
    with top[1]:
        st.markdown('<div class="ow-section">Predicted cases, DengAI test period</div>', unsafe_allow_html=True)
        if t is not None and len(t):
            st.plotly_chart(dengai_test_chart(t), width="stretch", config=cfg)
        else:
            st.caption("models/dengai_test_predictions.csv not found.")
    bottom = st.columns(2, gap="large")
    with bottom[0]:
        st.markdown('<div class="ow-section">What the model relies on</div>', unsafe_allow_html=True)
        st.plotly_chart(dengai_importance_chart(info.get("feature_importance", {})), width="stretch", config=cfg)
    with bottom[1]:
        st.markdown('<div class="ow-section">Seasonal profile</div>', unsafe_allow_html=True)
        if h is not None and len(h):
            week = min(int(pd.Timestamp.now().isocalendar().week), 52)
            st.plotly_chart(dengai_season_chart(h, week, est), width="stretch", config=cfg)
    st.caption(f"Model: {info.get('model_name', 'n/a')}" + (" (low skill: barely beats the seasonal baseline)" if low_skill else "")
               + f" · Data period: {city['period']} · Holdout MAE is the live-input MAE (Open-Meteo inputs). Live estimate uses "
               f"current Open-Meteo weather; climate-only model (no case-reporting feed).")


# ----------------------------------------------------------------------------
# 11. MAIN
# ----------------------------------------------------------------------------
def location_search() -> None:
    """Search box: geocode each submitted query once and select that place."""
    ss = st.session_state
    query = st.text_input("🔍 Search location / city:", placeholder="e.g., Chennai, Bangkok, Rio de Janeiro",
                          key="loc_search").strip()
    if query and query != ss.seen_search:
        ss.seen_search = query                 # handle each submitted query once
        hit = forward_geocode(query)
        if hit:
            ss.lat, ss.lon, ss.picked = round(hit[0], 6), round(hit[1], 6), True
            st.rerun()
        st.warning("Location not found. Try another city name.")


def select_from_map(m: folium.Map, cities: dict) -> None:
    """Render the map; a new click, marker, rectangle or city-marker click selects that point."""
    ss = st.session_state
    out = st_folium(m, height=470, use_container_width=True, key="risk_map",
                    returned_objects=["last_clicked", "last_object_clicked", "last_active_drawing"])
    new_pt = read_map_event(out, cities)       # MAP EVENT -> new coordinates
    if new_pt:
        ss.lat, ss.lon, ss.picked = new_pt[0], new_pt[1], True
        st.rerun()                             # refetch weather + recompute everything


def render_start_screen(cities: dict) -> None:
    """Before the first selection: world map and search only. No default place is analysed."""
    st.sidebar.markdown("### Local inputs")
    st.sidebar.info("Pick a location on the map, or search for one, to load live data and adjust local inputs.")
    col_map, col_side = st.columns([2.1, 1], gap="large")
    with col_map:
        location_search()
        select_from_map(build_start_map(cities), cities)
    with col_side:
        st.markdown('<div class="ow-loc"><b>No location selected yet</b><br><small>Click anywhere on the map, drop a marker '
                    'or draw a rectangle with the toolbar, search for a city, or click a DengAI city marker.</small></div>',
                    unsafe_allow_html=True)


def main() -> None:
    ss = st.session_state
    ss.setdefault("picked", False)       # stays False until the user picks a place: nothing is preselected
    ss.setdefault("seen_click", None)
    ss.setdefault("seen_draw", None)
    ss.setdefault("seen_obj", None)
    ss.setdefault("seen_search", "")
    bundle = get_bundle() if dengai else None
    cities = dengai.CITIES if bundle else {}

    components.html(CURSOR_TRAIL_HTML, height=0)
    st.markdown('<h1 class="ow-title">SENTINEL: <span class="ow-title-sub">Syndromic Early-warning Network Tracking '
                'Infections &amp; Neighborhood Epidemics</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="ow-sub">Click the map to pick a location: live weather, air-quality and water signals update the '
                'dengue and fever risk instantly.</p>', unsafe_allow_html=True)
    if not ss.picked:
        render_start_screen(cities)
        return

    # Round to ~1 km so nearby clicks share the cache.
    lat, lon = round(ss.lat, 2), round(ss.lon, 2)
    with st.spinner("Fetching live weather, air-quality and water signals..."):
        wx = fetch_environment(lat, lon)
        place, iso2 = reverse_geocode(lat, lon)
        wm = fetch_water_mgmt(iso2) if iso2 else None
        city = dengai_city(ss.lat, ss.lon, bundle)       # DengAI model city under the selected point, if any
        dg = dengai_live(city)                           # its live ML estimate (None if unavailable)
    conf = " · lower confidence" if dg and dg.get("weeks_used", 99) < 16 else ""   # forecast-API fallback = short history
    ml_note = f"DengAI model city · ML estimate ~{dg['weekly_cases']:.0f} cases/week{conf}" if dg else None
    src_line = source_line({**wx["sources"], "World Bank": wm is not None})

    p = sidebar_inputs(wx, wm, dg, ss.lat, ss.lon)
    r = compute_risk(p)
    fc = forecast_outbreak(p, r)
    zones, pts = build_zone_data(p, r)

    show_dengai = False
    col_map, col_side = st.columns([2.1, 1], gap="large")
    with col_map:
        location_search()
        select_from_map(build_map(p, r, zones, pts, place, ml_note, cities), cities)
        if dg:
            show_dengai = st.toggle(f"📊 Show DengAI model details for {dengai.CITIES[city]['name']}", key="dengai_details")
        elif city:
            st.caption("DengAI model city, but the live estimate is unavailable right now (weather data missing or unreachable).")
    with col_side:
        st.markdown(f'<div class="ow-loc"><b>{place}</b><br>'
                    f'<small>{p.lat:.4f}, {p.lon:.4f} · {src_line}</small>'
                    + (f'<br><small><b style="color:{ACCENT}">DengAI model city</b> · ML estimate ~{dg["weekly_cases"]:.0f} cases/week{conf}</small>'
                       if dg else "") + '</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="ow-badge{" crit" if r.level == "Critical" else ""}" style="background:{LEVEL_COLORS[r.level]}">{r.level} hazard</span>',
                    unsafe_allow_html=True)
        st.plotly_chart(gauge(r.overall, r.level), width="stretch", config={"displayModeBar": False})
        st.caption("Click again or draw a rectangle to analyse another area.")

    if show_dengai:
        dengai_panel(city, dg, bundle)

    peak = fc.loc[fc["cases"].idxmax()]
    top_zone = zones.sort_values("risk", ascending=False).iloc[0]
    cond_note, mgmt_note = water_notes(p, wx, wm)
    pm25 = f"{wx['pm25']:.0f} µg/m³" if wx["pm25"] is not None else "n/a"
    kpi_grid([
        kpi_card("Outbreak Hazard Score", r.overall, f"{r.level} hazard"),
        kpi_card("Vector Breeding Risk", r.vector, f"Stagnation {p.stagnation:.0f}% · {p.rainfall:.0f} mm rain"),
        kpi_card("Health System Load", r.health, f"AQI {p.aqi:.0f} · {p.past_cases} recent cases"),
        kpi_card("Environmental Vulnerability", r.env, f"{p.density:,.0f}/km² · drainage {p.drainage:.0f}%"),
        kpi_card("Supply Shortage Risk", r.supply, f"Stock {p.med_availability}% vs demand {p.med_demand}%"),
        kpi_card("Climate Impact Factor", r.climate, f"{p.temperature:.1f} °C 14-d avg (now {wx['temp_now']:.0f} °C) · {p.humidity:.0f}% RH"),
        kpi_card("Projected peak", float(peak["cases"]), f"Cases/week, {peak['week']}", suffix=""),
        kpi_card("Highest-risk zone", float(top_zone["risk"]), top_zone["zone"]),
        kpi_card("Air Quality Risk", r.factors["Air quality (AQI)"], f"US AQI {p.aqi:.0f} · PM2.5 {pm25}"),
        kpi_card("Water Condition Risk", r.factors["Water condition (live)"], cond_note),
        kpi_card("Water Management Gap", r.factors["Water management gap"], mgmt_note),
        kpi_card("Recent Case Pressure", r.factors["Recent cases"],
                 f"{p.past_cases} cases in the last 4 weeks" + (" · DengAI estimate" if dg else "")),
    ])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="ow-section">What is driving the risk</div>', unsafe_allow_html=True)
        st.plotly_chart(breakdown_chart(r.factors), width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown(f'<div class="ow-section">Predicted outbreak curve, next {p.horizon} weeks</div>', unsafe_allow_html=True)
        st.plotly_chart(forecast_chart(fc, r.level), width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="ow-section">Zone breakdown</div>', unsafe_allow_html=True)
    table = (zones.sort_values("risk", ascending=False)[["zone", "risk", "level", "est_cases_wk"]]
             .rename(columns={"zone": "Zone", "risk": "Risk %", "level": "Level", "est_cases_wk": "Est. cases/wk"}))
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download zone data (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="sentinel_zones.csv", mime="text/csv")

    st.markdown('<div class="ow-section">Recommended actions</div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for i, rec in enumerate(build_recommendations(p, r, zones, place, wx["river_ratio"])):
        cols[i % 2].markdown(
            f'<div class="ow-alert" style="--c:{LEVEL_COLORS[rec["sev"]]}"><small>{rec["who"]} · {rec["sev"]} priority</small>'
            f'<br/><b>{rec["title"]}</b><br/>{rec["detail"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="ow-foot">Weather, air quality and river flow come from Open-Meteo (weather, air-quality and flood APIs); '
                'water and sanitation access from the World Bank WDI (WHO/UNICEF JMP); place names from OpenStreetMap'
                + ('; the DengAI case model is trained on the DengAI challenge data' if bundle else '') + '. '
                'Zones and case counts are simulated, and the model is for demonstration only.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
