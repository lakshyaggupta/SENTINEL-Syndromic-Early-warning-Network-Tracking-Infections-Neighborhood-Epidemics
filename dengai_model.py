"""Shared DengAI helpers: weekly features, live Open-Meteo fetch and inference.

Imported by both train_dengai.py and the Streamlit app so features are identical
at train and serve time. Only weather variables available live are used.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

CITIES: dict[str, dict] = {
    "sj": {"name": "San Juan, Puerto Rico", "lat": 18.4655, "lon": -66.1057, "period": "1990–2008"},
    "iq": {"name": "Iquitos, Peru", "lat": -3.7437, "lon": -73.2516, "period": "2000–2010"},
}
NEAR_KM: float = 60.0
BUNDLE_PATH: Path = Path(__file__).resolve().parent / "models" / "dengai_bundle.joblib"

BASE_FEATURES: list[str] = [
    "precip_mm", "temp_mean_c", "temp_max_c", "temp_min_c", "dtr_c", "rh_pct", "dewpoint_c",
]
_DENGAI_COLS: dict[str, tuple[str, float]] = {  # DengAI column -> (name, Kelvin offset)
    "precipitation_amt_mm": ("precip_mm", 0.0),
    "reanalysis_avg_temp_k": ("temp_mean_c", 273.15),
    "reanalysis_max_air_temp_k": ("temp_max_c", 273.15),
    "reanalysis_min_air_temp_k": ("temp_min_c", 273.15),
    "reanalysis_tdtr_k": ("dtr_c", 0.0),
    "reanalysis_relative_humidity_percent": ("rh_pct", 0.0),
    "reanalysis_dew_point_temp_k": ("dewpoint_c", 273.15),
}
_ROLL_WINDOWS = (2, 4, 8, 12, 16)  # weeks
_LAG_WEEKS = (4, 8, 12)  # 4-week mean ending this many weeks back
_MAX_WEEK = 52
_DAILY_VARS = {
    "temperature_2m_mean": "temp_mean_c",
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "precipitation_sum": "precip_mm",
    "relative_humidity_2m_mean": "rh_pct",
    "dew_point_2m_mean": "dewpoint_c",
}
_WEEKLY_COLS = ["week_start_date", "weekofyear", *BASE_FEATURES]


def nearest_city(lat: float, lon: float) -> str | None:
    """Code of the closest DengAI city within NEAR_KM (haversine), else None."""
    best: tuple[float, str] | None = None
    for code, c in CITIES.items():
        p1, p2 = math.radians(lat), math.radians(c["lat"])
        a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(c["lon"] - lon) / 2) ** 2
        km = 2 * 6371.0088 * math.asin(math.sqrt(a))
        if km <= NEAR_KM and (best is None or km < best[0]):
            best = (km, code)
    return best[1] if best else None


def base_from_dengai(df: pd.DataFrame) -> pd.DataFrame:
    """DengAI rows -> city, week_start_date, weekofyear + 7 base features (ffill then bfill per city)."""
    out = df[["city", "week_start_date", "weekofyear"]].copy()
    out["week_start_date"] = pd.to_datetime(out["week_start_date"])
    out["weekofyear"] = out["weekofyear"].astype(int).clip(upper=_MAX_WEEK)
    for src, (dst, offset) in _DENGAI_COLS.items():
        out[dst] = df[src].astype(float) - offset
    out = out.sort_values(["city", "week_start_date"]).reset_index(drop=True)
    out[BASE_FEATURES] = out.groupby("city")[BASE_FEATURES].ffill()
    out[BASE_FEATURES] = out.groupby("city")[BASE_FEATURES].bfill()
    return out


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily Open-Meteo frame -> weekly rows in 7-day bins ending on the last date.

    The incomplete oldest bin is dropped. weekofyear is the ISO week of the bin's first day, capped at 52.
    """
    src = ["temp_mean_c", "temp_max_c", "temp_min_c", "precip_mm", "rh_pct", "dewpoint_c"]
    d = daily[["date", *src]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.dropna(subset=src, how="all").drop_duplicates("date").sort_values("date")
    if d.empty:
        return pd.DataFrame(columns=_WEEKLY_COLS)
    d["bin"] = (d["date"].max() - d["date"]).dt.days // 7
    d["dtr_c"] = d["temp_max_c"] - d["temp_min_c"]
    w = d.groupby("bin").agg(
        week_start_date=("date", "min"), n=("date", "size"), precip_mm=("precip_mm", "sum"),
        temp_mean_c=("temp_mean_c", "mean"), temp_max_c=("temp_max_c", "max"),
        temp_min_c=("temp_min_c", "min"), dtr_c=("dtr_c", "mean"),
        rh_pct=("rh_pct", "mean"), dewpoint_c=("dewpoint_c", "mean"),
    ).sort_index(ascending=False)  # oldest first
    if len(w) and w["n"].iloc[0] < 7:
        w = w.iloc[1:]
    w = w.reset_index(drop=True)
    w["weekofyear"] = w["week_start_date"].dt.isocalendar().week.astype(int).clip(upper=_MAX_WEEK)
    return w[_WEEKLY_COLS]


def build_features(base: pd.DataFrame, climatology: dict[int, float]) -> pd.DataFrame:
    """One city's chronologically sorted base rows -> model features (needs <= 16 weeks of history).

    Per base feature: rolling means (2/4/8/12/16 wk) and 4-wk means lagged by 4, 8 and 12 weeks.
    Plus weekofyear sin/cos and the seasonal climatology of log1p(cases).
    """
    b = base.reset_index(drop=True)
    cols: dict[str, pd.Series] = {}
    for w in _ROLL_WINDOWS:
        r = b[BASE_FEATURES].rolling(w, min_periods=1).mean()
        cols.update({f"{c}_r{w}": r[c] for c in BASE_FEATURES})
    r4 = b[BASE_FEATURES].rolling(4, min_periods=1).mean()
    for lag in _LAG_WEEKS:
        s = r4.shift(lag).fillna(r4.iloc[0])  # not enough history: use the oldest known week
        cols.update({f"{c}_l{lag}": s[c] for c in BASE_FEATURES})
    ang = 2 * np.pi * b["weekofyear"].astype(float) / _MAX_WEEK
    cols["woy_sin"], cols["woy_cos"] = np.sin(ang), np.cos(ang)
    default = float(np.mean(list(climatology.values()))) if climatology else 0.0
    cols["clim"] = b["weekofyear"].astype(int).map(climatology).astype(float).fillna(default)
    return pd.DataFrame(cols)


def load_bundle() -> dict | None:
    """Load the trained bundle; None if missing or corrupt."""
    try:
        bundle = joblib.load(BUNDLE_PATH)
        return bundle if isinstance(bundle, dict) and "cities" in bundle and "features" in bundle else None
    except Exception:
        return None


def _get_json(url: str, params: dict, timeout: float = 10) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _daily_frame(payload: dict | None) -> pd.DataFrame | None:
    daily = (payload or {}).get("daily")
    if not daily or "time" not in daily or any(v not in daily for v in _DAILY_VARS):
        return None
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_name, name in _DAILY_VARS.items():
        df[name] = pd.to_numeric(pd.Series(daily[api_name]), errors="coerce")
    df = df.dropna(subset=list(_DAILY_VARS.values()), how="all")
    return df.reset_index(drop=True) if len(df) >= 14 else None


def fetch_daily_range(lat: float, lon: float, start: date, end: date, timeout: float = 10) -> pd.DataFrame | None:
    """Daily weather for start..end from the Open-Meteo archive API; None on failure."""
    return _daily_frame(_get_json("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": lat, "longitude": lon, "daily": ",".join(_DAILY_VARS), "timezone": "auto",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    }, timeout))


def fetch_daily(lat: float, lon: float) -> pd.DataFrame | None:
    """Last ~26 weeks of daily weather from Open-Meteo (archive, then forecast API); None on failure."""
    today = date.today()
    df = fetch_daily_range(lat, lon, today - timedelta(days=190), today - timedelta(days=6))
    if df is not None:
        return df
    return _daily_frame(_get_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "daily": ",".join(_DAILY_VARS), "timezone": "auto",
        "past_days": 92, "forecast_days": 1,
    }))


def apply_calibration(weekly: pd.DataFrame, calibration: dict[str, dict[str, list[float]]] | None) -> pd.DataFrame:
    """Map Open-Meteo weekly features onto the DengAI (NCEP/satellite) scale by quantile mapping.

    DengAI inputs differ systematically from Open-Meteo/ERA5 (e.g. Iquitos temp_min, daily range), so
    each feature is mapped via the stored {"src": ERA5 quantiles, "dst": DengAI quantiles}.
    """
    out = weekly.copy()
    for col, q in (calibration or {}).items():
        out[col] = np.interp(out[col].astype(float), q["src"], q["dst"])
    return out


def predict_live(bundle: dict, city: str, daily: pd.DataFrame) -> dict:
    """Estimate weekly dengue cases from daily weather for a DengAI city.

    Returns weekly_cases (latest week), last4_cases (sum of last 4 weekly estimates), recent (last 8 weeks)
    and weeks_used. Weekly inputs are quantile-calibrated to the DengAI scale (when the bundle has a
    calibration) and clipped to the city's training ranges. Raises ValueError if no full week exists.
    """
    spec = bundle["cities"][city]
    weekly = weekly_from_daily(daily)
    if weekly.empty:
        raise ValueError("not enough daily weather to form a complete week")
    weekly = apply_calibration(weekly, spec.get("calibration"))
    for col, (lo, hi) in spec["train_ranges"].items():
        weekly[col] = weekly[col].astype(float).clip(lo, hi)
    x = build_features(weekly, spec["climatology"])[bundle["features"]]
    pred = np.clip(np.asarray(spec["model"].predict(x), dtype=float), 0.0, None)
    recent = [
        {"week_start_date": d.strftime("%Y-%m-%d"), "predicted": float(p)}
        for d, p in zip(weekly["week_start_date"].tail(8), pred[-8:])
    ]
    return {
        "weekly_cases": float(pred[-1]),
        "last4_cases": float(pred[-4:].sum()),
        "recent": recent,
        "weeks_used": int(len(weekly)),
    }
