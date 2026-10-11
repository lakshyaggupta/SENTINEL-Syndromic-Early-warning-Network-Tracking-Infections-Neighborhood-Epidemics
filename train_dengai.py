"""Train per-city DengAI case models (weather-only features) and write models/ artifacts.

Run: venv\\Scripts\\python.exe train_dengai.py
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from dengai_model import (
    BASE_FEATURES, BUNDLE_PATH, CITIES, apply_calibration, base_from_dengai, build_features, fetch_daily_range,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = BUNDLE_PATH.parent
HOLDOUT_FRAC = 0.25
CALIB_QUANTILES = np.linspace(0, 1, 21)
SEED = 0

XGB_CONFIGS = [
    dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.6, reg_lambda=5),
    dict(n_estimators=200, learning_rate=0.05, max_depth=2, min_child_weight=10, subsample=0.8, colsample_bytree=0.5, reg_lambda=10),
    dict(n_estimators=500, learning_rate=0.02, max_depth=4, min_child_weight=10, subsample=0.7, colsample_bytree=0.5, reg_lambda=10),
    dict(n_estimators=400, learning_rate=0.02, max_depth=3, min_child_weight=15, subsample=0.7, colsample_bytree=0.4, reg_lambda=20),
    dict(n_estimators=600, learning_rate=0.01, max_depth=5, min_child_weight=10, subsample=0.7, colsample_bytree=0.5, reg_lambda=10),
]
HGB_CONFIGS = [
    dict(max_iter=200, learning_rate=0.04, max_depth=3, min_samples_leaf=20, l2_regularization=1.0),
    dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=30, l2_regularization=5.0),
    dict(max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=15, l2_regularization=2.0),
    dict(max_iter=250, learning_rate=0.03, max_depth=5, min_samples_leaf=25, l2_regularization=5.0),
]
CANDIDATES = (
    [("XGBoost (Poisson)", p) for p in XGB_CONFIGS] + [("HistGradientBoosting (Poisson)", p) for p in HGB_CONFIGS]
)


def make_model(name: str, params: dict):
    if name.startswith("XGBoost"):
        return XGBRegressor(objective="count:poisson", random_state=SEED, n_jobs=4, **params)
    return HistGradientBoostingRegressor(loss="poisson", random_state=SEED, **params)


def climatology(train: pd.DataFrame) -> dict[int, float]:
    """Circularly smoothed (1-2-3-2-1) mean log1p(cases) per weekofyear, from the given rows only."""
    m = np.log1p(train["total_cases"]).groupby(train["weekofyear"]).mean().reindex(range(1, 53))
    v = m.interpolate(limit_direction="both").to_numpy()
    kernel = {-2: 1, -1: 2, 0: 3, 1: 2, 2: 1}
    smooth = sum(np.roll(v, s) * w for s, w in kernel.items()) / sum(kernel.values())
    return {week: float(x) for week, x in enumerate(smooth, start=1)}


def top_importance(model, x_hold: pd.DataFrame, y_hold: np.ndarray, features: list[str]) -> dict[str, float]:
    """Native importances if available, else permutation importance on holdout; top 12 normalised to 1."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        imp = permutation_importance(
            model, x_hold, y_hold, scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED
        ).importances_mean
    imp = np.clip(imp, 0.0, None)
    top = sorted(zip(features, imp), key=lambda t: t[1], reverse=True)[:12]
    total = sum(v for _, v in top) or 1.0
    return {f: float(v / total) for f, v in top}


def era5_weekly(code: str, starts: pd.Series) -> pd.DataFrame | None:
    """Open-Meteo weekly features for the 7-day window starting on each DengAI week_start_date (same
    aggregation as weekly_from_daily). None if the archive request fails."""
    c = CITIES[code]
    daily = fetch_daily_range(c["lat"], c["lon"], starts.min().date(),
                              (starts.max() + pd.Timedelta(days=6)).date(), timeout=120)
    if daily is None:
        return None
    d = daily.set_index("date").asfreq("D")
    w = pd.DataFrame({
        "precip_mm": d["precip_mm"].rolling(7).sum(), "temp_mean_c": d["temp_mean_c"].rolling(7).mean(),
        "temp_max_c": d["temp_max_c"].rolling(7).max(), "temp_min_c": d["temp_min_c"].rolling(7).min(),
        "dtr_c": (d["temp_max_c"] - d["temp_min_c"]).rolling(7).mean(),
        "rh_pct": d["rh_pct"].rolling(7).mean(), "dewpoint_c": d["dewpoint_c"].rolling(7).mean(),
    })
    return w.reindex(starts + pd.Timedelta(days=6)).reset_index(drop=True)


def fit_calibration(era5: pd.DataFrame, dengai: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Per-feature quantile maps from Open-Meteo values to the DengAI distribution (same weeks)."""
    return {
        c: {"src": np.quantile(era5[c].dropna(), CALIB_QUANTILES).tolist(),
            "dst": np.quantile(dengai[c], CALIB_QUANTILES).tolist()}
        for c in BASE_FEATURES
    }


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Base rows for train+test concatenated (so ffill/rolling windows span the boundary), plus raw frames."""
    train = pd.read_csv(ROOT / "dengue_features_train.csv")
    test = pd.read_csv(ROOT / "dengue_features_test.csv")
    labels = pd.read_csv(ROOT / "dengue_labels_train.csv")
    base = base_from_dengai(pd.concat([train, test], ignore_index=True))
    y = train.merge(labels, on=["city", "year", "weekofyear"])[["city", "week_start_date", "total_cases"]]
    y["week_start_date"] = pd.to_datetime(y["week_start_date"])
    return base.merge(y, on=["city", "week_start_date"], how="left"), train, test


def fit_city(code: str, rows: pd.DataFrame, era5: pd.DataFrame | None) -> dict:
    """Select, refit and score the model for one city. Returns bundle entry plus frames for the CSV outputs."""
    labelled = rows[rows["total_cases"].notna()].reset_index(drop=True)
    n = len(labelled)
    n_hold = int(round(n * HOLDOUT_FRAC))
    cut = n - n_hold
    fit, hold = labelled.iloc[:cut], labelled.iloc[cut:]

    # model selection: climatology from the fit portion only
    x_all = build_features(labelled, climatology(fit))
    features = list(x_all.columns)
    x_fit, x_hold = x_all.iloc[:cut], x_all.iloc[cut:]
    y_fit, y_hold = fit["total_cases"].to_numpy(float), hold["total_cases"].to_numpy(float)

    baseline = fit.groupby("weekofyear")["total_cases"].mean()
    pred_base = hold["weekofyear"].map(baseline).fillna(fit["total_cases"].mean()).to_numpy(float)
    mae_baseline = float(np.mean(np.abs(pred_base - y_hold)))

    best = None
    for name, params in CANDIDATES:
        model = make_model(name, params).fit(x_fit, y_fit)
        mae = float(np.mean(np.abs(model.predict(x_hold) - y_hold)))
        if best is None or mae < best[0]:
            best = (mae, name, params, model)
    mae_model, name, params, holdout_model = best
    holdout_pred = np.clip(holdout_model.predict(x_hold), 0, None)
    importance = top_importance(holdout_model, x_hold, y_hold, features)

    # live proxy: same holdout weeks, but Open-Meteo inputs calibrated using the fit portion only
    calibration, mae_live = None, None
    if era5 is not None:
        calibration = fit_calibration(era5, rows[BASE_FEATURES])
        era5_lab = apply_calibration(era5.iloc[:n], fit_calibration(era5.iloc[:cut], fit[BASE_FEATURES]))
        era5_lab = pd.concat([labelled[["weekofyear"]], era5_lab[BASE_FEATURES].ffill().bfill()], axis=1)
        x_live = build_features(era5_lab, climatology(fit))[features].iloc[cut:]
        mae_live = float(np.mean(np.abs(np.clip(holdout_model.predict(x_live), 0, None) - y_hold)))

    # refit on all labelled weeks; features for train+test together so windows span the boundary
    clim = climatology(labelled)
    x_full = build_features(rows.reset_index(drop=True), clim)[features]
    is_lab = rows["total_cases"].notna().to_numpy()
    model = make_model(name, params).fit(x_full[is_lab], rows.loc[is_lab, "total_cases"].to_numpy(float))
    if hasattr(model, "feature_importances_"):
        importance = top_importance(model, x_hold, y_hold, features)
    pred_full = np.clip(model.predict(x_full), 0, None)

    history = pd.concat([
        pd.DataFrame({"city": code, "week_start_date": fit["week_start_date"], "actual": y_fit,
                      "predicted": pred_full[is_lab][:cut], "split": "fit"}),
        pd.DataFrame({"city": code, "week_start_date": hold["week_start_date"], "actual": y_hold,
                      "predicted": holdout_pred, "split": "holdout"}),
    ], ignore_index=True)
    test_rows = rows.loc[~is_lab, ["city", "week_start_date"]].copy()
    test_rows["predicted"] = pred_full[~is_lab]

    entry = {
        "model": model,
        "model_name": name,
        "climatology": clim,
        "train_ranges": {c: (float(labelled[c].min()), float(labelled[c].max())) for c in BASE_FEATURES},
        "metrics": {
            "mae_model": mae_model, "mae_baseline": mae_baseline, "n_train_weeks": n,
            "mae_model_live_inputs": mae_live,  # same holdout, Open-Meteo inputs after calibration
            "holdout_start": hold["week_start_date"].iloc[0].strftime("%Y-%m-%d"),
        },
        "feature_importance": importance,
        "calibration": calibration,
    }
    return {"entry": entry, "features": features, "history": history, "test": test_rows, "params": params}


def main() -> None:
    t0 = time.time()
    warnings.filterwarnings("ignore")
    MODELS_DIR.mkdir(exist_ok=True)
    base, train_raw, test_raw = load_base()

    cities, histories, tests, features = {}, [], [], None
    for code in CITIES:
        rows = base[base["city"] == code].sort_values("week_start_date").reset_index(drop=True)
        era5 = era5_weekly(code, rows["week_start_date"])
        if era5 is None:
            print(f"WARNING: Open-Meteo history unavailable for {code}; no input calibration stored")
        res = fit_city(code, rows, era5)
        cities[code] = res["entry"]
        histories.append(res["history"])
        tests.append(res["test"])
        features = res["features"]

    bundle = {"trained_at": datetime.now().isoformat(timespec="seconds"), "features": features, "cities": cities}
    joblib.dump(bundle, BUNDLE_PATH)

    pd.concat(histories, ignore_index=True).to_csv(MODELS_DIR / "dengai_history.csv", index=False, float_format="%.3f")
    pred = pd.concat(tests, ignore_index=True)
    test_meta = test_raw[["city", "year", "weekofyear", "week_start_date"]].copy()
    test_meta["week_start_date"] = pd.to_datetime(test_meta["week_start_date"])
    out = test_meta.merge(pred, on=["city", "week_start_date"], how="left")
    out[["city", "week_start_date", "year", "weekofyear", "predicted"]].to_csv(
        MODELS_DIR / "dengai_test_predictions.csv", index=False, float_format="%.3f")

    sub = pd.read_csv(ROOT / "submission_format.csv")
    sub = sub[["city", "year", "weekofyear"]].merge(out[["city", "year", "weekofyear", "predicted"]],
                                                    on=["city", "year", "weekofyear"], how="left")
    assert sub["predicted"].notna().all(), "missing predictions for submission rows"
    sub["total_cases"] = np.rint(sub["predicted"].clip(lower=0)).astype(int)
    sub[["city", "year", "weekofyear", "total_cases"]].to_csv(ROOT / "submission.csv", index=False)

    print(f"{'city':<5}{'model':<32}{'MAE':>7}{'baseline':>10}{'gain':>8}{'MAE live':>10}{'train wk':>10}  holdout from")
    for code, c in cities.items():
        m = c["metrics"]
        gain = 100 * (1 - m["mae_model"] / m["mae_baseline"])
        live = m["mae_model_live_inputs"]
        print(f"{code:<5}{c['model_name']:<32}{m['mae_model']:>7.2f}{m['mae_baseline']:>10.2f}{gain:>7.1f}%"
              f"{('n/a' if live is None else format(live, '.2f')):>10}{m['n_train_weeks']:>10}  {m['holdout_start']}")
    print(f"{len(features)} features; bundle -> {BUNDLE_PATH}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
