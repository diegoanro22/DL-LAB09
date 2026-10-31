"""Baselines estadísticos (StatsForecast) y LightGBM global (sección 5).

Todos devuelven un `forecast_fn(hist, origin, H)` para `src.backtest.backtest`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.backtest import frame_from_matrix, tail_matrix
from src.data import SeriesScaler, calendar_features, pt_holidays


# --------------------------------------------------------------------------------------
# Naive y seasonal naive (sin parámetros)
# --------------------------------------------------------------------------------------
def naive_fn(hist, origin, H):
    """Repite el último valor observado (23:00 del día anterior)."""
    ids, x = tail_matrix(hist, 1)
    return frame_from_matrix(ids, origin, np.repeat(x[:, -1:, 0], H, axis=1))


def seasonal_naive_fn(season: int):
    """Repite el valor de hace `season` horas (24 = ayer, 168 = semana pasada)."""
    def fn(hist, origin, H):
        ids, x = tail_matrix(hist, season)
        y = x[:, :, 0]
        reps = int(np.ceil(H / season))
        return frame_from_matrix(ids, origin, np.tile(y, reps)[:, :H])
    return fn


# --------------------------------------------------------------------------------------
# MSTL (StatsForecast)
# --------------------------------------------------------------------------------------
def mstl_fn(weeks: int = 8, n_jobs: int = -1):
    """MSTL con estacionalidades diaria y semanal y tendencia con AutoETS.

    Es un modelo local (uno por cliente). Ajustarlo sobre los ~2.5 años completos en cada
    origen tarda >1 min por cliente, así que en cada origen se ajusta sobre las últimas
    `weeks` semanas (8 semanas = 8 ciclos semanales), que es lo usual en pronóstico de corto
    plazo. Solo usa datos anteriores al origen.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoETS

    def fn(hist, origin, H):
        tail = hist.loc[hist["ds"] >= origin - pd.Timedelta(weeks=weeks), ["unique_id", "ds", "y"]]
        sf = StatsForecast(
            models=[MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))],
            freq="h", n_jobs=n_jobs,
        )
        fc = sf.forecast(df=tail, h=H).reset_index()
        return pd.DataFrame({"unique_id": fc["unique_id"].astype(str), "ds": fc["ds"], "y_hat": fc["MSTL"]})

    return fn


# --------------------------------------------------------------------------------------
# LightGBM global con rezagos y calendario (mlforecast)
# --------------------------------------------------------------------------------------
LGB_LAGS = list(range(1, 25)) + [168]
LGB_DEFAULT = dict(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _with_holiday(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["unique_id", "ds", "y"]].copy()
    out["is_holiday"] = calendar_features(out["ds"])["is_holiday"].to_numpy()
    return out


def fit_lightgbm(train_panel: pd.DataFrame, params: dict | None = None, lags=LGB_LAGS,
                 history_start: pd.Timestamp | None = None):
    """Entrena un LightGBM global (todas las series) y devuelve (forecast_fn, info).

    - Objetivo normalizado por serie con SeriesScaler ajustado SOLO sobre `train_panel`.
    - Rezagos 1-24 y 168; calendario: hora, día de la semana, mes (de la fecha) y feriado
      como exógena conocida a futuro.
    - Estrategia recursiva: predice h=1, lo usa como rezago para h=2, etc. (mlforecast).
    - Se entrena una vez; en cada origen `predict(new_df=historia)` actualiza los rezagos
      sin reentrenar.
    """
    import lightgbm as lgb
    from mlforecast import MLForecast

    if history_start is not None:
        train_panel = train_panel.loc[train_panel["ds"] >= history_start]
    fit_end = train_panel["ds"].max()
    scaler = SeriesScaler(fit_end=fit_end).fit(train_panel)
    z = _with_holiday(scaler.transform(train_panel))

    model = lgb.LGBMRegressor(**{**LGB_DEFAULT, **(params or {})})
    mlf = MLForecast(models={"LightGBM": model}, freq="h", lags=lags,
                     date_features=["hour", "dayofweek", "month"])
    t = time.perf_counter()
    mlf.fit(z, static_features=[])
    fit_s = time.perf_counter() - t
    max_lag = max(lags)

    def fn(hist, origin, H):
        tail = hist.loc[hist["ds"] >= origin - pd.Timedelta(hours=max_lag + 24)]
        zt = _with_holiday(scaler.transform(tail))
        fut = pd.date_range(origin, periods=H, freq="h")
        ids = zt["unique_id"].unique()
        X_df = pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(fut, len(ids))})
        X_df["is_holiday"] = X_df["ds"].dt.normalize().isin(pt_holidays([origin.year, fut[-1].year])).astype("float32")
        pred = mlf.predict(h=H, new_df=zt, X_df=X_df)
        pred = pred.rename(columns={"LightGBM": "y_hat"})
        pred["unique_id"] = pred["unique_id"].astype(str)
        return scaler.inverse(pred)[["unique_id", "ds", "y_hat"]]

    # "Parámetros" de un ensamble de árboles: un valor por hoja (+ umbral por nodo interno)
    fitted = mlf.models_["LightGBM"]
    trees = fitted.booster_.trees_to_dataframe()
    info = {
        "fit_s": fit_s,
        "arboles": int(fitted.booster_.num_trees()),
        "params_total": int(len(trees)),           # nodos totales (hojas + divisiones)
        "filas_train": len(z),
        "historia_desde": str(z["ds"].min()),
    }
    return fn, info
