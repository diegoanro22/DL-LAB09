"""Métricas de la sección 4, siempre en kWh (escala original).

Todas reciben un DataFrame de pronósticos con columnas
    unique_id, ds, origin, y_hat   (+ q10, q50, q90 si es probabilístico)
y el panel con los valores reales (unique_id, ds, y).

MASE usa como escala, por cliente, el MAE del seasonal naive de 24 h dentro del bloque
de entrenamiento (ds <= 2014-06-30). MASE < 1 indica un error menor que la escala
estacional dentro de train; no demuestra que supere al seasonal naive en test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TRAIN_END

CHRISTMAS_WEEK = (pd.Timestamp("2014-12-22 00:00"), pd.Timestamp("2014-12-28 23:00"))
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


# --------------------------------------------------------------------------------------
# Escala del MASE
# --------------------------------------------------------------------------------------
def mase_scale(panel: pd.DataFrame, season: int = 24, end: pd.Timestamp = TRAIN_END) -> pd.Series:
    """MAE del seasonal naive (y_t - y_{t-season}) por cliente, con datos hasta `end`."""
    train = panel.loc[panel["ds"] <= end, ["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    diff = train.groupby("unique_id", observed=True)["y"].diff(season).abs()
    return diff.groupby(train["unique_id"], observed=True).mean().rename("scale")


# --------------------------------------------------------------------------------------
# Unión con los reales
# --------------------------------------------------------------------------------------
def attach_actuals(fc: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega y real, error, paso del horizonte h (1..H) y hora del día."""
    out = fc.merge(panel[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left", validate="m:1")
    assert out["y"].notna().all(), "hay pronósticos sin valor real (ds fuera del panel)"
    out["err"] = out["y_hat"] - out["y"]
    out["h"] = ((out["ds"] - out["origin"]) / pd.Timedelta(hours=1)).astype(int) + 1
    out["hour"] = out["ds"].dt.hour
    return out


# --------------------------------------------------------------------------------------
# Métricas puntuales
# --------------------------------------------------------------------------------------
def _point(df: pd.DataFrame, scale: pd.Series, mape_eps: float = 1e-3) -> dict:
    ae = df["err"].abs()
    y = df["y"].abs()
    nz = y > mape_eps
    per_series_mae = ae.groupby(df["unique_id"], observed=True).mean()
    mase_series = per_series_mae / scale.reindex(per_series_mae.index)
    return {
        "MAE": ae.mean(),
        "RMSE": np.sqrt((df["err"] ** 2).mean()),
        "WAPE": ae.sum() / y.sum(),
        # MAPE solo sobre horas con consumo > eps; con y ~ 0 el cociente explota
        "MAPE": (ae[nz] / y[nz]).mean(),
        "MAPE_excluidas": int((~nz).sum()),
        "MASE": mase_series.mean(),
        "n": len(df),
    }


def _probabilistic(df: pd.DataFrame) -> dict:
    """Pinball loss promedio de los cuantiles disponibles y cobertura del intervalo P10-P90."""
    qs = [c for c in QUANTILES if c in df.columns]
    if not qs:
        return {}
    out = {}
    losses = []
    for c in qs:
        tau = QUANTILES[c]
        diff = df["y"] - df[c]
        loss = np.maximum(tau * diff, (tau - 1) * diff).mean()
        out[f"pinball_{c}"] = loss
        losses.append(loss)
    out["pinball"] = float(np.mean(losses))
    if {"q10", "q90"} <= set(df.columns):
        out["cobertura_P10_P90"] = ((df["y"] >= df["q10"]) & (df["y"] <= df["q90"])).mean()
    return out


def evaluate(fc: pd.DataFrame, panel: pd.DataFrame, scale: pd.Series | None = None) -> dict:
    """Todas las métricas y desgloses que pide la sección 4.

    Devuelve un dict con:
        resumen      : MAE, RMSE, WAPE, MAPE, MASE (+ pinball y cobertura si aplica)
        por_cliente  : MAE, RMSE y MASE por cliente
        por_h        : MAE, RMSE y MASE por paso del horizonte (h = 1..H)
        por_hora     : MAE y MASE por hora del día
        navidad      : resumen sobre la semana del 22 al 28 de diciembre
    """
    scale = mase_scale(panel) if scale is None else scale
    df = attach_actuals(fc, panel)

    resumen = {**_point(df, scale), **_probabilistic(df)}

    g = df.groupby("unique_id", observed=True)
    por_cliente = pd.DataFrame(
        {
            "MAE": g["err"].apply(lambda e: e.abs().mean()),
            "RMSE": g["err"].apply(lambda e: np.sqrt((e ** 2).mean())),
        }
    )
    por_cliente["MASE"] = por_cliente["MAE"] / scale.reindex(por_cliente.index)

    def breakdown(key):
        rows = {k: _point(sub, scale) for k, sub in df.groupby(key)}
        return pd.DataFrame(rows).T[["MAE", "RMSE", "WAPE", "MASE"]].astype(float)

    lo, hi = CHRISTMAS_WEEK
    xmas = df[(df["ds"] >= lo) & (df["ds"] <= hi)]
    navidad = {**_point(xmas, scale), **_probabilistic(xmas)} if len(xmas) else {}

    return {
        "resumen": resumen,
        "por_cliente": por_cliente,
        "por_h": breakdown("h"),
        "por_hora": breakdown("hour"),
        "navidad": navidad,
    }


def summary_row(name: str, ev: dict) -> pd.Series:
    """Fila compacta para la tabla comparativa."""
    row = {k: v for k, v in ev["resumen"].items() if k not in ("n", "MAPE_excluidas")}
    if ev["navidad"]:
        row["MASE_navidad"] = ev["navidad"]["MASE"]
    return pd.Series(row, name=name)
