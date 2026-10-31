"""Adaptador de NeuralForecast al protocolo del lab (PatchTST, iTransformer, N-HiTS, TCN, DeepAR).

Flujo (igual que con los modelos propios):
1. `fit_nf(make, panel_z, val_size=2208)`: entrena con datos hasta 2014-09-30, usando los
   últimos 3 meses (jul-sep) SOLO para early stopping. Los pesos quedan entrenados con datos
   hasta el 30 de junio; con ellos se mide el MASE de validación (backtest sobre jul-sep).
2. `fit_nf(make, panel_z, val_size=0, max_steps=mejor_paso)`: reentrena una vez con todos
   los datos hasta el 30 de septiembre durante los pasos que eligió la validación.
3. `nf_forecast_fn(...)` -> forecast_fn para `backtest()`: en cada origen llama a
   `predict(df=últimas L horas)` sin reentrenar.

Se le pasa la y ya normalizada por serie (SeriesScaler) y se desnormaliza la salida, igual
que con los modelos propios.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.data import VAL_END, calendar_features

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="lightning")

QUIET = dict(enable_progress_bar=False, logger=False, enable_model_summary=False)


def _with_calendar(df: pd.DataFrame, futr: list[str]) -> pd.DataFrame:
    out = df[["unique_id", "ds", "y"]].copy()
    if futr:
        cal = calendar_features(out["ds"])
        for c in futr:
            out[c] = cal[c].to_numpy()
    return out


def fit_nf(
    make: Callable[..., object],
    panel_z: pd.DataFrame,
    val_size: int = 2208,
    max_steps: int = 1500,
    val_check_steps: int = 100,
    patience: int = 5,
    end: pd.Timestamp = VAL_END,
    history_start: pd.Timestamp | None = None,
    futr_exog: list[str] | None = None,
):
    """Entrena un modelo de NeuralForecast. `make(**kw)` construye el modelo con los
    argumentos de entrenamiento que se le pasan (max_steps, early stopping, etc.).

    Devuelve (nf, info) con pasos entrenados, mejor paso en validación, curvas, tiempo,
    parámetros y memoria pico de GPU.
    """
    from neuralforecast import NeuralForecast

    df = panel_z.loc[panel_z["ds"] <= end]
    if history_start is not None:
        df = df.loc[df["ds"] >= history_start]
    df = _with_calendar(df, futr_exog or [])
    kw = dict(max_steps=max_steps, val_check_steps=val_check_steps,
              early_stop_patience_steps=patience if val_size else -1, **QUIET)
    model = make(**kw)
    nf = NeuralForecast(models=[model], freq="h")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    nf.fit(df=df, val_size=val_size)
    fit_s = time.perf_counter() - t

    m = nf.models[0]
    train_traj = pd.DataFrame(getattr(m, "train_trajectories", []), columns=["step", "train_loss"])
    valid_traj = pd.DataFrame(getattr(m, "valid_trajectories", []), columns=["step", "val_loss"])
    steps = int(train_traj["step"].max()) + 1 if len(train_traj) else max_steps
    best_step = int(valid_traj.loc[valid_traj["val_loss"].idxmin(), "step"]) if len(valid_traj) else steps
    n_params = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    info = {
        "fit_s": fit_s,
        "pasos": steps,
        "mejor_paso": best_step,
        "params_total": n_params,
        "params_entrenables": n_train,
        "gpu_mem_pico_MB": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
        "historia_desde": str(df["ds"].min()),
        "train_traj": train_traj,
        "valid_traj": valid_traj,
    }
    return nf, info


def nf_forecast_fn(nf, scaler, L: int, model_col: str, futr_exog: list[str] | None = None,
                   quantile_cols: dict | None = None):
    """forecast_fn para backtest(): predict con las últimas L horas, sin reentrenar.

    quantile_cols: {"q10": "<col>", "q50": "<col>", "q90": "<col>"} para modelos
    probabilísticos (p. ej. DeepAR con level=[80]).
    """
    futr_exog = futr_exog or []

    def fn(hist, origin, H):
        tail = hist.loc[hist["ds"] >= origin - pd.Timedelta(hours=L)]
        z = _with_calendar(scaler.transform(tail), futr_exog)
        kw = {}
        if futr_exog:
            ids = z["unique_id"].unique()
            fut = pd.date_range(origin, periods=H, freq="h")
            futr = pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(fut, len(ids))})
            cal = calendar_features(futr["ds"])
            for c in futr_exog:
                futr[c] = cal[c].to_numpy()
            kw["futr_df"] = futr
        pred = nf.predict(df=z, **kw).reset_index()
        out = pd.DataFrame({"unique_id": pred["unique_id"].astype(str), "ds": pred["ds"], "y_hat": pred[model_col]})
        cols = ["y_hat"]
        for k, c in (quantile_cols or {}).items():
            out[k] = pred[c].to_numpy()
            cols.append(k)
        return scaler.inverse(out, cols=tuple(cols))

    return fn
