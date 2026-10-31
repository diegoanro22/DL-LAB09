"""Protocolo de evaluación único (sección 4): backtesting con origen móvil diario.

Todos los modelos se evalúan con `backtest(forecast_fn, panel)`:

    forecast_fn(hist, origin, H) -> DataFrame(unique_id, ds, y_hat [, q10, q50, q90])

- `hist` trae SOLO filas con ds < origin (hasta las 23:00 del día anterior). El backtest
  lo corta él mismo y verifica que el pronóstico caiga en [origin, origin + H - 1].
- El origen avanza 24 h: 92 orígenes del 2014-10-01 al 2014-12-31.
- Los pesos se entrenan una vez, antes, con datos hasta el 2014-09-30.

Adaptadores:
- `torch_forecast_fn` envuelve un nn.Module que recibe (x) o (x, x_fut).
- `from_nixtla_cv` convierte la salida de cross_validation de StatsForecast/NeuralForecast
  al mismo formato.
- `backtest_retrain` repite la prueba reentrenando al inicio de cada mes (ventana expandible).
"""

from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.data import CALENDAR_COLS, H_DEFAULT, ROOT, TEST_END, TEST_START, calendar_features

FORECASTS_DIR = ROOT / "outputs" / "forecasts"
RESULTS_DIR = ROOT / "outputs" / "results"

ForecastFn = Callable[[pd.DataFrame, pd.Timestamp, int], pd.DataFrame]


def daily_origins(start: pd.Timestamp = TEST_START, end: pd.Timestamp = TEST_END, H: int = H_DEFAULT):
    """Orígenes a las 00:00 cuyo horizonte completo cabe antes de `end` (92 en prueba)."""
    last = pd.Timestamp(end) - pd.Timedelta(hours=H - 1)
    return pd.date_range(pd.Timestamp(start).normalize(), last.normalize(), freq="1D")


def backtest(
    forecast_fn: ForecastFn,
    panel: pd.DataFrame,
    name: str = "modelo",
    H: int = H_DEFAULT,
    start: pd.Timestamp = TEST_START,
    end: pd.Timestamp = TEST_END,
    verbose: bool = True,
) -> pd.DataFrame:
    """Corre forecast_fn en cada origen diario y concatena los pronósticos.

    Devuelve unique_id, ds, origin, modelo, y_hat (+ cuantiles) y guarda el tiempo total
    de inferencia en `out.attrs["inference_s"]`.
    """
    panel = panel.sort_values(["unique_id", "ds"], kind="stable")
    ids = set(panel["unique_id"].unique())
    origins = daily_origins(start, end, H)
    parts = []
    t_start = time.perf_counter()
    for origin in origins:
        hist = panel.loc[panel["ds"] < origin]
        assert hist["ds"].max() == origin - pd.Timedelta(hours=1)
        fc = forecast_fn(hist, origin, H)
        # Validaciones del contrato: horizonte exacto y todos los clientes
        assert set(fc["unique_id"].unique()) == ids, f"{origin}: faltan clientes"
        assert fc["ds"].min() == origin and fc["ds"].max() == origin + pd.Timedelta(hours=H - 1), \
            f"{origin}: el pronóstico no cubre exactamente [origin, origin + H - 1]"
        assert len(fc) == len(ids) * H, f"{origin}: filas duplicadas o faltantes"
        assert not fc.duplicated(["unique_id", "ds"]).any(), f"{origin}: claves duplicadas"
        assert fc.groupby("unique_id", observed=True).size().eq(H).all(), f"{origin}: horizonte por cliente incompleto"
        assert fc["ds"].isin(pd.date_range(origin, periods=H, freq="h")).all(), f"{origin}: fechas fuera de la grilla horaria"
        assert np.isfinite(fc["y_hat"]).all(), f"{origin}: pronósticos no finitos"
        for c in ("q10", "q50", "q90"):
            if c in fc:
                assert np.isfinite(fc[c]).all(), f"{origin}: cuantiles no finitos"
        if {"q10", "q50", "q90"} <= set(fc):
            assert ((fc.q10 <= fc.q50) & (fc.q50 <= fc.q90)).all(), f"{origin}: cruce de cuantiles"
        fc = fc.copy()
        fc["origin"] = origin
        parts.append(fc)
    elapsed = time.perf_counter() - t_start

    out = pd.concat(parts, ignore_index=True)
    out["modelo"] = name
    cols = ["unique_id", "ds", "origin", "modelo", "y_hat"] + [c for c in ("q10", "q50", "q90") if c in out]
    out = out[cols]
    out.attrs["inference_s"] = elapsed
    if verbose:
        print(f"[{name}] {len(origins)} orígenes, {len(out):,} filas, inferencia {elapsed:.1f} s")
    return out


# --------------------------------------------------------------------------------------
# Reentrenamiento mensual (ventana expandible)
# --------------------------------------------------------------------------------------
def backtest_retrain(
    fit_fn: Callable[[pd.DataFrame], ForecastFn],
    panel: pd.DataFrame,
    name: str = "modelo",
    H: int = H_DEFAULT,
    start: pd.Timestamp = TEST_START,
    end: pd.Timestamp = TEST_END,
) -> pd.DataFrame:
    """Reentrena al inicio de cada mes con todo lo anterior y evalúa ese mes.

    fit_fn(train_panel) recibe filas con ds < inicio de mes y devuelve un forecast_fn.
    Con la prueba oct-dic son 3 reentrenamientos (1 oct, 1 nov, 1 dic).
    """
    months = pd.date_range(pd.Timestamp(start).normalize(), end, freq="MS")
    parts, fit_s = [], 0.0
    for i, m in enumerate(months):
        m_end = months[i + 1] - pd.Timedelta(hours=1) if i + 1 < len(months) else pd.Timestamp(end)
        t = time.perf_counter()
        fn = fit_fn(panel.loc[panel["ds"] < m])
        fit_s += time.perf_counter() - t
        parts.append(backtest(fn, panel, name=name, H=H, start=m, end=m_end, verbose=False))
    out = pd.concat(parts, ignore_index=True)
    out.attrs["inference_s"] = sum(p.attrs["inference_s"] for p in parts)
    out.attrs["fit_s"] = fit_s
    print(f"[{name}] reentrenado {len(months)} veces, {out['origin'].nunique()} orígenes")
    return out


# --------------------------------------------------------------------------------------
# Adaptadores
# --------------------------------------------------------------------------------------
def tail_matrix(hist: pd.DataFrame, L: int, cols=("y",)) -> tuple[list, np.ndarray]:
    """Últimas L horas de cada serie como arreglo (S, L, len(cols)), ids en orden."""
    last = hist["ds"].max()
    tail = hist.loc[hist["ds"] > last - pd.Timedelta(hours=L)]
    tail = tail.sort_values(["unique_id", "ds"], kind="stable")
    ids = tail["unique_id"].drop_duplicates().tolist()
    arr = tail[list(cols)].to_numpy(dtype=np.float32).reshape(len(ids), L, len(cols))
    return ids, arr


def frame_from_matrix(ids, origin: pd.Timestamp, yhat: np.ndarray, extra: dict | None = None) -> pd.DataFrame:
    """(S, H) -> DataFrame largo unique_id, ds, y_hat (+ columnas extra (S, H))."""
    S, H = yhat.shape
    ds = pd.date_range(origin, periods=H, freq="h")
    df = pd.DataFrame(
        {"unique_id": np.repeat(np.asarray(ids, dtype=object), H), "ds": np.tile(ds, S), "y_hat": yhat.ravel()}
    )
    for k, v in (extra or {}).items():
        df[k] = np.asarray(v).ravel()
    return df


def torch_forecast_fn(
    model: torch.nn.Module,
    scaler,
    L: int,
    H: int = H_DEFAULT,
    with_future: bool = False,
    use_calendar: bool = True,
    device: str | torch.device = "cpu",
) -> ForecastFn:
    """forecast_fn para un modelo PyTorch entrenado sobre WindowDataset.

    Arma la entrada igual que WindowDataset: y normalizada con el scaler (estadísticos de
    entrenamiento) + calendario de las L horas previas; x_fut = calendario del horizonte,
    calculado solo a partir de las fechas. La salida (S, H) se desnormaliza a kWh.
    """
    cols = ["y"] + (CALENDAR_COLS if use_calendar else [])

    @torch.no_grad()
    def fn(hist: pd.DataFrame, origin: pd.Timestamp, H_: int) -> pd.DataFrame:
        model.eval()
        z = scaler.transform(hist.loc[hist["ds"] > origin - pd.Timedelta(hours=L + 1)])
        ids, x = tail_matrix(z, L, cols)
        xt = torch.from_numpy(x).to(device)
        if with_future:
            fut = calendar_features(pd.date_range(origin, periods=H_, freq="h")).to_numpy(np.float32)
            xf = torch.from_numpy(np.broadcast_to(fut, (len(ids),) + fut.shape).copy()).to(device)
            pred = model(xt, xf)
        else:
            pred = model(xt)
        pred = pred.float().cpu().numpy()
        st = scaler.stats.loc[ids].to_numpy()
        pred = pred * st[:, [1]] + st[:, [0]]
        return frame_from_matrix(ids, origin, pred)

    return fn


def from_nixtla_cv(cv: pd.DataFrame, model_col: str, name: str | None = None,
                   quantile_cols: dict | None = None) -> pd.DataFrame:
    """Salida de cross_validation (unique_id, ds, cutoff, <modelo>...) -> formato del backtest.

    En Nixtla, `cutoff` es la última hora observada, así que origin = cutoff + 1h.
    quantile_cols: p. ej. {"q10": "DeepAR-lo-80", "q90": "DeepAR-hi-80", "q50": "DeepAR-median"}.
    """
    out = pd.DataFrame(
        {
            "unique_id": cv["unique_id"].astype(str).to_numpy(),
            "ds": pd.to_datetime(cv["ds"]).to_numpy(),
            "origin": (pd.to_datetime(cv["cutoff"]) + pd.Timedelta(hours=1)).to_numpy(),
            "modelo": name or model_col,
            "y_hat": cv[model_col].to_numpy(),
        }
    )
    for k, c in (quantile_cols or {}).items():
        out[k] = cv[c].to_numpy()
    return out


# --------------------------------------------------------------------------------------
# Guardado
# --------------------------------------------------------------------------------------
def save_forecasts(fc: pd.DataFrame, name: str) -> None:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(FORECASTS_DIR / f"{name}.parquet", index=False)


def save_results(name: str, info: dict) -> None:
    """info: L, estrategia, covariables, params, tiempos, hardware, métricas, etc."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def conv(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        return v

    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps({k: conv(v) for k, v in info.items()}, indent=2, ensure_ascii=False, default=str)
    )


def hardware() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    import platform
    from pathlib import Path

    try:
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    return f"CPU ({cpu})"


def finish(name: str, fc: pd.DataFrame, panel: pd.DataFrame, scale: pd.Series | None = None,
           save: bool = True, **info) -> tuple[pd.Series, dict]:
    """Evalúa un pronóstico de prueba, lo guarda y devuelve (fila_resumen, evaluación).

    info: familia, L, estrategia, covariables, params_total, params_entrenables, train_s,
    hardware, etc. El tiempo de inferencia se toma de fc.attrs si no se pasa.
    """
    from src import metrics as M

    ev = M.evaluate(fc, panel, scale)
    row = M.summary_row(name, ev)
    info = {"modelo": name, "hardware": hardware(),
            "inference_s": fc.attrs.get("inference_s"), **info, **row.to_dict()}
    if save:
        save_forecasts(fc, name)
        save_results(name, info)
    return pd.Series(info, name=name), ev
