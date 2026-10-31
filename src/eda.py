"""Exploración del panel horario (sección 2): perfiles, feriados, Navidad y escalas."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PERIOD_END, PERIOD_START, TRAIN_END, pt_holidays

DOW_NAMES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def panel_summary(panel: pd.DataFrame) -> pd.Series:
    """Cantidad de clientes y observaciones horarias."""
    n_clients = panel["unique_id"].nunique()
    per_series = panel.groupby("unique_id", observed=True).size()
    return pd.Series(
        {
            "clientes": n_clients,
            "obs_por_serie": int(per_series.iloc[0]),
            "series_con_largo_distinto": int((per_series != per_series.iloc[0]).sum()),
            "obs_panel": len(panel),
        }
    )


def scale_stats(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Consumo promedio por cliente (kWh/h) en entrenamiento: mínimo, mediana y máximo."""
    means = panel.loc[panel["ds"] <= TRAIN_END].groupby("unique_id", observed=True)["y"].mean()
    summary = means.describe(percentiles=[0.5])[["min", "50%", "max"]].rename(
        {"min": "mínimo", "50%": "mediana", "max": "máximo"}
    )
    summary["max/min"] = means.max() / max(means.min(), 1e-9)
    return summary, means


def _normalized(panel: pd.DataFrame) -> pd.DataFrame:
    """Cada cliente dividido entre su media de entrenamiento, para que los perfiles no
    queden dominados por los clientes grandes."""
    means = panel.loc[panel["ds"] <= TRAIN_END].groupby("unique_id", observed=True)["y"].mean()
    out = panel[["unique_id", "ds", "y"]].copy()
    out["y_rel"] = out["y"] / out["unique_id"].map(means).clip(lower=1e-9).to_numpy()
    return out


def plot_profiles(panel: pd.DataFrame, ax=None):
    """Perfil promedio por hora del día, una curva por día de la semana (consumo relativo)."""
    df = _normalized(panel)
    df["hour"] = df["ds"].dt.hour
    df["dow"] = df["ds"].dt.dayofweek
    prof = df.groupby(["dow", "hour"])["y_rel"].mean().unstack(0)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for d in prof.columns:
        ax.plot(prof.index, prof[d], label=DOW_NAMES[d], lw=2 if d >= 5 else 1.2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Consumo relativo (1 = media del cliente)")
    ax.set_title("Perfil promedio por hora y día de la semana")
    ax.legend(ncol=7, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return ax, prof


def daily_relative(panel: pd.DataFrame) -> pd.Series:
    """Consumo diario promedio del panel (relativo a la media de cada cliente)."""
    df = _normalized(panel)
    return df.set_index("ds")["y_rel"].groupby(level=0).mean().resample("1D").mean()


def plot_daily_with_holidays(panel: pd.DataFrame, ax=None):
    """Serie diaria del panel con los feriados de Portugal marcados."""
    daily = daily_relative(panel)
    hol = pt_holidays(range(PERIOD_START.year, PERIOD_END.year + 1))
    hol = hol[(hol >= daily.index.min()) & (hol <= daily.index.max())]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, lw=0.8, color="0.3", label="Consumo diario")
    ax.scatter(hol, daily.reindex(hol).values, color="C3", s=18, zorder=3, label="Feriado PT")
    ax.set_ylabel("Consumo relativo")
    ax.set_title("Consumo diario del panel y feriados de Portugal")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def christmas_effect(panel: pd.DataFrame) -> pd.DataFrame:
    """Consumo de 24-25 dic y 31 dic-1 ene contra un día típico del mismo día de la semana.

    La referencia es el promedio de las 4 semanas previas (mismo día de la semana) para
    aislar el efecto del feriado del efecto día de la semana.
    """
    daily = daily_relative(panel)
    rows = []
    for year in range(PERIOD_START.year, PERIOD_END.year + 1):
        for label, date in [
            ("Nochebuena", f"{year}-12-24"),
            ("Navidad", f"{year}-12-25"),
            ("Nochevieja", f"{year}-12-31"),
            ("Año Nuevo", f"{year}-01-01"),
        ]:
            d = pd.Timestamp(date)
            refs = [d - pd.Timedelta(weeks=k) for k in range(1, 5)]
            refs = [r for r in refs if r in daily.index]
            if d not in daily.index or not refs:
                continue
            base = daily.loc[refs].mean()
            rows.append(
                {"fecha": d.date(), "día": label, "dow": DOW_NAMES[d.dayofweek],
                 "relativo": daily.loc[d], "referencia": base,
                 "cambio_%": 100 * (daily.loc[d] / base - 1)}
            )
    return pd.DataFrame(rows)


def plot_christmas(panel: pd.DataFrame, ax=None):
    """Perfil horario del 15 dic al 5 ene de cada año (consumo relativo)."""
    df = _normalized(panel).set_index("ds")["y_rel"].groupby(level=0).mean()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    for year in range(PERIOD_START.year, PERIOD_END.year):
        start, end = pd.Timestamp(f"{year}-12-15"), pd.Timestamp(f"{year + 1}-01-05 23:00")
        seg = df.loc[start:end]
        if seg.empty:
            continue
        ax.plot((seg.index - start) / pd.Timedelta(days=1), seg.values, lw=0.9, label=f"{year}/{year + 1}")
    for day, name in [(9, "24 dic"), (10, "25 dic"), (16, "31 dic"), (17, "1 ene")]:
        ax.axvline(day, color="C3", ls="--", lw=0.8)
        ax.text(day + 0.1, ax.get_ylim()[1] * 0.97, name, fontsize=8, color="C3", va="top")
    ax.set_xlabel("Días desde el 15 de diciembre")
    ax.set_ylabel("Consumo relativo")
    ax.set_title("Navidad y Año Nuevo (consumo horario)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_scales(means: pd.Series, ax=None):
    """Histograma (escala log) del consumo promedio por cliente."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.log10(means.clip(lower=1e-3)), bins=30, color="0.4")
    ax.set_xlabel("log10(consumo promedio, kWh/h)")
    ax.set_ylabel("Clientes")
    ax.set_title("Escalas entre clientes")
    return ax
