"""Dataset de ventanas deslizantes para todas las series del panel (sección 2).

Cada ventana se identifica por (serie, t0), donde t0 es la PRIMERA hora del horizonte
(el "origen" del pronóstico). Para esa ventana:

    x      : (L, F)  valores en [t0 - L, t0 - 1]   -> y normalizada + calendario de esas horas
    x_fut  : (H, C)  calendario en [t0, t0 + H - 1] -> solo depende de la fecha (conocido a futuro)
    y      : (H,)    objetivo normalizado en [t0, t0 + H - 1]

La última hora de entrada es t0 - 1h: con t0 = 00:00, la red ve hasta las 23:00 del día
anterior, igual que el pronóstico day-ahead del enunciado.

Una ventana pertenece a un split si TODAS sus horas objetivo caen dentro de ese split.
La entrada puede venir de antes (por ejemplo, validación usa la última semana de
entrenamiento como contexto), lo cual no es fuga porque es historia conocida.

Uso:
    scaler = SeriesScaler().fit(panel)
    ds_tr = WindowDataset(scaler.transform(panel), split="train")
    for x, y in ds_tr.batches(512, shuffle=True, device="cuda"):  # rápido, vectorizado
        ...
    # o con DataLoader estándar:
    DataLoader(ds_tr, batch_size=512, shuffle=True)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import (
    CALENDAR_COLS, H_DEFAULT, L_DEFAULT, PERIOD_START, TEST_END, TEST_START, TRAIN_END,
    VAL_END, VAL_START,
)

SPLITS = {
    "train": (PERIOD_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}


class WindowDataset(Dataset):
    """Ventanas (x, y) de todas las series, con índice perezoso (no copia ventanas).

    Args:
        panel: formato largo con unique_id, ds, y (YA normalizada) y CALENDAR_COLS.
            Todas las series deben tener el mismo índice horario (lo garantiza build_panel).
        L, H: largo de la entrada y del horizonte.
        split: "train", "val" o "test"; define el rango de horas objetivo.
        stride: separación en horas entre orígenes consecutivos (1 = todas las horas).
        origin_hour: si se da (p. ej. 0), solo orígenes a esa hora del día
            (92 orígenes diarios en prueba con origin_hour=0, stride=1).
        target_start / target_end: sobrescriben el rango del split (reentrenamiento
            mensual, ventana expandible).
        history_start: descarta ventanas cuya entrada empieza antes de esta fecha
            (para la iteración de "meses de historia" de la sección 5.2).
        with_future: si True, cada muestra devuelve (x, x_fut, y) para modelos que
            usan el calendario del horizonte (seq2seq, TFT); si no, (x, y).
        use_calendar: si False, x solo trae la y (F = 1).
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        L: int = L_DEFAULT,
        H: int = H_DEFAULT,
        split: str | None = "train",
        stride: int = 1,
        origin_hour: int | None = None,
        target_start: pd.Timestamp | None = None,
        target_end: pd.Timestamp | None = None,
        history_start: pd.Timestamp | None = None,
        with_future: bool = False,
        use_calendar: bool = True,
    ):
        self.L, self.H = L, H
        self.with_future = with_future
        self.use_calendar = use_calendar

        panel = panel.sort_values(["unique_id", "ds"], kind="stable")
        self.ids = panel["unique_id"].drop_duplicates().tolist()
        n_series = len(self.ids)
        self.times = pd.DatetimeIndex(panel["ds"].iloc[: len(panel) // n_series])
        T = len(self.times)
        assert len(panel) == n_series * T, "todas las series deben tener el mismo largo"
        assert (self.times.to_series().diff().dropna() == pd.Timedelta(hours=1)).all(), \
            "el índice debe ser horario y sin huecos"

        # Tensor (S, T, 1 + C): columna 0 = y, columnas 1.. = calendario
        cols = ["y"] + CALENDAR_COLS
        self.data = torch.from_numpy(
            panel[cols].to_numpy(dtype=np.float32).reshape(n_series, T, len(cols))
        )

        # Rango de horas objetivo
        lo, hi = SPLITS[split] if split else (self.times[0], self.times[-1])
        lo = pd.Timestamp(target_start) if target_start is not None else lo
        hi = pd.Timestamp(target_end) if target_end is not None else hi

        # Orígenes válidos t0 (posiciones en self.times)
        t0 = np.arange(T)
        ok = (t0 - L >= 0) & (t0 + H - 1 < T)
        ok &= (self.times[np.clip(t0, 0, T - 1)] >= lo)
        ok &= (self.times[np.clip(t0 + H - 1, 0, T - 1)] <= hi)
        if history_start is not None:
            ok &= self.times[np.clip(t0 - L, 0, T - 1)] >= pd.Timestamp(history_start)
        if origin_hour is not None:
            ok &= self.times[np.clip(t0, 0, T - 1)].hour == origin_hour
        origins = t0[ok][::stride]

        # Índice perezoso: todas las combinaciones (serie, origen)
        self.series_idx = np.repeat(np.arange(n_series), len(origins))
        self.origin_idx = np.tile(origins, n_series)
        self._ar_in = torch.arange(-L, 0)
        self._ar_out = torch.arange(H)

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self.series_idx)

    def __getitem__(self, i: int):
        out = self._gather(np.asarray([i]))
        return tuple(t[0] for t in out)

    def __getitems__(self, idx: list[int]):
        """Lectura vectorizada que DataLoader usa automáticamente con batches."""
        out = self._gather(np.asarray(idx))
        return [tuple(t[j] for t in out) for j in range(len(idx))]

    # ------------------------------------------------------------------ utilidades
    @property
    def n_features(self) -> int:
        return self.data.shape[2] if self.use_calendar else 1

    @property
    def n_future_features(self) -> int:
        return self.data.shape[2] - 1

    def _gather(self, idx: np.ndarray, data: torch.Tensor | None = None):
        data = self.data if data is None else data
        dev = data.device
        s = torch.as_tensor(self.series_idx[idx], device=dev)[:, None]
        t0 = torch.as_tensor(self.origin_idx[idx], device=dev)[:, None]
        t_in = t0 + self._ar_in.to(dev)       # (B, L)  horas t0-L .. t0-1
        t_out = t0 + self._ar_out.to(dev)     # (B, H)  horas t0 .. t0+H-1
        x = data[s, t_in]                     # (B, L, 1+C)
        if not self.use_calendar:
            x = x[..., :1]
        y = data[s, t_out, 0]                 # (B, H)
        if self.with_future:
            x_fut = data[s, t_out, 1:]        # (B, H, C) solo calendario
            return x, x_fut, y
        return x, y

    def batches(
        self,
        batch_size: int = 512,
        shuffle: bool = False,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Iterador rápido de batches: copia el panel al device una vez y arma las
        ventanas con indexado avanzado (sin bucles de Python por muestra)."""
        data = self.data.to(device)
        n = len(self)
        if shuffle:
            g = np.random.default_rng(seed)
            order = g.permutation(n)
        else:
            order = np.arange(n)
        for start in range(0, n, batch_size):
            yield self._gather(order[start : start + batch_size], data)

    def window_info(self, idx=None) -> pd.DataFrame:
        """unique_id, origen (t0) y última hora de entrada de cada ventana."""
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        t0 = self.times[self.origin_idx[idx]]
        return pd.DataFrame(
            {
                "unique_id": np.asarray(self.ids, dtype=object)[self.series_idx[idx]],
                "origin": t0,
                "last_input": t0 - pd.Timedelta(hours=1),
            }
        )

    def target_times(self, i: int) -> pd.DatetimeIndex:
        """Las H horas objetivo de la ventana i (para reconstruir y_hat con su ds)."""
        t0 = self.origin_idx[i]
        return self.times[t0 : t0 + self.H]


def count_windows(panel: pd.DataFrame, L: int = L_DEFAULT, H: int = H_DEFAULT, stride: int = 1) -> pd.DataFrame:
    """Cantidad de ventanas por split (por serie y en todo el panel)."""
    rows = []
    for split in ("train", "val", "test"):
        ds = WindowDataset(panel, L=L, H=H, split=split, stride=stride)
        n_series = len(ds.ids)
        rows.append({"split": split, "por_serie": len(ds) // n_series, "panel": len(ds)})
    return pd.DataFrame(rows).set_index("split")
