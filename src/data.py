"""Carga y preparación del panel horario (sección 2 del lab).

Pipeline: LD2011_2014.txt (kW cada 15 min) -> kWh por hora -> filtro de clientes completos
2012-2014 -> corrección de cambio de horario -> subconjunto fijo -> formato largo
(unique_id, ds, y) + covariables de calendario.

Uso rápido:
    from src.data import build_panel
    panel = build_panel()          # lee o genera data/hourly.parquet
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_TXT = DATA_DIR / "LD2011_2014.txt"
RAW_ZIP = DATA_DIR / "electricity.zip"
PANEL_PARQUET = DATA_DIR / "hourly.parquet"
CLIENTS_JSON = DATA_DIR / "clients.json"

RAW_PARQUET = DATA_DIR / "hf_mirror" / "LD2011_2014.parquet"

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
HF_URL = ("https://huggingface.co/datasets/sxj1024/electricity_load_2011_2014/"
          "resolve/main/data/train-00000-of-00001.parquet")

# Splits fijos del enunciado (inclusive)
PERIOD_START = pd.Timestamp("2012-01-01 00:00")
PERIOD_END = pd.Timestamp("2014-12-31 23:00")
TRAIN_END = pd.Timestamp("2014-06-30 23:00")
VAL_START, VAL_END = pd.Timestamp("2014-07-01 00:00"), pd.Timestamp("2014-09-30 23:00")
TEST_START, TEST_END = pd.Timestamp("2014-10-01 00:00"), pd.Timestamp("2014-12-31 23:00")

L_DEFAULT, H_DEFAULT = 168, 24
N_CLIENTS, SEED = 150, 42

CALENDAR_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_holiday",
]


# --------------------------------------------------------------------------------------
# 1. Datos crudos
# --------------------------------------------------------------------------------------
def download_raw() -> Path:
    """Devuelve la ruta al crudo, descargándolo si hace falta.

    Prioridad: LD2011_2014.txt de UCI si ya está extraído; si no, la copia en parquet de
    Hugging Face (mismo contenido, verificado contra UCI; el servidor de UCI descarga a
    ~3 KB/s y el zip pesa ~250 MB). UCI queda como último recurso.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_TXT.exists():
        return RAW_TXT
    if RAW_ZIP.exists():
        zipfile.ZipFile(RAW_ZIP).extract("LD2011_2014.txt", DATA_DIR)
        return RAW_TXT
    if not RAW_PARQUET.exists():
        RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HF_URL, RAW_PARQUET)
        except Exception:
            RAW_PARQUET.unlink(missing_ok=True)
            urllib.request.urlretrieve(URL, RAW_ZIP)
            zipfile.ZipFile(RAW_ZIP).extract("LD2011_2014.txt", DATA_DIR)
            return RAW_TXT
    return RAW_PARQUET


def load_raw(path: Path = RAW_TXT) -> pd.DataFrame:
    """Matriz cruda: índice de timestamps de 15 min, una columna por cliente (kW)."""
    path = Path(path)
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
        raw = raw.set_index(pd.to_datetime(raw.pop("datetime")))
    else:
        raw = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
    raw.index.name = None
    assert raw.shape == (140_256, 370), f"crudo con forma inesperada: {raw.shape}"
    return raw.astype("float32")


# --------------------------------------------------------------------------------------
# 2. Agregación horaria y cambio de horario
# --------------------------------------------------------------------------------------
def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """kW cada 15 min -> kWh por hora.

    Cada registro marca el FINAL de su intervalo (00:15 cubre 00:00-00:15), así que la hora
    00:00 agrupa 00:15, 00:30, 00:45 y 01:00 (closed='right', label='left').
    Un registro de 15 min en kW equivale a kW * 0.25 kWh.
    """
    hourly = raw.resample("1h", closed="right", label="left").sum() * 0.25
    return hourly.astype("float32")


def dst_days(years) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Último domingo de marzo y de octubre (cambio de horario en Portugal/UE)."""
    march, october = [], []
    for y in years:
        for month, out in ((3, march), (10, october)):
            last = pd.Timestamp(year=y, month=month, day=31)
            out.append(last - pd.Timedelta(days=(last.dayofweek + 1) % 7))
    return march, october


def fix_dst(hourly: pd.DataFrame) -> pd.DataFrame:
    """Corrige los dos días de cambio de horario de cada año.

    Según la documentación del dataset, entre la 1:00 y las 2:00 de esos días:
    - marzo (día de 23 h): los valores son cero -> se interpolan linealmente entre 00:00 y 02:00.
    - octubre (día de 25 h): se registran dos horas sumadas -> se divide entre 2.
    Así toda serie queda con 24 valores por día y sin picos ni huecos artificiales.
    """
    out = hourly.copy()
    years = sorted(set(out.index.year))
    march, october = dst_days(years)
    for day in march:
        t = day + pd.Timedelta(hours=1)
        if t in out.index:
            out.loc[t] = np.nan
    out = out.interpolate(method="time", limit=1, limit_area="inside")
    for day in october:
        t = day + pd.Timedelta(hours=1)
        if t in out.index:
            out.loc[t] = out.loc[t] / 2
    return out


# --------------------------------------------------------------------------------------
# 3. Filtro y subconjunto de clientes
# --------------------------------------------------------------------------------------
def complete_clients(hourly: pd.DataFrame, max_zero_day_run: int = 2) -> list[str]:
    """Clientes con datos completos en 2012-01-01 .. 2014-12-31.

    En el dataset, los clientes que empezaron a medir tarde (o que se dieron de baja)
    aparecen con ceros. Se exige, dentro del periodo:
    - que el primer día (2012-01-01) tenga consumo > 0 (ya medía cuando arranca);
    - ninguna racha de días completos en cero más larga que `max_zero_day_run`
      (descarta altas tardías, bajas y cortes largos).
    """
    period = hourly.loc[PERIOD_START:PERIOD_END]
    daily_zero = (period.resample("1D").sum() == 0)
    keep = []
    for c in daily_zero.columns:
        if daily_zero[c].iloc[0]:
            continue
        z = daily_zero[c].to_numpy()
        # racha más larga de días en cero
        run = best = 0
        for v in z:
            run = run + 1 if v else 0
            best = max(best, run)
        if best <= max_zero_day_run:
            keep.append(c)
    return keep


def select_subset(clients: list[str], n: int = N_CLIENTS, seed: int = SEED) -> list[str]:
    """Subconjunto fijo de clientes (mismo para todos los modelos)."""
    if n is None or n >= len(clients):
        return sorted(clients)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(clients), size=n, replace=False).tolist())


# --------------------------------------------------------------------------------------
# 4. Covariables de calendario (conocidas a futuro)
# --------------------------------------------------------------------------------------
def pt_holidays(years) -> pd.DatetimeIndex:
    """Feriados nacionales de Portugal (la librería ya refleja los suspendidos en 2013-2015)."""
    return pd.DatetimeIndex(sorted(holidays.Portugal(years=list(years)).keys()))


def calendar_features(ds: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Covariables de calendario para cada timestamp.

    Solo dependen de la fecha/hora, así que se pueden calcular para el horizonte futuro
    sin fuga. Hora, día de la semana y mes con seno/coseno (cíclicos: las 23 h quedan
    cerca de las 0 h); feriado como binaria.
    """
    ds = pd.DatetimeIndex(ds)
    hol = pt_holidays(range(ds.year.min(), ds.year.max() + 1))
    two_pi = 2 * np.pi
    feats = pd.DataFrame(
        {
            "hour_sin": np.sin(two_pi * ds.hour / 24),
            "hour_cos": np.cos(two_pi * ds.hour / 24),
            "dow_sin": np.sin(two_pi * ds.dayofweek / 7),
            "dow_cos": np.cos(two_pi * ds.dayofweek / 7),
            "month_sin": np.sin(two_pi * (ds.month - 1) / 12),
            "month_cos": np.cos(two_pi * (ds.month - 1) / 12),
            "is_holiday": ds.normalize().isin(hol).astype("float32"),
        },
        index=ds,
    )
    return feats.astype("float32")


# --------------------------------------------------------------------------------------
# 5. Panel largo y normalización
# --------------------------------------------------------------------------------------
def to_long(hourly: pd.DataFrame) -> pd.DataFrame:
    """Matriz ancha -> formato largo (unique_id, ds, y) + covariables, ordenado."""
    long = (
        hourly.rename_axis("ds")
        .reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .sort_values(["unique_id", "ds"], kind="stable")
        .reset_index(drop=True)
    )
    cal = calendar_features(long["ds"]).reset_index(drop=True)
    long = pd.concat([long, cal], axis=1)
    long["unique_id"] = long["unique_id"].astype("string")
    long["y"] = long["y"].astype("float32")
    return long


def _cache_matches(n_clients: int | None, seed: int) -> bool:
    """True si hourly.parquet existe y corresponde a (n_clients, seed)."""
    if not (PANEL_PARQUET.exists() and CLIENTS_JSON.exists()):
        return False
    meta = json.loads(CLIENTS_JSON.read_text())
    return meta.get("seed") == seed and meta.get("n_requested", "missing") == n_clients


def build_panel(
    n_clients: int | None = N_CLIENTS,
    seed: int = SEED,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pipeline completo. Reusa el parquet solo si fue generado con la misma semilla y
    cantidad de clientes (según clients.json); si no, lo regenera desde el crudo."""
    if not force and _cache_matches(n_clients, seed):
        return pd.read_parquet(PANEL_PARQUET)

    raw = load_raw(download_raw())
    hourly = fix_dst(to_hourly(raw))
    complete = complete_clients(hourly)
    clients = select_subset(complete, n_clients, seed)
    hourly = hourly.loc[PERIOD_START:PERIOD_END, clients]
    assert not hourly.isna().any().any(), "quedaron NaN después de la limpieza"

    panel = to_long(hourly)
    panel.to_parquet(PANEL_PARQUET, index=False)
    CLIENTS_JSON.write_text(
        json.dumps(
            {"seed": seed, "n_requested": n_clients,
             "n_complete": len(complete), "n_selected": len(clients),
             "complete": complete, "selected": clients},
            indent=2,
        )
    )
    if verbose:
        n_obs = hourly.shape[0]
        print(f"Clientes en el crudo: {raw.shape[1]}")
        print(f"Clientes completos 2012-2014: {len(complete)}")
        print(f"Clientes usados (seed={seed}): {len(clients)}")
        print(f"Observaciones horarias por serie: {n_obs:,}")
        print(f"Observaciones del panel: {n_obs * len(clients):,}")
    return panel


def split_masks(ds: pd.Series) -> dict[str, pd.Series]:
    """Máscaras booleanas train/val/test sobre la columna ds."""
    return {
        "train": ds <= TRAIN_END,
        "val": (ds >= VAL_START) & (ds <= VAL_END),
        "test": (ds >= TEST_START) & (ds <= TEST_END),
    }


@dataclass
class SeriesScaler:
    """z-score por serie con estadísticos SOLO del bloque de entrenamiento.

    Evita la fuga de "escalar con toda la serie": media y desviación se calculan con
    ds <= fit_end y luego se aplican igual a validación y prueba.
    """

    fit_end: pd.Timestamp = TRAIN_END
    eps: float = 1e-6
    stats: pd.DataFrame | None = None

    def fit(self, panel: pd.DataFrame) -> "SeriesScaler":
        train = panel.loc[panel["ds"] <= self.fit_end]
        self.stats = train.groupby("unique_id", observed=True)["y"].agg(mu="mean", sigma="std")
        self.stats["sigma"] = self.stats["sigma"].clip(lower=self.eps)
        return self

    def transform(self, panel: pd.DataFrame, col: str = "y") -> pd.DataFrame:
        s = self.stats.loc[panel["unique_id"]].to_numpy()
        out = panel.copy()
        out[col] = ((panel[col].to_numpy() - s[:, 0]) / s[:, 1]).astype("float32")
        return out

    def inverse(self, df: pd.DataFrame, cols=("y_hat",)) -> pd.DataFrame:
        s = self.stats.loc[df["unique_id"]].to_numpy()
        out = df.copy()
        for c in cols:
            out[c] = df[c].to_numpy() * s[:, 1] + s[:, 0]
        return out


if __name__ == "__main__":
    build_panel(force=True)
