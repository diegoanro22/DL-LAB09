"""Loop de entrenamiento genérico para los modelos propios en PyTorch (sección 5.2).

`fit()` registra, por iteración, todo lo que pide el lab:
- configuración (L, H, meses de historia y lo que se pase en `config`);
- pérdida de entrenamiento y validación por epoch;
- MAE (kWh) y MASE en validación al final de cada epoch;
- parámetros totales y entrenables;
- tiempo por epoch y total, memoria pico de GPU.

Early stopping SOLO con validación: `fit` rechaza un val_ds cuyas horas objetivo lleguen
al periodo de prueba (auditoría de fugas).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import ROOT, TEST_START
from src.dataset import WindowDataset

RUNS_DIR = ROOT / "outputs" / "runs"


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_entrenables": trainable}


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame
    info: dict = field(default_factory=dict)


def _forward(model, batch, with_future):
    if with_future:
        x, xf, y = batch
        if model.training and hasattr(model, "forward_train"):
            return model.forward_train(x, xf, y), y
        return model(x, xf), y
    x, y = batch
    return model(x), y


@torch.no_grad()
def evaluate_windows(model, ds: WindowDataset, scaler, mase_scale: pd.Series, loss_fn,
                     batch_size: int = 2048, device="cpu") -> dict:
    """Pérdida (escala normalizada), MAE en kWh y MASE sobre las ventanas de ds."""
    model.eval()
    data = ds.data.to(device)
    stats = torch.tensor(scaler.stats.loc[ds.ids].to_numpy(), dtype=torch.float32, device=device)
    scale = torch.tensor(mase_scale.reindex(ds.ids).to_numpy(), dtype=torch.float32, device=device)
    S = len(ds.ids)
    abs_err_sum = torch.zeros(S, device=device)
    count = torch.zeros(S, device=device)
    loss_sum, n = 0.0, 0
    for start in range(0, len(ds), batch_size):
        idx = np.arange(start, min(start + batch_size, len(ds)))
        batch = ds._gather(idx, data)
        pred, y = _forward(model, batch, ds.with_future)
        loss_sum += loss_fn(pred, y).item() * len(y)
        n += len(y)
        s = torch.as_tensor(ds.series_idx[idx], device=device)
        mu, sigma = stats[s, 0:1], stats[s, 1:2]
        ae = ((pred - y) * sigma).abs()          # error en kWh
        abs_err_sum.index_add_(0, s, ae.sum(1))
        count.index_add_(0, s, torch.full_like(s, ae.shape[1], dtype=torch.float32))
    mae_series = abs_err_sum / count
    return {
        "loss": loss_sum / n,
        "MAE": (abs_err_sum.sum() / count.sum()).item(),
        "MASE": (mae_series / scale).mean().item(),
    }


def fit(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    scaler,
    mase_scale: pd.Series,
    name: str,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 3,
    weight_decay: float = 0.0,
    loss: str = "mse",
    max_batches_per_epoch: int | None = None,
    forecast_start: pd.Timestamp = TEST_START,
    device: str | torch.device | None = None,
    seed: int = 42,
    config: dict | None = None,
    verbose: bool = True,
) -> FitResult:
    """Entrena con Adam y early stopping sobre la pérdida de validación.

    max_batches_per_epoch: recorta cada epoch a una muestra aleatoria de batches (el
        panel completo tiene 3.25 M ventanas; útil para iteraciones rápidas).
    forecast_start: primera hora que se va a pronosticar; ninguna hora objetivo de
        validación puede ser >= esa fecha.
    """
    # --- Auditoría: early stopping nunca con datos de prueba ---------------------------
    val_last_target = val_ds.times[val_ds.origin_idx.max() + val_ds.H - 1]
    assert val_last_target < pd.Timestamp(forecast_start), (
        f"validación llega a {val_last_target}, que es >= {forecast_start}: "
        "early stopping estaría usando el periodo de prueba"
    )
    train_last_target = train_ds.times[train_ds.origin_idx.max() + train_ds.H - 1]
    val_first_target = val_ds.times[val_ds.origin_idx.min()]
    assert train_last_target < val_first_target, "train y val se solapan en el tiempo"

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}[loss]
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows, best, best_state, bad = [], float("inf"), None, 0
    t_total = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tr_sum, tr_n = 0.0, 0
        for b, batch in enumerate(train_ds.batches(batch_size, shuffle=True, device=device, seed=seed + epoch)):
            if max_batches_per_epoch and b >= max_batches_per_epoch:
                break
            pred, y = _forward(model, batch, train_ds.with_future)
            l = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += l.item() * len(y)
            tr_n += len(y)
        if device != "cpu":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter() - t0
        val = evaluate_windows(model, val_ds, scaler, mase_scale, loss_fn, device=device)
        rows.append({"epoch": epoch, "train_loss": tr_sum / tr_n, "val_loss": val["loss"],
                     "val_MAE": val["MAE"], "val_MASE": val["MASE"], "epoch_s": t_epoch})
        if verbose:
            r = rows[-1]
            print(f"[{name}] epoch {epoch:2d}  train {r['train_loss']:.4f}  val {r['val_loss']:.4f}  "
                  f"MAE {r['val_MAE']:.2f} kWh  MASE {r['val_MASE']:.3f}  ({t_epoch:.1f}s)")
        if val["loss"] < best - 1e-5:
            best, bad = val["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[{name}] early stopping en epoch {epoch}")
                break
    total_s = time.perf_counter() - t_total
    model.load_state_dict(best_state)

    history = pd.DataFrame(rows)
    best_row = history.loc[history["val_loss"].idxmin()]
    info = {
        "nombre": name,
        "L": train_ds.L,
        "H": train_ds.H,
        "historia_desde": str(train_ds.times[train_ds.origin_idx.min() - train_ds.L]),
        "ventanas_train": len(train_ds),
        "ventanas_val": len(val_ds),
        "usa_calendario": train_ds.use_calendar,
        "usa_futuro": train_ds.with_future,
        "lr": lr, "batch_size": batch_size, "epochs_max": epochs, "patience": patience,
        "loss": loss, "seed": seed,
        **count_params(model),
        "epochs_corridos": len(history),
        "mejor_epoch": int(best_row["epoch"]),
        "val_MAE": float(best_row["val_MAE"]),
        "val_MASE": float(best_row["val_MASE"]),
        "train_s_total": total_s,
        "train_s_por_epoch": float(history["epoch_s"].mean()),
        "gpu_mem_pico_MB": (torch.cuda.max_memory_allocated() / 2**20) if device != "cpu" else 0.0,
        **(config or {}),
    }
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / "history.csv", index=False)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    return FitResult(model=model, history=history, info=info)


def refit(
    model: nn.Module,
    full_ds: WindowDataset,
    epochs: int,
    name: str,
    lr: float = 1e-3,
    batch_size: int = 512,
    weight_decay: float = 0.0,
    loss: str = "mse",
    max_batches_per_epoch: int | None = None,
    forecast_start: pd.Timestamp = TEST_START,
    device: str | torch.device | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Entrena un modelo nuevo durante `epochs` fijos (los que eligió el early stopping)
    sobre entrenamiento + validación (datos hasta 2014-09-30), sin mirar la prueba.

    Es el paso "los pesos se entrenan una vez con datos hasta el 2014-09-30" del protocolo.
    """
    last_target = full_ds.times[full_ds.origin_idx.max() + full_ds.H - 1]
    assert last_target < pd.Timestamp(forecast_start), "el reentrenamiento toca la prueba"

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss()}[loss]
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_total = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for b, batch in enumerate(full_ds.batches(batch_size, shuffle=True, device=device, seed=seed + epoch)):
            if max_batches_per_epoch and b >= max_batches_per_epoch:
                break
            pred, y = _forward(model, batch, full_ds.with_future)
            l = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += l.item() * len(y)
            tr_n += len(y)
        if verbose:
            print(f"[{name} refit] epoch {epoch:2d}  train {tr_sum / tr_n:.4f}")
    total_s = time.perf_counter() - t_total
    info = {
        "refit_epochs": epochs,
        "refit_s": total_s,
        "refit_ventanas": len(full_ds),
        "gpu_mem_pico_MB_refit": (torch.cuda.max_memory_allocated() / 2**20) if device != "cpu" else 0.0,
        **count_params(model),
    }
    return model, info
