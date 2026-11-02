"""Ejecución reanudable de la parte B: python -m src.run_cisco --stage all.

Los límites CPU son explícitos y no reducen clientes ni orígenes de evaluación.
La selección se hace sobre julio-septiembre; octubre-diciembre queda reservado.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import backtest as B
from src import data as D
from src import metrics as M
from src import train as T
from src.dataset import WindowDataset
from src.models.rnn import LSTMAutoregressive, LSTMMultiOutput, Seq2Seq

RUNS = D.ROOT / "outputs" / "runs"
CLASSES = {
    "LSTM_AR": LSTMAutoregressive,
    "LSTM_Multi": LSTMMultiOutput,
    "Seq2Seq": Seq2Seq,
}
CPU = not torch.cuda.is_available()
DEVICE = "cpu" if CPU else "cuda"
EPOCHS, BATCHES, BATCH_SIZE = (6, 50, 128) if CPU else (15, 1000, 512)
# En GPU se usa el mismo presupuesto que los modelos de la parte A.
HIDDENS = (24, 48) if CPU else (64, 128)
NF_STEPS, NF_CHECK, NF_PATIENCE = (250, 50, 3) if CPU else (1500, 100, 5)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def load_context():
    panel = D.build_panel()
    expected = json.loads(D.CLIENTS_JSON.read_text())["selected"]
    assert sorted(panel.unique_id.unique()) == expected
    scaler = D.SeriesScaler().fit(panel)
    scale = M.mase_scale(panel)
    assert (scale > 0).all()
    return panel, scaler, scale


def datasets(panel, scaler, L=168, months=30, H_train=24, end=D.TRAIN_END):
    # Excluir físicamente las etiquetas de prueba del tensor de entrenamiento.
    z = scaler.transform(panel.loc[panel.ds <= D.VAL_END])
    start = (
        pd.Timestamp(end).normalize()
        + pd.Timedelta(days=1)
        - pd.DateOffset(months=months)
    )
    tr = WindowDataset(z, L=L, H=H_train, split="train", history_start=start)
    va = WindowDataset(z, L=L, H=24, split="val", origin_hour=0, with_future=True)
    tr.with_future = True
    return tr, va


def select_rnn(
    panel,
    scaler,
    scale,
    family,
    L=168,
    months=30,
    hidden=32,
    seed=42,
    tag=None,
    epochs=EPOCHS,
    batches=BATCHES,
):
    name = tag or f"{family}_h{hidden}_L{L}_m{months}_s{seed}"
    tr, va = datasets(
        panel, scaler, L, months, H_train=1 if family == "LSTM_AR" else 24
    )
    T.set_seed(seed)  # Inicialización reproducible, antes de construir el módulo.
    model = CLASSES[family](hidden=hidden)
    res = T.fit(
        model,
        tr,
        va,
        scaler,
        scale,
        name=name,
        epochs=epochs,
        patience=2,
        batch_size=BATCH_SIZE,
        max_batches_per_epoch=batches,
        seed=seed,
        config={
            "familia": family,
            "hidden": hidden,
            "meses_historia": months,
            "max_batches_per_epoch": batches,
            "hardware": B.hardware(),
        },
    )
    out = RUNS / name
    torch.save(res.model.state_dict(), out / "selected.pt")
    del tr, va
    gc.collect()
    return res


def refit_rnn(panel, scaler, family, info, end=D.VAL_END, name=None):
    L, months, seed = info["L"], info["meses_historia"], info["seed"]
    z = scaler.transform(panel.loc[panel.ds <= end])
    start = (
        pd.Timestamp(end).normalize()
        + pd.Timedelta(days=1)
        - pd.DateOffset(months=months)
    )
    ds = WindowDataset(
        z,
        L=L,
        H=1 if family == "LSTM_AR" else 24,
        split=None,
        target_end=end,
        history_start=start,
        with_future=True,
    )
    T.set_seed(seed)
    model = CLASSES[family](hidden=info["hidden"])
    model, rinfo = T.refit(
        model,
        ds,
        epochs=info["mejor_epoch"],
        name=name or family,
        batch_size=BATCH_SIZE,
        max_batches_per_epoch=info["max_batches_per_epoch"],
        forecast_start=end + pd.Timedelta(hours=1),
        seed=seed,
    )
    del ds
    gc.collect()
    return model, rinfo


def recurrentes(panel, scaler, scale):
    for family in CLASSES:
        if (B.RESULTS_DIR / f"{family}.json").exists():
            continue
        candidates = [
            select_rnn(panel, scaler, scale, family, hidden=h) for h in HIDDENS
        ]
        best = min(candidates, key=lambda r: r.info["val_MASE"])
        info = best.info
        del candidates, best
        gc.collect()
        model, ri = refit_rnn(panel, scaler, family, info)
        out = RUNS / family
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "final.pt")
        write_json(out / "selection.json", info)
        fn = B.torch_forecast_fn(
            model, scaler, info["L"], with_future=True, device=DEVICE
        )
        fc = B.backtest(fn, panel, name=family)
        B.finish(
            family,
            fc,
            panel,
            scale,
            familia="Recurrentes",
            L=info["L"],
            estrategia="recursiva"
            if family == "LSTM_AR"
            else "seq2seq"
            if family == "Seq2Seq"
            else "multi-salida",
            covariables=True,
            params_total=ri["params_total"],
            params_entrenables=ri["params_entrenables"],
            train_s=info["train_s_total"] + ri["refit_s"],
            train_s_search_total=sum(
                json.loads(
                    (RUNS / f"{family}_h{h}_L168_m30_s42" / "info.json").read_text()
                )["train_s_total"]
                for h in HIDDENS
            )
            + ri["refit_s"],
            val_MASE=info["val_MASE"],
            seed=42,
            hidden=info["hidden"],
            gpu_mem_pico_MB=max(info["gpu_mem_pico_MB"], ri["gpu_mem_pico_MB_refit"]),
            epochs=info["mejor_epoch"],
            max_batches_per_epoch=BATCHES,
        )
        del model, fc
        gc.collect()


def libraries(panel, scaler, scale):
    from neuralforecast.losses.pytorch import MAE, DistributionLoss
    from neuralforecast.models import NHITS, TCN, DeepAR

    from src.models.nf import fit_nf, nf_forecast_fn

    z = scaler.transform(panel.loc[panel.ds <= D.VAL_END])
    for family, cls in (("NHITS", NHITS), ("TCN", TCN), ("DeepAR", DeepAR)):
        if (B.RESULTS_DIR / f"{family}.json").exists():
            continue
        torch.manual_seed(42)
        extra = {
            "batch_size": 16 if CPU else 32,
            "valid_batch_size": 16 if CPU else 32,
            "windows_batch_size": 64 if CPU else 1024,
            "inference_windows_batch_size": 128 if CPU else 1024,
            "learning_rate": 1e-3,
            "accelerator": DEVICE,
            "devices": 1,
            "enable_checkpointing": False,
        }
        if family == "NHITS":
            w = 64 if CPU else 512
            extra.update(mlp_units=[[w, w]] * 3, n_blocks=[1, 1, 1])
        elif family == "TCN":
            extra.update(
                encoder_hidden_size=32 if CPU else 128,
                context_size=24,
                decoder_hidden_size=32 if CPU else 128,
                dilations=[1, 2, 4, 8, 16, 32],
                kernel_size=3,
            )
        else:
            extra.update(
                lstm_hidden_size=32 if CPU else 128,
                lstm_n_layers=1,
                trajectory_samples=100,
                loss=DistributionLoss(distribution="Normal", level=[80]),
                valid_loss=MAE(),
            )
        futr = D.CALENDAR_COLS if family != "NHITS" else []

        def make(_cls=cls, _futr=futr, _extra=extra, **kw):
            return _cls(
                h=24,
                input_size=168,
                scaler_type="identity",
                random_seed=42,
                futr_exog_list=_futr,
                **_extra,
                **kw,
            )

        steps = NF_STEPS
        nf, info = fit_nf(
            make, z, max_steps=steps, val_check_steps=NF_CHECK, patience=NF_PATIENCE,
            futr_exog=futr,
        )
        best_steps = max(1, info["mejor_paso"])
        out = RUNS / family
        out.mkdir(parents=True, exist_ok=True)
        info["train_traj"].to_csv(out / "train.csv", index=False)
        info["valid_traj"].to_csv(out / "validation.csv", index=False)
        qs = (
            {"q10": "DeepAR-lo-80", "q50": "DeepAR-median", "q90": "DeepAR-hi-80"}
            if family == "DeepAR"
            else None
        )
        val_fc = B.backtest(
            nf_forecast_fn(nf, scaler, 168, family, futr_exog=futr, quantile_cols=qs),
            panel,
            name=f"{family}_val",
            start=D.VAL_START,
            end=D.VAL_END,
        )
        val_score = M.evaluate(val_fc, panel, scale)["resumen"]["MASE"]
        write_json(
            out / "selection.json",
            {
                "val_MASE": val_score,
                "best_steps": best_steps,
                "max_steps": steps,
                "architecture": {k: str(v) for k, v in extra.items()},
                "futr_exog": futr,
            },
        )
        del val_fc
        sel_s = info["fit_s"]
        del nf
        gc.collect()
        nf, final = fit_nf(
            make,
            z,
            val_size=0,
            max_steps=best_steps,
            val_check_steps=50,
            futr_exog=futr,
        )
        qs = (
            {"q10": "DeepAR-lo-80", "q50": "DeepAR-median", "q90": "DeepAR-hi-80"}
            if family == "DeepAR"
            else None
        )
        fn = nf_forecast_fn(nf, scaler, 168, family, futr_exog=futr, quantile_cols=qs)
        fc = B.backtest(fn, panel, name=family)
        B.finish(
            family,
            fc,
            panel,
            scale,
            familia="Probabilísticos" if family == "DeepAR" else "Convolucionales",
            L=168,
            estrategia="probabilística AR" if family == "DeepAR" else "multi-salida",
            covariables=bool(futr),
            params_total=final["params_total"],
            params_entrenables=final["params_entrenables"],
            train_s=sel_s + final["fit_s"],
            pasos_seleccion=steps,
            pasos_final=best_steps,
            val_MASE=val_score,
            validation_step_size=info.get("validation_step_size", 1),
            gpu_mem_pico_MB=max(info["gpu_mem_pico_MB"], final["gpu_mem_pico_MB"]),
        )
        del nf, fc
        gc.collect()


def chronos(panel, scaler, scale):
    if (B.RESULTS_DIR / "ChronosBolt.json").exists():
        return
    from chronos import BaseChronosPipeline

    model_id = "amazon/chronos-bolt-small" if CPU else "amazon/chronos-bolt-base"
    t = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=DEVICE, torch_dtype=torch.float32
    )
    load_s = time.perf_counter() - t

    def fn(hist, origin, H):
        ids, x = B.tail_matrix(hist, 168)
        qs, _mean = pipe.predict_quantiles(
            torch.from_numpy(x[..., 0].copy()),
            prediction_length=H,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        # Una salida directa por cuantil; mediana como pronóstico puntual.
        # Bolt estima cuantiles con cabezas separadas y puede cruzarlos. El
        # reordenamiento monótono por hora garantiza un intervalo coherente.
        q = np.sort(qs.float().cpu().numpy(), axis=-1)
        return B.frame_from_matrix(
            ids,
            origin,
            q[:, :, 1],
            {"q10": q[:, :, 0], "q50": q[:, :, 1], "q90": q[:, :, 2]},
        )

    fc = B.backtest(fn, panel, name="ChronosBolt")
    B.finish(
        "ChronosBolt",
        fc,
        panel,
        scale,
        familia="Fundacional",
        L=168,
        estrategia="zero-shot",
        covariables=False,
        train_s=0.0,
        model_id=model_id,
        load_s=load_s,
        params_total=sum(p.numel() for p in pipe.model.parameters()),
        cuantiles_postproceso="reordenamiento monótono de q10/q50/q90 por cliente y hora",
        params_entrenables=0,
        gpu_mem_pico_MB=0.0 if CPU else torch.cuda.max_memory_allocated() / 2**20,
        contaminacion_preentrenamiento="Electricity puede formar parte del preentrenamiento; no equivale a prueba temporal libre de contaminación externa.",
    )


def experimentos(panel, scaler, scale):
    out = RUNS / "experimentos_B"
    out.mkdir(parents=True, exist_ok=True)
    baseline_window_sensitivity(panel, scale)
    infos = {f: json.loads((RUNS / f / "selection.json").read_text()) for f in CLASSES}
    family = min(infos, key=lambda f: infos[f]["val_MASE"])
    base = infos[family]
    write_json(
        out / "selected_family.json",
        {"familia": family, "criterio": "mínimo MASE en validación", "config": base},
    )
    # Ventana y meses: comparación exclusivamente en validación.
    rows = []
    csv = out / "sensibilidad.csv"
    if not csv.exists():
        for L, months in [(48, 30), (168, 30), (336, 30), (168, 3), (168, 12)]:
            if (L, months) == (168, 30):
                info = base
            else:
                info = select_rnn(
                    panel,
                    scaler,
                    scale,
                    family,
                    L=L,
                    months=months,
                    hidden=base["hidden"],
                    tag=f"sweep_B_L{L}_m{months}",
                    epochs=EPOCHS,
                ).info
            rows.append(
                {
                    "modelo": family,
                    "L": L,
                    "meses": months,
                    "val_MAE": info["val_MAE"],
                    "val_MASE": info["val_MASE"],
                    "train_s": info["train_s_total"],
                    "epochs_max": EPOCHS,
                    "max_batches_per_epoch": info["max_batches_per_epoch"],
                }
            )
            pd.DataFrame(rows).to_csv(out / "sensibilidad.partial.csv", index=False)
        pd.DataFrame(rows).to_csv(csv, index=False)
    # Local vs global: mismos 10 clientes elegidos con semilla fija y mismos targets.
    if not (out / "local_global.csv").exists():
        ids = (
            np.random.default_rng(42)
            .choice(sorted(panel.unique_id.unique()), 10, replace=False)
            .tolist()
        )
        write_json(out / "local_clients.json", ids)
        p10 = panel.loc[panel.unique_id.isin(ids)]
        T.set_seed(42)
        global_model = CLASSES[family](hidden=base["hidden"])
        global_model.load_state_dict(
            torch.load(
                RUNS / family / "final.pt", map_location=DEVICE, weights_only=True
            )
        )
        # Para comparar ambos entrenados solo en train, cargar checkpoint de selección.
        global_model.load_state_dict(
            torch.load(
                RUNS / base["nombre"] / "selected.pt",
                map_location=DEVICE,
                weights_only=True,
            )
        )
        global_model.to(DEVICE)
        global_fc = B.backtest(
            B.torch_forecast_fn(
                global_model, scaler, 168, with_future=True, device=DEVICE
            ),
            p10,
            name="Global10_val",
            start=D.VAL_START,
            end=D.VAL_END,
        )
        grow = M.evaluate(global_fc, p10, scale)["resumen"]
        local_parts, elapsed, nparams = [], 0.0, 0
        for uid in ids:
            p = panel.loc[panel.unique_id == uid]
            r = select_rnn(
                p,
                scaler,
                scale,
                family,
                hidden=base["hidden"],
                tag=f"local_{uid}",
                epochs=4,
                batches=20,
            )
            fc = B.backtest(
                B.torch_forecast_fn(
                    r.model, scaler, 168, with_future=True, device=DEVICE
                ),
                p,
                name="Local10_val",
                start=D.VAL_START,
                end=D.VAL_END,
                verbose=False,
            )
            local_parts.append(fc)
            elapsed += r.info["train_s_total"]
            nparams += r.info["params_total"]
        lrow = M.evaluate(pd.concat(local_parts), p10, scale)["resumen"]
        pd.DataFrame(
            [
                {
                    "alcance": "global_150_evaluado_10",
                    **grow,
                    "train_s": base["train_s_total"],
                    "params": base["params_total"],
                },
                {"alcance": "local_10", **lrow, "train_s": elapsed, "params": nparams},
            ]
        ).to_csv(out / "local_global.csv", index=False)
    # Mejor baseline según validación (A), no por resultados de prueba.
    from src.models.baselines import fit_lightgbm

    hist_a = pd.read_csv(RUNS / "barrido_historia.csv")
    baseline = (
        hist_a.loc[hist_a.modelo.isin(["LightGBM", "SeasonalNaive24"])]
        .sort_values("val_MASE")
        .iloc[0]
    )
    write_json(out / "baseline_selection.json", baseline.to_dict())
    name = "LightGBM_mensual"
    if not (B.RESULTS_DIR / f"{name}.json").exists():
        monthly_fits = []

        def fit_fn(p):
            config = json.loads((B.RESULTS_DIR / "LightGBM.json").read_text())["config"]
            fn, fit_info = fit_lightgbm(p, params={**config, "n_jobs": 4})
            monthly_fits.append({"hasta": str(p.ds.max()), **fit_info})
            return fn

        fc = B.backtest_retrain(fit_fn, panel, name=name)
        total_params = max(x["params_total"] for x in monthly_fits)
        B.finish(
            name,
            fc,
            panel,
            scale,
            familia="Reentrenamiento",
            L=168,
            estrategia="mensual expandible",
            covariables=True,
            train_s=fc.attrs["fit_s"],
            params_total=total_params,
            params_entrenables=total_params,
            gpu_mem_pico_MB=0.0,
            seleccion="validación; configuración congelada antes de prueba",
            ajustes_mensuales=monthly_fits,
        )
    if not all(
        (B.RESULTS_DIR / f"{f}.json").exists() for f in ("NHITS", "TCN", "DeepAR")
    ):
        print(
            "Experimentos recurrentes listos; selección DL global pendiente de terminar librerías.",
            flush=True,
        )
        return
    best_dl_experiments(panel, scaler, scale, infos)


def baseline_window_sensitivity(panel, scale):
    """L en baselines: explicitar qué rezagos caben y cuándo L no cambia el modelo."""
    out = RUNS / "experimentos_B"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "baseline_L.csv"
    if path.exists():
        return
    from src.models.baselines import LGB_DEFAULT, fit_lightgbm

    historical = pd.read_csv(RUNS / "barrido_historia.csv")
    rows = []
    sn = float(
        historical.loc[
            (historical.modelo == "SeasonalNaive24") & (historical.meses == 30),
            "val_MASE",
        ].iloc[0]
    )
    lgb168 = float(
        historical.loc[
            (historical.modelo == "LightGBM") & (historical.meses == 30), "val_MASE"
        ].iloc[0]
    )
    for L in (48, 168, 336):
        rows.append(
            {
                "modelo": "SeasonalNaive24",
                "L": L,
                "L_efectivo": 24,
                "val_MASE": sn,
                "fuente": "A: barrido_historia.csv; invariancia exacta para L>=24",
                "train_s": 0.0,
            }
        )
    # Para L=48 no puede usarse el rezago 168. Se elimina y se informa esta ablación.
    fn, info = fit_lightgbm(
        panel.loc[panel.ds <= D.TRAIN_END],
        params={**LGB_DEFAULT, "n_jobs": 4},
        lags=list(range(1, 25)),
    )
    fc = B.backtest(
        fn, panel, name="LightGBM_L48_val", start=D.VAL_START, end=D.VAL_END
    )
    score = M.evaluate(fc, panel, scale)["resumen"]["MASE"]
    rows.append(
        {
            "modelo": "LightGBM",
            "L": 48,
            "L_efectivo": 24,
            "val_MASE": score,
            "fuente": "B CPU: mismos parámetros de A/historia, sin rezago 168",
            "train_s": info["fit_s"],
        }
    )
    for L in (168, 336):
        rows.append(
            {
                "modelo": "LightGBM",
                "L": L,
                "L_efectivo": 168,
                "val_MASE": lgb168,
                "fuente": "A: barrido_historia.csv; rezagos fijos 1-24,168, L=336 no agrega features",
                "train_s": float(
                    historical.loc[
                        (historical.modelo == "LightGBM") & (historical.meses == 30),
                        "train_s",
                    ].iloc[0]
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def best_dl_experiments(panel, scaler, scale, rnn_infos):
    """Semillas y actualización del mejor DL entre A y B, elegido con validación."""
    candidates = []
    for p in B.RESULTS_DIR.glob("*.json"):
        row = json.loads(p.read_text())
        if (
            row.get("familia")
            in (
                "Recurrentes",
                "Atención",
                "MLP y lineales",
                "Convolucionales",
                "Probabilísticos",
            )
            and row.get("val_MASE") is not None
        ):
            candidates.append(row)
    best = min(candidates, key=lambda x: x["val_MASE"])
    family = best["modelo"]
    out = RUNS / "experimentos_B"
    write_json(
        out / "best_dl_selection.json",
        {
            "criterio": "MASE validación, todas las familias A y B",
            "mejor": best,
            "candidatos": [
                {"modelo": x["modelo"], "val_MASE": x["val_MASE"]} for x in candidates
            ],
        },
    )
    if family in CLASSES:
        info = rnn_infos[family]
        for seed in (7, 123):
            name = f"{family}_seed{seed}"
            if (B.RESULTS_DIR / f"{name}.json").exists():
                continue
            r = select_rnn(
                panel, scaler, scale, family, hidden=info["hidden"], seed=seed, tag=name
            )
            model, ri = refit_rnn(panel, scaler, family, r.info)
            fc = B.backtest(
                B.torch_forecast_fn(
                    model, scaler, 168, with_future=True, device=DEVICE
                ),
                panel,
                name=name,
            )
            B.finish(
                name,
                fc,
                panel,
                scale,
                familia="Semillas",
                L=168,
                estrategia=family,
                seed=seed,
                params_total=ri["params_total"],
                params_entrenables=ri["params_entrenables"],
                train_s=r.info["train_s_total"] + ri["refit_s"],
                gpu_mem_pico_MB=r.info["gpu_mem_pico_MB"],
            )
        name = f"{family}_mensual"
        if not (B.RESULTS_DIR / f"{name}.json").exists():

            def fit_fn(p):
                model, _ = refit_rnn(p, scaler, family, info, end=p.ds.max(), name=name)
                return B.torch_forecast_fn(
                    model, scaler, 168, with_future=True, device=DEVICE
                )

            fc = B.backtest_retrain(fit_fn, panel, name=name)
            B.finish(
                name,
                fc,
                panel,
                scale,
                familia="Reentrenamiento",
                L=168,
                estrategia="mensual",
                covariables=True,
                train_s=fc.attrs["fit_s"],
                params_total=info["params_total"],
                params_entrenables=info["params_entrenables"],
            )
        return
    # En A, iTransformer es el mejor DL en validación. Mantener su arquitectura.
    from neuralforecast.models import NHITS, PatchTST, iTransformer

    from src.models.nf import fit_nf, nf_forecast_fn

    if family == "iTransformer":
        cls = iTransformer
        architecture = {
            "n_series": 150,
            "hidden_size": 256,
            "n_heads": 8,
            "e_layers": 3,
            "d_ff": 512,
            "dropout": 0.1,
            "learning_rate": 5e-4,
        }
    elif family == "PatchTST":
        cls = PatchTST
        architecture = {
            "patch_len": 16,
            "stride": 8,
            "encoder_layers": 3,
            "n_heads": 16,
            "hidden_size": 128,
            "linear_hidden_size": 256,
            "dropout": 0.2,
            "revin": True,
            "learning_rate": 5e-4,
        }
    elif family == "NHITS":
        cls = NHITS
        w = 64 if CPU else 512
        architecture = {"mlp_units": [[w, w]] * 3, "n_blocks": [1, 1, 1], "learning_rate": 1e-3}
    else:
        raise ValueError(
            f"Agregar arquitectura seleccionada para semillas/reentrenamiento: {family}"
        )

    def factory(seed):
        return lambda **kw: cls(
            h=24,
            input_size=168,
            scaler_type="identity",
            random_seed=seed,
            batch_size=16 if CPU else 32,
            windows_batch_size=32 if CPU else 1024,
            inference_windows_batch_size=32 if CPU else 1024,
            accelerator=DEVICE,
            devices=1,
            enable_checkpointing=False,
            **architecture,
            **kw,
        )

    z = scaler.transform(panel.loc[panel.ds <= D.VAL_END])
    cfg_path = out / "best_dl_cpu_budget.json"
    if family == "NHITS" and not cfg_path.exists():
        # N-HiTS ya pasó por selección con validación en libraries(): se reutilizan sus pasos.
        sel = json.loads((RUNS / "NHITS" / "selection.json").read_text())
        write_json(cfg_path, {"modelo": family, "steps": sel["best_steps"],
                              "architecture": architecture, "windows_batch_size": 1024,
                              "hardware": B.hardware()})
    if cfg_path.exists():
        selected = json.loads(cfg_path.read_text())
    else:
        nf, info = fit_nf(
            factory(42),
            z,
            max_steps=200 if CPU else 1500,
            val_check_steps=50,
            patience=2,
        )
        selected = {
            "modelo": family,
            "steps": max(1, info["mejor_paso"]),
            "selection_s": info["fit_s"],
            "architecture": architecture,
            "windows_batch_size": 32,
            "hardware": B.hardware(),
        }
        write_json(cfg_path, selected)
        del nf
        gc.collect()
    for seed in (42, 7, 123):
        name = f"{family}_seed{seed}"
        if (B.RESULTS_DIR / f"{name}.json").exists():
            continue
        nf, info = fit_nf(
            factory(seed),
            z,
            val_size=0,
            max_steps=selected["steps"],
            val_check_steps=50,
        )
        fc = B.backtest(nf_forecast_fn(nf, scaler, 168, family), panel, name=name)
        B.finish(
            name,
            fc,
            panel,
            scale,
            familia="Semillas",
            L=168,
            estrategia="multi-salida",
            seed=seed,
            covariables=False,
            train_s=info["fit_s"],
            params_total=info["params_total"],
            params_entrenables=info["params_entrenables"],
            pasos=selected["steps"],
            gpu_mem_pico_MB=info["gpu_mem_pico_MB"],
        )
        del nf
        gc.collect()
    name = f"{family}_mensual"
    if not (B.RESULTS_DIR / f"{name}.json").exists():

        def fit_fn(p):
            nf, _info = fit_nf(
                factory(42),
                scaler.transform(p),
                val_size=0,
                max_steps=selected["steps"],
                val_check_steps=50,
                end=p.ds.max(),
            )
            return nf_forecast_fn(nf, scaler, 168, family)

        fc = B.backtest_retrain(fit_fn, panel, name=name)
        B.finish(
            name,
            fc,
            panel,
            scale,
            familia="Reentrenamiento",
            L=168,
            estrategia="mensual expandible",
            covariables=False,
            train_s=fc.attrs["fit_s"],
            params_total=best["params_total"],
            params_entrenables=best["params_entrenables"],
            pasos=selected["steps"],
            gpu_mem_pico_MB=0.0 if CPU else torch.cuda.max_memory_allocated() / 2**20,
        )


def verify(panel, scaler, scale):
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    from src.models.baselines import seasonal_naive_fn

    # Un cliente basta para comprobar la igualdad matemática de ambos protocolos.
    p = panel.loc[panel.unique_id == panel.unique_id.iloc[0], ["unique_id", "ds", "y"]]
    ours = B.backtest(seasonal_naive_fn(24), p, name="SeasonalNaive24", verbose=False)
    cv = StatsForecast(models=[SeasonalNaive(24)], freq="h", n_jobs=1).cross_validation(
        df=p, h=24, n_windows=92, step_size=24, refit=False
    )
    theirs = B.from_nixtla_cv(cv, "SeasonalNaive", name="SeasonalNaive24")
    keys = ["unique_id", "origin", "ds"]
    a = ours.sort_values(keys).reset_index(drop=True)
    b = theirs.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
    np.testing.assert_allclose(a.y_hat, b.y_hat, rtol=0, atol=0)
    checked = []
    for path in sorted(B.FORECASTS_DIR.glob("*.parquet")):
        if path.stem == "todos":
            continue
        fc = pd.read_parquet(path)
        assert len(fc) == 92 * 24 * 150, (path, len(fc))
        assert fc.origin.nunique() == 92
        assert not fc.duplicated(keys).any()
        assert set(fc.unique_id) == set(panel.unique_id)
        assert fc.groupby(["unique_id", "origin"], observed=True).size().eq(24).all()
        assert ((fc.ds - fc.origin) / pd.Timedelta(hours=1)).between(0, 23).all()
        assert np.isfinite(fc.y_hat).all()
        for c in ("q10", "q50", "q90"):
            if c in fc:
                assert np.isfinite(fc[c]).all()
        if "q10" in fc:
            assert ((fc.q10 <= fc.q50) & (fc.q50 <= fc.q90)).all()
        checked.append({"modelo": path.stem, "filas": len(fc), "origenes": 92})
    assert scaler.fit_end == D.TRAIN_END
    assert len(scaler.stats) == 150
    write_json(
        RUNS / "verification.json",
        {
            "statsforecast_equivalencia_exacta": True,
            "clientes_equivalencia": 1,
            "origenes_equivalencia": 92,
            "normalizacion_hasta": str(scaler.fit_end),
            "modelos": checked,
        },
    )
    print(
        f"Verificados {len(checked)} archivos de pronósticos; equivalencia exacta StatsForecast.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "rnn",
            "libraries",
            "chronos",
            "experiments",
            "verify",
            "report",
        ],
        default="all",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    panel, scaler, scale = load_context()
    stages = {
        "rnn": recurrentes,
        "libraries": libraries,
        "chronos": chronos,
        "experiments": experimentos,
        "verify": verify,
    }
    for stage, fn in stages.items():
        if args.stage in ("all", stage):
            print(f"\n=== {stage} ({B.hardware()}) ===", flush=True)
            fn(panel, scaler, scale)
            gc.collect()
    if args.stage in ("all", "report"):
        from src.reporting import build_report

        build_report(panel, scale)


if __name__ == "__main__":
    main()
