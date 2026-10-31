# Lab 9 · Series de tiempo y forecasting (CC3092)

Pronóstico day-ahead (H = 24 h) del consumo eléctrico horario de 150 clientes de Portugal
(ElectricityLoadDiagrams20112014, UCI). Todos los modelos se evalúan con el mismo
`backtest()`: 92 orígenes diarios en prueba (2014-10-01 a 2014-12-31).

## Cómo correrlo

**Local (uv):**
```bash
uv sync
uv run jupyter lab
```

**Google Colab:**
```python
!git clone <url-del-repo>
%cd DL-LAB09
!pip install -q -r requirements-colab.txt
import sys; sys.path.insert(0, ".")
```
Los datos se descargan solos la primera vez (`build_panel()`, ~2 s desde una copia en
Hugging Face verificada contra UCI). Colab se desconecta: **subir `outputs/` al repo apenas
termina cada modelo.**

## Notebooks

| Notebook | Contenido | Quién |
|---|---|---|
| `01_datos_eda` | carga, limpieza, cambio de horario, exploración, normalización, covariables, `WindowDataset`, conteo de ventanas, 2 fugas | Diego |
| `02_backtest_metricas` | `backtest()` validado contra StatsForecast, métricas, 2 fugas, reentrenamiento mensual | Diego |
| `03_baselines_arboles` | Naive, Seasonal naive 24/168, MSTL, LightGBM | Diego |
| `04_lineales_mlp` | MLP, DLinear, NLinear (propios) | Diego |
| `05_rnn_seq2seq` | LSTM (autorregresiva, multi-salida), Seq2seq | Cisco |
| `06a_patchtst_itransformer` | PatchTST, iTransformer | Diego |
| `06b_nhits_tcn_deepar_chronos` | N-HiTS, TCN, DeepAR, Chronos-Bolt | Cisco |
| `07a_iteraciones` | 5.2: ventana L y meses de historia | Diego |
| `07b_experimentos` | 5.1 estrategias, 5.3 local vs global, reentrenamiento, semillas | Cisco |
| `08_comparacion` | tabla comparativa y gráficas de la sección 6 | Cisco |

## Piezas reutilizables (`src/`)

```python
from src import data as D, metrics as M, backtest as B, train as T
from src.dataset import WindowDataset

panel = D.build_panel()                       # unique_id, ds, y (kWh) + calendario
scale = M.mase_scale(panel)                   # escala del MASE (seasonal naive 24 h en train)
scaler = D.SeriesScaler().fit(panel)          # z-score por serie, solo con datos de train
z = scaler.transform(panel)
```

**Modelo propio en PyTorch** (ver `04_lineales_mlp` como ejemplo completo):
```python
train_ds = WindowDataset(z, L=168, H=24, split="train", with_future=True)
val_ds   = WindowDataset(z, L=168, H=24, split="val", origin_hour=0, with_future=True)
full_ds  = WindowDataset(z, L=168, H=24, split=None, target_end=D.VAL_END, with_future=True)

# forward(x, x_fut) -> (B, 24);  x: (B, L, 8) = y + calendario,  x_fut: (B, 24, 7)
res = T.fit(MiModelo(), train_ds, val_ds, scaler, scale, name="LSTM_h128", epochs=15)
model, rinfo = T.refit(MiModelo(), full_ds, epochs=res.info["mejor_epoch"], name="LSTM_h128")

fn = B.torch_forecast_fn(model, scaler, L=168, with_future=True, device="cuda")
fc = B.backtest(fn, panel, name="LSTM")
row, ev = B.finish("LSTM", fc, panel, scale, familia="Recurrentes", L=168,
                   estrategia="multi-salida", covariables=True,
                   params_total=rinfo["params_total"], params_entrenables=rinfo["params_entrenables"],
                   train_s=res.info["train_s_total"] + rinfo["refit_s"])
```
`fit` guarda en `outputs/runs/<nombre>/` el historial por epoch (pérdidas, MAE y MASE de
validación, tiempo) y la info de la iteración (parámetros, memoria pico de GPU, etc.).

**NeuralForecast** (ver `06a_patchtst_itransformer`):
```python
from src.models.nf import fit_nf, nf_forecast_fn
from neuralforecast.models import NHITS
make = lambda **kw: NHITS(h=24, input_size=168, scaler_type="identity", random_seed=42, **kw)
nf, info = fit_nf(make, z[["unique_id", "ds", "y"]], val_size=2208)       # selección
nf, info = fit_nf(make, z[["unique_id", "ds", "y"]], val_size=0, max_steps=info["mejor_paso"])  # final
fc = B.backtest(nf_forecast_fn(nf, scaler, 168, "NHITS"), panel, name="NHITS")
```
Para DeepAR, pasar `quantile_cols={"q10": ..., "q50": ..., "q90": ...}` a `nf_forecast_fn`
y `finish` calcula pinball loss y cobertura P10–P90 solo.

**Cualquier otro modelo** (Chronos, etc.): escribir una función
`forecast_fn(hist, origin, H) -> DataFrame(unique_id, ds, y_hat [, q10, q50, q90])` que use
solo `hist` (filas con `ds < origin`) y pasarla a `B.backtest`.

**Reentrenamiento mensual:** `B.backtest_retrain(fit_fn, panel)`, donde
`fit_fn(datos_hasta_inicio_de_mes)` devuelve un `forecast_fn`.

## Reglas del protocolo

- Selección de hiperparámetros **solo con validación** (jul–sep 2014). La prueba se corre
  una vez por modelo, con `backtest()` y la mejor configuración.
- `fit` se niega a usar la prueba para early stopping.
- Métricas siempre en kWh. `finish()` guarda `outputs/forecasts/<modelo>.parquet`
  (`unique_id, ds, origin, modelo, y_hat`) y `outputs/results/<modelo>.json`.
