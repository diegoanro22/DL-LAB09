"""Tablas, figuras y reporte reproducible a partir de pronósticos guardados."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import backtest as B
from src import data as D
from src import metrics as M

FIGS = D.ROOT / "outputs" / "figs"
TABLES = D.ROOT / "outputs" / "tables"
REPORT = D.ROOT / "informe"


def load_results():
    return pd.DataFrame(
        [json.loads(p.read_text()) for p in sorted(B.RESULTS_DIR.glob("*.json"))]
    ).sort_values("MASE")


def chart(name):
    plt.tight_layout()
    plt.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_tables_and_figures(panel, scale):
    for directory in (FIGS, TABLES, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    results = load_results()
    main = results.loc[~results.familia.isin(["Semillas", "Reentrenamiento"])].copy()
    results.to_csv(TABLES / "todos_resultados.csv", index=False)
    main.to_csv(TABLES / "comparacion.csv", index=False)
    monthly_comparison = []
    for row in results.loc[results.familia == "Reentrenamiento"].to_dict("records"):
        family = row["modelo"].removesuffix("_mensual")
        reference = family if family == "LightGBM" else f"{family}_seed42"
        fixed = results.loc[results.modelo == reference]
        if len(fixed):
            fixed = fixed.iloc[0]
            monthly_comparison.append(
                {
                    "modelo": family,
                    "referencia_fija": reference,
                    "MASE_fijo": fixed.MASE,
                    "MASE_mensual": row["MASE"],
                    "delta_MASE": row["MASE"] - fixed.MASE,
                    "MASE_navidad_fijo": fixed.MASE_navidad,
                    "MASE_navidad_mensual": row["MASE_navidad"],
                    "train_s_mensual": row["train_s"],
                }
            )
    pd.DataFrame(monthly_comparison).to_csv(
        TABLES / "reentrenamiento_comparacion.csv", index=False
    )
    actuals = panel.loc[panel.ds >= D.TEST_START, ["unique_id", "ds", "y"]]
    by_h, prob, xmas = [], [], []
    for row in main.to_dict("records"):
        name = row["modelo"]
        fc = pd.read_parquet(B.FORECASTS_DIR / f"{name}.parquet")
        ev = M.evaluate(fc, actuals, scale)
        ev["por_cliente"].to_csv(TABLES / f"{name}_por_cliente.csv")
        ev["por_h"].to_csv(TABLES / f"{name}_por_h.csv")
        ev["por_hora"].to_csv(TABLES / f"{name}_por_hora.csv")
        by_h.append(ev["por_h"].assign(modelo=name).rename_axis("h").reset_index())
        xmas.append({"modelo": name, **ev["navidad"]})
        if "q10" in fc:
            prob.append(
                {
                    "modelo": name,
                    **{
                        k: v
                        for k, v in ev["resumen"].items()
                        if k.startswith(("pinball", "cobertura"))
                    },
                }
            )
    horizons = pd.concat(by_h, ignore_index=True)
    horizons.to_csv(TABLES / "mae_por_h.csv", index=False)
    pd.DataFrame(prob).to_csv(TABLES / "probabilistica.csv", index=False)
    pd.DataFrame(xmas).to_csv(TABLES / "navidad.csv", index=False)
    # Pareto se presenta por hardware; los tiempos no son directamente comparables.
    fig, ax = plt.subplots(figsize=(9, 4.4))
    offsets = {
        "ChronosBolt": (-40, -15),
        "iTransformer": (12, -12),
        "MLP": (10, -12),
        "LightGBM": (-63, 16),
        "PatchTST": (20, 27),
        "DLinear": (-18, 32),
        "NLinear": (28, 10),
        "SeasonalNaive24": (-5, 18),
        "SeasonalNaive168": (10, 30),
    }
    for hw, g in main.groupby("hardware"):
        x = (g.train_s.fillna(0) + g.inference_s.fillna(0)).clip(lower=0.1)
        points = ax.scatter(x, g.MASE, label=hw, s=35)
        frontier = pd.DataFrame({"seconds": x, "MASE": g.MASE}).sort_values("seconds")
        frontier = frontier.loc[
            frontier.MASE.lt(frontier.MASE.cummin().shift(fill_value=np.inf))
        ]
        ax.plot(
            frontier.seconds,
            frontier.MASE,
            "--",
            color=points.get_facecolor()[0],
            alpha=0.5,
            lw=0.8,
        )
        for name, xx, yy in zip(g.modelo, x, g.MASE):
            ax.annotate(
                name,
                (xx, yy),
                fontsize=7,
                xytext=offsets.get(name, (3, 3)),
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": ".6", "lw": 0.5},
            )
    ax.set_xscale("log")
    ax.set_xlabel("Entrenamiento + inferencia (s, log)")
    ax.set_ylabel("MASE prueba")
    ax.set_ylim(bottom=0.35)
    ax.set_title("MASE de prueba contra tiempo de entrenamiento e inferencia")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    chart("pareto")
    fig, ax = plt.subplots(figsize=(9, 4))
    wanted = list(main.head(5).modelo) + [
        "Naive",
        "SeasonalNaive24",
        "LSTM_AR",
        "LSTM_Multi",
        "Seq2Seq",
    ]
    for name in dict.fromkeys(wanted):
        g = horizons.loc[horizons.modelo == name]
        if len(g):
            ax.plot(g.h, g.MAE, label=name, linewidth=1.3)
    ax.set_xlabel("Paso h (horas)")
    ax.set_ylabel("MAE (kWh)")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_title(
        "MAE de prueba por paso del horizonte"
    )
    ax.legend(ncol=3, fontsize=7)
    ax.grid(alpha=0.2)
    chart("mae_por_h")
    hist = pd.read_csv(D.ROOT / "outputs" / "runs" / "barrido_historia.csv").rename(
        columns={"meses": "meses_historia"}
    )
    sweep = D.ROOT / "outputs" / "runs" / "experimentos_B" / "sensibilidad.csv"
    if sweep.exists():
        b = pd.read_csv(sweep).rename(columns={"meses": "meses_historia"})
        hist = pd.concat([hist, b.loc[b.L == 168]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, g in hist.groupby("modelo"):
        g = g.sort_values("meses_historia")
        ax.plot(g.meses_historia, g.val_MASE, "o-", label=name)
    chronos_val = D.ROOT / "outputs" / "runs" / "experimentos_B" / "chronos_val.json"
    if chronos_val.exists():
        cv = json.loads(chronos_val.read_text())["val_MASE"]
        ax.axhline(cv, color="black", ls="--", lw=1, label=f"Chronos zero-shot ({cv:.3f})")
    ax.set_xlabel("Meses de historia")
    ax.set_ylabel("MASE validación")
    ax.set_xticks([3, 12, 30])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    chart("mase_vs_historia")
    # Mostrar un cliente elegido por escala de train, sin escoger por desempeño en test.
    means = (
        panel.loc[panel.ds <= D.TRAIN_END]
        .groupby("unique_id", observed=True)
        .y.mean()
        .sort_values()
    )
    uid = means.index[len(means) // 2]
    lo, hi = M.CHRISTMAS_WEEK
    real = actuals.loc[(actuals.unique_id == uid) & actuals.ds.between(lo, hi)]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(
        real.ds,
        real.y,
        color="black",
        label=f"Real ({uid}, cliente mediano en train)",
        lw=1.4,
    )
    for name in list(main.head(2).modelo) + ["SeasonalNaive24"]:
        fc = pd.read_parquet(B.FORECASTS_DIR / f"{name}.parquet")
        p = fc.loc[(fc.unique_id == uid) & fc.ds.between(lo, hi)].sort_values("ds")
        ax.plot(p.ds, p.y_hat, label=name, lw=0.9)
        if "q10" in p:
            ax.fill_between(p.ds, p.q10, p.q90, alpha=0.15)
    ax.set_ylabel("kWh/h")
    ax.set_title("Semana 22–28 diciembre; pronósticos day-ahead diarios")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    chart("semana_navidad_comparacion")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, family in zip(axes, ["LSTM_AR", "LSTM_Multi", "Seq2Seq"]):
        selection = D.ROOT / "outputs" / "runs" / family / "selection.json"
        if selection.exists():
            info = json.loads(selection.read_text())
            h = pd.read_csv(
                D.ROOT / "outputs" / "runs" / info["nombre"] / "history.csv"
            )
            ax.plot(h.epoch, h.train_loss, label="train")
            ax.plot(h.epoch, h.val_loss, label="val")
            ax.set_title(family)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE normalizada")
            ax.legend(fontsize=7)
    chart("curvas_recurrentes")
    # Archivo único del contrato: schema común, escritura por partes para acotar RAM.
    writer = None
    try:
        for p in sorted(B.FORECASTS_DIR.glob("*.parquet")):
            if p.stem == "todos":
                continue
            fc = pd.read_parquet(p)
            for c in ("q10", "q50", "q90"):
                if c not in fc:
                    fc[c] = np.nan
            for c in ("y_hat", "q10", "q50", "q90"):
                fc[c] = fc[c].astype("float64")
            for c in ("unique_id", "modelo"):
                fc[c] = fc[c].astype(str)
            table = pa.Table.from_pandas(
                fc[
                    [
                        "unique_id",
                        "ds",
                        "origin",
                        "modelo",
                        "y_hat",
                        "q10",
                        "q50",
                        "q90",
                    ]
                ],
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    B.FORECASTS_DIR / "todos.parquet", table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return main, results


def create_docx(main, results):
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt
    from pypdf import PdfReader

    doc = Document()
    doc.core_properties.title = "Lab 9 · Series de tiempo y forecasting"
    doc.core_properties.author = "Diego y Cisco"
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(1.5)
    section.left_margin = section.right_margin = Cm(1.6)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.0

    def p(text):
        doc.add_paragraph(str(text))

    def title(text):
        doc.add_heading(text, level=1)

    def image(name, width=6.7):
        doc.add_picture(str(FIGS / f"{name}.png"), width=Inches(width))

    def table(df, cols, headers=None):
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Light Shading Accent 1"
        for c, name in zip(t.rows[0].cells, headers or cols):
            c.text = str(name)
        for row in df.to_dict("records"):
            cells = t.add_row().cells
            for cell, key in zip(cells, cols):
                v = row.get(key, "")
                cell.text = (
                    str(int(v))
                    if key
                    in ("seed", "L", "params", "params_total", "params_entrenables")
                    and isinstance(v, (int, float, np.number))
                    and np.isfinite(v)
                    else (f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
                )
        for row in t.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(8)

    best = main.iloc[0]
    doc.add_heading("Lab 9 · Series de tiempo y forecasting", level=0)
    p("CC3092 · Diego y Cisco · Entrega: 1 de noviembre de 2026")
    title("1. Datos y protocolo común")
    p(
        "Se pronostican las siguientes 24 horas de electricidad de 150 clientes portugueses. El panel contiene 26,304 horas por cliente (2012–2014); los clientes y seed=42 coinciden en todas las familias. Cada registro de 15 min representa kW al final del intervalo: se suma con peso 0.25 y etiquetas horarias a la izquierda para obtener kWh/h."
    )
    p(
        "Train termina el 30/06/2014, validación abarca julio–septiembre y prueba octubre–diciembre. Cada modelo base tiene un único backtest de 92 orígenes a medianoche: 331,200 predicciones. Las ventanas usan L=168, H=24 y stride de entrenamiento 1 h. El z-score de las redes y la escala del MASE se calculan con train; LightGBM reajusta su escalado con la historia que recibe, sin fechas futuras. El reajuste final incluye train+validación, con presupuesto elegido en validación."
    )
    p(
        "El código existente interpola la hora de marzo y divide por dos la hora sumada de octubre. Se conserva esta transformación en B para mantener compatibilidad con los resultados de A; difiere del borrador PLAN.md, que proponía conservar la suma de octubre. Calendario futuro: hora, día y mes cíclicos, y feriados nacionales portugueses."
    )
    title("2. Fundamentos y auditoría")
    p(
        "LSTM devuelve la secuencia de estados y el estado final (h,c); GRU usa solo h. La cabeza directa emite 24 valores; la estrategia recursiva aprende un paso y realimenta sus propias salidas. Seq2seq separa encoder y decoder, agrega calendario conocido y usa teacher forcing de 0.5 solo en train; validación y prueba decodifican sin etiquetas futuras. RevIN normaliza cada ventana y revierte su salida con estadísticos de esa misma entrada."
    )
    p(
        "Una TCN usa convoluciones causales dilatadas. Con kernel 3 y dilataciones 1,2,4,8,16,32, la implementación instalada usa una convolución por nivel y cubre 127 posiciones: R=1+Σ(k−1)d. Conv1d recibe (batch, canales, tiempo). Un TransformerEncoderLayer combina atención y MLP; requiere posición explícita para distinguir el orden temporal. iTransformer atiende entre clientes y PatchTST entre parches de tiempo."
    )
    p(
        "Se auditan cuatro fugas: normalizar con toda la serie, desalinear calendario, dividir ventanas aleatoriamente y usar test para early stopping. La historia se corta en origin−1 h. El contrato rechaza claves duplicadas, clientes/horizontes incompletos, valores no finitos y cruces de cuantiles. SeasonalNaive24 coincide exactamente con StatsForecast.cross_validation en los 92 orígenes de un cliente de control, con refit=False y step_size=24."
    )
    doc.add_page_break()
    title("3. Comparación de modelos")
    p(
        f"Se completaron {len(main)} modelos base. El menor MASE de prueba fue {best.modelo}: {best.MASE:.3f}, MAE={best.MAE:.2f} kWh, WAPE={best.WAPE:.1%}. Esta clasificación describe la prueba; no se usó para elegir hiperparámetros."
    )
    compact = main.copy()
    compact["Equipo"] = compact.hardware.map(
        lambda x: "RTX5070" if "5070" in x else "CPU"
    )
    compact["modelo"] = compact["modelo"].replace(
        {"SeasonalNaive24": "SNaive24", "SeasonalNaive168": "SNaive168"}
    )
    table(
        compact,
        ["modelo", "MAE", "RMSE", "WAPE", "MASE", "train_s", "inference_s", "Equipo"],
        ["Modelo", "MAE", "RMSE", "WAPE", "MASE", "Train s", "Infer s", "Equipo"],
    )
    p(
        "Parámetros totales/entrenables, covariables, estrategia, memoria y tiempos completos: outputs/tables/comparacion.csv y outputs/results/*.json. LightGBM cuenta nodos, no pesos neuronales; el hardware heredado identifica el equipo y no implica entrenamiento GPU de árboles. En MSTL, inferencia incluye ajustes por origen sobre ocho semanas."
    )
    image("pareto")
    p(
        "Los tiempos de A se midieron en RTX 5070 y los de B en AMD Ryzen AI 7 350 (CPU). B usó redes menores y presupuestos limitados explícitos; esta figura no permite concluir qué arquitectura sería más rápida con el mismo equipo y presupuesto."
    )
    doc.add_page_break()
    title("4. Estrategias y sensibilidad")
    image("mae_por_h")
    p(
        "La recursión puede acumular error al reutilizar predicciones. La salida directa evita esta realimentación; seq2seq comparte dinámica entre pasos y admite calendario del decoder. Las curvas comparan arquitecturas y presupuestos diferentes, por lo que no aíslan causalmente el efecto de la estrategia."
    )
    image("mase_vs_historia")
    sweep = D.ROOT / "outputs" / "runs" / "experimentos_B" / "sensibilidad.csv"
    if sweep.exists():
        s = pd.read_csv(sweep)
        r = s.loc[s.val_MASE.idxmin()]
        p(
            f"Para el mejor recurrente por validación, se ensayaron L=48/168/336 y 3/12/30 meses. La configuración con menor MASE de este barrido fue L={int(r.L)}, {int(r.meses)} meses: {r.val_MASE:.3f}. El barrido se reporta en validación y no implica una segunda selección sobre prueba. Los experimentos de A conservan sus registros originales."
        )
    p(
        "En los baselines, L se interpreta según sus rezagos: SeasonalNaive24 no cambia para L≥24; LightGBM con rezagos fijos 1–24,168 no cambia entre L=168 y 336. Para L=48 se mide una ablación sin el rezago 168. Los valores y su procedencia están en baseline_L.csv."
    )
    lg = D.ROOT / "outputs" / "runs" / "experimentos_B" / "local_global.csv"
    if lg.exists():
        df = pd.read_csv(lg)
        table(df, ["alcance", "MAE", "MASE", "train_s", "params"])
        p(
            "Local: diez redes separadas; global: una red entrenada con los 150 clientes y evaluada sobre los mismos diez IDs fijados con seed=42. La evaluación de este experimento es validación. El presupuesto por cliente y el volumen de datos difieren; se informa el costo total de los diez ajustes."
        )
    doc.add_page_break()
    title("5. Incertidumbre, estabilidad y actualización")
    prob = pd.read_csv(TABLES / "probabilistica.csv")
    if len(prob):
        table(
            prob,
            [
                "modelo",
                "pinball_q10",
                "pinball_q50",
                "pinball_q90",
                "cobertura_P10_P90",
            ],
            ["Modelo", "Pinball .1", "Pinball .5", "Pinball .9", "Cobertura .1–.9"],
        )
    p(
        "La pérdida pinball penaliza asimétricamente los residuos según τ. El intervalo P10–P90 tiene cobertura nominal de 80%; se mide frecuencia real de inclusión. MASE promedia por cliente el MAE dividido por el MAE estacional de train; WAPE divide la suma de errores absolutos entre consumo absoluto total y enfatiza clientes grandes. El MAPE omite |y|≤0.001 y no se interpreta sin esa exclusión."
    )
    p(
        "DeepAR aprende una distribución Normal sobre valores normalizados y obtiene cuantiles mediante trayectorias. Chronos-Bolt-small funciona zero-shot en CPU y devuelve (series, horizonte, cuantiles); se reordenan q10/q50/q90 por hora para evitar cruces y se usa q50 como pronóstico puntual. No se demuestra que Electricity estuviera excluido de su preentrenamiento; su evaluación externa puede contener contaminación."
    )
    seeds = results.loc[results.familia == "Semillas"]
    if len(seeds):
        p(
            f"Tres semillas con configuración y presupuesto CPU iguales: MASE promedio {seeds.MASE.mean():.3f}, desviación estándar muestral {seeds.MASE.std(ddof=1):.3f}. La selección del modelo y del presupuesto se fijó con validación. Una media de tres réplicas no constituye un intervalo de confianza robusto."
        )
        table(seeds, ["modelo", "seed", "MAE", "MASE", "train_s"])
    monthly = results.loc[results.familia == "Reentrenamiento"]
    if len(monthly):
        p(
            "La actualización mensual reentrena al inicio de octubre, noviembre y diciembre con historia anterior al mes. Hiperparámetros y pasos quedan congelados; el escalado del DL permanece ajustado a train. La referencia fija del DL es la réplica CPU seed=42, con igual arquitectura y presupuesto, no la corrida original de A."
        )
        table(
            monthly, ["modelo", "MAE", "MASE", "MASE_navidad", "train_s", "inference_s"]
        )
        comparison = pd.read_csv(TABLES / "reentrenamiento_comparacion.csv")
        for row in comparison.to_dict("records"):
            p(
                f"{row['modelo']}: MASE fijo {row['MASE_fijo']:.3f} → mensual {row['MASE_mensual']:.3f} (Δ={row['delta_MASE']:+.3f})."
            )
    image("semana_navidad_comparacion")
    doc.add_page_break()
    title("6. Discusión, reproducibilidad y referencias")
    xmas = pd.read_csv(TABLES / "navidad.csv").sort_values("MASE")
    p(
        f"En Navidad (22–28 diciembre), el menor MASE fue {xmas.iloc[0].modelo}: {xmas.iloc[0].MASE:.3f}. Los horarios festivos difieren de días laborales y elevan el error de modelos que dependen de periodicidad. La figura usa el cliente de consumo medio mediano en train; no fue elegido por un error favorable en test."
    )
    image("curvas_recurrentes")
    p(
        "Las curvas de tres recurrentes muestran convergencia y brecha de generalización. En CPU se acotó cada epoch a una muestra reproducible de 50 batches de 128 ventanas, hasta 6 epochs; el barrido mantiene ese mismo máximo y los modelos locales usan un presupuesto menor. Se conserva stride=1 en las ventanas y evaluación completa de 150 clientes. N-HiTS, TCN y DeepAR usan hasta 250 pasos y refit con el paso seleccionado. Los límites exactos quedan en cada JSON."
    )
    p(
        "Los pronósticos preservan unique_id, ds, origin, modelo, y_hat y cuantiles cuando corresponden. outputs/forecasts/todos.parquet reúne las corridas. main.ipynb carga artefactos guardados y se ejecuta de principio a fin sin reentrenar. python -m src.run_cisco --stage all reanuda modelos faltantes. requirements-cpu-lock.txt registra el entorno ejecutado. Las pruebas automatizadas verifican aislamiento de etiquetas futuras y el contrato del backtest."
    )
    p(
        "Límites: presupuestos y hardware heterogéneos; muestras de batches en entrenamiento; incertidumbre externa de preentrenamiento Chronos; comparación local/global con distinto volumen de información; selección de clientes completos usa disponibilidad del panel entero. No se extrapola la clasificación a clientes nuevos ni se atribuyen diferencias de tiempo exclusivamente al modelo."
    )
    p(
        "Referencias técnicas consultadas: PyTorch, nn.LSTM/nn.GRU y TransformerEncoderLayer (docs.pytorch.org); Nixtla, código de NeuralForecast y StatsForecast (github.com/Nixtla); Amazon Science, chronos-forecasting y modelo chronos-bolt-small (huggingface.co/amazon/chronos-bolt-small); UCI ElectricityLoadDiagrams20112014 (archive.ics.uci.edu/dataset/321). En informe/investigacion.md se amplían las notas y enlaces primarios."
    )
    repo_url = "https://github.com/diegoanro22/DL-LAB09"
    paragraph = doc.add_paragraph("Repositorio: ")
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(
        qn("r:id"),
        doc.part.relate_to(repo_url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True),
    )
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    style = OxmlElement("w:rStyle")
    style.set(qn("w:val"), "Hyperlink")
    props.append(style)
    run.append(props)
    text = OxmlElement("w:t")
    text.text = repo_url
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    doc.save(REPORT / "informe.docx")
    with tempfile.TemporaryDirectory(prefix="lab09_lo_") as tmp:
        subprocess.run(
            [
                "libreoffice",
                f"-env:UserInstallation={Path(tmp).as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(REPORT),
                str(REPORT / "informe.docx"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
    pages = len(PdfReader(REPORT / "informe.pdf").pages)
    if pages > 5:
        raise AssertionError(
            f"El PDF tiene {pages} páginas; ajustar a cinco antes de entregar"
        )
    (REPORT / "verificacion.json").write_text(
        json.dumps(
            {"paginas_pdf": pages, "maximo": 5, "modelos_base": len(main)}, indent=2
        )
    )
    print(f"Informe exportado: {pages} páginas.", flush=True)


def build_report(panel, scale):
    main, results = build_tables_and_figures(panel, scale)
    create_docx(main, results)
