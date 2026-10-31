# Investigación técnica y decisiones del laboratorio

Estos apuntes documentan la implementación existente de A y la ampliación de B. Las
interfaces concretas se verificaron también en el código instalado: PyTorch 2.9.1,
NeuralForecast 3.3.0 y Chronos Forecasting 2.3.2. Las versiones del entorno CPU quedan
en `requirements-cpu-lock.txt`; no se supone una GPU T4 cuando la ejecución fue CPU.

## LSTM, GRU y estrategias de salida

Con `batch_first=True`, LSTM recibe `(B,L,F)` y devuelve `output: (B,L,D*hidden)`
y `(h_n,c_n)`, donde los estados finales tienen ejes capas/direcciones, batch y hidden.
GRU conserva una sola familia de estados, `h_n`. En este trabajo todas las redes
propias son unidireccionales: no se introducen observaciones futuras del horizonte.

`LSTMMultiOutput` toma el último estado y el calendario futuro, y emite H valores.
`LSTMAutoregressive` se ajusta con objetivos de una hora y, durante el pronóstico,
incorpora la predicción y el calendario del paso correspondiente antes de continuar.
`Seq2Seq` tiene encoder y decoder separados: el decoder recibe el valor anterior y
el calendario de la hora que está prediciendo.

En teacher forcing se usa la etiqueta del paso anterior como entrada del decoder,
exclusivamente al entrenar. Nuestra probabilidad es 0.5, con máscara por muestra.
Durante validación y prueba no se entregan etiquetas al decoder. La prueba
automatizada modifica los targets de evaluación y exige predicciones idénticas.
La distribución de entradas de train y de inferencia puede diferir (exposure bias).

Fuentes primarias: [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html),
[tutorial seq2seq y teacher forcing](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

## Conv1d, causalidad y campo receptivo

`Conv1d` opera sobre `(B,C,T)`. Para que una salida no dependa de horas posteriores
se usa relleno izquierdo o relleno simétrico seguido de recorte derecho. Las
dilataciones amplían el campo receptivo sin que el número de parámetros crezca
proporcionalmente al contexto. Con stride 1, kernel k y dilataciones d_i:

`R = 1 + sum_i (k_i - 1) * d_i`.

La TCN instalada usa una convolución causal por nivel en
`TemporalConvolutionEncoder`, no dos. Para kernel 3 y d=[1,2,4,8,16,32], R=127.
L=168 permite construir ese contexto, pero una salida del encoder no mezcla por sí
sola las 168 posiciones. El adaptador/contexto y el decoder también afectan qué
estados consume el pronóstico. No se aplica automáticamente la fórmula de dos
convoluciones por bloque de otra implementación de TCN.

Fuentes: [Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html),
[módulos de NeuralForecast](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_modules.py),
[TCN](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/tcn.py).

## Evaluación temporal en Nixtla

La validación cruzada temporal usa orígenes móviles. `h=24`, `n_windows=92`,
`step_size=24` y `refit=False` mantienen los mismos cortes diarios y pesos fijos.
`cutoff` es la última observación, por eso nuestro `origin=cutoff+1h`.
El adaptador de cada modelo recibe solo `hist[ds < origin]`.

La comprobación ejecutada compara SeasonalNaive24 con StatsForecast en un cliente
del panel, los 92 orígenes y todos los valores predichos, con tolerancia cero. La
validación estructural se aplica a los archivos de los 150 clientes: 331,200 filas,
92 orígenes, 24 horas por cliente/origen y ausencia de claves duplicadas.

El early stopping solo observa julio–septiembre. En el adaptador de B se guarda en
RAM el estado de mínimo loss de validación y se restaura al terminar. El refit se
hace desde una inicialización nueva, con train+validación y el número de pasos
elegido. La normalización de las redes mantiene estadísticas del train original.
TCN, DeepAR y los experimentos posteriores usan stride de validación 24, sin
modificar el stride 1 del entrenamiento. N-HiTS se ajustó primero con validación
horaria; su comparación externa también usa los 92 orígenes diarios.

El LightGBM heredado ajusta su normalizador con todo el bloque que recibe en
`fit_lightgbm`: train para selección, train+validación para el ajuste final, e
historia disponible para cada actualización mensual. Es un ajuste causal, pero
difiere del acuerdo inicial de conservar la media/std del train original para
todas las familias. Se conserva la implementación de A y se explicita la diferencia.

MSTL, también heredado de A, se vuelve a ajustar dentro de cada llamada al
forecast_fn sobre las últimas ocho semanas. Su tiempo queda incluido en inferencia,
con train_s=0 en el registro original. No equivale al refit=False de la comprobación
de SeasonalNaive24 ni a los pesos fijos de las redes base.

Fuentes: [StatsForecast core](https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/core.py),
[NeuralForecast core](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/core.py).

## Atención y codificación posicional

`TransformerEncoderLayer` incluye atención propia, conexión residual, normalización
y red densa por posición. La atención sin posición no conoce el orden temporal;
pueden sumarse codificaciones sinusoidales o embeddings aprendidos. PatchTST
agrupa segmentos contiguos en parches y aplica atención sobre esos tokens.
iTransformer invierte los ejes habituales: los tokens representan series/clientes,
con su historia temporal proyectada a embeddings. Por ello exige el mismo conjunto
de 150 series y la misma alineación temporal también al inferir.

Fuentes: [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html),
[PatchTST](https://arxiv.org/abs/2211.14730), [iTransformer](https://arxiv.org/abs/2310.06625).

## Pérdida pinball, DeepAR y cobertura

Para residuo `u=y-q_tau`, la pinball loss es `max(tau*u,(tau-1)*u)`.
La mediana minimiza error absoluto; cuantiles 0.1 y 0.9 forman un intervalo nominal
de 80%. Se reportan pérdidas por cuantil y promedio, además de la proporción de y
observados dentro del intervalo. Cobertura alta sin considerar el ancho no implica
un buen pronóstico; intervalos excesivamente amplios pueden cubrir casi todo.

DeepAR parametriza una distribución mediante una recurrente y obtiene cuantiles
de trayectorias simuladas. Se usa Normal en z-scores, que pueden ser negativos;
no corresponde usar directamente una distribución restringida a valores positivos
después de un z-score. Una Normal también puede asignar masa a consumos negativos
al desnormalizar; no se recortan silenciosamente las predicciones.

Fuente: [DeepAR](https://arxiv.org/abs/1704.04110),
[pérdidas de NeuralForecast](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/losses/pytorch.py).

## Chronos-Bolt

`BaseChronosPipeline.from_pretrained` selecciona el pipeline a partir del checkpoint.
`predict_quantiles(context, prediction_length=24, quantile_levels=[.1,.5,.9])`
devuelve cuantiles `(S,H,3)` y una salida puntual. El contexto es `(S,L)`, sin
calendario ni etiquetas futuras. Elegimos q50 como pronóstico puntual y guardamos
q10/q50/q90; sus unidades ya son las del contexto original. Como las cabezas
de Bolt pueden cruzar cuantiles, se aplica reordenamiento monótono de los tres
cuantiles por cliente y hora antes de calcular pinball y cobertura; este
postprocesamiento se registra explícitamente en los resultados.

La ejecución CPU usa `amazon/chronos-bolt-small`, alternativa expresamente prevista
en el plan, y no hace fine-tuning. Sus pesos preentrenados cuentan como parámetros
totales, pero hay cero parámetros entrenados en este laboratorio. Tiempo de carga,
inferencia y entrenamiento se registran por separado. No afirmamos que el conjunto
Electricity estuviera excluido del preentrenamiento: esa contaminación externa no
se detecta cortando correctamente la historia en nuestro backtest.

Fuentes: [modelo de Amazon](https://huggingface.co/amazon/chronos-bolt-small),
[implementación de predict_quantiles](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py),
[Chronos](https://arxiv.org/abs/2403.07815).

## Local, global y RevIN

Un modelo local tiene pesos propios por cliente; uno global comparte pesos entre
clientes y puede aprovechar patrones transferibles. La comparación usa diez IDs
fijados con seed=42 y las mismas fechas de validación. El global recibe 150 series
en entrenamiento y el local solo una por ajuste: es una comparación de esos
procedimientos, con tamaños de información y costos distintos, no una intervención
que mantenga todo lo demás constante.

RevIN obtiene media y desviación de cada ventana de entrada, normaliza y después
revierte la salida con esos mismos valores. No necesita estadísticas futuras.
El z-score por cliente se ajusta exclusivamente hasta junio; RevIN es una operación
adicional por ventana. Se evita interpretar el uso de ambas normalizaciones como
dos ajustes sobre toda la serie.

Fuente: [artículo y código RevIN](https://github.com/ts-kim/RevIN).

## MASE frente a WAPE

MASE divide el MAE de cada cliente entre su MAE estacional de 24 h calculado en
train y luego promedia entre clientes. WAPE suma errores absolutos sobre todos los
clientes y divide por consumo absoluto total, de modo que clientes grandes pesan
más. MASE<1 significa superar la escala estacional **de train**; para afirmar que
un modelo supera al seasonal naive **en test**, se comparan sus métricas en test.
MAPE excluye horas de consumo absoluto menor o igual a 0.001 y registra cuántas.

La semana navideña y los desgloses por horizonte/hora se evalúan en kWh, después
de revertir cualquier normalización. Seleccionar clientes completos usa la
disponibilidad observada del periodo entero, una limitación de selección del panel
que se conserva para reproducir A.

Fuente: [Forecasting: Principles and Practice, accuracy](https://otexts.com/fpp3/accuracy.html).

## Reparto y entrega

Se mantienen los archivos y resultados originales de Diego. La ampliación de Cisco
incluye recurrentes, N-HiTS/TCN/DeepAR/Chronos, experimentos, comparación, notebooks
y documento final. El nombre de una herramienta en PLAN.md no implica que esté
instalada: no se encontró la skill `formato-tarea-docx`; se generó el documento con
python-docx y se exportó con LibreOffice, verificando su número de páginas.
