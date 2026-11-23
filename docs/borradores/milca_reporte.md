# Borrador de reporte · Milca · Datos, EDA, estado y filtro de Kalman

Las cifras salen de código que corrió el 2026-11-23 (semilla 42): `data.py` y `artifacts/feature_analysis_fold_{1,2,3,4}.csv`. Son de **train**, salvo la tabla de ventanas del fold 4 (valid y test), que solo cuenta barras y horas, sin precios ni retornos. Cubre las preguntas 1 y 2 y las solicitudes de Arturo de `docs/borradores/arturo_solicitudes_milca.md`.

**Fuentes:** `code/data.py`, `code/plots.py` (figuras 2 y 3 se regeneran con `plots.plot_price_regimes` y `plots.plot_feature_information`; el PDF completo es `artifacts/btcusdt_eda.pdf`) y `tests/test_causality.py`.

## 1. Datos y limpieza

Archivo de Canvas con barras horarias de BTCUSDT, del 2017-08-17 04:00 al 2026-04-26 23:00 UTC. Lo limpia `data.py` y deja una barra por hora, con una columna `gap_hours` que dice cuántas horas faltan justo antes de cada barra.

- **Filas:** el archivo trae 75,588; el enunciado dice 75,558 (30 menos) [PENDIENTE: confirmar con el profesor que el archivo de Canvas es el correcto].
- **Duplicados:** 506 filas repiten fecha y OHLCV de otra fila (todas en 2025; solo cambia la escala de la columna `Unix`, que no usamos). Se eliminan y quedan **75,082 barras únicas**.
- **Hora fuera de punto:** 43 barras de febrero de 2018 traen minutos y segundos. Se redondean hacia abajo a la hora: la barra marcada 03:28:14 cierra en 7,976.74 y la de las 04:00 abre en 7,976.74, así que es la barra de 03:00 a 04:00 y su cierre se conoce a las 04:00 (sin mirar al futuro).
- **Huecos:** no se rellenan. Contra una malla horaria completa de 76,196 horas faltan 1,114, repartidas en 376 huecos. Desde abril de 2025 falta casi a diario la barra de las 00:00 UTC (230 huecos de 1 h en 2025 y 103 en 2026), y hay huecos largos (el mayor, 145 h en julio de 2025).
- **Volumen cero:** 4 barras, que se conservan.

| Periodo (fold 4) | Barras | Horas faltantes | Barras con hueco previo | Hueco más largo (h) |
|---|---|---|---|---|
| Train | 64,525 | 127 | 28 | 32 |
| Valid | 8,073 | 687 | 240 | 145 |
| Test | 2,484 | 300 | 108 | 73 |

**Folds walk-forward.** Cuatro folds expansivos, cada uno con partición cronológica train / valid / test y sin mezclar épocas. Todo lo que se estima o se elige (normalización, hiperparámetros, configuración) usa solo train o valid; el test se toca una sola vez.

| Fold | Train | Valid | Test |
|---|---|---|---|
| 1 | 2017-08 a 2021-12 (38,222 barras) | 2022-01 a 2022-12 (8,760 barras) | 2023-01 a 2023-12 (8,759 barras) |
| 2 | 2017-08 a 2022-12 (46,982 barras) | 2023-01 a 2023-12 (8,759 barras) | 2024-01 a 2024-12 (8,784 barras) |
| 3 | 2017-08 a 2023-12 (55,741 barras) | 2024-01 a 2024-12 (8,784 barras) | 2025-01 a 2025-12 (8,073 barras) |
| 4 | 2017-08 a 2024-12 (64,525 barras) | 2025-01 a 2025-12 (8,073 barras) | 2026-01 a 2026-04 (2,484 barras) |

*Figura 1. Los cuatro folds. Cada fold agrega un año al train y avanza valid y test un año (el test del fold 4 son 4 meses).* (se dibuja en el PDF)

## 2. Pregunta 1 · Regímenes del mercado

Partimos el train del fold 4 (2017-08 a 2024-12, que contiene a los demás trains; el precio pasa de 4,309 a 93,576 USDT) con un **zigzag**: un máximo se confirma cuando el precio cae 60% desde él, y un mínimo cuando sube 60%. Se calcula sobre el cierre diario (UTC) con `zigzag_turning_points`. Sale una secuencia que alterna alcista y bajista.

*Figura 2. Precio de cierre (escala log) y regímenes R1–R7 · train del fold 4 · verde = alcista, naranja = bajista.* (se dibuja en el PDF)

|  | Régimen | Fechas | Retorno total | Volatilidad anualizada | Drawdown máximo | Curtosis en exceso |
|---|---|---|---|---|---|---|
| R1 | Burbuja 2017 | 2017-08-17 a 2017-12-16 | +343% | 130% | -41% | 12.1 |
| R2 | Invierno cripto 2018 | 2017-12-17 a 2018-12-15 | -83% | 107% | -84% | 13.8 |
| R3 | Recuperación 2019 | 2018-12-16 a 2019-06-26 | +307% | 63% | -19% | 25.9 |
| R4 | Corrección y COVID | 2019-06-27 a 2020-03-12 | -64% | 75% | -64% | 127.8 |
| R5 | Ciclo alcista 2020-21 | 2020-03-13 a 2021-11-08 | +1351% | 82% | -55% | 27.0 |
| R6 | Bajista 2022 | 2021-11-09 a 2022-11-21 | -77% | 66% | -77% | 9.6 |
| R7 | Recuperación y ETF 2023-24 | 2022-11-22 a 2024-12-31 | +490% | 47% | -32% | 14.3 |

**Qué dicen las cifras.** Ningún régimen representa a todo el train: la volatilidad anualizada va de 47% a 130%, el drawdown llega a -84% y la curtosis en exceso va de 9.6 a 127.8. Por eso el estado del agente incluye volatilidad y una creencia de tendencia con su incertidumbre, y cada fold se normaliza solo con su train.

**Por qué 60%.** Con 40% el zigzag marca 16 giros y con 50%, 10: el ciclo 2020-21 se parte en la corrección de abril a julio de 2021. Con 70% marca 4 y se pierden la recuperación de 2019 y la caída del COVID. Con 60% marca 8 giros; unimos a mano el rebote de un mes de 2018 (2018-02-05 a 2018-03-04), que el zigzag sí marca. Separa ciclos de mercado y no correcciones intermedias.

| Umbral | Giros (train fold 4) | Regímenes (train fold 4) | Confirmados con solo train del fold 1 | …fold 2 | …fold 3 | …fold 4 |
|---|---|---|---|---|---|---|
| 40% | 16 | 17 | 14 | 15 | 16 | 16 |
| 50% | 10 | 11 | 8 | 9 | 10 | 10 |
| 60% | 8 | 9 | 6 | 7 | 8 | 8 |
| 70% | 4 | 5 | 2 | 3 | 4 | 4 |

> **Los regímenes son solo descriptivos.** No entran al estado del agente ni a ninguna decisión. El zigzag no es causal: un giro solo se conoce meses después, cuando el precio ya se alejó. Las fronteras salen del train del fold 4, que incluye valid y test del fold 1; por eso mostramos que el train del fold 1 (hasta 2021-12-31) ya contiene completos R1 a R5 (el último termina el 2021-11-08) y que con solo ese train se confirman 6 giros a 60%.

## 3. Pregunta 2 · Estado del agente y alternativa rechazada

El estado de información tiene **21 variables causales** en 6 familias, todas calculadas solo con datos hasta la barra de decisión (lo prueba `tests/test_causality.py`): **retornos** (`ret_1`, `ret_4`, `ret_24`, `ret_168`), **volatilidad** (`vol_24`, `vol_168`, `range_1`, `parkinson_24`), **volumen** (`volume_ratio_24`, `volume_ratio_168`, `trades_ratio_24`), **tendencia/momentum** (`dist_ema_24`, `dist_ema_168`, `rsi_14`), **tiempo** (seno y coseno de la hora y del día) y **creencia** (`trend_mean`, `trend_std`, `trend_snr`, del filtro de Kalman). Cada fold las normaliza con la mediana y el IQR de su train y recorta a ±5.

**Por qué queda así** (train del fold 4):

- **Colas gordas:** curtosis en exceso 34.0 y 0.95% de las horas superan 4 desviaciones (una normal daría 0.0063%). Por eso se normaliza con mediana e IQR y se recorta a ±5, no con media y desviación.
- **Dirección contra magnitud:** la autocorrelación del retorno es pequeña (máximo |ACF| = 0.044) mientras la de |retorno| supera la banda de 95% en 168 de 168 rezagos (rezago 1 = 0.31). No usamos 168 retornos crudos; usamos pocos retornos resumidos (1, 4, 24 y 168 h) más variables de volatilidad.
- **Hora del día:** el |retorno| medio va de 0.39% a 0.60% según la hora (×1.5); por eso hay familia de tiempo.

**Alternativa rechazada: variables en nivel** (precio, log-precio y volumen sin transformar). La tabla muestra que el precio no es estacionario en ningún fold (el ADF no rechaza la raíz unitaria y el KPSS rechaza la estacionariedad):

| Serie (train) | Fold 1: ADF p / KPSS p | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| Precio (nivel) | 0.87 / ≤ 0.01 | 0.56 / ≤ 0.01 | 0.63 / ≤ 0.01 | 0.98 / ≤ 0.01 |
| Log-precio | 0.83 / ≤ 0.01 | 0.51 / ≤ 0.01 | 0.59 / ≤ 0.01 | 0.76 / ≤ 0.01 |
| Retorno horario | < 0.001 / ≥ 0.10 | < 0.001 / ≥ 0.10 | < 0.001 / ≥ 0.10 | < 0.001 / ≥ 0.10 |
| Volatilidad (24 h) | < 0.001 / ≤ 0.01 | < 0.001 / ≤ 0.01 | < 0.001 / ≤ 0.01 | < 0.001 / ≤ 0.01 |

El ADF prueba que hay raíz unitaria (un p-valor alto no la rechaza); el KPSS prueba que la serie es estacionaria (un p-valor bajo la rechaza). El retorno pasa las dos pruebas en los 4 folds. El KPSS reporta como mínimo 0.01 y como máximo 0.10.

**El nivel sí tiene información mutua alta con el retorno a 24 h, y eso hay que explicarlo.** Para separar información real de "la variable solo identifica la época" comparamos la MI con dos nulos: el objetivo barajado hora por hora y el objetivo barajado por bloques de 30 días (cada bloque conserva su volatilidad y tendencia lenta, pero cae en otra época).

| Variable (h = 24, train) | MI con el objetivo | Nulo hora por hora | Nulo de bloques de 30 días | % de la MI que conserva el nulo de bloques |
|---|---|---|---|---|
| log_close | 0.176–0.186 | 0.000–0.006 | 0.095–0.133 | 54–74% |
| close | 0.152–0.174 | 0.000–0.001 | 0.089–0.121 | 55–79% |
| vol_24 | 0.094–0.106 | 0.000–0.000 | 0.034–0.054 | 36–51% |
| dist_ema_168 | 0.051–0.058 | 0.000–0.000 | 0.016–0.026 | 31–44% |
| ret_24 | 0.036–0.044 | 0.000–0.001 | 0.003–0.018 | 7–50% |

Cada rango es el mínimo y el máximo entre los 4 folds. El log-precio conserva entre 54–74% de su MI con el nulo de bloques, contra 31–44% de `dist_ema_168`: una parte grande de la MI del nivel viene de la época. **La prueba respalda el rechazo solo en parte** (`vol_24` conserva 36–51%, un valor parecido): un nulo de bloques de 30 días es exigente y le quita MI a cualquier variable lenta. El rechazo se apoya sobre todo en que el nivel **no es estacionario**.

*Figura 3. Información mutua máxima por familia con el retorno futuro a 1, 4 y 24 h · train del fold 4 · la marca negra es el nulo de bloques de 30 días de esa misma variable.* (se dibuja en el PDF)

### Volatilidad y otras variables muy persistentes que sí entran al estado

En el train del fold 1, `vol_24`, `vol_168`, `range_1`, `parkinson_24` y `trend_std` salen "no estacionarias" solo porque el KPSS rechaza (p = 0.01, el mínimo de la tabla). El ADF sí rechaza la raíz unitaria (p < 0.001): son series con memoria muy larga que vuelven a su nivel con lentitud, no caminatas aleatorias. Las dejamos en el estado porque son información de **costo**, no de dirección: las barreras miden ±vol·√H, así que la volatilidad decide si el costo de ida y vuelta (0.25%) cabe en el movimiento.

| Variable (train fold 1) | ADF p | KPSS p | ACF rezago 1 | MI h=1 | Nulo hora h=1 | Nulo bloques h=1 | MI h=24 | Nulo bloques h=24 |
|---|---|---|---|---|---|---|---|---|
| vol_24 | < 0.001 | ≤ 0.01 | 0.991 | 0.121 | 0.000 | 0.006 | 0.106 | 0.054 |
| vol_168 | < 0.001 | ≤ 0.01 | 0.999 | 0.105 | 0.000 | 0.007 | 0.181 | 0.115 |
| range_1 | < 0.001 | ≤ 0.01 | 0.636 | 0.097 | 0.005 | 0.012 | 0.057 | 0.005 |
| parkinson_24 | < 0.001 | ≤ 0.01 | 0.995 | 0.126 | 0.000 | 0.008 | 0.108 | 0.049 |
| trend_std | < 0.001 | ≤ 0.01 | 0.999 | 0.114 | 0.000 | 0.007 | 0.110 | 0.049 |

A 1 h la MI es 0.097 a 0.126, contra un nulo hora por hora ≤ 0.005 y un nulo de bloques ≤ 0.012: esa información no es solo de época. A 24 h una parte sí lo es (por ejemplo, `vol_168` conserva 63% de su MI con el nulo de bloques). **Riesgo que declaramos:** el normalizador del train queda descalibrado si valid o test tienen una escala de volatilidad distinta, y recortamos a ±5.

## 4. Filtro de Kalman de la tendencia (componente de creencia B_t)

Estimamos una tendencia latente μ_t, el log-retorno horario esperado, con un modelo de nivel local: **estado** μ_t = μ_{t−1} + η con Var(η) = q·R_t por hora de reloj, y **observación** y_t = μ_t + ε con Var(ε) = R_t. Aquí y_t es el log-retorno horario y R_t = vol_24² es la varianza móvil de las últimas 24 barras.

**Cómo se fijan q y R.** R_t se mide con los datos (varianza móvil de 24 barras). q no se estima: se fija para que la ganancia estacionaria sin huecos, k, olvide la mitad de la información en 24 horas, es decir, (1 − k)^24 = 0.5. De ahí k = 1 − 0.5^(1/24) = **0.02847** y q = k²/(1 − k) = **0.000834**. Una ganancia de 2.8% por hora hace al filtro lento: confía mucho en su estimación previa y poco en cada retorno, porque el retorno de una hora es casi todo ruido.

**Huecos.** En una barra con hueco solo se hace la predicción: la media no cambia y la varianza crece q·R_t por cada hora faltante, porque el retorno de esa barra abarca varias horas en las que μ pudo moverse. Así el hueco sube la incertidumbre en lugar de rellenar datos. Pasa en los 376 de 376 huecos; en el de 145 h (2025-07-22) la desviación posterior de la tendencia sube de 0.00079 a 0.00222 (×2.8).

En la formulación de Powell, μ̂_t y σ̂_t son **creencia**, no estado físico: son lo que el sistema estima sobre la deriva del precio, junto con su incertidumbre. La razón μ̂_t/σ̂_t (`trend_snr`) es la tercera variable de la familia.

## 5. Limitaciones de datos y variables

**Las ventanas cuentan barras, no horas.** Las variables de 24 y 168 "horas" son ventanas de 24 y 168 barras observadas. Como desde abril de 2025 falta casi a diario la barra de las 00:00 UTC (230 huecos de 1 h en 2025 y 103 en 2026 hasta abril), esas ventanas abarcan más reloj en valid y test del fold 4:

| Periodo (fold 4) | Barras | Horas de reloj de la ventana de 24 barras (media) | Filas con ventana > 24 h | Horas de reloj de la ventana de 168 barras (media) | Filas con ventana > 168 h |
|---|---|---|---|---|---|
| Train | 64,525 | 24.05 | 1.0% | 168.33 | 6.8% |
| Valid | 8,073 | 25.96 | 67.9% | 182.15 | 67.9% |
| Test | 2,484 | 26.78 | 100.0% | 188.17 | 100.0% |

El efecto es moderado (hasta un 12% más de reloj en el test del fold 4) y no cambiamos las variables para no invalidar el entrenamiento ya hecho. La corrección sería usar ventanas por tiempo (`rolling("24h")`). Esta tabla solo cuenta barras y tiempos; no usa precios ni retornos de valid ni de test.

**Otras limitaciones:** (1) las fronteras de los regímenes salen del train del fold 4, que incluye valid y test del fold 1, y por eso son solo descriptivas; (2) los datos tienen una sola serie de BTC, así que cuatro folds con trains que se traslapan no son cuatro experimentos independientes; (3) probamos muchas configuraciones y variables sobre los mismos datos, lo que es un riesgo de *data snooping* (la corrección formal sería un Deflated Sharpe Ratio o un Reality Check, que no aplicamos).

## 6. Estado de las solicitudes de Arturo

| Solicitud | Estado | Dónde |
|---|---|---|
| 1. Cargador del S&P 500 | Hecho (PR #3) | `data/sp500_daily.csv` (FRED, serie `SP500`); `evaluate.load_sp500` lo lee; fuente declarada en el README. Pendiente la confirmación del profesor. |
| 2. Textos de las preguntas 1 y 2 | Hecho | Este archivo, secciones 2 y 3. |
| 3. Merge a `main` | Hecho (PRs #2 y #4) | `main` ya contiene los commits de los tres. |
| 4. Variables no estacionarias en el estado | Opción B (argumento escrito, sin cambiar la observación) | Sección 3, "Volatilidad y otras variables muy persistentes". |
| 5. Rechazo del precio en nivel con un nulo de bloques | Hecho | `data.target_relations` (columnas `mi_null_bloques_h1/h4/h24` de `feature_analysis_fold_{k}.csv`) y hoja 8 del EDA. Respalda el rechazo solo en parte. |
| 6. Regímenes por fold y umbral de 60% | Hecho | `data.zigzag_turning_points`, `data.regime_threshold_table` y hoja 2 del EDA. |
| 7. Diferencia de filas (75,588 contra 75,558) | Pendiente con el profesor | Sección 1; está documentada en el docstring de `data.py`. |
| 8. Justificar q y R del filtro | Hecho | Sección 4. |
| 9. Ventanas por tiempo | Declarada como limitación, sin cambiar código | Sección 5, con la medición en el fold 4. |
| 10. Creencia de régimen con un HMM | No se hizo (opcional) | Cambiaría la observación que ya entrenó Paula. |
