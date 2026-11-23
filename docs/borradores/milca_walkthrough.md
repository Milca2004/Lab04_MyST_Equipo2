# Walkthrough · Milca · `data.py`, `plots.py` y `tests/test_causality.py`

Recorrido del código de datos y EDA, las decisiones no evidentes y un mapa de preguntas probables. Cada nombre citado existe en el código (se comprobó con `ast`).

## Datos y EDA: `data.py`, `plots.py` y `tests/test_causality.py`

`data.py` carga y limpia el archivo, define los folds, calcula las 21 variables causales y el análisis del EDA. `plots.py` dibuja el EDA (una función por figura) y `write_eda_pdf` arma `artifacts/btcusdt_eda.pdf`. `python main.py` los llama en su etapa 1 y regenera `artifacts/feature_analysis_fold_{1..4}.csv` y el PDF, idénticos byte por byte entre corridas (el PDF no lleva fecha de creación en sus metadatos).

| Bloque | Qué hace | Decisión no evidente |
|---|---|---|
| `data.read_raw`, `data.clean_bars`, `data.load_clean` | Lee el CSV (primera línea con una URL, orden descendente, fechas con y sin fracción), redondea a la hora, elimina duplicados y calcula `gap_hours`. | Usa solo la columna `Date`, porque `Unix` no es confiable (en 672 filas de 2025 viene en microsegundos). Redondea hacia abajo (floor): el cierre se conoce a las 04:00 sin fuga; con ceil chocaría con la barra existente. Falla si dos barras distintas caen en la misma hora. No rellena huecos. |
| `data.Fold`, `data.get_folds` | Cuatro folds expansivos: train hasta 2021, 2022, 2023 y 2024; valid y test de un año cada uno (el test del fold 4 son 4 meses). | `Fold.split` corta con fechas, nunca con porcentajes. |
| `data.log_return_features`, `data.volatility_features`, `data.volume_features`, `data.trend_momentum_features`, `data.time_features`, `data.build_features` | Calculan las 21 variables en 6 familias y las devuelven junto con el mapa de familias. | Todo es causal (solo mira hacia atrás) y las ventanas cuentan barras observadas; las primeras 168 filas son NaN (calentamiento). El volumen se usa como razón (log1p), nunca en nivel. |
| `data.kalman_trend`, `data.belief_features` | Filtro de Kalman de nivel local sobre el log-retorno horario: media, desviación y razón de la tendencia. | q no se estima: k = 1 − 0.5^(1/24), q = k²/(1 − k). En un hueco solo predice, así que la incertidumbre sube. R_t es la varianza móvil de 24 barras. |
| `data.Normalizer` | `fit(train)` guarda mediana e IQR; `transform` normaliza y recorta a ±5. | Se ajusta solo con el train de cada fold. No escala las variables de tiempo (seno y coseno). |
| `data.gap_report`, `data.cleaning_summary` | Cifras de calidad: duplicados, horas faltantes y huecos por fold y periodo. | Solo cuenta barras y tiempos; no mira precios de valid ni de test. |
| `data.REGIMES`, `data.zigzag_turning_points`, `data.regime_threshold_table`, `data.characterize_regimes` | Regímenes del train del fold 4 por un zigzag de 60%, su tabla de umbrales y su caracterización. | Solo descriptivos. Se unió a mano el rebote de 2018. El zigzag no es causal. |
| `data.return_distribution_stats`, `data.autocorrelation_summary`, `data.stationarity_tests`, `data.stationarity_table` | Colas, autocorrelación, ADF y KPSS por fold (train). | ADF con `maxlag` 24 y AIC; KPSS con constante. El KPSS reporta como mínimo 0.01 y como máximo 0.10. |
| `data.build_targets`, `data.rejected_level_features`, `data.block_permutation`, `data.target_relations`, `data.feature_analysis`, `data.write_feature_analysis` | Relación de cada variable con el retorno futuro (Pearson, Spearman, información mutua y dos nulos) en el train de cada fold. | El objetivo es log(close[t+h] / open[t+1]), NaN si hay un hueco en el tramo. El nulo de bloques baraja bloques de 720 barras con semilla 42. |
| `plots.write_eda_pdf` | Arma las 9 hojas del EDA con la decisión de diseño que apoya cada una. | Cada título dice el fold y el periodo. |
| `tests/test_causality.py` | 12 pruebas con 4 índices (una barra de 2018, la barra tras el hueco de 145 h, la siguiente y la última). | Comparan la fila i calculada con datos hasta i contra la calculada con toda la serie, y alteran el futuro para comprobar que la fila i no cambia. |

## Mapa: pregunta probable → archivo y función

| Pregunta del profesor | Dónde está |
|---|---|
| ¿Cómo evitan usar el futuro en las variables? | `data.build_features` (todo causal); `tests/test_causality.py` |
| ¿Por qué redondean hacia abajo las 43 barras de 2018? | `data.clean_bars` (docstring del módulo) |
| ¿Qué hacen con los huecos? | `data.clean_bars` (`gap_hours`), `data.kalman_trend` (solo predice), `env.TradingEnv._first_exit` |
| ¿Con qué datos se normaliza? | `data.Normalizer.fit` con el train del fold; lo llama `env.load_split` |
| ¿Cómo definieron los regímenes y por qué 60%? | `data.zigzag_turning_points`, `data.REGIMES`, `data.regime_threshold_table` |
| ¿Por qué no usar el precio en nivel? | `data.stationarity_table`, `data.rejected_level_features`, `data.target_relations` (`mi_null_bloques`) |
| ¿De dónde salen q y R del filtro de Kalman? | `data.kalman_trend` (q = k²/(1 − k)) y `data.belief_features` (R_t = varianza móvil de 24 barras) |
