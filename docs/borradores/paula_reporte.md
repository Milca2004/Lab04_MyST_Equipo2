# Borrador de Paula: entrenamiento, evaluación y resultados

Rama `paula/entrenamiento`. Todas las cifras salen de los CSV de `artifacts/` generados por `python main.py` (semilla 42). Cada cifra indica si es de **train**, **valid** o **test**.

**Resumen en una línea:** ningún agente supera a comprar y mantener BTC ni gana dinero después de costos en test. Sí pierden menos que el agente aleatorio y que "siempre largo", pero su ventaja bruta por operación no es estadísticamente distinta de cero. Esto confirma el diagnóstico de Arturo sobre el train (t = 0.17 en el fold 1).

---

## 1. Entrenamiento (sección 3.5)

### 1.1 Procedimiento

- **Algoritmos:** PPO y DQN de Stable-Baselines3 2.9.2, ambos con `MlpPolicy` y `Discrete(3)`.
- **DQN:** siempre usa `SMDPReplayBuffer` con `gamma_per_hour = 0.999`, es decir, descuento 0.999^τ con τ en horas reales.
- **Un modelo por (fold, algoritmo, configuración, semilla).** Ningún modelo se reutiliza entre folds.
- **Episodio de train:** ventana de 720 h con inicio al azar (diseño de Arturo). En valid y test se usa la partición completa.
- **Sin `VecNormalize`:** las observaciones ya vienen normalizadas por el `Normalizer` de Milca, ajustado solo con el train de cada fold. La recompensa ya está escalada (100·log).
- **Hiperparámetros fijos (no se buscan):**
  - PPO usa los valores por defecto de SB3.
  - DQN usa `buffer_size = 100k`, `learning_starts = 1,000`, `target_update_interval = 1,000` y `exploration_fraction = 0.2`.
  - **Por qué no se usan los valores por defecto de DQN:** están pensados para Atari con millones de pasos. Con 100k pasos, la red objetivo solo se actualizaría 10 veces.
- **Metadatos:** cada corrida guarda en `artifacts/run_metadata.csv` las versiones de Python, SB3, Gymnasium, torch y numpy, además de la semilla, los hiperparámetros, la fecha y la duración. Las versiones son Python 3.12, SB3 2.9.2, Gymnasium 1.4.0 y torch 2.15.0.

### 1.2 Búsqueda de hiperparámetros (solo con valid)

**Rejilla (fijada antes de correr):**
- `learning_rate` ∈ {1e-4, 3e-4, 1e-3}.
- Número de pasos ∈ {25k, 50k, 100k}.
- 3 semillas (42, 43, 44).
- Total: 4 folds × 2 algoritmos × 3 lr × 3 semillas = **72 corridas**.

**Cómo se evalúan los pasos:** no requieren corridas extra. Cada corrida guarda el modelo y lo evalúa en valid al llegar a 25k, 50k y 100k pasos. Eso da 216 evaluaciones, todas en `validation_results_fold_{k}.csv`.

**Criterio de selección (fijado antes de correr):**
- Se elige la **mediana entre semillas del log del capital final en valid**, log(C_T/C_0).
- Es la misma cantidad que optimiza la contribución terminal (100·log del cambio de capital). Así se selecciona con lo que se entrena.
- Desempate: el mayor mínimo entre semillas.

**Cómputo:** con 4 procesos en paralelo, la búsqueda tardó 20 min. El pipeline completo (búsqueda, semillas finales, ablación, test y sensibilidad) tarda unos 18 min si se reutiliza la búsqueda guardada en `models/`.

**Configuración elegida por fold** (`hyperparameter_selection.csv`, valid, 3 semillas):

| Fold | Algoritmo | lr | Pasos | Mediana log C_T/C_0 | Mín – máx |
|---|---|---|---|---|---|
| 1 | PPO | 1e-3 | 25k | −0.33 | −0.90 a −0.31 |
| 1 | DQN | 1e-4 | 25k | −0.98 | −1.38 a −0.94 |
| 2 | PPO | 3e-4 | 100k | −0.38 | −1.13 a +0.01 |
| 2 | DQN | 1e-3 | 100k | −0.63 | −1.65 a −0.59 |
| 3 | PPO | 1e-3 | 25k | −0.49 | −0.74 a −0.20 |
| 3 | DQN | 1e-4 | 25k | −0.19 | −0.52 a −0.17 |
| 4 | PPO | 3e-4 | 25k | −0.52 | −1.41 a −0.51 |
| 4 | DQN | 1e-4 | 25k | −0.80 | −1.48 a −0.66 |

**Lectura de la rejilla:**
- **Todas las configuraciones pierden en valid.** Como referencia, Arturo midió en valid −1.34 para el agente aleatorio y −1.44 para "siempre largo" (promedio de folds, `sensitivity_baselines.csv`).
- **Óptimos en el borde de la rejilla:** en 8 de 8 casos el número de pasos elegido es un extremo, y en 6 de 8 lo es el lr.
  - Lo más frecuente son **25k pasos**, el mínimo. Entrenar más empeora valid.
  - Las curvas de entrenamiento lo explican: entre 25k y 100k pasos la recompensa por episodio de **train** sube (en el fold 1, PPO pasa de ≈0 a +10/+40 por ventana de 720 h), mientras la de valid baja. Es sobreajuste al train, no aprendizaje de una regularidad.
  - Un óptimo en el borde indica que el mejor punto podría estar fuera de la rejilla: menos pasos, que se acerca a "no operar".
  - **No ampliamos la rejilla** para no seguir buscando sobre valid.

**Algoritmo primario (elegido con valid):** **PPO**, con una media entre folds de la mediana de valid de −0.43 contra −0.65 de DQN. En test resultó al revés (sección 2). Lo reportamos tal cual: cambiar de algoritmo después de ver el test invalidaría el protocolo.

### 1.3 Modelos finales y dispersión entre semillas

- **Semillas por modelo final:** 5 (42 a 46). Las 3 de la búsqueda se reutilizan y se entrenaron 2 más con la configuración elegida: 16 corridas.
- **Modelo representativo:** la semilla con la mediana de valid entre las 5 (`final_models.csv`). Ese modelo genera la bitácora, la curva de capital y la sensibilidad.
- **Dispersión en test entre las 5 semillas** (`test_seed_dispersion.csv`, log C_T/C_0):

| Fold | PPO media ± desv (mín – máx) | DQN media ± desv (mín – máx) |
|---|---|---|
| 1 | −0.60 ± 0.13 (−0.73 a −0.40) | −0.84 ± 0.30 (−1.35 a −0.63) |
| 2 | −0.91 ± 0.48 (−1.59 a −0.42) | −1.21 ± 0.66 (−1.84 a −0.26) |
| 3 | −1.34 ± 0.09 (−1.45 a −1.24) | −0.66 ± 0.39 (−1.19 a −0.22) |
| 4 | −0.37 ± 0.21 (−0.66 a −0.13) | −0.17 ± 0.22 (−0.39 a **+0.17**) |

- De 40 modelos evaluados en test, **solo uno termina con ganancia**: DQN, fold 4, +0.17 de log. Una ganancia en 40 intentos es lo que se espera por azar, no evidencia de ventaja.
- Arturo ya había visto en train que DQN iba de 0.11× a 2.93× según la semilla. En test la dispersión es menor porque todos los modelos aprendieron a operar menos, pero sigue siendo del mismo orden que la diferencia entre algoritmos.

---

## 2. Resultados en test (secciones 3.6 y 3.7)

`metrics_summary.csv`: por fold y agregados con media y desviación entre los 4 folds. Para los modelos de RL se usa el representativo de cada fold.

**Convenciones:**
- Retornos diarios simples, anualizados con 365 días porque cripto opera todos los días. El S&P 500 se anualiza con 252.
- Retorno anualizado con días de calendario.
- Sharpe y Sortino con tasa libre de riesgo cero.
- **Alfa:** retorno anualizado de la estrategia menos el del benchmark.
- **Rotación:** nocional operado (entrada más salida) dividido entre el capital medio y entre los años.
- **Fold 4:** su test son solo 4 meses (enero a abril de 2026), así que sus cifras anualizadas amplifican el resultado.

**Media (desviación) entre los 4 folds de test:**

| Métrica | PPO (primario) | DQN | Comprar y mantener BTC | Aleatoria | Siempre largo | Siempre fuera |
|---|---|---|---|---|---|---|
| Retorno total | −51.4% (19.3) | −28.1% (17.5) | +64.2% (85.8) | −59.8% (26.4) | −58.9% (17.4) | 0% |
| Retorno anualizado | −60.2% (9.4) | −31.9% (12.7) | +59.2% (91.7) | −68.0% (12.5) | −68.8% (11.7) | 0% |
| Volatilidad anualizada | 37.8% (2.3) | 35.7% (7.0) | 48.2% (6.1) | 43.0% (6.4) | 46.5% (7.4) | 0 |
| Sharpe | −2.31 (0.64) | −0.97 (0.58) | 0.93 (1.32) | −2.75 (1.57) | −2.38 (0.76) | — |
| Sortino | −3.02 (0.67) | −1.37 (0.85) | 1.60 (2.23) | −3.34 (1.69) | −3.05 (0.79) | — |
| Drawdown máximo | −56.9% (17.9) | −40.3% (19.1) | −28.3% (6.7) | −63.0% (25.0) | −64.4% (15.6) | 0 |
| Calmar | −1.16 (0.48) | −0.93 (0.49) | 2.81 (4.09) | −1.23 (0.54) | −1.14 (0.45) | — |
| Hit rate | 45.2% (1.7) | 48.3% (3.6) | — | 48.2% (3.2) | 47.0% (1.4) | — |
| Duración promedio (h) | 15.6 (1.0) | 15.4 (0.8) | todo el periodo | 15.8 (1.1) | 15.6 (1.5) | — |
| Operaciones por fold | 302 (132) | 323 (187) | 1 | 454 (198) | 478 (216) | 0 |
| Rotación anual (veces el capital) | 755 (139) | 725 (244) | 3.2 | 1,087 (68) | 1,136 (105) | 0 |
| Log C_T/C_0 | −0.78 (0.41) | −0.35 (0.26) | +0.38 (0.56) | −1.07 (0.65) | −0.96 (0.43) | 0 |
| Correlación diaria con BTC | 0.05 (0.35) | 0.12 (0.16) | 1.00 | 0.00 (0.10) | 0.81 (0.03) | — |
| Alfa vs BTC | −119 pp (84) | −91 pp (100) | 0 | −127 pp (100) | −128 pp (82) | −59 pp (92) |
| Alfa vs S&P 500 | −80 pp (6) | −52 pp (16) | +39 pp (87) | −88 pp (16) | −89 pp (8) | −20 pp (5) |

**S&P 500:** tomado de FRED (serie `SP500`, el índice y no un ETF) solo como benchmark, en `data/sp500_daily.csv` (columnas `fecha` y `close`). Ya está incluido en `artifacts/metrics_summary.csv` (fila `sp500` y columna `alfa_vs_sp500`) y en las curvas `equity_curves_test_fold_{k}.png`. **Supuesto declarado:** el enunciado restringe los datos externos de precio, y por eso este archivo se usa solo como benchmark: no entra al estado ni al entrenamiento del agente.

**Por fold (test, log C_T/C_0 y retorno total del modelo representativo):**

| Fold (test) | PPO | DQN | Comprar y mantener BTC | Aleatoria | Siempre largo |
|---|---|---|---|---|---|
| 1 (2023) | −0.73 (−51.6%) | −0.69 (−49.9%) | +0.94 (+155.2%) | −1.83 | −0.73 |
| 2 (2024) | −0.80 (−55.1%) | −0.26 (−22.8%) | +0.79 (+119.8%) | −0.95 | −1.21 |
| 3 (2025) | −1.31 (−73.0%) | −0.38 (−31.7%) | −0.08 (−7.4%) | −1.25 | −1.42 |
| 4 (ene–abr 2026) | −0.30 (−26.0%) | −0.08 (−8.0%) | −0.11 (−10.8%) | −0.26 | −0.47 |

---

## 3. Pregunta 4: ¿cómo varía el desempeño entre folds y qué explica el fold más débil?

- **Ningún fold es positivo para ninguno de los dos algoritmos.** No es el caso de "gana en tres y pierde mucho en uno": pierde en los cuatro.
- **Variación:** PPO va de −0.30 (fold 4) a −1.31 (fold 3), con desviación de 0.41 entre folds. DQN va de −0.08 a −0.69, con desviación de 0.26.
- **Cuánto pierde depende de cuánto opera.** El fold 4 pierde menos sobre todo porque su test dura 4 meses y tiene menos operaciones (126 en PPO y 74 en DQN). Por operación pierde lo mismo: −0.24% neto en PPO.

**Fold más débil: fold 3 con PPO (test 2025, −1.31 de log, −73%).** Tres razones medibles:

1. **Operó con ventaja bruta negativa.** Su retorno bruto medio por operación fue de −0.097% (t = −0.98), y después de costos queda en −0.348% por operación con 376 operaciones. 376 × −0.35% ≈ −1.31 de log: la pérdida es casi exactamente costo más ruido.
2. **Régimen de 2025:** BTC subió hasta ~126k en octubre y luego cayó a ~88k (comprar y mantener −7.4%). Es un año lateral con caídas fuertes, sin la tendencia sostenida de 2023 y 2024. Las barreras simétricas (stop y TP a 1·σ₂₄·√24) recortan los movimientos grandes en ambos sentidos.
3. **Su drawdown es el mayor del test** (sección 6): del 19 de enero al 31 de diciembre de 2025, −75.6%, sin recuperación.
   - Con la misma configuración, DQN perdió −0.38 en el mismo fold. La diferencia entre algoritmos es del tamaño de la dispersión entre semillas (PPO del fold 3: −1.45 a −1.24), así que no se puede atribuir a un algoritmo mejor.

**Folds 1 y 2 (2023 y 2024, alcistas):** el agente se queda muy atrás del buy & hold (alfa de −207 y −174 pp en PPO) porque no puede mantener una posición más de 24 h. El horizonte, el stop y el TP le impiden capturar la tendencia. Es una limitación del diseño del entorno, no del entrenamiento.

---

## 4. Pregunta 5: ¿supera a comprar y mantener BTC y al S&P 500 después de costos?

**No.**
- **Contra BTC:** PPO tiene un alfa medio de −119 pp por año y DQN de −91 pp. Solo en el fold 4 DQN (−8.0%) queda ligeramente por encima de comprar y mantener (−10.8%), y en un periodo de 4 meses esa diferencia no es significativa.
- **Contra el S&P 500:** el alfa medio es −80 pp por año en PPO y −52 pp en DQN, y es negativo en los 4 folds para los dos agentes (el S&P 500 rindió +24.6%, +23.9%, +16.5% y +4.3% en los folds 1 a 4, `metrics_summary.csv`).

**¿Qué mide entonces el resultado?**

1. **Principalmente el costo de transacción.** La ventaja bruta del agente en test es pequeña y no significativa:

   | Fold | Ventaja bruta PPO (t) | Ventaja bruta DQN (t) |
   |---|---|---|
   | 1 | +0.080% (t = 0.95) | +0.101% (t = 1.30) |
   | 2 | −0.036% (t = −0.26) | +0.195% (t = 2.00) |
   | 3 | −0.097% (t = −0.98) | +0.116% (t = 1.09) |
   | 4 | +0.011% (t = 0.06) | +0.137% (t = 0.53) |

   - El costo de ida y vuelta es 0.25%, así que el neto queda negativo en los 8 casos.
   - El único t ≈ 2 (DQN, fold 2) es uno de 8 contrastes y no sobrevive a una corrección por pruebas múltiples.

2. **Que el agente aprendió a operar menos, no a predecir.** El agente aleatorio opera ~454 veces por fold y pierde −1.07. PPO opera ~302 y pierde −0.78; DQN ~323 y pierde −0.35. Perder menos que el aleatorio se explica en gran parte por pagar menos comisiones.

3. **No es una posición larga disfrazada.** Hay que revisarlo explícitamente:
   - La correlación diaria con BTC es 0.05 (PPO) y 0.12 (DQN), contra 0.81 de "siempre largo".
   - El agente reparte entre largos y cortos (entre 21% y 66% largos según fold y algoritmo).
   - No es un largo apalancado ni filtrado.

4. **La mejor política del espacio es "siempre fuera"** (log 0). El agente debería converger a ella y no lo hace del todo: en 25k pasos sigue operando entre 20% y 50% de las decisiones.

---

## 5. Pregunta 7: ablación por familia y segundo algoritmo

### 5.1 Ablación

**Diseño:**
- Se quita una familia a la vez, con `make_env(..., drop_families=(familia,))`.
- Se usa el algoritmo primario (PPO) con la configuración elegida en cada fold y 3 semillas (42 a 44).
- Se evalúa **solo en valid**: el test no se usa para decisiones de diseño.
- 72 corridas. Archivos: `ablation_valid.csv` y `ablation_summary_valid.csv`.

**Δ de la mediana de log C_T/C_0 en valid frente al modelo completo:**

| Familia quitada | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Media |
|---|---|---|---|---|---|
| creencia | −1.15 | −0.05 | −0.63 | −0.42 | **−0.56** |
| tendencia_momentum | −1.04 | −0.66 | −0.06 | −0.69 | **−0.61** |
| volumen | −1.28 | +0.09 | −0.04 | −0.77 | −0.50 |
| volatilidad | −0.48 | −0.02 | −0.15 | −0.50 | −0.29 |
| tiempo | −0.66 | +0.06 | −0.55 | +0.11 | −0.26 |
| retornos | −0.42 | −0.16 | +0.05 | −0.45 | −0.25 |

**Advertencia de lectura (importante):** el modelo completo tiene ventaja de entrada en esta tabla. Su lr y su número de pasos se eligieron como **el máximo de 9 combinaciones en estas mismas semillas de valid**, mientras que los modelos sin una familia usan esa configuración sin reoptimizarla. Por eso todas las Δ están sesgadas hacia abajo (es la "maldición del ganador").

**Lo que sí se puede decir:**
- Quitar **tendencia_momentum** o **creencia** (el filtro de Kalman de Milca) es lo que más empeora y lo hace en 3 de 4 folds. Son las familias que el agente más usa.
- **Volumen** y **tiempo** cambian de signo entre folds: su aporte no es estable.
- Ninguna familia convierte la estrategia en ganadora. Con todas las familias, el resultado en valid sigue siendo negativo en los 4 folds.

### 5.2 Segundo algoritmo (DQN frente a PPO)

- **En valid**, PPO fue mejor: mediana media de −0.43 contra −0.65. Por eso es el primario.
- **En test**, DQN fue mejor: log medio de −0.35 contra −0.78, y Sharpe de −0.97 contra −2.31.
- **Por qué cambia el orden:** la diferencia está dentro de la dispersión entre semillas en 3 de 4 folds (sección 1.3). Que el orden se invierta entre valid y test es justo lo que se espera cuando ninguno de los dos tiene ventaja real.
- **Diferencia de diseño relevante:** DQN usa el descuento semi-Markov correcto (0.999^τ) y PPO no, porque su GAE descuenta por decisión. DQN tiene ventaja bruta positiva en los 4 folds, aunque nunca significativa salvo el fold 2. Es consistente con una mejor asignación de crédito, pero no alcanza para cubrir costos.

---

## 6. Análisis de operaciones (sección 3.9)

### 6.1 Las tres operaciones con mayor pérdida en test (PPO, todos los folds)

Fuente: `worst_trades_test.csv`. Las variables son las del estado de información en la barra de decisión (una hora antes de la entrada), sin normalizar. El archivo trae también las normalizadas.

| # | Fold | Entrada (decisión) | Entrada / precio | Salida / precio | Motivo | Retorno neto |
|---|---|---|---|---|---|---|
| 1 | 2 | 2024-03-05 20:00 | 2024-03-05 21:00 · corto a 61,886 | 2024-03-06 06:00 · 66,245 | stop | −7.30% |
| 2 | 1 | 2023-03-13 16:00 | 2023-03-13 17:00 · corto a 23,954 | 2023-03-14 13:00 · 25,596 | stop | −7.11% |
| 3 | 3 | 2025-03-07 00:00 | 2025-03-07 01:00 · corto a 85,529 | 2025-03-07 15:00 · 91,039 | stop | −6.70% |

**Variables de estado en la decisión:**

| # | ret_1 | ret_24 | vol_24 | RSI 14 | dist. EMA 168 | trend_mean | trend_snr |
|---|---|---|---|---|---|---|---|
| 1 | +0.77% | −8.57% | 1.39% | 30.4 | −0.36% | +0.0003 | 0.21 |
| 2 | −0.11% | +16.12% | 1.35% | 82.5 | +11.42% | +0.0019 | 1.35 |
| 3 | −5.02% | −5.15% | 1.27% | 31.2 | −3.39% | +0.0000 | 0.00 |

**Interpretación:**
- **Patrón común:** las tres son **cortos que tocaron el stop** después de un movimiento extremo. La pérdida (≈ −7%) es la distancia del stop, 1·σ₂₄·√24 ≈ 6.3%–6.8%, más los costos y el deslizamiento de la barra.
- **Op. 1:** corto tras una caída de −8.6% en 24 h con RSI ≈ 30. Es decir, apuesta a que la caída continúa justo cuando BTC rebotaba hacia su máximo histórico de marzo de 2024.
- **Op. 2:** corto tras +16% en 24 h (rebote posterior a SVB), con RSI 82 y precio 11% sobre la EMA 168. Apuesta a reversión contra un momentum muy fuerte, con un filtro de Kalman que sí marcaba tendencia positiva (snr 1.35).
- **Op. 3:** corto tras −5% en una hora (7 de marzo de 2025, noticia de la reserva estratégica de EE. UU.) que se revirtió el mismo día.
- **Por qué las tomó:** en las tres, σ₂₄ estaba en el decil superior del train de su fold (percentiles 88 a 93; `vol_24_norm` entre 1.38 y 1.61) y el retorno de 24 h a 1.6–4.2 unidades normalizadas de la mediana. Son estados poco frecuentes en el entrenamiento, donde la política es más ruidosa. Además, la distancia del stop crece con σ₂₄, así que justo en esos estados cada error cuesta más.

**Qué tendría que cambiar:**
- **Estado:** agregar la magnitud del último movimiento en unidades de σ (un z-score de ret_24) y una variable de régimen. Así el agente distinguiría "después de un choque" de "tendencia normal".
- **Recompensa:** no cambiaríamos la contribución terminal (justificada por Arturo). Una penalización por drawdown no es markoviana sin el pico en el estado.
- **Riesgo:** no abrir operaciones durante k horas después de un movimiento mayor a 3σ, o dimensionar la posición inversamente a la volatilidad. Discrete(3) no puede expresar fracciones; es una limitación ya declarada.

### 6.2 Periodo del mayor drawdown en test

- **Cuándo:** fold 3, PPO (representativo, semilla 44), **del 19 de enero de 2025 al 31 de diciembre de 2025, −75.6%** (`drawdown_period_test.csv` y `drawdown_period_test.png`).
- **Qué se ve en la gráfica:**
  - El capital no cae por un evento puntual, sino de forma casi lineal durante todo el año: 376 operaciones de ~15 h, cada una con −0.35% de retorno neto medio.
  - Las entradas largas y cortas se reparten en subidas y bajadas sin patrón visible.
  - Las semanas con mayor caída del capital (−8% a −11% cada una) cierran el 16 de febrero, el 9 de marzo, el 25 de mayo, el 26 de octubre y el 21 de diciembre de 2025. Ninguna explica sola el drawdown.
- **Lectura:** es un drawdown por desgaste (costo × frecuencia), no por una apuesta grande equivocada. El mismo agente con costo cero habría perdido −0.097% por operación en bruto, cerca de cero.

### 6.3 Correlación con BTC (test, retornos diarios)

| Fold | PPO | DQN | Siempre largo |
|---|---|---|---|
| 1 | 0.28 | 0.33 | 0.82 |
| 2 | −0.43 | −0.05 | 0.84 |
| 3 | −0.00 | 0.07 | 0.77 |
| 4 | 0.35 | 0.14 | 0.83 |
| Media | 0.05 | 0.12 | 0.81 |

La correlación es baja y cambia de signo entre folds (en el fold 2 PPO fue corto el 79% de las veces). **El agente no es una posición larga apalancada ni filtrada.** Su resultado no viene de la exposición a BTC, y tampoco de una señal que funcione.

---

## 7. Producción (sección 3.10, con los modelos finales en test)

Complementa la sección 5 del reporte de Arturo, que usa políticas base. Fuente: `sensitivity_test_fold_{k}.csv`, una corrida por fold con el modelo representativo. Las cifras son el promedio de los 4 folds de test, en log C_T/C_0.

| Escenario | PPO | Δ vs base | DQN | Δ vs base |
|---|---|---|---|---|
| base (stop primero, sin retraso, sin impacto) | −0.79 | — | −0.35 | — |
| tp_first (TP primero en barras ambiguas) | −0.79 | 0.00 | −0.34 | +0.01 |
| retraso_1_barra | −0.59 | **+0.20** | −0.49 | −0.14 |
| impacto 0.1 M USDT, Y = 1 | −1.03 | −0.25 | −0.78 | −0.43 |
| **impacto 1 M USDT, Y = 1** | **−1.68** | **−0.89** | **−1.59** | **−1.24** |
| impacto 10 M USDT, Y = 1 | −3.44 | −2.65 | −3.56 | −3.21 |
| impacto 50 M USDT, Y = 1 | −7.56 | −6.77 | −6.83 | −6.48 |

**Pregunta 6, sensibilidad intrabarra:**
- Las operaciones ambiguas (stop y TP en la misma barra) son 0% en PPO y entre 0% y 0.22% en DQN, según el fold.
- Cambiar el supuesto mueve el log del capital en ≤ 0.01. **El supuesto conservador no está creando ni ocultando el resultado.**

**Pregunta 6, retraso de una barra:**
- **A PPO le va mejor con datos retrasados una hora** (+0.20). Si el agente usara información útil de la última barra, retrasarla debería costarle.
- Que mejore confirma que su política no depende de la información reciente de forma productiva. La diferencia es ruido de trayectoria: al cambiar el estado, cambia la secuencia de operaciones.
- A DQN el retraso le cuesta −0.14, una diferencia del mismo orden que la dispersión entre semillas.

**Supuestos de ejecución:**
- El simulador llena al open de la barra siguiente sin spread ni impacto.
- Con la ley de raíz cuadrada calibrada en el train, operar 1 M USDT agrega 0.14%–0.24% por lado.
- Con ese costo realista, la pérdida se duplica o triplica (de −0.79 a −1.68 en PPO, y de −0.35 a −1.59 en DQN).

**Escenario `impacto_1M_Y1`:**
- No hay ningún resultado positivo que defender: el agente ya pierde en el escenario base.
- Lo que muestra el escenario es que **la distancia entre el simulador y la operación real juega en contra**. El simulador es optimista y aun así el agente pierde.

**Latencia:** 2 segundos de inferencia equivalen a un movimiento típico de ≈0.02% (`execution_assumptions.csv`). Es irrelevante frente a una pérdida de −0.24% a −0.35% por operación.

**Capacidad:** con una ventaja bruta estadísticamente nula no hay ventaja que repartir entre 50 fondos. Con 50 M USDT agregados, el impacto (≈1%–1.7% por lado) lleva el log del capital a −7 en un año.

**Cambio de régimen:**
- Si los próximos 12 meses se parecen a **2022** (bajista con choques: Luna, FTX), el agente seguiría perdiendo por costo, con más operaciones tocando el stop como las de la sección 6.1. El valid del fold 1 (2022) dio entre −0.31 y −0.90 para PPO.
- Si se parecen a **2018** (bajista sostenido, −70%), "siempre fuera" o el corto puro le ganarían.
- **Las variables que más se degradarían:**
  - Las de volatilidad y la familia creencia: σ₂₄ y la incertidumbre del filtro de Kalman están normalizadas con los cuantiles del train. En un régimen de volatilidad fuera de rango se recortan a ±5 y el agente deja de distinguir entre niveles extremos.
  - La familia tiempo (estacionalidad intradía): ya cambia de signo entre folds en la ablación.

---

## 8. Advertencia de interpretación (obligatoria)

**Supuestos del simulador:** modela la **política** y supone que el resto del sistema es gratuito, instantáneo y perfecto:
- Llenado completo al open.
- Sin spread, impacto ni deslizamiento del stop (salvo en huecos).
- Datos sin retraso.
- Sin financiamiento de cortos ni caídas del exchange.

**Magnitud estimada con nuestros modelos en test:**
- **Impacto de 1 M USDT:** −0.89 (PPO) y −1.24 (DQN) de log por año.
- **Intrabarra:** ≤ 0.01.
- **Retraso de una barra:** −0.14 a +0.20.

**Qué demuestra y qué supone:**
- **Lo que el modelo demuestra:** con estas 21 variables y este gestor de posiciones, un agente de RL no encuentra en 2017-2024 una regularidad que cubra el 0.25% de ida y vuelta en 2023-2026.
- **Lo que supone:** que el costo real es el del simulador. En la realidad sería mayor.

---

## 9. Supuestos y pendientes

- **S&P 500:** resuelto. `data/sp500_daily.csv` (FRED, serie `SP500`) alimenta `alfa_vs_sp500` y las curvas de capital. Supuesto declarado: se usa solo como benchmark.
- **Datos:** el CSV de Canvas se usa tal cual (75,588 filas contra 75,558 del enunciado), como supuesto declarado.
- **Reproducibilidad de PPO y DQN.** Con la semilla 42, los datos, el EDA y las políticas sin entrenamiento (aleatoria, siempre largo, siempre fuera, comprar y mantener BTC y S&P 500) se reproducen hasta el redondeo (diferencias ≤ 1e-12). PPO y DQN no: varían entre máquinas y versiones de torch. En una corrida limpia en un clon nuevo (macOS, Python 3.12.2, torch 2.15.0 sin CUDA) dieron PPO −59.0% y DQN −47.9% de retorno total (media de los 4 folds de test); los documentos reportan la corrida de `artifacts/` (Python 3.12.3, torch 2.15.0+cu132), con PPO −51.4% y DQN −28.1%. La conclusión se mantiene: 0 de 40 modelos ganan en la corrida limpia (1 de 40 en la de `artifacts/`), |t| ≤ 2 de la ventaja bruta en las dos corridas y, en promedio, los dos pierden contra comprar y mantener BTC y contra el S&P 500. Lo que no es estable entre corridas son las cifras puntuales y el orden de la ablación por familia.
- **Gráfica del drawdown:** si Milca prefiere que esté en `plots.py`, se puede mover. Hoy la genera `evaluate.plot_drawdown_period`.
