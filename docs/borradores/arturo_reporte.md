# Borrador de reporte · Arturo · Entorno, contribución y simulador

Las cifras salen de código que corrió el 2026-11-19, con semilla 42, y solo de **train** (y de **valid** donde se indica). Ninguna cifra usa test.

**Fuentes:**
- `artifacts/sensitivity_baselines.csv`, `execution_assumptions.csv`, `gross_edge_test.csv` (salen de `python code/sensitivity.py baselines`).
- `artifacts/reward_timing_fold1_train.csv` (sale de `python code/sensitivity.py reward-timing`).
- Diagnósticos de diseño previos a `env.py` (sección 0).

## 0. Cifras de diagnóstico previas al diseño (train)

| Cifra | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| σ del log-retorno horario | 0.95% | 0.91% | 0.85% | 0.82% |
| Mediana de \|ret\| a 1 h | 0.31% | 0.30% | 0.27% | 0.27% |
| Mediana de \|ret\| a 24 h | 1.88% | 1.81% | 1.63% | 1.59% |
| Costo ida y vuelta / mediana de \|ret\| a 1 h | 0.80 | 0.83 | 0.92 | 0.93 |
| Costo ida y vuelta / mediana de \|ret\| a 24 h | 0.13 | 0.14 | 0.15 | 0.16 |
| ACF de rezago 1 del retorno horario | −0.052 | −0.046 | −0.046 | −0.044 |
| Barras con \|open(t+1)/close(t) − 1\| > 0.1% | 3.7% | 3.0% | 2.6% | 2.2% |
| Volumen horario mediano (USDT) | 15.4 M | 23.0 M | 28.0 M | 34.9 M |

Entradas guionadas cada 6 barras en el train del fold 1, con barreras simétricas a k·vol_24·√H:

| H | k | Stop | TP | Horizonte | Ambiguas intrabarra | Duración media |
|---|---|---|---|---|---|---|
| 12 | 0.5 | 47–48% | 45–46% | 7% | 1.75% | 4.3 h |
| 24 | 0.5 | 46–47% | 45–46% | 8% | 0.95% | 8.1 h |
| 24 | 1.0 | 29–31% | 29–31% | 40% | 0.24% | 16.2 h |
| 48 | 2.0 | 11–12% | 11–12% | 76% | 0.00% | 42.8 h |

## 1. Formulación de Powell

Índice de tiempo: t recorre los **instantes de decisión**, que son los cierres de barras horarias observadas en los que el agente está fuera del mercado. La frecuencia de decisión es **endógena**: depende de cuándo se cierra cada operación, y por eso el modelo es semi-Markov.

### 1.1 Variable de estado S_t = (R_t, I_t, B_t)

- **Componente física R_t** (recursos):
  - capital C_t;
  - capital máximo del episodio C_t^max;
  - posición q_t, precio de entrada p_t^e y tiempo en posición h_t.
  - Como solo se decide estando fuera, en todo instante de decisión q_t = 0, p_t^e = ∅ y h_t = 0. Lo declaramos explícitamente en lugar de rellenar el vector con constantes.
  - El agente observa log(C_t / C_0) y el drawdown C_t / C_t^max − 1.
  - La posición abierta existe **entre** decisiones y la maneja la transición (sección 1.4).
- **Componente de información I_t:**
  - las 18 variables causales de precio, volumen y tiempo de `data.py` en la barra t − d, donde d = `data_delay_bars`, 0 por defecto. Son 4 familias de retornos, volatilidad, volumen y tendencia/momentum, más el calendario. Las normaliza el Normalizer del fold, con mediana e IQR del train;
  - las horas faltantes antes de la barra observada, log1p(gap)/log1p(145);
  - vol_24 sin normalizar, que fija las barreras.
  - La justificación con ACF y ADF/KPSS es la sección de Milca (pregunta 2).
- **Componente de creencia B_t:** media posterior μ̂_t, desviación posterior σ̂_t y μ̂_t/σ̂_t de la tendencia latente, estimadas con el filtro de Kalman causal de `data.py`. Es lo que el sistema **estima** sobre la deriva del precio, junto con su incertidumbre. En un hueco, el filtro solo predice (σ̂ crece con las horas reales).

### 1.2 Decisión x_t

x_t ∈ X = {0: quedarse fuera, 1: abrir largo, 2: abrir corto} (`Discrete(3)`).

- No hay acciones inválidas: no existe "cerrar" (las barreras cierran) y nunca se decide estando dentro.
- Política X^π(S_t): la red de PPO o de DQN.

### 1.3 Información exógena W_{t+1}

La trayectoria OHLCV de las barras posteriores a t: (o_j, h_j, l_j, c_j, v_j) para j > t. Incluye cuándo llega la siguiente barra, ya que los huecos de datos también son exógenos.

- En el modo de diseño, W_{t+1} abarca todas las barras hasta el cierre de la operación.
- La duración τ_{t+1} (horas hasta la siguiente decisión) es una función de W_{t+1} y de x_t.

### 1.4 Función de transición S_{t+1} = S^M(S_t, x_t, W_{t+1})

- **Si x_t = 0:** la siguiente decisión es la siguiente barra observada; C no cambia y τ = horas hasta esa barra.
- **Si x_t ∈ {1, 2}** (lado s = +1 o −1):
  1. **Entrada:** p^e = o_{t+1}, el open de la siguiente barra.
  2. **Barreras:** stop = p^e · exp(−s · k_s · σ_t · √H) y TP = p^e · exp(+s · k_tp · σ_t · √H), con σ_t = vol_24 en la barra t − d, k_s = k_tp = 1 y H = 24 h.
  3. **Salida:** en la primera barra j ≥ t+1 donde ocurra, en este orden:
     1. Si la barra empieza después de entrada + H (vence dentro de un hueco): sale en o_j, motivo horizonte.
     2. Si el open ya atravesó una barrera (hueco de precio): sale en o_j, que es peor que el stop si fue el stop.
     3. Si h_j y l_j tocan las dos barreras: barra **ambigua**; por defecto gana el stop (`intrabar_priority`).
     4. Si solo toca una: sale en esa barrera.
     5. Si la barra cierra en entrada + H: sale en c_j, motivo horizonte.
  4. **Capital:** C_{t+1} = C_t + pnl_neto (sección 1.5). La siguiente decisión es al cierre de la barra de salida, y τ = horas reales.
- **Embargo:** no se abren operaciones a menos de H del fin de la partición, así que ninguna barrera usa precios del periodo siguiente.
- **Información y creencia:** I y B se leen de la siguiente barra de decisión. Son exógenas al agente, que no mueve el precio (supuesto de la sección 5).

### 1.5 Función de contribución C(S_t, x_t, W_{t+1})

- **Contabilidad:**
  - nocional N = C_t (100% del capital, sin apalancamiento);
  - qty = N / p^e;
  - bruto = s · qty · (p^s − p^e);
  - costos = 0.00125 · N + 0.00125 · qty · p^s;
  - neto = bruto − costos.
- **Contribución terminal**, pagada al cierre de la operación: C = 100 · log(C_{t+1} / C_t). Si x_t = 0, C = 0.

### 1.6 Función objetivo

max_π E [ Σ_t γ_h^{T_t} · C(S_t, X^π(S_t), W_{t+1}) | S_0 ], donde T_t son las horas transcurridas hasta la decisión t.

- Con γ_h → 1, la suma de las contribuciones es 100 · log(C_final / C_0). El objetivo equivale a **maximizar el crecimiento compuesto del capital**, que es el criterio de Kelly.
- Las métricas de evaluación (Sharpe, drawdown, Calmar) no son el objetivo. Se reportan aparte, y declaramos esa diferencia.

## 2. Decisiones de diseño (council (a), antes de escribir `env.py`)

Council con 5 consejeros, 5 revisiones anónimas y chairman. Los 5 revisores eligieron al Contrarian como la respuesta más fuerte. Verifiqué su afirmación central en el código de SB3 2.9.2 y era falsa (ver la fila de γ^τ).

| Decisión | Elección | Alternativa rechazada y cifra (train) |
|---|---|---|
| Espacio de acciones | Discrete(3); solo se decide estando fuera | **Cierre por barra:** el costo de ida y vuelta es el 80% del movimiento mediano de 1 h (fold 1). **Menú (H, k):** pruebas múltiples sin evidencia de que hagan falta. **Solo largo:** la asimetría largo +0.10% contra corto −0.60% por operación (H=48, k=2) es la tendencia de la muestra. |
| Acciones inválidas | No existen por construcción | **Máscara:** el DQN de SB3 no la soporta. **Penalización:** un coeficiente de moldeo más sin justificación. |
| Barreras | H = 24 h; k = 1 → ±1·vol_24·√24 (≈ ±3.0% con la vol_24 mediana del fold 1) | **k = 0.5:** de 0.95 a 1.75% de barras ambiguas (4 a 7 veces más) y solo 7–8% de operaciones llegan al horizonte. **H de 1 a 4 h:** el costo se come del 40 al 80% del movimiento típico. |
| Unidad del horizonte | Horas de reloj | **Barras:** desde abril de 2025 falta casi a diario la barra de 00:00, y contar barras alargaría las operaciones de valid y test frente a train. |
| Convención temporal | Decidir al cierre de t, llenar en el open de t+1 | **Llenar al close de t:** usa un precio ya pasado. La diferencia es pequeña (mediana \|o_{t+1}/c_t − 1\| = 2·10⁻⁶), pero el open de t+1 es el primer precio ejecutable. |
| Hueco de precio | Llenar en el open (peor que el stop) | **Llenar en el nivel del stop:** supone una ejecución que no existió. |
| Episodio | En train, ventanas de 720 h con inicio al azar y `truncated=True`. En valid y test, la partición completa. | **Todo el train como un episodio:** unas 38,000 decisiones por episodio en el fold 1 y pocas actualizaciones. **Una operación por episodio:** un bandido sin capital como estado. **Qué se pierde:** dependencias de más de 30 días (por ejemplo, el ciclo de un régimen) y el efecto del capital acumulado más allá de la ventana. |
| Embargo | No abrir a menos de 24 h del fin de la partición | **Sin embargo:** una operación abierta el 2021-12-31 se cerraría con precios de 2022 (valid). |
| γ y duración | `info["tau_horas"]`; DQN con `SMDPReplayBuffer` (γ_h^τ, γ_h = 0.999/h) | El Contrarian decía que había que reescribir `DQN.train()`. Es falso: `dqn.py` usa `replay_data.discounts`. **PPO:** su GAE usa un γ fijo por decisión. Es una limitación declarada; τ queda registrado. |
| Una posición a la vez | Sí | **Posiciones traslapadas:** más reglas de capital. Se pierden señales consecutivas mientras hay una operación abierta. |

## 3. Contribución (council (1), antes de cerrar `reward.py`)

**Veredicto:**
- Mantener la contribución **terminal** 100·log(C_después / C_antes), neta de costos. Es la función objetivo de Powell y equivale a Kelly.
- **Rechazados por unanimidad:**
  - **Costo de oportunidad por estar fuera:** obligaría a operar con ventaja negativa.
  - **Penalización por drawdown:** no es markoviana sin el máximo histórico en el estado, y el log ya castiga la varianza (E[log] ≈ μ − σ²/2).
  - **Repesar el replay sin importance sampling:** sesga Q.
- **Escala:** se mantiene el factor 100. Una recompensa típica queda en ±3 (σ = 3.34 en el fold 1). Con la pérdida Huber de SB3 (δ = 1), esos errores caen en la zona lineal y el gradiente queda acotado. Lo declaramos.
- **Proceso:** las cinco revisiones se hicieron en una sola llamada y la síntesis del chairman la escribí yo, para respetar el límite de costo.

**Sin ventaja bruta que aprender** (`gross_edge_test.csv`, "siempre largo" con las mismas barreras, train):

| Fold | Operaciones | Media bruta (log) | Error estándar | t | Media neta | Kelly f* = μ/σ² |
|---|---|---|---|---|---|---|
| 1 | 2,537 | +0.011% | 0.066% | 0.17 | −0.239% | −2.14 |
| 2 | 3,122 | −0.020% | 0.057% | −0.35 | −0.271% | −2.63 |
| 3 | 3,741 | +0.005% | 0.049% | 0.10 | −0.246% | −2.73 |
| 4 | 4,342 | +0.011% | 0.044% | 0.24 | −0.240% | −2.87 |

Para cubrir el costo de ida y vuelta (0.25%) haría falta una ventaja bruta de unos 3.8 errores estándar en el fold 1, y no la hay.

**Tamaño de posición:** 100% del capital es la cota superior sin apalancamiento, no un óptimo. Con f* < 0, el tamaño de Kelly es cero. Discrete(3) no puede expresar fracciones; se rechazó Discrete(5) con ±0.5 y ±1 porque obligaba a reentrenar todo y agregaba pruebas múltiples. Es una limitación declarada.

### 3.1 Pregunta 3: ¿por paso o terminal?

**Terminal.** La contribución se paga completa al cierre, y la transición del replay es (estado al abrir, acción, contribución al cerrar, estado al cerrar, τ). El crédito de la operación cae entero en la única decisión que la causó.

La alternativa por paso (`reward_timing="per_bar"`) paga el cambio marcado a mercado de cada barra mientras la operación sigue abierta, en pasos forzados donde la acción se ignora. Suma exactamente lo mismo: lo prueba `test_variante_por_paso_suma_lo_mismo_que_la_terminal`. Solo cambia cómo se reparte el crédito.

**Experimento** (`reward_timing_fold1_train.csv`): DQN con 5 semillas, 30,000 pasos cada una, train del fold 1. La métrica común se fijó antes de correr.

| Mediana (mín–máx) | Terminal | Por paso |
|---|---|---|
| Replay: transiciones que abren operación | 30.5% (12.4–43.0%) | **5.9% (5.1–6.3%)** |
| Replay: transiciones forzadas sin decisión | 0% | **85.1% (72.6–89.0%)** |
| Decisiones libres que abren operación | 29.0% (5.5–43.0%) | 31.3% (7.5–69.5%) |
| Retorno neto medio por operación | +0.06% (−0.12% a +0.11%) | −0.15% (−0.21% a −0.05%) |
| Log del capital final, train completo | +0.07 (−2.21 a +1.17) | −3.85 (−5.16 a −2.27) |

**Comportamiento indeseado de la opción por paso:**
1. El replay se llena de transiciones donde la acción no importa (85%), y las decisiones reales bajan a ~6%. Con el mismo presupuesto, el agente ve unas 5 veces menos decisiones.
2. El costo de entrada (−0.125) cae entero y sin descontar en la primera barra, mientras el resultado llega disperso y descontado. Eso empuja a decidir por el ruido de las primeras barras.
3. La frecuencia de operación se vuelve errática entre semillas: de 7% a 70%.

**Advertencia:** el +0.07 de la terminal es **en muestra** y con un t bruto de 0.17. No es evidencia de ventaja, solo de una mejor asignación de crédito. La dispersión entre semillas (de −2.21 a +1.17) es justo la que pide reportar la sección 3.5.

## 4. Líneas base del simulador (con costos)

`sensitivity_baselines.csv`, escenario `base`, promedio de los 4 folds:

| Periodo | Política | Operaciones | Retorno neto medio por operación | Log del capital final | Sharpe anual |
|---|---|---|---|---|---|
| Train | Aleatoria | 3,274 | −0.27% | −10.35 | −2.46 |
| Train | Siempre largo | 3,436 | −0.20% | −8.56 | −1.91 |
| Train | Siempre fuera | 0 | — | 0 | — |
| Valid | Aleatoria | 551 | −0.22% | −1.34 | −3.07 |
| Valid | Siempre largo | 584 | −0.23% | −1.44 | −2.80 |
| Valid | Siempre fuera | 0 | — | 0 | — |

El agente aleatorio pierde en los 4 folds de train y de valid, como exige la sección 3.5. Con miles de operaciones el retorno total se satura en −100%, por eso comparamos con el log del capital y el retorno medio por operación. "Siempre largo" con barreras **no** es comprar y mantener: el benchmark de buy & hold lo calcula `evaluate.py`.

## 5. Pregunta 6 y sección 3.10: sensibilidad y supuestos de ejecución

### 5.1 Intrabarra (stop primero contra TP primero)

- **Operaciones afectadas:** entre 0.16% y 0.25% de las operaciones del train, según fold y política. En valid, entre 0% (fold 4) y 0.36%.
- **Cambio al suponer TP primero** (promedio de folds): el retorno medio por operación sube 0.01 pp; el log del capital, +0.19 a +0.22 en train y +0.03 a +0.04 en valid; el Sharpe, +0.04 a +0.08.
- **Conclusión:** el supuesto conservador cuesta muy poco. Lo que mueve el resultado es el costo, no la resolución intrabarra.

### 5.2 Retraso de una barra en los datos (`data_delay_bars=1`)

Observación y volatilidad de las barreras de t − 1; el llenado sigue en o_{t+1}.

- **"Siempre largo":** Δ log del capital = −0.02 (train) y −0.07 (valid); ΔSharpe = +0.01 y −0.15.
- **Aleatoria:** las diferencias (−0.64 ± 0.80 en train) son ruido de trayectoria. Al desplazarse las decisiones, la secuencia de operaciones cambia.
- **Conclusión:** las políticas base no usan la información, así que esto mide solo el efecto mecánico del simulador, que es pequeño. La prueba importante es con los modelos de Paula: si su resultado cae mucho con una barra de retraso, dependía de información que en la práctica llega tarde.

### 5.3 Supuestos de ejecución

`execution_assumptions.csv`, train:

- **A qué precio llena:** al open de la barra siguiente, sin spread, sin deslizamiento y sin impacto, con la orden completa. Los stops y TP se llenan exactamente en su nivel salvo un hueco, que llena en el open. En la realidad un stop es una orden de mercado que se llena peor que el nivel.
- **Impacto si se opera tamaño a ese precio.** Ley de raíz cuadrada, impacto por lado ≈ Y · σ_1h · √(Q / V_1h), con Y = 1:

  | Orden | Fold 1 | Fold 4 |
  |---|---|---|
  | 0.1 M USDT | 0.08% | 0.04% |
  | 1 M USDT | 0.24% | 0.14% |
  | 10 M USDT | 0.77% | 0.44% |
  | 50 M USDT | 1.71% | 0.98% |

  - Con 1 M USDT el impacto **iguala o supera la comisión** de 0.125%. En el escenario `impacto_1M_Y1`, el retorno medio por operación de las políticas base cae otros 0.36 pp y el log del capital entre −11.5 y −12.1 (train) y entre −2.0 y −2.1 (valid).
- **Latencia:**
  - 2 segundos de inferencia equivalen a un movimiento típico de σ_1h · √(2/3600) ≈ 0.02% (fold 1), una sexta parte de la comisión. Sí se puede operar al precio supuesto, salvo en momentos de noticias.
  - Una barra de retraso en los datos es lo de la sección 5.2.
- **Capacidad:**
  - Si 50 fondos operaran la misma señal con 1 M cada uno, sería como una orden de 50 M: entre 1% y 1.7% de impacto por lado, mucho más que cualquier ventaja bruta medida (≤ 0.011% por operación).
  - La contraparte son creadores de mercado y participantes con horizontes distintos. Con una ventaja bruta estadísticamente nula, no hay razón para que nos paguen.

### 5.4 Advertencia obligatoria

El simulador modela la **política** y supone que el resto del sistema es gratuito, instantáneo y perfecto:

- llenado completo al open sin impacto ni spread;
- stops sin deslizamiento (salvo huecos);
- datos sin retraso;
- sin caídas del exchange, límites de posición, financiamiento de cortos ni riesgo de contraparte.

**Distancia estimada:**
- **Impacto:** de 0.14% a 0.24% por lado para 1 M USDT, del orden de la comisión. Es decir, el costo real de ida y vuelta podría ser **de 2 a 3 veces** el que simulamos.
- **Intrabarra:** cambia el log del capital en ≤ 0.22.
- **Retraso de una barra:** en las políticas base cambia ≤ 0.07 en "siempre largo".

Cualquier resultado positivo del agente debe sobrevivir al escenario de impacto antes de llamarlo ventaja.

## 6. Pendientes y límites conocidos

- PPO usa un γ fijo por decisión: no descuenta con la duración real.
- El tamaño fijo de 100% no puede expresar Kelly fraccional.
- Las ventanas de `vol_24` cuentan barras, no horas (lo pedí a Milca como opcional). Mi horizonte sí va en horas de reloj.
- El experimento de la pregunta 3 es en muestra y con 5 semillas. Sirve para mostrar el mecanismo de asignación de crédito, no una diferencia de desempeño fuera de muestra.
