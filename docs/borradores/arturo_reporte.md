# Borrador de reporte · Arturo · Entorno y función de contribución

Estado: borrador de trabajo. Las cifras salen solo de los datos de **train**. Ninguna cifra de este documento usa valid ni test.

## 0. Cifras de diagnóstico previas al diseño (train)

Calculadas antes de escribir `env.py`, sobre el train de cada fold. Se decide al cierre de t y se entra en el open de t+1.

| Cifra | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| σ del log-retorno horario | 0.95% | 0.91% | 0.85% | 0.82% |
| Mediana de \|ret\| a 1 h | 0.31% | 0.30% | 0.27% | 0.27% |
| Mediana de \|ret\| a 24 h | 1.88% | 1.81% | 1.63% | 1.59% |
| Costo ida y vuelta / mediana de \|ret\| a 1 h | 0.80 | 0.83 | 0.92 | 0.93 |
| Costo ida y vuelta / mediana de \|ret\| a 24 h | 0.13 | 0.14 | 0.15 | 0.16 |
| ACF de rezago 1 del retorno horario | −0.052 | −0.046 | −0.046 | −0.044 |
| Barras con \|open(t+1)/close(t) − 1\| > 0.1% | 3.7% | 3.0% | 2.6% | 2.2% |

Entradas guionadas cada 6 barras en el train del fold 1, con barreras simétricas a k·vol_24·√H:

| H | k | Stop | TP | Horizonte | Ambiguas intrabarra | Duración media |
|---|---|---|---|---|---|---|
| 12 | 0.5 | 47–48% | 45–46% | 7% | 1.75% | 4.3 h |
| 24 | 0.5 | 46–47% | 45–46% | 8% | 0.95% | 8.1 h |
| 24 | 1.0 | 29–31% | 29–31% | 40% | 0.24% | 16.2 h |
| 48 | 2.0 | 11–12% | 11–12% | 76% | 0.00% | 42.8 h |

En los largos las dos primeras columnas dan stop y TP en ese orden; en los cortos se invierten. Con H=48 y k=2 el largo deja +0.10% neto por operación y el corto −0.60%. Esa asimetría es la tendencia alcista de la muestra, no una señal.

## 1. Veredicto del council (a): diseño del entorno

Corrí el council (`/llm-council`) antes de escribir `env.py`, con 5 consejeros, 5 revisiones anónimas y un chairman. Los 5 revisores eligieron al Contrarian como la respuesta más fuerte. Verifiqué en el código de SB3 2.9.2 una de sus afirmaciones, y era falsa (ver la decisión de γ^τ).

| Decisión | Elección | Alternativa rechazada y por qué (cifra de train) |
|---|---|---|
| Espacio de acciones | Discrete(3): fuera, abrir largo, abrir corto. Solo se decide estando fuera. | **Acción de cierre por barra:** el costo de ida y vuelta es el 80% del movimiento mediano de 1 h (fold 1). **Menú de (H, k) en la acción:** multiplica las pruebas múltiples sin evidencia de que haga falta. **Solo largo:** el sesgo largo (+0.10% contra −0.60%) es de régimen. |
| Acciones inválidas | No existen por construcción, así que no hace falta máscara ni penalización. | **Máscara:** DQN de SB3 no la soporta. **Penalización:** agrega un coeficiente de moldeo más que habría que justificar. |
| Barreras | H = 24 h de reloj; stop y TP a 1·vol_24·√24 (≈ ±3.0% con la vol_24 mediana del fold 1: 0.62%·√24). | **k = 0.5:** de 4 a 7 veces más barras ambiguas (0.95–1.75%) y solo 7–8% de operaciones llegan al horizonte. **H = 1 a 4 h:** el costo se come del 40 al 80% del movimiento típico. |
| Unidad del horizonte | Horas de reloj. | **Barras:** desde abril de 2025 falta casi a diario la barra de 00:00, y contar barras haría más largas las operaciones de valid y test que las de train. Ese cambio de distribución lo causaríamos nosotros. |
| Convención temporal | Se decide al cierre de t y se llena en el open de t+1. Las barreras se fijan con ese precio y con vol_24 en t. | **Llenar al close de t:** usa un precio que ya pasó cuando se termina de calcular la observación. **Llenar en open de t+2:** se deja como experimento de retraso (`data_delay_bars`). |
| Hueco de precio | Un stop atravesado por un hueco se llena en el open, peor que el nivel. | **Llenar en el nivel del stop:** supone una ejecución que no existió. |
| Intrabarra | Stop primero por defecto. `intrabar_priority='tp_first'` permite el contraste. | La cuantificación va en la sección de la pregunta 6. |
| Episodio | En train, ventanas de 720 h (30 días) con inicio al azar, cerradas con `truncated=True` para que SB3 haga bootstrap. En valid y test, la partición completa. | **El train completo como un episodio:** muy pocas actualizaciones por época. **Una operación por episodio:** convierte el problema en un bandido y pierde el capital como estado. **Qué se pierde:** cualquier dependencia de régimen de más de 30 días. |
| Embargo | No se abren operaciones a menos de 24 h del final de la partición. | **Sin embargo:** una barrera abierta al final del train se cerraría con precios de validación. |
| Componente física R_t | Capital, drawdown del episodio y horas del último hueco. La posición, el precio de entrada y el tiempo en posición valen 0 al decidir, y se declara así. | Meter la tasa de aciertos o el último resultado en R_t: eso es información o creencia, no un recurso físico. |
| Creencia B_t | trend_mean, trend_std y trend_snr del filtro de Kalman de `data.py` (media y desviación posteriores de la tendencia latente). | Pendiente para la Fase 2: justificarla contra quitarla, con ablación. |
| γ y duración real | `info["tau_horas"]` en cada paso. DQN usa `SMDPReplayBuffer`, que descuenta con γ_h^τ (γ_h = 0.999 por hora). | El Contrarian sostenía que había que reescribir `DQN.train()`. Es falso: en SB3 2.9.2 `dqn.py` usa `replay_data.discounts` cuando existe. **PPO:** su GAE usa un γ fijo por decisión. Es una limitación declarada; τ queda registrado. |

**Cordura del simulador (train completo, política aleatoria, semilla 42):** retorno neto medio por operación de −0.30% (fold 1), −0.27% (fold 2), −0.26% (fold 3) y −0.27% (fold 4). Pierde en los 4 folds, en línea con el costo de ida y vuelta de 0.25%.

**Pendientes para el council (b):** unidades y escala de la recompensa (hoy es 100·log del cambio de capital), tamaño de posición (hoy 100% del capital, sin apalancamiento), desbalance del replay (unas 16 transiciones "fuera" por cada operación) y la diferencia entre el objetivo de Powell y las métricas de evaluación.
