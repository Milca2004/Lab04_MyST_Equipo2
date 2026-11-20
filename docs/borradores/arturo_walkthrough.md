# Walkthrough · Arturo · Entorno, contribución y sensibilidad

Recorrido de `code/env.py`, `code/reward.py` y `code/sensitivity.py`. Los números de línea corresponden a la rama `arturo/entorno`.

**Desviación declarada de la estructura 2.2:** `code/sensitivity.py` es un archivo extra. Agrupa los experimentos de la pregunta 3, la pregunta 6 y la sección 3.10, que no pertenecen al entorno (`env.py`) ni a la evaluación de los modelos (`evaluate.py`).

## 1. Cómo fluye un paso

```
data.py (Milca)               env.py                                   reward.py
load_clean / build_features → load_split(fold, split) → SplitData
  Normalizer (solo train)        TradingEnv.step(acción)
                                   ├─ 0 fuera: avanza una barra, recompensa 0
                                   └─ 1/2 abrir: _run_trade
                                        ├─ barrier_prices (vol_24 en t)
                                        ├─ _first_exit (stop, TP, horizonte)
                                        └─ trade_accounting ──────────────→ costos y P&L
                                      log_return_contribution ─────────────→ recompensa
                                   info = {tau_horas, capital, operacion}
```

## 2. `code/env.py`

| Bloque | Qué hace | Decisión no evidente |
|---|---|---|
| `SplitData`, `load_split` (L56, L85) | Arma OHLC, huecos, `vol_24` sin escalar y las 21 variables normalizadas de un fold y partición. | El Normalizer se ajusta con el train del fold y nunca con valid ni test. `vol_24` se guarda sin escalar porque fija la distancia de las barreras en precio. |
| `market_data` (L78) | Carga y calcula las variables una sola vez (caché). | Las variables son causales, así que calcularlas sobre la serie completa y recortar después da lo mismo. Lo prueba `test_causality.py` de Milca. |
| `TradingEnv.__init__` (L181) | Define parámetros, espacios y el rango de decisiones. | `last_decision` aplica el **embargo**: no se abre nada a menos de un horizonte del fin de la partición. |
| `reset` (L219) | Reinicia capital, bitácora y ventana. | En train la ventana es de 720 h con inicio al azar; en valid y test es la partición completa. `options={'start_index': i}` sirve para las pruebas. |
| `step` (L236) | Una decisión libre o un paso forzado (solo en `per_bar`). | Nunca devuelve `terminated=True`. El fin de ventana es `truncated`, para que SB3 haga bootstrap: el mercado no se acaba. |
| `_observation` (L307) | Variables en t − `data_delay_bars` más la componente física. | La posición no entra en el modo de diseño porque siempre es 0 al decidir. Solo `per_bar` agrega lado y tiempo transcurrido. |
| `barrier_prices` (L330) | Stop y TP = entrada · exp(∓k · vol_24 · √H). | La volatilidad es la de la barra de decisión, nunca una posterior. |
| `_run_trade` (L338) | Abre en el open de t+1, busca la salida y llena la bitácora. | Nocional = `position_fraction` · capital (100%, sin apalancamiento). |
| `_first_exit` (L376) | Recorre las barras: horizonte vencido en un hueco, hueco de precio, barra ambigua, horizonte normal. | El orden importa: (1) si la barra empieza después del horizonte, sale en su open; (2) si el open ya atravesó una barrera, llena en el open; (3) si las dos barreras caen en la barra, decide `intrabar_priority`. |
| `rollout`, `BASELINES` (L474, L471) | Corre un episodio con cualquier política `obs -> acción`. | Las políticas son fábricas `f(env)`, porque la aleatoria necesita el `action_space` sembrado. |
| `SMDPReplayBuffer` (L501) | Guarda `tau_horas` y devuelve `discounts = γ_h^τ`. | El DQN de SB3 2.9.2 usa `replay_data.discounts` si existe (`dqn.py` ~L201), así que `DQN.train()` no se toca. PPO no lo usa (limitación declarada). |

## 3. `code/reward.py`

| Función | Qué hace |
|---|---|
| `trade_accounting` (L48) | qty = nocional / precio de entrada. Costo de entrada = 0.125% del nocional; costo de salida = 0.125% de qty · precio de salida. Neto = bruto − costos. |
| `log_return_contribution` (L62) | Contribución terminal = 100 · log(capital después / capital antes). |
| `per_bar_contributions` (L76) | Alternativa por paso para la pregunta 3: la misma operación marcada a mercado barra por barra. La suma telescopa al mismo total. |

## 4. `code/sensitivity.py`

| Función | Para qué |
|---|---|
| `run_sensitivity` (L102) | Tabla de los escenarios base, `tp_first`, `retraso_1_barra` e impacto (4 tamaños × 2 valores de Y) para cualquier política. Calcula `delta_retorno` y `delta_sharpe`. |
| `execution_assumptions` (L128) | σ horaria, volumen horario mediano, impacto por lado según la ley de raíz cuadrada y movimiento típico en 2 s de inferencia. |
| `gross_edge_test` (L144) | Prueba t de la ventaja bruta por operación y fracción de Kelly. |
| `reward_timing_experiment` (L182) | Pregunta 3: DQN con 5 semillas en cada modo, contribución terminal contra por paso, solo en el train del fold 1. |

## 5. Mapa "pregunta probable → dónde ocurre"

| Pregunta del profe | Archivo y función |
|---|---|
| ¿A qué precio se llena la orden? | `env.py` `_run_trade`: `entry_price = d.open[t + 1]` |
| ¿Qué pasa si el stop y el TP caen en la misma barra? | `env.py` `_first_exit`, rama `hit_stop and hit_take`, con `intrabar_priority` |
| ¿Y si un hueco salta el stop? | `env.py` `_gap_through`, llamado desde `_first_exit` |
| ¿Dónde está la lógica semi-Markov? | `env.py` `step`: una operación avanza de `t` a `exit_bar`, e `info["tau_horas"]` trae la duración |
| ¿Cómo se usa la duración real en el aprendizaje? | `env.py` `SMDPReplayBuffer._get_samples`: `discounts = gamma_per_hour ** tau` |
| ¿Dónde se cobran los costos? | `reward.py` `trade_accounting` (0.125% por lado) |
| ¿La contribución es por paso o terminal? | `reward.py` `log_return_contribution` (terminal). La alternativa es `per_bar_contributions` |
| ¿Cómo evitan acciones inválidas? | `env.py` `step`: solo se decide estando fuera, con `Discrete(3)` |
| ¿Cómo evitan usar el futuro en la observación? | `env.py` `_observation` usa la barra t − retraso. Las variables son causales (`data.py`, `test_causality.py`) |
| ¿Dónde se normaliza y con qué datos? | `env.py` `load_split` → `data.Normalizer(...).fit(train del fold)` |
| ¿Qué pasa al final del episodio? | `env.py` `_is_truncated` (`truncated=True`). El embargo vive en `__init__` (`last_decision`) |
| ¿Dónde está la bitácora? | `env.py` `_run_trade` (registro), `trade_log_frame`, `TRADE_LOG_COLUMNS` |
| ¿Cómo prueban que el entorno es correcto? | `tests/test_reference_trade.py` (cálculos a mano) y `tests/test_env.py` (check_env, cordura, identidad) |
| ¿Cómo midieron la sensibilidad al retraso? | `sensitivity.py` `run_sensitivity`, escenario `retraso_1_barra` (`data_delay_bars=1`) |
| ¿Cuánto cuesta el impacto de mercado? | `sensitivity.py` `impact_per_side` y `execution_assumptions` |
