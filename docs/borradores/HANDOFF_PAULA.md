# Handoff para Paula: cómo usar el entorno (rama `arturo/entorno`)

Todo lo de aquí está probado en la rama `arturo/entorno`, que ya incluye los commits de Milca: 28 pruebas pasan con `pytest`. Los ejemplos suponen que el script corre desde la raíz del repo con `code/` en `sys.path`, como hace `main.py`.

## 1. Crear el entorno

```python
import env as E

train_env = E.make_env(fold, "train")   # ventanas de 720 h con inicio al azar (episodio)
valid_env = E.make_env(fold, "valid")   # la partición completa, desde su primera barra
test_env  = E.make_env(fold, "test")    # SOLO para la corrida final, con todo congelado
```

- `make_env(fold, split, drop_families=(), **kwargs)`. Cualquier parámetro de `TradingEnv` se puede sobrescribir, por ejemplo `intrabar_priority="tp_first"` o `data_delay_bars=1`.
- `load_split(fold, split)` devuelve un `SplitData`, con las barras y las 21 variables ya normalizadas por el `Normalizer` de Milca (ajustado solo con el train del fold).
- Para evaluar un modelo usa `E.rollout(env, policy)`, con `policy = lambda obs: model.predict(obs, deterministic=True)[0]`. Devuelve `capital_inicial`, `capital_final`, `decisiones`, `bitacora` (DataFrame) y `capital` (curva por decisión).
- **Políticas base:** `E.BASELINES = {"aleatoria", "siempre_largo", "siempre_fuera"}`. Son fábricas: `E.BASELINES["aleatoria"](env)` devuelve la política.

## 2. Contrato de `info` en cada `step`

| Clave | Contenido |
|---|---|
| `tau_horas` | Horas reales entre esta decisión y la siguiente: 1 si se quedó fuera, la duración hasta el cierre si operó. |
| `capital` | Capital después del paso (USDT). |
| `operacion` | `None` si no cerró operación; si cerró, el registro de la bitácora (las mismas claves de la tabla siguiente). |

`terminated` siempre es `False`. El fin de ventana o de partición es `truncated=True`. `DummyVecEnv` lo pasa como `TimeLimit.truncated` y SB3 hace bootstrap.

## 3. Bitácora → `trade_log_fold_{k}.csv`

`env.trade_log_frame()` devuelve exactamente estas columnas (`E.TRADE_LOG_COLUMNS`):

`entrada_ts, salida_ts, lado, motivo_salida, precio_entrada, precio_salida, stop, take_profit, tamano_btc, nocional_usdt, pnl_bruto, costos, pnl_neto, capital_antes, capital_despues, duracion_horas, ambigua, cruza_hueco, entrada_post_hueco`

- Cubre todo lo que pide la sección 3.8: marcas de tiempo, razón de salida (`stop | take_profit | horizonte | fin_split`), precios, tamaño, P&L bruto, costos y P&L neto.
- **Identidad contable:** `capital_final == capital_inicial + bitacora.pnl_neto.sum()` (verificada en `tests/test_env.py`, con error de ~1e-9).
- Para la prueba obligatoria 4 basta afirmar eso sobre un rollout.

## 4. DQN con el descuento semi-Markov

```python
from stable_baselines3 import DQN
model = DQN("MlpPolicy", train_env, seed=seed,
            replay_buffer_class=E.SMDPReplayBuffer,
            replay_buffer_kwargs={"gamma_per_hour": 0.999})
```

- `SMDPReplayBuffer` descuenta cada transición con 0.999^τ, donde τ son sus horas reales. El `gamma` del constructor de DQN queda sin efecto.
- No uses `optimize_memory_usage=True`: el buffer lo rechaza.
- **PPO:** su GAE usa un γ fijo por decisión. Es una limitación declarada en mi reporte; no hace falta hacer nada.

## 5. VecNormalize: cuidado con la doble normalización

- Las observaciones **ya vienen normalizadas** (mediana/IQR del train del fold, recortadas a ±5) y las físicas son pequeñas. **No uses `VecNormalize(norm_obs=True)`**: normalizarías dos veces y además con estadísticos que cambian durante el entrenamiento.
- Si quieres normalizar la recompensa (`norm_reward=True`), congela el wrapper en valid y test (`training=False`, `norm_reward=False`) y guarda sus estadísticos con el modelo. No hace falta: la recompensa ya está escalada (100 · log, típicamente ±3).

## 6. Sensibilidad con tus modelos (pregunta 6 y sección 3.10)

```python
from sensitivity import run_sensitivity, model_policy
# UNA vez por fold, con el modelo congelado de ese fold:
run_sensitivity({"ppo": model_policy(ppo_k), "dqn": model_policy(dqn_k)},
                folds=[k], splits=["test"],
                out_path=E.data.ARTIFACTS_DIR / f"sensitivity_test_fold_{k}.csv")
```

- Ya existen las cifras con las políticas base en train y valid (`artifacts/sensitivity_baselines.csv`), más `execution_assumptions.csv` y `gross_edge_test.csv`. Se regeneran con `python code/sensitivity.py baselines`.
- Escenarios incluidos: `base`, `tp_first`, `retraso_1_barra` y 8 de impacto (0.1, 1, 10 y 50 M USDT × Y = 0.5 y 1).

## 7. Ablación por familia (pregunta 7)

```python
E.make_env(fold, "train", drop_families=("volumen",))
```

- Las 6 familias de `data.build_features` son: `retornos`, `volatilidad`, `volumen`, `tendencia_momentum`, `tiempo` y `creencia`.
- Las barreras siguen usando `vol_24` aunque quites la familia de volatilidad, así que el simulador es el mismo y solo cambia lo que observa el agente.

## 8. Los dos algoritmos

PPO y DQN de SB3, los dos con `MlpPolicy` y `Discrete(3)`; no hace falta máscara. DQN con `SMDPReplayBuffer`.

## 9. Estimación de cómputo (medida en esta máquina, 2026-11-19)

**Velocidades medidas:**
- Entorno solo: ≈ 3,850 pasos/s.
- PPO: 5,000 pasos en 14.9 s, ≈ 335 pasos/s.
- DQN: 5,000 pasos en 7.8 s, ≈ 640 pasos/s.
- Un rollout de evaluación en una partición de un año tarda unos segundos.

**Ejemplo de plan:** 200k pasos por modelo.
- Un PPO ≈ 10 min y un DQN ≈ 5.2 min.

| Bloque | Corridas | Tiempo secuencial |
|---|---|---|
| Búsqueda de hiperparámetros: 4 folds × 2 algoritmos × 4 configuraciones × 3 semillas | 96 | 48 × 10 + 48 × 5.2 ≈ 12.2 h |
| Ablación: 4 folds × 6 familias × PPO × 1 semilla con la mejor configuración | 24 | ≈ 4 h |
| Modelos finales: 4 folds × 2 algoritmos × 5 semillas | 40 | 20 × 10 + 20 × 5.2 ≈ 5.1 h |
| **Total** | 160 | **≈ 21 h** secuencial |

Cómo bajarlo:
- Usar 100k pasos (la mitad de tiempo).
- Correr varios procesos a la vez, uno por fold. Con 4 núcleos queda en ≈ 5–6 h.
- Hacer la búsqueda de hiperparámetros solo en el fold 1 y reutilizarla, declarándolo.

## 10. Qué te falta y de qué depende

| Pendiente | Depende de |
|---|---|
| `train.py`: entrenamiento por fold y semilla, búsqueda en valid, `validation_results_fold_{k}.csv`, `training_curves_fold_{k}.png` | Nada: el entorno ya está listo. |
| `evaluate.py`: métricas de la sección 3.7, bitácora, `equity_curves_test_fold_{k}.png`, `metrics_summary.csv` | El benchmark de S&P 500, que depende de Milca y de la respuesta del profe. |
| `main.py`: pipeline completo con semilla 42, que incluye `data.write_feature_analysis`, `plots.write_eda_pdf`, entrenamiento, evaluación y `sensitivity` | Que `milca/datos` y `arturo/entorno` estén en `main`. |
| Prueba de identidad contable | Nada (ver la sección 3). |
| Metadatos de cada corrida con las versiones de SB3, Gymnasium y torch (sección 2.2) | Nada: `stable_baselines3.__version__` y equivalentes. |
| Corrida final en test (una sola vez) y sensibilidad con tus modelos | Haber congelado los hiperparámetros con valid. |

**Cosas a revisar en tus resultados** (veredicto del council 2):
1. Compara siempre contra el agente aleatorio y contra "siempre largo" con las mismas barreras. En el train del fold 1, la ventaja bruta de "siempre largo" es +0.011% por operación, con t = 0.17: es ruido.
2. Reporta la dispersión entre semillas. En pruebas cortas de DQN en el train del fold 1 salió un capital final de 0.11× a 2.93× según la semilla.
3. El impacto estimado de una orden de 1 M USDT (0.14–0.24% por lado) es del orden de la comisión. Si un resultado no sobrevive al escenario `impacto_1M_Y1`, es un artefacto del simulador.
