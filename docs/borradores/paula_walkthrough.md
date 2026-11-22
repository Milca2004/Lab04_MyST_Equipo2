# Walkthrough de Paula: `train.py`, `evaluate.py` y `main.py`

Recorrido del código de entrenamiento, evaluación y del pipeline. Aquí no se repite el entorno: eso está en el walkthrough de Arturo (`env.py`, `reward.py`, `sensitivity.py`) y en los docstrings de Milca (`data.py`, `plots.py`).

## 1. Flujo de `python main.py`

```
main.py
 ├─ stage_data             data.write_feature_analysis + plots.write_eda_pdf   (Milca)
 ├─ stage_search_and_select
 │    train.search_tasks  → 72 corridas (4 folds × PPO/DQN × 3 lr × 3 semillas)
 │    train.run_many      → cada corrida: train_run (entrena y evalúa en VALID en 25k/50k/100k)
 │    train.select_configs → mejor (lr, pasos) por fold y algoritmo: mediana entre semillas
 │    train.final_tasks   → 2 semillas más (45, 46) con la configuración elegida
 │    choose_primary_algorithm → PPO o DQN según valid
 │    escribe validation_results_fold_k.csv, training_curves_fold_k.png,
 │            hyperparameter_selection.csv, final_models.csv, run_metadata.csv
 ├─ stage_ablation          quita una familia a la vez (VALID)  → ablation_valid.csv
 ├─ stage_test              ÚNICA vez que se toca el test
 │    evaluate.evaluate_test por fold (5 semillas × 2 algoritmos + políticas base + benchmarks)
 │    escribe trade_log_fold_k.csv, equity_curves_test_fold_k.png, metrics_summary.csv,
 │            test_seed_dispersion.csv, worst_trades_test.csv, drawdown_period_test.*
 └─ stage_sensitivity       sensitivity.run_sensitivity (Arturo) con los modelos representativos
                            → sensitivity_test_fold_k.csv
```

- **Semilla:** 42, fijada en `main.SEED`. `main` verifica que coincida con `data.SEED`. `set_global_seed` fija random, numpy y torch, y cada modelo de SB3 recibe además su propia semilla (42 a 46).
- **Windows:** todo corre dentro de `if __name__ == "__main__":`. Los entornos usan `DummyVecEnv` y el paralelismo usa `ProcessPoolExecutor` con contexto `"spawn"`, el único que existe en Windows.
- **Caché:** cada corrida deja `models/fold_k/<nombre>.json` con metadatos, filas de valid y episodios. Si existe con los mismos checkpoints, `train_run` lo reutiliza; `--force` reentrena todo. `models/` está en `.gitignore`.
- **Paralelismo:** `--jobs N` (por defecto, el mínimo entre 4 y el número de núcleos). Cada proceso usa un hilo de torch.

## 2. `code/train.py`

| Función | Qué hace | Decisión no evidente |
|---|---|---|
| `FIXED_KWARGS` | Hiperparámetros que no se buscan. | DQN con buffer de 100k, `learning_starts` = 1k y target cada 1k pasos, porque con 100k pasos los valores por defecto (pensados para Atari) solo actualizarían el objetivo 10 veces. `SMDPReplayBuffer` con 0.999^τ. |
| `build_model` | Crea PPO o DQN con `MlpPolicy`, `Monitor` y `DummyVecEnv`. | Sin `VecNormalize`: las observaciones ya están normalizadas. |
| `TrainingLogger` | Callback que registra cada episodio de train (recompensa, fracción de decisiones que operan) y, en cada checkpoint, guarda el modelo y lo evalúa en valid. | Los pasos son parte de la rejilla sin costo extra: una corrida de 100k da tres puntos de la rejilla. |
| `evaluate_on_valid` | Rollout determinista en toda la partición de valid. | Reutiliza `sensitivity.equity_metrics` de Arturo (no se duplica la métrica). |
| `train_run` | Una corrida completa más su JSON de metadatos. | Fija `torch.set_num_threads(1)`, porque las corridas van en paralelo. |
| `run_many` | Ejecuta corridas en serie o en paralelo y junta resultados. | Contexto `"spawn"` para que funcione igual en Mac, Linux y Windows. |
| `select_configs` | Mejor (lr, pasos) por fold y algoritmo. | Criterio fijado antes de correr: mediana entre semillas de log C_T/C_0 en valid; desempate por mayor mínimo. Marca `lr_en_borde` y `pasos_en_borde`. |
| `final_tasks` / `final_models_table` | Completa 5 semillas y marca el representativo. | El representativo es la semilla con la **mediana** de valid, no la mejor: elegir la mejor sería sobreajustar a valid. |
| `plot_training_curves` | `training_curves_fold_k.png`. | Anota los checkpoints, el número de pasos elegido y el punto de mayor cambio en la frecuencia de operación. |

## 3. `code/evaluate.py`

| Función | Qué hace | Decisión no evidente |
|---|---|---|
| `daily_equity` | Curva de capital del entorno a capital diario. | El capital solo cambia al cerrar operaciones; se antepone el capital inicial el día anterior al inicio. |
| `buy_and_hold_btc` | Benchmark: comprar en el primer close y vender en el último. | Paga 0.125% en la entrada y en la salida, igual que el agente. |
| `load_sp500` / `sp500_index` | Benchmark del S&P 500. | Listo para `data/sp500_daily.csv`; si no existe, devuelve `None` y las métricas quedan en NaN. |
| `equity_curve_metrics` | Retorno total y anualizado, volatilidad, Sharpe, Sortino, drawdown máximo y Calmar. | Anualización con 365 días (cripto); 252 solo para el S&P. Retorno anualizado con días de calendario, para que el alfa compare lo mismo. |
| `trade_metrics` | Hit rate, duración promedio, número de operaciones y rotación. | Rotación = nocional de entrada más salida, entre el capital medio y entre los años. |
| `strategy_row` | Junta todas las métricas, la correlación diaria con BTC y el alfa contra cada benchmark. | Alfa = retorno anualizado en exceso, no el intercepto de una regresión. |
| `evaluate_test` | La **única** función que corre el test: modelos finales, `E.BASELINES` y benchmarks. | El parámetro `split` existe solo para depurar sobre valid; `main.py` siempre usa `"test"`. |
| `summarize_metrics` / `seed_dispersion` | `metrics_summary.csv` (por fold, media y desviación) y `test_seed_dispersion.csv`. | El resumen usa el modelo representativo; la dispersión entre las 5 semillas va aparte. |
| `write_trade_log` | `trade_log_fold_k.csv` con `E.TRADE_LOG_COLUMNS` más una columna `algoritmo`. | Se usa `env.trade_log_frame()`; no se reconstruye la bitácora. |
| `worst_trades` / `state_at_entry` | Las 3 operaciones con mayor pérdida y el estado en la decisión. | La decisión ocurre una barra antes de `entrada_ts`, porque el llenado es en el open de t+1. Se guardan los valores crudos y los normalizados. |
| `max_drawdown_period` / `plot_drawdown_period` | Mayor drawdown en test, con precio, acciones y capital. | Se mide sobre la curva por decisión, no la diaria, para fijar el pico y el valle exactos. |

## 4. `tests/test_accounting.py` (prueba obligatoria 4)

- `capital_final == capital_inicial + Σ pnl_neto` con rtol 1e-9, con las políticas aleatoria y siempre largo, en una ventana fija de 30 días del train del fold 1.
- `pnl_neto = pnl_bruto − costos` en cada fila, y el capital se encadena sin huecos entre operaciones.
- La identidad se cumple también con la contribución por paso: cambia la recompensa, no la contabilidad.

## 5. Mapa "pregunta probable → archivo y función"

| Pregunta del profesor | Dónde está |
|---|---|
| ¿Dónde se fija la semilla? | `main.SEED`, `main.set_global_seed` y `seed=` en `train.build_model` |
| ¿Cómo garantizan que el test se toca una sola vez? | `main.stage_test`, que llama a `evaluate.evaluate_test`; ninguna otra función usa `"test"`. La selección ocurre antes, en `train.select_configs`, solo con valid. |
| ¿Cuál es el criterio de selección de hiperparámetros? | `train.SELECTION_METRIC` y `train.select_configs` |
| ¿Por qué 4 modelos y no uno? | `train.train_run(fold, ...)` crea entornos con `make_env(fold, ...)`; `final_models.csv` tiene un modelo distinto por fold. |
| ¿Cómo se descuenta en DQN según la duración? | `train.FIXED_KWARGS["dqn"]` → `env.SMDPReplayBuffer` (Arturo) |
| ¿Por qué no usan VecNormalize? | Docstring de `train.py` y `train.build_model` |
| ¿Cómo calculan Sharpe, Sortino y la anualización? | `evaluate.equity_curve_metrics` y `evaluate.DAYS_PER_YEAR` |
| ¿El benchmark paga costos? | `evaluate.buy_and_hold_btc` y `evaluate.sp500_index` (`cost_rate`) |
| ¿Qué es la rotación y el alfa? | `evaluate.trade_metrics` y `evaluate.strategy_row` |
| ¿De dónde salen las 3 peores operaciones y su estado? | `evaluate.worst_trades` y `evaluate.state_at_entry` → `worst_trades_test.csv` |
| ¿Dónde se verifica la identidad contable? | `tests/test_accounting.py` |
| ¿Cómo se hizo la ablación? | `main.stage_ablation` → `env.make_env(..., drop_families=...)` |
| ¿De dónde sale la dispersión entre semillas? | `evaluate.seed_dispersion` → `test_seed_dispersion.csv` |
| ¿Cómo corre la sensibilidad con sus modelos? | `main.stage_sensitivity` → `sensitivity.run_sensitivity` (Arturo) |
| ¿Cuánto tarda y cómo se reanuda? | `train.train_run` (caché JSON) y `main.parse_args` (`--jobs`, `--force`) |
