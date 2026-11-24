# Lab04_MyST_Equipo2 · Agente de RL para trading de BTCUSDT

Microestructuras y Sistemas de Trading (ITESO), Laboratorio 04. En este laboratorio se entrena un agente de reinforcement learning (Stable-Baselines3) para operar BTCUSDT con datos horarios. El problema se formula en el marco de Powell como un problema semi-Markov y se evalúa con un esquema walk-forward de 4 folds.

## Integrantes

| Nombre | Usuario de GitHub | Módulos |
|---|---|---|
| Milca Lilian Correa Sanchez | Milca2004 | `code/data.py`, `code/plots.py` (EDA), `tests/test_causality.py` |
| Arturo Santillanes Llamas | ArturoSantillanes | `code/env.py`, `code/reward.py` |
| Paula Ines Pelayo Morales | paulapelayo | `code/train.py`, `code/evaluate.py`, `main.py` |

## Nivel de alcance

Desarrollamos el enunciado completo.

## Instalación

Requiere **Python 3.12** (las versiones se verificaron con Python 3.12.2). Las versiones de `requirements.txt` se comprobaron en dos sistemas: se instalaron en macOS y se confirmó que todas tienen instalador para Windows de 64 bits.

**Mac / Linux**

```bash
git clone https://github.com/Milca2004/Lab04_MyST_Equipo2.git
cd Lab04_MyST_Equipo2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Milca2004/Lab04_MyST_Equipo2.git
cd Lab04_MyST_Equipo2
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación, ejecuta antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Reproducción

Con el entorno activado, desde la raíz del repo y en cualquier sistema:

```bash
python main.py
```

Este comando corre el pipeline completo y escribe todos los artefactos en `artifacts/`.

**Tiempo de ejecución.** En un clon limpio no existe `models/` (está en .gitignore), así que `python main.py` entrena todos los modelos: 88 corridas que suman ~1.5 h de cómputo; con `--jobs 4` tardó ~26 min (artifacts/run_metadata.csv), más ablación y evaluación, en total ~30–45 min. Si `models/` existe, reutiliza los modelos y solo recalcula la evaluación. `--force` reentrena todo.

**Semilla:** 42. Está fijada en `main.py` y se usa en todo lo que tiene aleatoriedad. Aun con las semillas fijas, los números pueden variar un poco entre sistemas operativos y procesadores, porque las operaciones de punto flotante de numpy y torch no son idénticas en todas las plataformas.

**Nota de reproducibilidad.** Con la semilla 42, los datos, el EDA y las políticas sin entrenamiento (aleatoria, siempre largo, siempre fuera, comprar y mantener BTC y S&P 500) se reproducen hasta el redondeo (diferencias ≤ 1e-12). PPO y DQN no: varían entre máquinas y versiones de torch. En una corrida limpia en un clon nuevo (macOS, Python 3.12.2, torch 2.15.0 sin CUDA) dieron PPO −59.0% y DQN −47.9% de retorno total (media de los 4 folds de test); los documentos reportan la corrida de `artifacts/` (Python 3.12.3, torch 2.15.0+cu132), con PPO −51.4% y DQN −28.1%. La conclusión se mantiene: 0 de 40 modelos ganan en la corrida limpia (1 de 40 en la de `artifacts/`), |t| ≤ 2 de la ventaja bruta en las dos corridas y, en promedio, los dos pierden contra comprar y mantener BTC y contra el S&P 500. Lo que no es estable entre corridas son las cifras puntuales y el orden de la ablación por familia.

## Pruebas

```bash
pytest
```

Son 36 pruebas y tardan unos 12 segundos. Funciona igual en Mac, Linux y Windows. `tests/conftest.py` agrega `code/` a la ruta de importación.

## Correspondencia con la clase S19

Mapa entre las piezas de un `gym.Env` que vimos en la clase S19 ("Deep RL en la práctica") y nuestro entorno.

| Pieza de la clase | En nuestro entorno (`code/env.py`) |
|---|---|
| Mapa de lo que ocurre y acciones | Barras horarias de BTCUSDT (OHLCV, con huecos). Acciones: quedarse fuera, abrir largo o abrir corto. Cerrar lo hacen las barreras (decisión a). |
| `__init__` | `action_space = Discrete(3)` y `observation_space = Box(-10, 10, (24,), float32)`. Parámetros: horizonte 24 h, stop y take-profit a ±1·vol_24·√24, costo 0.00125 por lado, capital inicial 1,000,000. |
| `_obs` | `_observation()`: las 21 variables normalizadas de `data.py` en la barra t − `data_delay_bars`, más 3 valores de estado (decisión c). |
| `reset` | `super().reset(seed=seed)` crea `self.np_random`. En train, `_sample_start` elige con `self.np_random.integers` una barra de inicio al azar y la ventana dura 720 h; en valid y test empieza en la primera barra. |
| `step` | Un paso es una decisión. Si abre, simula la operación hasta su cierre (`_run_trade`) y paga 100·log(capital después / antes), neto de 0.125% por lado. Si se queda fuera, la recompensa es 0. |
| `terminated` / `truncated` | Decisión d. |
| Validación en 3 niveles | `check_env` sin advertencias (`tests/test_env.py`); operación de referencia calculada a mano (`tests/test_reference_trade.py`); líneas base con costos (aleatoria en `tests/test_env.py`, siempre fuera y siempre largo en `tests/test_baselines.py`). Guardar y cargar da las mismas acciones (`tests/test_save_load.py`). |

Decisiones de diseño frente a la clase:

| | Decisión de diseño | Por qué |
|---|---|---|
| a | **No hay acción "cerrar".** El stop, el take-profit o las 24 h cierran la operación. | Diseño semi-Markov de Powell: el agente decide solo cuando está fuera. El PDF del lab dice "si su diseño la permite", y el nuestro no la necesita. |
| b | **No hay acciones inválidas.** | Solo se decide estando fuera, así que abrir, quedarse fuera o abrir al otro lado siempre es válido. No hace falta máscara ni penalización. |
| c | **La observación no trae posición ni precio de entrada.** | Al decidir siempre está fuera (posición 0, sin precio de entrada ni tiempo en posición). Los 3 valores de estado son: `log(C_t / C_0)` (capital acumulado), `C_t / C_máx − 1` (drawdown del episodio, ≤ 0) y `log1p(horas faltantes) / log1p(145)` (cuánto hueco de datos hubo antes de la barra observada). Los dos primeros son componente física R_t; el tercero está en el código junto a ellos, pero en la formulación de Powell es información I_t (estado del calendario de datos). La variante `per_bar` agrega lado y fracción del horizonte transcurrida; no es el diseño. |
| d | **terminated siempre es False; truncated marca el fin de la ventana (720 h en train) o de la partición.** | No hay ruina ni límite de riesgo, y el mercado no se acaba, por eso el fin de datos es truncamiento y SB3 hace bootstrap. Nunca se trunca con una operación abierta: en el diseño la operación se cierra dentro del mismo `step`, y en `per_bar` `_is_truncated` devuelve `False` mientras haya pasos pendientes. |
| e | **No se usa VecNormalize.** | Normalizador propio (`data.Normalizer`, mediana e IQR) ajustado solo con el train de cada fold y aplicado sin cambios a valid y test. |
| f | **Se usa env.rollout en lugar de evaluate_policy.** | Necesitamos la bitácora de operaciones y la curva de capital, y cada paso dura una duración variable τ (de 1 a 24 h, o más con huecos), así que la recompensa media por paso no es comparable. La política es la misma: `predict(deterministic=True)`. |

## Compatibilidad Mac / Windows

- Todas las rutas usan `pathlib` y son relativas a la raíz del repo. El código no llama comandos de shell.
- Toda lectura y escritura de archivos usa `encoding="utf-8"`.
- Las figuras usan el backend `Agg` de matplotlib, que no abre ventanas.
- Los entornos vectorizados usan `DummyVecEnv` y el pipeline corre dentro de `if __name__ == "__main__":`, porque Windows arranca los subprocesos con "spawn".

## Datos

El archivo `data/Binance_BTCUSDT_1h.csv` es una copia del archivo publicado en Canvas. Se incluye en el repo para que el pipeline corra con un solo comando. Es la única carpeta fuera de la estructura del enunciado, y está pendiente confirmarlo con el profesor.

`data/sp500_daily.csv` tiene los cierres diarios del índice S&P 500 (2022-12-01 a 2026-04-30, 855 días hábiles; se quitaron los festivos sin cierre). Fuente: FRED, Federal Reserve Bank of St. Louis, serie `SP500` (https://fred.stlouisfed.org/series/SP500), descargada el 2026-11-22.
Se usa solo para el benchmark de indexar al S&P 500 (`code/evaluate.py`); no entra al estado ni al entrenamiento del agente.

## Uso de asistencia de IA

- **Milca:** usé Claude Code como asistente para escribir `code/data.py`, mis funciones de `code/plots.py`, `tests/test_causality.py`, `tests/test_save_load.py` y `tests/test_baselines.py` (estas dos últimas a partir de la clase S19; comprobé corriendo `pytest` que pasan), y para generar `artifacts/btcusdt_eda.pdf` y `artifacts/feature_analysis_fold_{1,2,3,4}.csv`. Las cifras de limpieza del CSV, los folds y los huecos se verificaron corriendo el código. El reporte, el walkthrough y la presentación (`docs/reporte.pdf`, `docs/walkthrough.pdf` y `docs/presentacion.pdf`) se armaron con Claude Code a partir de los borradores y los CSV de `artifacts/`.
- **Arturo:** usé Claude Code como asistente para escribir `code/env.py`, `code/reward.py`, `code/sensitivity.py`, `tests/test_env.py` y `tests/test_reference_trade.py`, y como apoyo de diseño mediante la skill `llm-council` (sus veredictos y las correcciones que les hice están en mi borrador de reporte). Verifiqué corriendo código: `check_env` sin advertencias, las operaciones de referencia contra cálculos a mano, la identidad contable, que el agente aleatorio pierde en el train de los 4 folds y todas las cifras de diseño (solo con train). También revisé en el código fuente de Stable-Baselines3 2.9.2 cómo DQN aplica el descuento.
- **Paula:** usé Claude Code como asistente para escribir `code/train.py`, `code/evaluate.py`, `main.py` y `tests/test_accounting.py`, y para correr el pipeline y redactar mis borradores (`docs/borradores/paula_*.md`). Verifiqué corriendo código: las 32 pruebas previas más la identidad contable, que el test se evalúa una sola vez después de fijar todo con valid, y que cada cifra de mis borradores sale de los CSV de `artifacts/`.
