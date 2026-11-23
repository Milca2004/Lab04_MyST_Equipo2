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

**Semilla:** 42. Está fijada en `main.py` y se usa en todo lo que tiene aleatoriedad. Aun con las semillas fijas, los números pueden variar un poco entre sistemas operativos y procesadores, porque las operaciones de punto flotante de numpy y torch no son idénticas en todas las plataformas.

## Pruebas

```bash
pytest
```

Funciona igual en Mac, Linux y Windows. `tests/conftest.py` agrega `code/` a la ruta de importación.

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

- **Milca:** usé Claude Code como asistente para escribir `code/data.py`, mis funciones de `code/plots.py` y `tests/test_causality.py`, y para generar `artifacts/btcusdt_eda.pdf` y `artifacts/feature_analysis_fold_{1,2,3,4}.csv`. Las cifras de limpieza del CSV, los folds y los huecos se verificaron corriendo el código.
- **Arturo:** usé Claude Code como asistente para escribir `code/env.py`, `code/reward.py`, `code/sensitivity.py`, `tests/test_env.py` y `tests/test_reference_trade.py`, y como apoyo de diseño mediante la skill `llm-council` (sus veredictos y las correcciones que les hice están en mi borrador de reporte). Verifiqué corriendo código: `check_env` sin advertencias, las operaciones de referencia contra cálculos a mano, la identidad contable, que el agente aleatorio pierde en el train de los 4 folds y todas las cifras de diseño (solo con train). También revisé en el código fuente de Stable-Baselines3 2.9.2 cómo DQN aplica el descuento.
- **Paula:** usé Claude Code como asistente para escribir `code/train.py`, `code/evaluate.py`, `main.py` y `tests/test_accounting.py`, y para correr el pipeline y redactar mis borradores (`docs/borradores/paula_*.md`). Verifiqué corriendo código: las 28 pruebas previas más la identidad contable, que el test se evalúa una sola vez después de fijar todo con valid, y que cada cifra de mis borradores sale de los CSV de `artifacts/`.
