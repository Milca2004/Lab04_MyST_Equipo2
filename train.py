"""Entrenamiento con Stable-Baselines3 por fold, algoritmo, configuración y semilla.

Dueña: Paula.

Protocolo (sección 3.5 y 3.6 del enunciado):

* Un modelo por (fold, algoritmo, configuración, semilla). Nunca se reutiliza
  un modelo entre folds.
* La búsqueda de hiperparámetros usa SOLO la partición de validación. El test
  no se toca en este módulo (lo hace ``evaluate.py`` una sola vez).
* Rejilla (fijada antes de correr): ``learning_rate`` en LEARNING_RATES y número
  de pasos en CHECKPOINTS. Los pasos no requieren corridas extra: cada corrida
  guarda el modelo y lo evalúa en valid al llegar a cada checkpoint.
* Criterio de selección (fijado antes de correr): la MEDIANA entre semillas del
  log del capital final en valid, log(C_T / C_0). Es la función objetivo del
  diseño (contribución terminal = 100 * log del cambio de capital), así que se
  selecciona con la misma métrica que se optimiza. Desempate: mayor mínimo entre
  semillas (más robusto).
* Modelos finales: la configuración elegida con FINAL_SEEDS semillas (las de la
  búsqueda se reutilizan; se entrenan las que falten). El "modelo representativo"
  del fold es la semilla con la mediana de valid entre las finales.

Hiperparámetros FIJOS (no se buscan, para que la rejilla sea pequeña):
  - PPO: los valores por defecto de SB3 2.9 (n_steps=2048, batch 64, gamma=0.99
    por decisión, GAE 0.95). Limitación declarada por Arturo: el gamma de PPO es
    por decisión, no por hora.
  - DQN: SMDPReplayBuffer con gamma_per_hour=0.999 (descuento 0.999**tau),
    buffer de 100k, learning_starts=1,000, target_update_interval=1,000 y
    exploration_fraction=0.2. Los valores por defecto de SB3 (buffer de 1M,
    target cada 10,000 pasos) están pensados para Atari con millones de pasos;
    con 100k pasos el objetivo solo se actualizaría 10 veces.
  - Sin VecNormalize: las observaciones ya vienen normalizadas por el
    Normalizer de data.py (ajustado con el train del fold) y la recompensa ya
    está escalada (100 * log, típicamente ±3).
"""
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import data
import env as E
from sensitivity import equity_metrics

ARTIFACTS_DIR = data.ARTIFACTS_DIR
MODELS_DIR = data.REPO_ROOT / "models"

ALGORITHMS = ("ppo", "dqn")
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
CHECKPOINTS = (25_000, 50_000, 100_000)
SEARCH_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
SELECTION_METRIC = "log_capital_final"
GAMMA_PER_HOUR = 0.999

FIXED_KWARGS = {
    "ppo": {},
    "dqn": {
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.2,
        "replay_buffer_class": E.SMDPReplayBuffer,
        "replay_buffer_kwargs": {"gamma_per_hour": GAMMA_PER_HOUR},
    },
}
ALGORITHM_CLASSES = {"ppo": PPO, "dqn": DQN}


# ---------------------------------------------------------------------------
# Metadatos y rutas
# ---------------------------------------------------------------------------

def library_versions():
    """Versiones de las librerías que exige registrar la sección 2.2."""
    return {
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def run_name(algo, lr, seed, drop_families=()):
    """Nombre único de una corrida (sin el número de pasos)."""
    suffix = "_sin_" + "_".join(drop_families) if drop_families else ""
    return f"{algo}_lr{lr:g}_s{seed}{suffix}"


def checkpoint_path(fold, name, steps):
    """Ruta del modelo guardado al llegar a ``steps`` pasos."""
    return MODELS_DIR / f"fold_{fold}" / f"{name}_{steps}.zip"


def load_model(fold, algo, lr, seed, steps, drop_families=()):
    """Carga un checkpoint guardado por ``train_run``."""
    return ALGORITHM_CLASSES[algo].load(checkpoint_path(fold, run_name(algo, lr, seed, drop_families), steps),
                                        device="cpu")


# ---------------------------------------------------------------------------
# Evaluación en validación (NUNCA en test)
# ---------------------------------------------------------------------------

def evaluate_on_valid(model, valid_env):
    """Corre la política determinista en toda la partición de validación.

    Devuelve las métricas de ``sensitivity.equity_metrics`` (retorno total, log
    del capital final, Sharpe diario anualizado con 365 días, drawdown máximo) y
    un resumen de operaciones.
    """
    result = E.rollout(valid_env, lambda obs: model.predict(obs, deterministic=True)[0])
    log = result["bitacora"]
    return {
        **equity_metrics(result["capital"]),
        "operaciones": len(log),
        "frac_decisiones_que_operan": len(log) / max(result["decisiones"], 1),
        "retorno_neto_medio_op": float((log["pnl_neto"] / log["capital_antes"]).mean()) if len(log) else np.nan,
        "hit_rate": float((log["pnl_neto"] > 0).mean()) if len(log) else np.nan,
    }


class TrainingLogger(BaseCallback):
    """Registra cada episodio de train y evalúa en valid en cada checkpoint.

    Episodio de train = ventana de 720 h con inicio al azar. Su recompensa es la
    suma de contribuciones (100 * log del cambio de capital en la ventana).
    """

    def __init__(self, fold, name, valid_env, checkpoints):
        super().__init__()
        self.fold, self.name, self.valid_env = fold, name, valid_env
        self.pending = sorted(checkpoints)
        self.episodes, self.valid_rows = [], []
        self._trades = 0
        self._decisions = 0

    def _on_step(self):
        """Acumula operaciones por episodio y dispara los checkpoints."""
        for info in self.locals["infos"]:
            self._decisions += 1
            self._trades += info.get("operacion") is not None
            if "episode" in info:
                self.episodes.append({
                    "pasos": self.num_timesteps,
                    "recompensa_episodio": info["episode"]["r"],
                    "decisiones": self._decisions,
                    "frac_opera": self._trades / max(self._decisions, 1),
                })
                self._trades = self._decisions = 0
        while self.pending and self.num_timesteps >= self.pending[0]:
            steps = self.pending.pop(0)
            path = checkpoint_path(self.fold, self.name, steps)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            self.valid_rows.append({"pasos": steps, **evaluate_on_valid(self.model, self.valid_env)})
        return True


# ---------------------------------------------------------------------------
# Una corrida
# ---------------------------------------------------------------------------

def build_model(algo, train_env, lr, seed):
    """Crea el modelo SB3 con MlpPolicy, los hiperparámetros fijos y la semilla."""
    venv = DummyVecEnv([lambda: Monitor(train_env)])
    return ALGORITHM_CLASSES[algo]("MlpPolicy", venv, learning_rate=lr, seed=seed, verbose=0,
                                   device="cpu", **FIXED_KWARGS[algo])


def train_run(fold, algo, lr, seed, checkpoints=CHECKPOINTS, drop_families=(), force=False):
    """Entrena una corrida (fold, algoritmo, lr, semilla) y la evalúa en valid en cada checkpoint.

    Guarda los modelos en models/fold_k/ y un JSON con los metadatos (versiones,
    semilla, hiperparámetros, fecha y duración), las filas de validación y los
    episodios de train. Si ese JSON ya existe con los mismos checkpoints y
    ``force`` es False, se reutiliza en lugar de reentrenar (el pipeline es
    reanudable). Devuelve un dict con "valid", "episodios" y "metadatos".
    """
    name = run_name(algo, lr, seed, drop_families)
    meta_path = MODELS_DIR / f"fold_{fold}" / f"{name}.json"
    if not force and meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        models_ok = all(checkpoint_path(fold, name, s).exists() for s in checkpoints)
        if cached["metadatos"]["checkpoints"] == list(checkpoints) and models_ok:
            return cached
    torch.set_num_threads(1)  # varias corridas en paralelo: un hilo por proceso
    train_env = E.make_env(fold, "train", drop_families=drop_families)
    valid_env = E.make_env(fold, "valid", drop_families=drop_families)
    train_env.reset(seed=seed)
    model = build_model(algo, train_env, lr, seed)
    logger = TrainingLogger(fold, name, valid_env, checkpoints)
    start = time.perf_counter()
    model.learn(total_timesteps=max(checkpoints), callback=logger)
    metadata = {
        "fold": fold, "algoritmo": algo, "learning_rate": lr, "semilla": seed,
        "pasos_totales": max(checkpoints), "checkpoints": list(checkpoints),
        "drop_families": list(drop_families),
        "hiperparametros_fijos": {k: (v.__name__ if isinstance(v, type) else v)
                                  for k, v in FIXED_KWARGS[algo].items()},
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round(time.perf_counter() - start, 1),
        **library_versions(),
    }
    rows = [{"fold": fold, "periodo": "valid", "algoritmo": algo, "learning_rate": lr, "semilla": seed,
             "familias_quitadas": ",".join(drop_families), **row} for row in logger.valid_rows]
    episodes = [{"fold": fold, "algoritmo": algo, "learning_rate": lr, "semilla": seed, **ep}
                for ep in logger.episodes]
    result = {"valid": rows, "episodios": episodes, "metadatos": metadata}
    meta_path.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return result


def train_run_task(kwargs):
    """Adaptador de una sola entrada para ProcessPoolExecutor.map."""
    return train_run(**kwargs)


def run_many(tasks, jobs=1):
    """Ejecuta varias corridas, en paralelo si ``jobs > 1``, y junta sus resultados.

    Usa procesos con "spawn" (el único método disponible en Windows) y cada
    proceso entrena con DummyVecEnv y un hilo de torch. Devuelve
    ``(valid, episodios, metadatos)`` como DataFrames.
    """
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(train_run_task, tasks))
    else:
        results = [train_run_task(task) for task in tasks]
    valid = pd.DataFrame([row for r in results for row in r["valid"]])
    episodes = pd.DataFrame([ep for r in results for ep in r["episodios"]])
    metadata = pd.DataFrame([r["metadatos"] for r in results])
    return valid, episodes, metadata


def search_tasks(folds, algorithms=ALGORITHMS, learning_rates=LEARNING_RATES, seeds=SEARCH_SEEDS,
                 checkpoints=CHECKPOINTS):
    """Lista de corridas de la búsqueda de hiperparámetros."""
    return [{"fold": f, "algo": a, "lr": lr, "seed": s, "checkpoints": checkpoints}
            for f in folds for a in algorithms for lr in learning_rates for s in seeds]


# ---------------------------------------------------------------------------
# Selección (solo con valid)
# ---------------------------------------------------------------------------

def summarize_configs(valid_results):
    """Mediana, mínimo, máximo y desviación entre semillas de cada (fold, algoritmo, lr, pasos)."""
    keys = ["fold", "algoritmo", "learning_rate", "pasos"]
    grouped = valid_results.groupby(keys)[SELECTION_METRIC]
    summary = grouped.agg(mediana="median", minimo="min", maximo="max", desv="std", semillas="count")
    summary["operaciones_mediana"] = valid_results.groupby(keys)["operaciones"].median()
    return summary.reset_index()


def select_configs(valid_results):
    """Mejor (lr, pasos) por fold y algoritmo según el criterio declarado.

    Agrega la columna ``en_borde`` si el lr o el número de pasos elegido es un
    extremo de la rejilla (el óptimo podría estar fuera de ella).
    """
    summary = summarize_configs(valid_results)
    ordered = summary.sort_values(["fold", "algoritmo", "mediana", "minimo"], ascending=[True, True, False, False])
    best = ordered.groupby(["fold", "algoritmo"]).head(1).copy()
    lrs, steps = valid_results["learning_rate"].unique(), valid_results["pasos"].unique()
    best["lr_en_borde"] = best["learning_rate"].isin([min(lrs), max(lrs)])
    best["pasos_en_borde"] = best["pasos"].isin([min(steps), max(steps)])
    return best.reset_index(drop=True)


def final_tasks(best, search_seeds=SEARCH_SEEDS, final_seeds=FINAL_SEEDS):
    """Corridas que faltan para tener FINAL_SEEDS semillas con la configuración elegida."""
    tasks = []
    for row in best.itertuples():
        for seed in final_seeds:
            if seed not in search_seeds:
                tasks.append({"fold": row.fold, "algo": row.algoritmo, "lr": row.learning_rate, "seed": seed,
                              "checkpoints": (int(row.pasos),)})
    return tasks


def final_models_table(best, valid_results):
    """Tabla de modelos finales: todas las semillas de la configuración elegida, con su valid.

    Marca como ``representativo`` la semilla con la mediana de valid (con 5
    semillas, la tercera). Ese modelo genera la bitácora y la curva de capital.
    """
    rows = []
    for row in best.itertuples():
        mask = ((valid_results["fold"] == row.fold) & (valid_results["algoritmo"] == row.algoritmo)
                & (valid_results["learning_rate"] == row.learning_rate) & (valid_results["pasos"] == row.pasos))
        chosen = valid_results[mask].sort_values(SELECTION_METRIC).reset_index(drop=True)
        chosen["representativo"] = False
        chosen.loc[(len(chosen) - 1) // 2, "representativo"] = True
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------

def write_validation_results(valid_results, fold, out_dir=ARTIFACTS_DIR):
    """Escribe validation_results_fold_{k}.csv: cada configuración, semilla y checkpoint en valid."""
    path = Path(out_dir) / f"validation_results_fold_{fold}.csv"
    table = valid_results[valid_results["fold"] == fold].sort_values(
        ["algoritmo", "learning_rate", "pasos", "semilla"])
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def plot_training_curves(episodes, best, fold, out_dir=ARTIFACTS_DIR, window=25):
    """training_curves_fold_{k}.png: recompensa por episodio contra pasos, por algoritmo.

    Se grafican las semillas de la configuración elegida (media móvil de
    ``window`` episodios) y, en un segundo eje, la fracción de decisiones que
    abren operación. Anotaciones: los checkpoints evaluados en valid, el número
    de pasos elegido y el punto donde la frecuencia de operación cambia más.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, algo in zip(axes, ALGORITHMS):
        choice = best[(best["fold"] == fold) & (best["algoritmo"] == algo)]
        if choice.empty:
            ax.set_visible(False)
            continue
        lr, chosen_steps = choice["learning_rate"].iloc[0], int(choice["pasos"].iloc[0])
        sub = episodes[(episodes["fold"] == fold) & (episodes["algoritmo"] == algo)
                       & (episodes["learning_rate"] == lr)]
        ax2 = ax.twinx()
        trade_rates = []
        for seed, group in sub.groupby("semilla"):
            group = group.sort_values("pasos")
            smooth = group["recompensa_episodio"].rolling(window, min_periods=1).mean()
            line, = ax.plot(group["pasos"], smooth, lw=1.3, label=f"semilla {seed}")
            rate = group["frac_opera"].rolling(window, min_periods=1).mean()
            ax2.plot(group["pasos"], rate, lw=0.8, ls=":", color=line.get_color())
            trade_rates.append(rate.set_axis(group["pasos"].to_numpy()))
        ax.axhline(0, color="grey", lw=0.8)
        for steps in CHECKPOINTS:
            ax.axvline(steps, color="grey", lw=0.6, ls="--")
        ax.axvline(chosen_steps, color="black", lw=1.5, ls="--")
        ax.annotate(f"elegido en valid: {chosen_steps // 1000}k pasos", xy=(chosen_steps, 0.03),
                    xycoords=ax.get_xaxis_transform(), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, fontweight="bold")
        if trade_rates:
            mean_rate = pd.concat(trade_rates, axis=1).sort_index().ffill().mean(axis=1)
            change = mean_rate.diff(window).abs()
            change = change[change.index >= CHECKPOINTS[0] // 5]  # se ignora el arranque aleatorio
            if change.notna().any():
                at = change.idxmax()
                ax2.annotate(f"mayor cambio en frecuencia\nde operación (~{at / 1000:.0f}k pasos)",
                             xy=(at, mean_rate.loc[at]), xytext=(25, 30), textcoords="offset points",
                             fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.7},
                             bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
        ax.set_title(f"Fold {fold} · {algo.upper()} · lr={lr:g} · train (ventanas de 720 h)")
        ax.set_xlabel("Pasos de entrenamiento (decisiones)")
        ax.set_ylabel(f"Recompensa por episodio (media móvil {window})\n= 100·log(C_fin/C_ini) de la ventana")
        ax2.set_ylabel("Fracción de decisiones que abren operación (punteada)")
        ax2.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
    fig.suptitle(f"Curvas de entrenamiento · fold {fold} · líneas grises: checkpoints evaluados en valid")
    fig.tight_layout()
    path = Path(out_dir) / f"training_curves_fold_{fold}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
