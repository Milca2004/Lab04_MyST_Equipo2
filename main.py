"""Punto de entrada único del pipeline: `python main.py`. Semilla global 42.

Dueña: Paula.

Etapas (cada una escribe sus artefactos en artifacts/):
  1. Datos y EDA (Milca): feature_analysis_fold_{k}.csv y btcusdt_eda.pdf.
  2. Búsqueda de hiperparámetros en VALID (train.py): 4 folds x {PPO, DQN} x
     3 learning rates x 3 semillas, checkpoints a 25k/50k/100k pasos.
  3. Selección en VALID con el criterio fijado de antemano y semillas
     adicionales para llegar a 5 por modelo final: validation_results_fold_{k}.csv,
     training_curves_fold_{k}.png, hyperparameter_selection.csv, final_models.csv
     y run_metadata.csv.
  4. Ablación por familia de variables en VALID (pregunta 7): ablation_valid.csv.
  5. TEST, una sola vez y con todo congelado (evaluate.py): trade_log_fold_{k}.csv,
     equity_curves_test_fold_{k}.png, metrics_summary.csv, test_seed_dispersion.csv,
     worst_trades_test.csv y drawdown_period_test.png.
  6. Sensibilidad con los modelos finales en test (sensitivity.py de Arturo):
     sensitivity_test_fold_{k}.csv.

Los modelos se guardan en models/ (ignorada por git). Cada corrida deja un JSON
con sus metadatos y resultados de valid; si ya existe, se reutiliza en lugar de
reentrenar (``--force`` reentrena todo). Con la misma semilla los resultados se
repiten en la misma máquina.

Compatibilidad con Windows: Windows arranca subprocesos con "spawn", por lo
que todo el pipeline se ejecuta dentro de `if __name__ == "__main__":` y los
entornos vectorizados de SB3 usan DummyVecEnv (no SubprocVecEnv). Las corridas
en paralelo usan procesos "spawn" (``--jobs``).
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

# La carpeta se llama code/ (exigido por el enunciado) y Python ya trae un
# módulo estándar llamado `code`; por eso se agrega code/ a la ruta y se
# importan los módulos por su nombre (data, env, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

SEED = 42
FOLDS = (1, 2, 3, 4)
ABLATION_SEEDS = (42, 43, 44)


def set_global_seed(seed=SEED):
    """Fija la semilla de random, numpy y torch (cada modelo de SB3 recibe además la suya)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message, start):
    """Mensaje de avance con el tiempo transcurrido."""
    print(f"[{(time.perf_counter() - start) / 60:6.1f} min] {message}", flush=True)


def stage_data(start):
    """Etapa 1: análisis de variables por fold y EDA (código de Milca)."""
    import data
    import plots

    bars = data.load_clean()
    data.write_feature_analysis(bars)
    plots.write_eda_pdf(bars)
    log("datos: feature_analysis_fold_{1..4}.csv y btcusdt_eda.pdf", start)


def stage_search_and_select(jobs, force, start):
    """Etapas 2 y 3: búsqueda en valid, selección y semillas finales. Devuelve (best, finales, primario)."""
    import pandas as pd

    import train as T

    tasks = [{**task, "force": force} for task in T.search_tasks(FOLDS)]
    valid, episodes, metadata = T.run_many(tasks, jobs)
    log(f"búsqueda: {len(tasks)} corridas, {len(valid)} evaluaciones en valid", start)

    best = T.select_configs(valid)
    extra = [{**task, "force": force} for task in T.final_tasks(best)]
    valid_extra, episodes_extra, metadata_extra = T.run_many(extra, jobs)
    log(f"semillas finales: {len(extra)} corridas adicionales", start)

    valid_all = pd.concat([valid, valid_extra], ignore_index=True)
    episodes_all = pd.concat([episodes, episodes_extra], ignore_index=True)
    metadata_all = pd.concat([metadata, metadata_extra], ignore_index=True)
    finals = T.final_models_table(best, valid_all)
    primary = choose_primary_algorithm(best)

    out = T.ARTIFACTS_DIR
    for fold in FOLDS:
        T.write_validation_results(valid_all, fold)
        T.plot_training_curves(episodes_all, best, fold)
    best.assign(algoritmo_primario=best["algoritmo"] == primary).to_csv(
        out / "hyperparameter_selection.csv", index=False, encoding="utf-8")
    finals.to_csv(out / "final_models.csv", index=False, encoding="utf-8")
    metadata_all.drop(columns=["hiperparametros_fijos"]).to_csv(
        out / "run_metadata.csv", index=False, encoding="utf-8")
    log(f"selección en valid lista; algoritmo primario: {primary.upper()}", start)
    return best, finals, primary


def choose_primary_algorithm(best):
    """Algoritmo con mayor media entre folds de la mediana de valid de su mejor configuración."""
    return best.groupby("algoritmo")["mediana"].mean().idxmax()


def stage_ablation(best, primary, jobs, force, start):
    """Etapa 4: quitar una familia a la vez con el algoritmo primario y su configuración elegida (valid)."""
    import pandas as pd

    import env as E
    import train as T

    families = E.market_data()[2]  # las 6 familias de data.build_features
    tasks = []
    for row in best[best["algoritmo"] == primary].itertuples():
        for family in families:
            for seed in ABLATION_SEEDS:
                tasks.append({"fold": row.fold, "algo": primary, "lr": row.learning_rate, "seed": seed,
                              "checkpoints": (int(row.pasos),), "drop_families": (family,), "force": force})
    valid, _, _ = T.run_many(tasks, jobs)
    reference = pd.read_csv(T.ARTIFACTS_DIR / "final_models.csv")
    reference = reference[(reference["algoritmo"] == primary) & reference["semilla"].isin(ABLATION_SEEDS)]
    table = pd.concat([reference.assign(familias_quitadas="ninguna"), valid], ignore_index=True)
    table = table.drop(columns=["representativo"], errors="ignore")
    summary = table.groupby(["fold", "familias_quitadas"])["log_capital_final"].agg(["median", "min", "max"])
    summary = summary.reset_index()
    base = summary[summary["familias_quitadas"] == "ninguna"].set_index("fold")["median"]
    summary["delta_mediana_vs_completo"] = summary["median"] - summary["fold"].map(base)
    table.to_csv(T.ARTIFACTS_DIR / "ablation_valid.csv", index=False, encoding="utf-8")
    summary.assign(periodo="valid").to_csv(T.ARTIFACTS_DIR / "ablation_summary_valid.csv", index=False,
                                           encoding="utf-8")
    log(f"ablación: {len(tasks)} corridas en valid", start)


def stage_test(finals, primary, start):
    """Etapa 5: corrida ÚNICA en test con modelos e hiperparámetros congelados. Devuelve los modelos representativos."""
    import pandas as pd

    import evaluate as V
    import train as T

    sp500 = V.load_sp500()
    rows, logs, equity, representatives = [], {}, {}, {}
    for fold in FOLDS:
        items = []
        for row in finals[finals["fold"] == fold].itertuples():
            model = T.load_model(fold, row.algoritmo, row.learning_rate, row.semilla, int(row.pasos))
            items.append({"algoritmo": row.algoritmo, "semilla": row.semilla,
                          "representativo": bool(row.representativo), "modelo": model})
            if row.representativo:
                representatives[(fold, row.algoritmo)] = model
        result = V.evaluate_test(items, fold, sp500)
        rows += result["filas"]
        V.write_trade_log(result["bitacoras"], fold)
        V.plot_equity_curves(result["curvas"], fold, sp500_missing=sp500 is None)
        logs[fold] = result["bitacoras"][primary]
        equity[fold] = result["horario"][primary]
        log(f"test fold {fold}: {len(items)} modelos + políticas base + benchmarks", start)

    out = V.ARTIFACTS_DIR
    V.summarize_metrics(rows).to_csv(out / "metrics_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "test_results_all_seeds.csv", index=False, encoding="utf-8")
    V.seed_dispersion(rows).to_csv(out / "test_seed_dispersion.csv", index=False, encoding="utf-8")
    V.worst_trades(logs).assign(algoritmo=primary).to_csv(out / "worst_trades_test.csv", index=False,
                                                          encoding="utf-8")
    period = V.max_drawdown_period(equity)
    V.plot_drawdown_period(period, logs[period["fold"]], equity[period["fold"]])
    pd.DataFrame([{**period, "algoritmo": primary, "periodo": "test"}]).to_csv(
        out / "drawdown_period_test.csv", index=False, encoding="utf-8")
    return representatives


def stage_sensitivity(representatives, start):
    """Etapa 6: sensibilidad (intrabarra, retraso, impacto) con los modelos finales, una vez por fold en test."""
    from sensitivity import ARTIFACTS_DIR, model_policy, run_sensitivity

    for fold in FOLDS:
        policies = {algo: model_policy(representatives[(fold, algo)]) for algo in ("ppo", "dqn")}
        run_sensitivity(policies, folds=[fold], splits=["test"],
                        out_path=ARTIFACTS_DIR / f"sensitivity_test_fold_{fold}.csv")
    log("sensibilidad: sensitivity_test_fold_{1..4}.csv", start)


def parse_args(argv=None):
    """Opciones: número de procesos en paralelo y reentrenar sin caché."""
    parser = argparse.ArgumentParser(description="Pipeline completo del Lab 04 (semilla 42).")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="corridas de entrenamiento en paralelo (por defecto: min(4, núcleos))")
    parser.add_argument("--force", action="store_true", help="reentrena aunque existan modelos en models/")
    return parser.parse_args(argv)


def main(argv=None):
    """Ejecuta el pipeline completo y escribe todos los artefactos de la sección 3.8."""
    import data

    args = parse_args(argv)
    assert data.SEED == SEED, "la semilla de data.py y la de main.py deben coincidir"
    set_global_seed(SEED)
    data.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    stage_data(start)
    best, finals, primary = stage_search_and_select(args.jobs, args.force, start)
    stage_ablation(best, primary, args.jobs, args.force, start)
    representatives = stage_test(finals, primary, start)
    stage_sensitivity(representatives, start)
    log("pipeline completo", start)


if __name__ == "__main__":
    main()
