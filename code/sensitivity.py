"""Experimentos de sensibilidad del simulador (pregunta 6, sección 3.10) y de la contribución (pregunta 3).

Dueño: Arturo. Desviación menor de la estructura 2.2 (archivo extra en code/),
declarada en el walkthrough.

Todas las funciones aceptan cualquier política con la forma
``fabrica(env) -> (obs -> acción)``, como las de ``env.BASELINES``. Para un
modelo de SB3: ``model_policy(model)``.

Uso:
    python code/sensitivity.py baselines      # políticas base, train y valid de los 4 folds
    python code/sensitivity.py reward-timing  # pregunta 3: terminal vs por paso (train fold 1)

Corrida final (UNA vez, con los modelos congelados de Paula y el test):
    from sensitivity import run_sensitivity, model_policy
    run_sensitivity({"ppo": model_policy(ppo), ...}, folds=[k], splits=["test"],
                    out_path=ARTIFACTS_DIR / f"sensitivity_test_fold_{k}.csv")
    (un modelo por fold: llamar una vez por fold con su modelo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import env as E

ARTIFACTS_DIR = data.ARTIFACTS_DIR
SEED = data.SEED
DAYS_PER_YEAR = 365  # cripto opera todos los días
IMPACT_SIZES_USDT = (100_000, 1_000_000, 10_000_000, 50_000_000)
IMPACT_Y = (0.5, 1.0)  # constante de la ley de raíz cuadrada (rango de la literatura)
INFERENCE_SECONDS = 2.0


def model_policy(model, deterministic=True):
    """Convierte un modelo de SB3 en una fábrica de políticas compatible con este módulo."""
    return lambda env: (lambda obs: model.predict(obs, deterministic=deterministic)[0])


def equity_metrics(equity):
    """Retorno total, Sharpe y drawdown máximo de la curva de capital remuestreada por día."""
    daily = equity["capital"].resample("D").last().ffill()
    returns = np.log(daily).diff().dropna()
    sd = returns.std()
    return {
        "retorno_total": float(daily.iloc[-1] / daily.iloc[0] - 1),
        "log_capital_final": float(np.log(daily.iloc[-1] / daily.iloc[0])),
        "sharpe_anual": float(returns.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else np.nan,
        "max_drawdown": float((daily / daily.cummax() - 1).min()),
    }


def policy_metrics(environment, policy_factory, seed=SEED):
    """Corre un episodio completo y resume resultado y operaciones."""
    result = E.rollout(environment, policy_factory(environment), seed=seed)
    log = result["bitacora"]
    return {
        **equity_metrics(result["capital"]),
        "operaciones": len(log),
        "frac_ambiguas": float(log["ambigua"].mean()) if len(log) else 0.0,
        "frac_cruza_hueco": float(log["cruza_hueco"].mean()) if len(log) else 0.0,
        "retorno_neto_medio_op": float((log["pnl_neto"] / log["capital_antes"]).mean()) if len(log) else np.nan,
    }


def scenario_rows(split_data, policy_factory, scenarios, seed=SEED):
    """Métricas de una política bajo varios escenarios {nombre: kwargs de TradingEnv}."""
    return [{"escenario": name, **policy_metrics(E.TradingEnv(split_data, **kwargs), policy_factory, seed)}
            for name, kwargs in scenarios.items()]


def impact_per_side(split_data, size_usdt, y):
    """Impacto estimado por lado con la ley de raíz cuadrada: Y * sigma_1h * sqrt(Q / V_1h).

    sigma_1h y V_1h (volumen horario mediano en USDT) se miden en la partición
    recibida; para fijar el supuesto se usa el train del fold.
    """
    bars = E.market_data()[0].loc[split_data.index]
    sigma = np.log(bars["close"]).diff().std()
    return float(y * sigma * np.sqrt(size_usdt / bars["volume_usdt"].median()))


def execution_scenarios(train_data):
    """Escenarios de ejecución: costo por lado = 0.125% + impacto estimado con datos del train."""
    scenarios = {}
    for size in IMPACT_SIZES_USDT:
        for y in IMPACT_Y:
            impact = impact_per_side(train_data, size, y)
            scenarios[f"impacto_{size / 1e6:g}M_Y{y:g}"] = {
                "cost_rate": E.reward.COST_RATE + impact, "initial_capital": float(size)}
    return scenarios


SENSITIVITY_SCENARIOS = {
    "base": {},
    "tp_first": {"intrabar_priority": "tp_first"},
    "retraso_1_barra": {"data_delay_bars": 1},
}


DELTA_COLUMNS = {
    "retorno_total": "delta_retorno",
    "log_capital_final": "delta_log_capital",
    "retorno_neto_medio_op": "delta_retorno_medio_op",
    "sharpe_anual": "delta_sharpe",
}


def run_sensitivity(policies, folds=(1, 2, 3, 4), splits=("train", "valid"), out_path=None, seed=SEED):
    """Tabla de sensibilidad: intrabarra, retraso de una barra y supuestos de ejecución.

    ``policies``: {nombre: fábrica de política}. Devuelve un DataFrame con una
    fila por (fold, partición, política, escenario), más columnas ``delta_*``
    contra el escenario base. Con miles de operaciones el retorno total se
    satura en -100%; por eso también se comparan el log del capital final y
    el retorno neto medio por operación.
    """
    rows = []
    for fold in folds:
        train_data = E.load_split(fold, "train")
        scenarios = {**SENSITIVITY_SCENARIOS, **execution_scenarios(train_data)}
        for split in splits:
            split_data = train_data if split == "train" else E.load_split(fold, split)
            for name, factory in policies.items():
                for row in scenario_rows(split_data, factory, scenarios, seed):
                    rows.append({"fold": fold, "periodo": split, "politica": name, **row})
    table = pd.DataFrame(rows)
    base = table[table["escenario"] == "base"].set_index(["fold", "periodo", "politica"])
    keys = table.set_index(["fold", "periodo", "politica"]).index
    for column, delta in DELTA_COLUMNS.items():
        table[delta] = table[column].to_numpy() - base[column].reindex(keys).to_numpy()
    if out_path is not None:
        table.to_csv(out_path, index=False, encoding="utf-8")
    return table


def execution_assumptions(folds=(1, 2, 3, 4)):
    """Cifras de los supuestos de ejecución por fold (solo train): impacto y latencia de inferencia."""
    rows = []
    for fold in folds:
        train_data = E.load_split(fold, "train")
        bars = E.market_data()[0].loc[train_data.index]
        sigma = float(np.log(bars["close"]).diff().std())
        row = {"fold": fold, "periodo": "train", "sigma_1h": sigma,
               "volumen_1h_mediano_usdt": float(bars["volume_usdt"].median()),
               "mov_tipico_en_2s": sigma * np.sqrt(INFERENCE_SECONDS / 3600)}
        for size in IMPACT_SIZES_USDT:
            row[f"impacto_lado_{size / 1e6:g}M_Y1"] = impact_per_side(train_data, size, 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def gross_edge_test(fold=1, policy_factory=E.always_long, seed=SEED):
    """Prueba t de la media del log-retorno bruto por operación (¿hay ventaja antes de costos?)."""
    environment = E.TradingEnv(E.load_split(fold, "train"))
    log = E.rollout(environment, policy_factory(environment), seed=seed)["bitacora"]
    gross = np.log1p(log["pnl_bruto"] / log["capital_antes"])
    net = np.log1p(log["pnl_neto"] / log["capital_antes"])
    se = gross.std() / np.sqrt(len(gross))
    return {"fold": fold, "periodo": "train", "operaciones": len(log), "media_bruta": float(gross.mean()),
            "error_estandar": float(se), "t_bruta": float(gross.mean() / se),
            "media_neta": float(net.mean()), "sd": float(gross.std()),
            "kelly_f": float(net.mean() / net.var())}


def replay_composition(model, reward_timing):
    """Fracción de transiciones del replay que abren operación, que son forzadas o que son 'fuera'."""
    rb = model.replay_buffer
    n = rb.buffer_size if rb.full else rb.pos
    actions = rb.actions[:n, 0, 0]
    if reward_timing == "per_bar":
        forced = rb.observations[:n, 0, -2] != 0  # lado de la posición abierta
    else:
        forced = np.zeros(n, dtype=bool)
    opening = (actions != E.ACTION_FLAT) & ~forced
    return {"frac_abre_operacion": float(opening.mean()), "frac_forzadas": float(forced.mean()),
            "frac_fuera": float((~opening & ~forced).mean())}


def count_free_decisions(environment, policy, seed):
    """Corre un episodio y cuenta las decisiones libres (sin operación abierta en per_bar)."""
    obs, _ = environment.reset(seed=seed)
    free, done = 0, False
    while not done:
        free += not environment.in_trade
        obs, _, terminated, truncated, _ = environment.step(policy(obs))
        done = terminated or truncated
    return free


def reward_timing_experiment(seeds=(42, 43, 44, 45, 46), timesteps=30_000, fold=1):
    """Pregunta 3: DQN con contribución terminal vs por paso, mismo presupuesto de pasos.

    Solo train del fold. Métrica común fijada de antemano: log del capital
    final en un rollout determinista sobre el train completo del fold, más la
    fracción de decisiones libres que abren operación y la composición del
    replay buffer.
    """
    from stable_baselines3 import DQN

    rows = []
    for timing in E.REWARD_TIMINGS:
        for seed in seeds:
            train_env = E.make_env(fold, "train", reward_timing=timing)
            model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                        replay_buffer_class=E.SMDPReplayBuffer,
                        replay_buffer_kwargs={"gamma_per_hour": 0.999})
            model.learn(timesteps)
            evaluation = E.TradingEnv(train_env.d, reward_timing=timing)
            free_decisions = count_free_decisions(evaluation, model_policy(model)(evaluation), seed)
            log = evaluation.trade_log_frame()
            rows.append({
                "fold": fold, "periodo": "train", "contribucion": timing, "semilla": seed,
                "log_capital_final": float(np.log(evaluation.capital / evaluation.initial_capital)),
                "operaciones": len(log),
                "frac_decisiones_que_operan": len(log) / max(free_decisions, 1),
                "retorno_neto_medio_op": float((log["pnl_neto"] / log["capital_antes"]).mean()) if len(log) else np.nan,
                **replay_composition(model, timing),
            })
    return pd.DataFrame(rows)


def main(argv):
    """Punto de entrada de línea de comandos (ver docstring del módulo)."""
    command = argv[1] if len(argv) > 1 else "baselines"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if command == "baselines":
        table = run_sensitivity(E.BASELINES, out_path=ARTIFACTS_DIR / "sensitivity_baselines.csv")
        execution_assumptions().to_csv(ARTIFACTS_DIR / "execution_assumptions.csv", index=False, encoding="utf-8")
        pd.DataFrame([gross_edge_test(f) for f in (1, 2, 3, 4)]).to_csv(
            ARTIFACTS_DIR / "gross_edge_test.csv", index=False, encoding="utf-8")
        print(table.groupby(["periodo", "politica", "escenario"])[list(DELTA_COLUMNS.values())].mean())
    elif command == "reward-timing":
        table = reward_timing_experiment()
        table.to_csv(ARTIFACTS_DIR / "reward_timing_fold1_train.csv", index=False, encoding="utf-8")
        print(table)
    else:
        raise SystemExit(f"Comando desconocido: {command}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv)
