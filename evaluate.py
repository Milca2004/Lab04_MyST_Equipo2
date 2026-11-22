"""Métricas, benchmarks, bitácora y análisis de operaciones en el conjunto de prueba.

Dueña: Paula.

Convenciones (declaradas aquí para que el reporte las cite):

* Anualización: cripto opera los 365 días, así que los retornos diarios de la
  estrategia y de BTC se anualizan con 365. El S&P 500 solo opera ~252 días
  al año; sus métricas de riesgo se anualizan con 252 y su retorno anualizado
  usa días de calendario, igual que el de la estrategia, para que el alfa
  compare lo mismo.
* La curva de capital del entorno cambia solo al cerrar operaciones. Se
  remuestrea al último valor de cada día y se antepone el capital inicial el
  día anterior al inicio, para que el primer día también cuente.
* Retornos diarios simples. Sharpe y Sortino con tasa libre de riesgo 0.
* Alfa = retorno anualizado de la estrategia − retorno anualizado del benchmark
  (retorno en exceso, no el intercepto de una regresión).
* Rotación = nocional operado (entrada + salida) / capital medio / años.
* Benchmarks con el MISMO costo de 0.125% por lado: comprar en el primer
  close del periodo y vender en el último (dos operaciones).

El test se toca en ``evaluate_test`` y SOLO ahí. ``main.py`` lo llama una vez,
con los modelos y los hiperparámetros congelados.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import env as E
import reward

ARTIFACTS_DIR = data.ARTIFACTS_DIR
SP500_PATH = data.REPO_ROOT / "data" / "sp500_daily.csv"
DAYS_PER_YEAR = 365
SP500_DAYS_PER_YEAR = 252
METRIC_COLUMNS = [
    "retorno_total", "retorno_anualizado", "volatilidad_anualizada", "sharpe", "sortino",
    "max_drawdown", "calmar", "hit_rate", "duracion_promedio_h", "operaciones", "rotacion_anual",
    "log_capital_final", "corr_btc_diaria", "alfa_vs_btc", "alfa_vs_sp500",
]


# ---------------------------------------------------------------------------
# Curvas de capital
# ---------------------------------------------------------------------------

def daily_equity(equity, initial_capital):
    """Capital al cierre de cada día, con el capital inicial el día anterior al inicio."""
    capital = equity["capital"] if isinstance(equity, pd.DataFrame) else equity
    daily = capital.resample("D").last().ffill()
    start = daily.index[0] - pd.Timedelta(days=1)
    return pd.concat([pd.Series([initial_capital], index=[start]), daily])


def buy_and_hold_btc(split_data, initial_capital=1_000_000.0, cost_rate=reward.COST_RATE):
    """Comprar y mantener BTCUSDT en todo el periodo, con costo de entrada y de salida.

    Devuelve (capital diario, resumen de la única operación).
    """
    close = pd.Series(split_data.close, index=split_data.index)
    quantity = initial_capital * (1 - cost_rate) / close.iloc[0]
    hourly = quantity * close
    hourly.iloc[-1] *= (1 - cost_rate)
    trade = {"operaciones": 1, "nocional": initial_capital + quantity * close.iloc[-1],
             "duracion_h": (close.index[-1] - close.index[0]) / pd.Timedelta(hours=1),
             "ganadora": hourly.iloc[-1] > initial_capital}
    return daily_equity(hourly, initial_capital), trade


def load_sp500(path=SP500_PATH):
    """Serie diaria de cierres del S&P 500 (columnas ``fecha`` y ``close``) o None si no hay archivo.

    El archivo depende de Milca y de la respuesta del profesor (el enunciado
    prohíbe datos externos de precio pero pide este benchmark).
    """
    path = Path(path)
    if not path.exists():
        return None
    table = pd.read_csv(path, encoding="utf-8", parse_dates=["fecha"])
    series = table.set_index("fecha")["close"].sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    return series


def sp500_index(series, start, end, initial_capital=1_000_000.0, cost_rate=reward.COST_RATE):
    """Indexar al S&P 500 entre ``start`` y ``end`` (días hábiles), con costo de entrada y salida."""
    if series is None:
        return None, None
    window = series[(series.index >= start.normalize()) & (series.index <= end)]
    if len(window) < 2:
        return None, None
    capital = initial_capital * (1 - cost_rate) * window / window.iloc[0]
    capital.iloc[-1] *= (1 - cost_rate)
    trade = {"operaciones": 1, "nocional": initial_capital + capital.iloc[-1] / (1 - cost_rate),
             "duracion_h": (window.index[-1] - window.index[0]) / pd.Timedelta(hours=1),
             "ganadora": capital.iloc[-1] > initial_capital}
    return capital, trade


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def annualized_return(total_return, start, end):
    """Retorno anualizado con días de calendario (igual para cripto y S&P)."""
    years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
    return (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0


def equity_curve_metrics(daily, periods_per_year=DAYS_PER_YEAR):
    """Retorno total y anualizado, volatilidad, Sharpe, Sortino, drawdown máximo y Calmar."""
    returns = daily.pct_change().dropna()
    total = daily.iloc[-1] / daily.iloc[0] - 1
    annual = annualized_return(total, daily.index[0], daily.index[-1])
    sd = returns.std()
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    mdd = (daily / daily.cummax() - 1).min()
    root = np.sqrt(periods_per_year)
    return {
        "retorno_total": float(total),
        "retorno_anualizado": float(annual),
        "volatilidad_anualizada": float(sd * root),
        "sharpe": float(returns.mean() / sd * root) if sd > 0 else np.nan,
        "sortino": float(returns.mean() / downside * root) if downside > 0 else np.nan,
        "max_drawdown": float(mdd),
        "calmar": float(annual / abs(mdd)) if mdd < 0 else np.nan,
        "log_capital_final": float(np.log(daily.iloc[-1] / daily.iloc[0])),
    }


def trade_metrics(trade_log, mean_capital, years):
    """Hit rate, duración promedio, número de operaciones y rotación anual de una bitácora."""
    if len(trade_log) == 0:
        return {"hit_rate": np.nan, "duracion_promedio_h": np.nan, "operaciones": 0, "rotacion_anual": 0.0}
    exit_notional = trade_log["tamano_btc"] * trade_log["precio_salida"]
    traded = trade_log["nocional_usdt"].sum() + exit_notional.sum()
    return {
        "hit_rate": float((trade_log["pnl_neto"] > 0).mean()),
        "duracion_promedio_h": float(trade_log["duracion_horas"].mean()),
        "operaciones": int(len(trade_log)),
        "rotacion_anual": float(traded / mean_capital / years),
    }


def btc_daily_returns(split_data):
    """Retornos diarios simples del close de BTCUSDT en la partición."""
    close = pd.Series(split_data.close, index=split_data.index).resample("D").last().ffill()
    return close.pct_change().dropna()


def strategy_row(name, daily, trade_log, split_data, benchmarks, periods_per_year=DAYS_PER_YEAR):
    """Todas las métricas de la sección 3.7 para una curva diaria y su bitácora.

    ``benchmarks``: {"btc": capital diario, "sp500": capital diario o None}.
    """
    row = {"estrategia": name, **equity_curve_metrics(daily, periods_per_year)}
    years = (daily.index[-1] - daily.index[0]) / pd.Timedelta(days=DAYS_PER_YEAR)
    if isinstance(trade_log, pd.DataFrame):
        row.update(trade_metrics(trade_log, daily.mean(), years))
    else:  # benchmark de una sola operación (comprar y mantener)
        row.update({"hit_rate": float(trade_log["ganadora"]), "duracion_promedio_h": trade_log["duracion_h"],
                    "operaciones": 1, "rotacion_anual": trade_log["nocional"] / daily.mean() / years})
    agent = daily.pct_change().dropna()
    btc = btc_daily_returns(split_data)
    common = agent.index.intersection(btc.index)
    row["corr_btc_diaria"] = float(agent[common].corr(btc[common])) if agent[common].std() > 0 else np.nan
    for key, curve in benchmarks.items():
        if curve is None:
            row[f"alfa_vs_{key}"] = np.nan
            continue
        bench_total = curve.iloc[-1] / curve.iloc[0] - 1
        bench_annual = annualized_return(bench_total, curve.index[0], curve.index[-1])
        row[f"alfa_vs_{key}"] = row["retorno_anualizado"] - bench_annual
    return row


# ---------------------------------------------------------------------------
# Evaluación en test (se llama UNA vez desde main.py)
# ---------------------------------------------------------------------------

def run_policy(fold, split, policy_factory, seed=data.SEED, **env_kwargs):
    """Rollout determinista de una política en la partición completa."""
    environment = E.make_env(fold, split, **env_kwargs)
    result = E.rollout(environment, policy_factory(environment), seed=seed)
    return environment, result


def evaluate_test(final_models, fold, sp500=None, split="test"):
    """Corre en el test del fold los modelos finales, las políticas base y los benchmarks.

    ``final_models``: lista de dicts {algoritmo, semilla, representativo, modelo}.
    Devuelve un dict con:
      filas       métricas por estrategia (y por semilla en los modelos de RL)
      bitacoras   {algoritmo: bitácora del modelo representativo}
      curvas      {nombre: capital diario} para la gráfica
      horario     {algoritmo: curva de capital por decisión del representativo}

    ``split`` solo se cambia para depurar el código sobre valid sin tocar el test.
    """
    split_data = E.load_split(fold, split)
    btc_curve, btc_trade = buy_and_hold_btc(split_data)
    sp_curve, sp_trade = sp500_index(sp500, split_data.index[0], split_data.index[-1])
    benchmarks = {"btc": btc_curve, "sp500": sp_curve}
    rows, logs, curves, raw = [], {}, {"Comprar y mantener BTC": btc_curve}, {}
    for item in final_models:
        model = item["modelo"]
        policy = lambda env, m=model: (lambda obs: m.predict(obs, deterministic=True)[0])
        environment, result = run_policy(fold, split, policy)
        daily = daily_equity(result["capital"], result["capital_inicial"])
        row = strategy_row(item["algoritmo"], daily, result["bitacora"], split_data, benchmarks)
        rows.append({"fold": fold, "periodo": split, "semilla": item["semilla"],
                     "representativo": item["representativo"], **row})
        if item["representativo"]:
            logs[item["algoritmo"]] = result["bitacora"]
            curves[f"{item['algoritmo'].upper()} (semilla {item['semilla']})"] = daily
            raw[item["algoritmo"]] = result["capital"]
    for name, factory in E.BASELINES.items():
        _, result = run_policy(fold, split, factory)
        daily = daily_equity(result["capital"], result["capital_inicial"])
        rows.append({"fold": fold, "periodo": split, "semilla": data.SEED, "representativo": True,
                     **strategy_row(name, daily, result["bitacora"], split_data, benchmarks)})
        if name == "siempre_largo":
            curves["Siempre largo (mismas barreras)"] = daily
    rows.append({"fold": fold, "periodo": split, "semilla": np.nan, "representativo": True,
                 **strategy_row("comprar_mantener_btc", btc_curve, btc_trade, split_data, benchmarks)})
    if sp_curve is not None:
        curves["S&P 500"] = sp_curve
        rows.append({"fold": fold, "periodo": split, "semilla": np.nan, "representativo": True,
                     **strategy_row("sp500", sp_curve, sp_trade, split_data, benchmarks, SP500_DAYS_PER_YEAR)})
    return {"filas": rows, "bitacoras": logs, "curvas": curves, "horario": raw}


def summarize_metrics(rows):
    """metrics_summary.csv: métricas por fold y agregadas (media y desviación entre los 4 folds).

    Para los modelos de RL se usa el modelo representativo de cada fold; la
    dispersión entre semillas va aparte (``seed_dispersion``).
    """
    table = pd.DataFrame(rows)
    table = table[table["representativo"].astype(bool)]
    per_fold = table[["estrategia", "fold", *METRIC_COLUMNS]].copy()
    per_fold["fold"] = per_fold["fold"].astype(str)
    agg = per_fold.groupby("estrategia")[METRIC_COLUMNS]
    mean = agg.mean().reset_index().assign(fold="media")
    std = agg.std().reset_index().assign(fold="desv")
    summary = pd.concat([per_fold, mean, std], ignore_index=True)
    summary.insert(1, "periodo", table["periodo"].iloc[0])
    return summary.sort_values(["estrategia", "fold"]).reset_index(drop=True)


def seed_dispersion(rows):
    """Media, desviación, mínimo y máximo entre semillas por fold y algoritmo (test)."""
    table = pd.DataFrame(rows)
    table = table[table["estrategia"].isin(["ppo", "dqn"])]
    cols = ["log_capital_final", "sharpe", "max_drawdown", "operaciones"]
    out = table.groupby(["fold", "estrategia"])[cols].agg(["mean", "std", "min", "max"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index().assign(periodo=table["periodo"].iloc[0])


def write_trade_log(logs, fold, out_dir=ARTIFACTS_DIR):
    """trade_log_fold_{k}.csv: cada operación de test del modelo representativo de cada algoritmo."""
    frames = [log.assign(algoritmo=algo)[["algoritmo", *E.TRADE_LOG_COLUMNS]] for algo, log in logs.items()]
    path = Path(out_dir) / f"trade_log_fold_{fold}.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8")
    return path


def plot_equity_curves(curves, fold, out_dir=ARTIFACTS_DIR, sp500_missing=False, split="test"):
    """equity_curves_test_fold_{k}.png: estrategias contra los benchmarks, capital normalizado a 1."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    styles = {"Comprar y mantener BTC": {"color": "black", "lw": 2},
              "S&P 500": {"color": "tab:green", "lw": 2},
              "Siempre largo (mismas barreras)": {"color": "grey", "lw": 1, "ls": ":"}}
    for name, curve in curves.items():
        ax.plot(curve.index, curve / curve.iloc[0], label=name, **styles.get(name, {"lw": 1.5}))
    ax.axhline(1, color="grey", lw=0.6)
    ax.set_yscale("log")
    ax.set_title(f"Curvas de capital · fold {fold} · {split} ({curves['Comprar y mantener BTC'].index[1]:%Y-%m-%d} "
                 f"a {curves['Comprar y mantener BTC'].index[-1]:%Y-%m-%d}) · costos de 0.125% por lado")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Capital / capital inicial (escala log)")
    if sp500_missing:
        ax.text(0.01, 0.02, "S&P 500 pendiente: falta data/sp500_daily.csv", transform=ax.transAxes,
                fontsize=8, color="tab:red")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = Path(out_dir) / f"equity_curves_test_fold_{fold}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Análisis de operaciones (sección 3.9)
# ---------------------------------------------------------------------------

def state_at_entry(fold, entry_time, split="test"):
    """Variables de estado de información en la decisión que abrió la operación.

    La orden se llena en el open de la barra siguiente a la decisión, así que
    la decisión ocurrió en la barra observada anterior a ``entrada_ts``.
    Devuelve los valores SIN normalizar (de data.build_features) y normalizados.
    """
    split_data = E.load_split(fold, split)
    position = int(np.searchsorted(split_data.index, entry_time)) - 1
    decision_time = split_data.index[position]
    raw = E.market_data()[1].loc[decision_time]
    out = {"decision_ts": decision_time}
    out.update({f"{name}": float(raw[name]) for name in split_data.feature_names})
    out.update({f"{name}_norm": float(v) for name, v in zip(split_data.feature_names, split_data.features[position])})
    return out


def worst_trades(logs_by_fold, n=3, split="test"):
    """Las ``n`` operaciones con mayor pérdida (retorno neto) del test entre todos los folds."""
    frames = [log.assign(fold=fold) for fold, log in logs_by_fold.items() if len(log)]
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    trades["retorno_neto_op"] = trades["pnl_neto"] / trades["capital_antes"]
    worst = trades.nsmallest(n, "retorno_neto_op")
    rows = [{**trade.to_dict(), **state_at_entry(trade["fold"], trade["entrada_ts"], split)}
            for _, trade in worst.iterrows()]
    return pd.DataFrame(rows).assign(periodo=split)


def max_drawdown_period(equity_by_fold):
    """Fold, pico, valle y profundidad del mayor drawdown del agente en test."""
    best = None
    for fold, equity in equity_by_fold.items():
        capital = equity["capital"]
        drawdown = capital / capital.cummax() - 1
        trough = drawdown.idxmin()
        peak = capital[:trough].idxmax()
        if best is None or drawdown.min() < best["drawdown"]:
            best = {"fold": fold, "pico": peak, "valle": trough, "drawdown": float(drawdown.min())}
    return best


def plot_drawdown_period(period, log, equity, out_dir=ARTIFACTS_DIR, margin_days=3, split="test"):
    """Precio de BTC y acciones del agente (arriba) y su capital (abajo) durante su mayor drawdown."""
    split_data = E.load_split(period["fold"], split)
    start = period["pico"] - pd.Timedelta(days=margin_days)
    end = period["valle"] + pd.Timedelta(days=margin_days)
    close = pd.Series(split_data.close, index=split_data.index)[start:end]
    window = log[(log["entrada_ts"] >= start) & (log["entrada_ts"] <= end)]
    fig, (ax, ax_cap) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, height_ratios=(3, 1.3))
    ax.plot(close.index, close, color="black", lw=1, label="Close BTCUSDT")
    ax.axvspan(period["pico"], period["valle"], color="tab:red", alpha=0.08, label="Pico → valle del capital")
    for side, marker, color in (("largo", "^", "tab:green"), ("corto", "v", "tab:red")):
        sub = window[window["lado"] == side]
        ax.scatter(sub["entrada_ts"], sub["precio_entrada"], marker=marker, color=color, s=40,
                   label=f"Entrada {side}", zorder=3)
    ax.scatter(window["salida_ts"], window["precio_salida"], marker="x", color="tab:blue", s=25,
               label="Salida (stop / TP / horizonte)", zorder=3)
    ax.set_title(f"Mayor drawdown del agente en {split} · fold {period['fold']} · "
                 f"{period['pico']:%Y-%m-%d} a {period['valle']:%Y-%m-%d} · {period['drawdown']:.1%}")
    ax.set_ylabel("Precio (USDT)")
    ax.legend(loc="best", fontsize=8)
    capital = equity["capital"][start:end]
    ax_cap.step(capital.index, capital / 1e6, where="post", color="tab:blue", lw=1.2, label="Capital del agente")
    ax_cap.axvspan(period["pico"], period["valle"], color="tab:red", alpha=0.08)
    ax_cap.set_xlabel("Fecha")
    ax_cap.set_ylabel("Capital (M USDT)")
    ax_cap.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / f"drawdown_period_{split}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
