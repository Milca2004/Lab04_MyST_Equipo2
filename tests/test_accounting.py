"""Prueba obligatoria 4 (sección 3.11): identidad contable de la bitácora.

capital final = capital inicial + suma del P&L neto de todas las operaciones
registradas. Se verifica con políticas que operan mucho (aleatoria y siempre
largo) sobre una ventana corta de train, y también con la variante de
contribución por paso, que reparte la recompensa pero no debe cambiar la
contabilidad. Dueña: Paula.
"""
import numpy as np
import pytest

import env as E

REL_TOL = 1e-9


@pytest.fixture(scope="module")
def train_split():
    """Train del fold 1 (las variables se calculan una vez por módulo)."""
    return E.load_split(1, "train")


def short_env(split_data, **kwargs):
    """Ventana de 30 días desde una barra fija, para que la prueba sea rápida y determinista."""
    return E.TradingEnv(split_data, episode_hours=720, random_start=False, **kwargs)


@pytest.mark.parametrize("policy_name", ["aleatoria", "siempre_largo"])
def test_capital_final_igual_a_inicial_mas_pnl_neto(train_split, policy_name):
    """La suma del P&L neto de la bitácora explica todo el cambio de capital."""
    environment = short_env(train_split)
    result = E.rollout(environment, E.BASELINES[policy_name](environment), options={"start_index": 5_000})
    log = result["bitacora"]
    assert len(log) > 5, "la ventana debe tener varias operaciones para que la prueba diga algo"
    assert list(log.columns) == E.TRADE_LOG_COLUMNS
    expected = result["capital_inicial"] + log["pnl_neto"].sum()
    assert np.isclose(result["capital_final"], expected, rtol=REL_TOL, atol=0)


def test_pnl_neto_es_bruto_menos_costos_y_encadena_capital(train_split):
    """Cada fila cumple pnl_neto = pnl_bruto - costos y el capital se encadena sin huecos."""
    environment = short_env(train_split)
    log = E.rollout(environment, E.always_long(environment), options={"start_index": 5_000})["bitacora"]
    assert np.allclose(log["pnl_neto"], log["pnl_bruto"] - log["costos"], rtol=REL_TOL, atol=1e-6)
    assert np.allclose(log["capital_despues"], log["capital_antes"] + log["pnl_neto"], rtol=REL_TOL, atol=0)
    assert np.allclose(log["capital_antes"].iloc[1:].to_numpy(), log["capital_despues"].iloc[:-1].to_numpy(),
                       rtol=REL_TOL, atol=0)
    assert (log["costos"] > 0).all(), "toda operación paga costos de entrada y salida"


def test_identidad_contable_con_contribucion_por_paso(train_split):
    """La variante por paso cambia la recompensa, no la contabilidad."""
    environment = short_env(train_split, reward_timing="per_bar")
    result = E.rollout(environment, E.always_long(environment), options={"start_index": 5_000})
    expected = result["capital_inicial"] + result["bitacora"]["pnl_neto"].sum()
    assert np.isclose(result["capital_final"], expected, rtol=REL_TOL, atol=0)
