"""Prueba obligatoria 1 (sección 3.11) y verificaciones de cordura del simulador.

* check_env de Stable-Baselines3 pasa SIN advertencias (las advertencias se
  convierten en errores).
* El agente aleatorio con costos no gana dinero en el train de ningún fold
  (sección 3.5: si ganara, el simulador tendría un error).
* La bitácora cuadra con el capital: capital final = capital inicial + suma
  del P&L neto (apoya la prueba contable de evaluate.py).
"""
import warnings

import pytest
from stable_baselines3.common.env_checker import check_env

import env as E


@pytest.mark.parametrize("split", E.SPLITS)
def test_check_env_sin_advertencias(split):
    """check_env no emite ninguna advertencia en train, valid ni test del fold 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(E.make_env(1, split), warn=True, skip_render_check=True)


@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_agente_aleatorio_pierde_y_la_bitacora_cuadra(fold):
    """Train completo del fold con la política aleatoria (semilla 42)."""
    env = E.TradingEnv(E.load_split(fold, "train"))
    result = E.rollout(env, E.random_policy(env))
    log = result["bitacora"]
    assert len(log) > 1000
    assert (log["pnl_neto"] / log["capital_antes"]).mean() < 0
    assert result["capital_final"] < result["capital_inicial"]
    assert result["capital_final"] == pytest.approx(
        result["capital_inicial"] + log["pnl_neto"].sum(), rel=1e-9, abs=1e-6)
