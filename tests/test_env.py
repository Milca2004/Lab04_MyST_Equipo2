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


def test_check_env_variante_por_paso():
    """La variante por paso de la pregunta 3 también pasa check_env sin advertencias."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(E.make_env(1, "train", reward_timing="per_bar"), warn=True, skip_render_check=True)


def test_ablacion_quita_la_familia_y_no_cambia_las_barreras():
    """drop_families reduce la observación (pregunta 7) sin cambiar el simulador."""
    full = E.make_env(1, "valid")
    ablated = E.make_env(1, "valid", drop_families=("volatilidad",))
    assert ablated.observation_space.shape[0] == full.observation_space.shape[0] - 4
    assert "vol_24" not in ablated.d.feature_names
    assert (ablated.d.vol_24 == full.d.vol_24).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(ablated, warn=True, skip_render_check=True)


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
