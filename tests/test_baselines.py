"""Líneas base con costos (clase S19, nivel 3 de validación del simulador).

Complementa a test_env.py, que ya prueba que la aleatoria pierde en el train de
los 4 folds. Aquí: "siempre fuera" deja el capital exactamente igual y
"siempre largo" no gana dinero con costos de 0.125% por lado.
"""
import env as E


def test_siempre_fuera_deja_el_capital_exactamente_igual():
    """Sin operaciones no hay costos ni P&L: el capital no se mueve ni un centavo."""
    environment = E.make_env(1, "valid")
    result = E.rollout(environment, E.always_flat(environment))
    assert result["capital_final"] == result["capital_inicial"]
    assert len(result["bitacora"]) == 0


def test_siempre_largo_no_gana_con_costos_en_train():
    """Abrir largo en cada decisión pierde en el train completo de cada fold."""
    for fold in (1, 2, 3, 4):
        environment = E.TradingEnv(E.load_split(fold, "train"))
        result = E.rollout(environment, E.always_long(environment))
        assert result["capital_final"] < result["capital_inicial"], f"fold {fold}"
