"""Prueba obligatoria 3 (sección 3.11): operaciones de referencia calculadas a mano.

Datos sintéticos (no del archivo) para que cada número se pueda verificar
con calculadora. Supuestos comunes a todos los casos:

* Barras horarias desde 2024-01-01 00:00 UTC. Se decide al cierre de la
  barra 0 y se entra en el open de la barra 1 (01:00) a 100.
* horizon_hours = 4, k_stop = k_take = 1, vol_24 = 0.005 en todas las barras
  -> distancia log = 0.005 * sqrt(4) = 0.01.
      stop largo = 100 * exp(-0.01) = 99.0049833749168
      take largo = 100 * exp(+0.01) = 101.00501670841679
  (en un corto se intercambian).
* Capital 1,000,000, position_fraction = 1 -> nocional 1,000,000 y
  qty = 1,000,000 / 100 = 10,000 BTC.
* Costo 0.125% por lado: entrada = 0.00125 * 1,000,000 = 1,250;
  salida = 0.00125 * 10,000 * precio_salida.
* Recompensa = 100 * log(1 + pnl_neto / 1,000,000).

Cálculo de cada caso (P&L bruto - 1,250 - costo de salida = P&L neto):

1. Take-profit (largo). La barra 2 (02:00) tiene high 101.2 >= take.
   Bruto = 10,000 * (101.00501670841679 - 100) = 10,050.167084167895
   Salida = 0.00125 * 1,010,050.1670841679 = 1,262.5627088552098
   Neto = 7,537.604375312685. Duración: 01:00 -> cierre de la barra 2
   (03:00) = 2 h. tau = 02:00 - 00:00 = 2 h.
2. Stop (largo). La barra 2 tiene low 98.9 <= stop.
   Bruto = 10,000 * (99.0049833749168 - 100) = -9,950.166250831955
   Salida = 1,237.5622921864601. Neto = -12,437.728543018415. Duración 2 h.
3. Horizonte (largo). Las barras 1 a 4 se quedan dentro de 99.6-100.5.
   El horizonte vence a las 05:00 = cierre de la barra 4 (close 100.4).
   Bruto = 4,000; salida = 1,255; neto = 1,495. Duración 4 h, tau 4 h.
4. Ambigua (corto). La barra 2 toca high 101.2 y low 98.9: ambas barreras.
   stop_first: sale en el stop 101.00501670841679, neto -12,562.729793023105.
   tp_first: sale en el take 99.0049833749168, neto 7,462.603958645495.
5. Hueco que atraviesa el stop (largo). Falta la barra de 02:00; la de
   03:00 abre en 98.5 < stop, así que se llena en 98.5 (peor que el stop).
   Bruto = -15,000; salida = 0.00125 * 985,000 = 1,231.25;
   neto = -17,481.25. Duración 01:00 -> 03:00 = 2 h.
"""
import numpy as np
import pandas as pd
import pytest

import env as E

ENTRY = 100.0
QUIET = (100.0, 100.5, 99.6, 100.2)  # (open, high, low, close) sin tocar barreras
CAPITAL = 1_000_000.0


def make_split(bar2, n_bars=10, drop_bar2=False, close_bar4=100.2):
    """SplitData sintético: barra 0 cierra en 100, barra 1 abre en 100, ``bar2`` a elección.

    Con ``drop_bar2`` falta la barra de 02:00 y ``bar2`` pasa a ser la de 03:00.
    """
    rows = [(99.8, 100.1, 99.7, 100.0), (ENTRY, 100.5, 99.6, 100.2), bar2]
    rows += [QUIET] * (n_bars - 3)
    rows[4] = (100.0, 100.5, 99.6, close_bar4)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    if drop_bar2:
        rows[3] = bar2
        keep = [i for i in range(n_bars) if i != 2]
        rows, index = [rows[i] for i in keep], index[keep]
    o, h, l, c = (np.array(col, dtype=float) for col in zip(*rows))
    gap = np.r_[0.0, (index[1:] - index[:-1]) / pd.Timedelta(hours=1) - 1]
    return E.SplitData(fold=0, split="sintetico", index=index, open=o, high=h, low=l, close=c,
                       gap_hours=gap, vol_24=np.full(len(index), 0.005),
                       features=np.zeros((len(index), 21), np.float32),
                       feature_names=tuple(f"x{i}" for i in range(21)))


def open_and_close(split, action, **kwargs):
    """Abre una operación al cierre de la barra 0 y devuelve (recompensa, info)."""
    env = E.TradingEnv(split, horizon_hours=4, initial_capital=CAPITAL, **kwargs)
    env.reset(options={"start_index": 0})
    _, contribution, _, _, info = env.step(action)
    return contribution, info


CASES = {
    # nombre: (bar2, acción, opciones, motivo, precio_salida, neto, duración, recompensa)
    "take_profit": ((100.3, 101.2, 100.2, 101.0), E.ACTION_LONG, {}, "take_profit",
                    101.00501670841679, 7537.604375312685, 2.0, 0.7509338584159044),
    "stop": ((100.1, 100.3, 98.9, 99.2), E.ACTION_LONG, {}, "stop",
             99.0049833749168, -12437.728543018415, 2.0, -1.2515724491767133),
    "horizonte": (QUIET, E.ACTION_LONG, {"close_bar4": 100.4}, "horizonte",
                  100.4, 1495.0, 4.0, 0.14938836000401726),
}


@pytest.mark.parametrize("name", CASES)
def test_operacion_de_referencia(name):
    """El entorno reproduce motivo, precio, P&L neto, duración y recompensa calculados a mano."""
    bar2, action, opts, reason, exit_price, net, hours, expected_reward = CASES[name]
    contribution, info = open_and_close(make_split(bar2, **opts), action)
    trade = info["operacion"]
    assert trade["motivo_salida"] == reason
    assert trade["precio_entrada"] == ENTRY
    assert trade["tamano_btc"] == pytest.approx(10_000.0, rel=1e-12)
    assert trade["precio_salida"] == pytest.approx(exit_price, rel=1e-12)
    assert trade["pnl_neto"] == pytest.approx(net, rel=1e-9)
    assert trade["duracion_horas"] == hours
    assert info["tau_horas"] == hours
    assert contribution == pytest.approx(expected_reward, rel=1e-9)
    assert info["capital"] == pytest.approx(CAPITAL + net, rel=1e-12)


@pytest.mark.parametrize("priority, reason, exit_price, net", [
    ("stop_first", "stop", 101.00501670841679, -12562.729793023105),
    ("tp_first", "take_profit", 99.0049833749168, 7462.603958645495),
])
def test_barra_ambigua_segun_supuesto_intrabarra(priority, reason, exit_price, net):
    """Corto con ambas barreras en la barra 2: el resultado depende solo de intrabar_priority."""
    split = make_split((100.0, 101.2, 98.9, 100.0))
    _, info = open_and_close(split, E.ACTION_SHORT, intrabar_priority=priority)
    trade = info["operacion"]
    assert trade["ambigua"] is True
    assert trade["motivo_salida"] == reason
    assert trade["precio_salida"] == pytest.approx(exit_price, rel=1e-12)
    assert trade["pnl_neto"] == pytest.approx(net, rel=1e-9)


def test_hueco_que_atraviesa_el_stop_se_llena_en_el_open():
    """Sin barra de 02:00 y open de 98.5 a las 03:00: stop llenado en 98.5, peor que el nivel."""
    split = make_split((98.5, 98.9, 98.2, 98.6), drop_bar2=True)
    _, info = open_and_close(split, E.ACTION_LONG)
    trade = info["operacion"]
    assert trade["motivo_salida"] == "stop"
    assert trade["precio_salida"] == 98.5
    assert trade["pnl_neto"] == pytest.approx(-17481.25, rel=1e-12)
    assert trade["duracion_horas"] == 2.0
    assert trade["cruza_hueco"] is True
