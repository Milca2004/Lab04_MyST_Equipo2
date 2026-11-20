"""Contabilidad de una operación y función de contribución (recompensa).

Dueño: Arturo.

Convención de costos (sección 3.1 del enunciado): 0.125% por lado sobre el
nocional de cada lado.

* Entrada: se compra (o vende en corto) ``qty = nocional / precio_entrada``
  y se paga ``c * nocional``.
* Salida: se cierra ``qty`` al precio de salida y se paga
  ``c * qty * precio_salida``.
* P&L bruto = lado * qty * (precio_salida - precio_entrada).
* P&L neto = P&L bruto - costo_entrada - costo_salida.

El capital después de la operación es ``capital + pnl_neto``; así la
identidad contable del enunciado (capital final = capital inicial + suma del
P&L neto de la bitácora) se cumple por construcción.

La contribución C(S_t, x_t) del diseño es TERMINAL: se paga completa en el
paso en que la operación se cierra (formulación semi-Markov, sección 3.4).
``per_bar_contributions`` implementa la alternativa por paso (marcada a
mercado en cada barra) solo para el experimento de la pregunta 3: suma lo
mismo que la terminal, pero reparte el crédito barra por barra.
"""
import math
from dataclasses import dataclass

COST_RATE = 0.00125  # 0.125% por lado


@dataclass(frozen=True)
class TradeAccounting:
    """Resultado contable de una operación cerrada (todo en USDT salvo ``qty``)."""

    qty: float
    notional: float
    gross_pnl: float
    entry_cost: float
    exit_cost: float
    net_pnl: float

    @property
    def costs(self):
        """Costo total de la operación (entrada + salida)."""
        return self.entry_cost + self.exit_cost


def trade_accounting(side, entry_price, exit_price, notional, cost_rate=COST_RATE):
    """Calcula qty, P&L bruto, costos por lado y P&L neto de una operación.

    side: +1 largo, -1 corto. notional: USDT comprometidos a la entrada.
    """
    if side not in (1, -1):
        raise ValueError(f"side debe ser +1 o -1, no {side}")
    qty = notional / entry_price
    gross = side * qty * (exit_price - entry_price)
    entry_cost = cost_rate * notional
    exit_cost = cost_rate * qty * exit_price
    return TradeAccounting(qty, notional, gross, entry_cost, exit_cost, gross - entry_cost - exit_cost)


def log_return_contribution(capital_before, net_pnl, scale=100.0):
    """Contribución terminal = scale * log(capital_después / capital_antes).

    La suma de log-retornos de un episodio es el log del capital final entre
    el inicial, así que maximizar la suma de contribuciones equivale a
    maximizar el crecimiento compuesto del capital (no la suma de P&L en
    dólares, que premia arriesgar más cuando el capital ya creció).
    ``scale`` = 100 deja la recompensa en "puntos porcentuales log": un trade
    típico de ±0.25% da ±0.25 en lugar de ±0.0025, una escala que las redes
    de PPO y DQN aprenden mejor.
    """
    return scale * math.log1p(net_pnl / capital_before)


def per_bar_contributions(capital_before, accounting, side, entry_price, interim_closes, scale=100.0):
    """Alternativa por paso: contribución de cada barra de la operación, marcada a mercado.

    Valor de la cuenta al cierre de cada barra intermedia k (aún abierta):
        V_k = capital_antes - costo_entrada + lado * qty * (close_k - precio_entrada)
    y en la barra de salida V = capital_antes + pnl_neto. La contribución de
    cada barra es scale * log(V_k / V_{k-1}) con V_{-1} = capital_antes, así
    que la suma telescopa a ``log_return_contribution`` (mismo total, crédito
    repartido). El costo de entrada cae entero en la primera barra.
    ``interim_closes``: closes de las barras entre la entrada y la de salida
    (sin incluir esta última).
    """
    open_value = capital_before - accounting.entry_cost
    values = [open_value + side * accounting.qty * (close - entry_price) for close in interim_closes]
    values.append(capital_before + accounting.net_pnl)
    previous = [capital_before] + values[:-1]
    return [scale * math.log(v / p) for v, p in zip(values, previous)]
