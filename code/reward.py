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

La contribución C(S_t, x_t) es TERMINAL: se paga completa en el paso en que
la operación se cierra (formulación semi-Markov, sección 3.4).
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
