"""Entorno Gymnasium de trading BTCUSDT con cierre semi-Markov de posiciones.

Dueño: Arturo.

Resumen de la formulación (detalle y justificación con cifras en el reporte):

* Decisión x_t ∈ {0: fuera, 1: abrir largo, 2: abrir corto}. Solo se decide
  estando fuera, así que no existen acciones inválidas (no hay máscara ni
  penalización, y DQN funciona sin cambios).
* Una operación abierta se cierra en el PRIMERO de: stop-loss, take-profit u
  horizonte de ``horizon_hours`` horas de RELOJ. No hay acción de cierre.
* Convención temporal: se decide al cierre de la barra t con información
  hasta t; la orden se llena en el open de la siguiente barra observada
  (t+1). Las barreras se fijan con ese precio de entrada y la volatilidad
  ``vol_24`` conocida en t.
* Un paso de ``step`` = una decisión. "Fuera" avanza a la siguiente barra
  observada; una operación avanza hasta la barra en que se cierra
  (frecuencia de decisión endógena). ``info["tau_horas"]`` trae las horas
  reales transcurridas, para descontar con gamma**tau (ver
  ``SMDPReplayBuffer``).
* La contribución es terminal: se paga al cierre de la operación
  (``reward.log_return_contribution``) e incluye los costos de 0.125% por
  lado.
"""
from dataclasses import dataclass
from functools import lru_cache

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

import data
import reward

ONE_HOUR = pd.Timedelta(hours=1)
SPLITS = ("train", "valid", "test")
UNSCALED_FAMILIES = ("tiempo",)  # ya están en [-1, 1] (seno y coseno)
OBS_CLIP = 10.0
MAX_GAP_LOG = np.log1p(145.0)  # hueco más largo del archivo (julio de 2025)

ACTION_FLAT, ACTION_LONG, ACTION_SHORT = 0, 1, 2
SIDE_BY_ACTION = {ACTION_LONG: 1, ACTION_SHORT: -1}
SIDE_NAME = {1: "largo", -1: "corto"}
INTRABAR_PRIORITIES = ("stop_first", "tp_first")
REWARD_TIMINGS = ("terminal", "per_bar")


# ---------------------------------------------------------------------------
# Datos de un fold y partición (usa la interfaz de data.py, no la recalcula)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitData:
    """Barras y observaciones normalizadas de un fold y partición.

    ``features`` son las 21 variables de data.build_features, escaladas con el
    Normalizer del fold (ajustado SOLO con su train). ``vol_24`` se guarda sin
    normalizar porque fija la distancia de las barreras en precio.
    """

    fold: int
    split: str
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    gap_hours: np.ndarray
    vol_24: np.ndarray
    features: np.ndarray
    feature_names: tuple


@lru_cache(maxsize=1)
def market_data():
    """Barras limpias y variables causales del archivo completo (se calculan una vez)."""
    bars = data.load_clean()
    features, families = data.build_features(bars)
    return bars, features, families


def load_split(fold_number, split, drop_families=()):
    """Devuelve ``SplitData`` del fold (1-4) y la partición ('train', 'valid', 'test').

    Se descartan las filas sin variables completas (las 168 barras de
    calentamiento de 2017-08, solo en train). El Normalizer se ajusta con el
    train del fold y se aplica sin cambios a la partición pedida.
    ``drop_families`` quita familias de data.build_features de la observación
    (ablación, pregunta 7); las barreras siguen usando vol_24 aunque se quite
    la familia volatilidad, para que el simulador no cambie.
    """
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}")
    bars, features, families = market_data()
    unknown = set(drop_families) - set(families)
    if unknown:
        raise ValueError(f"Familias desconocidas: {sorted(unknown)}; válidas: {sorted(families)}")
    fold = data.get_folds()[fold_number - 1]
    complete = features.dropna()
    scaled = [c for fam, cols in families.items() if fam not in UNSCALED_FAMILIES for c in cols]
    normalizer = data.Normalizer(scaled).fit(fold.split(complete)[0])
    part = fold.split(complete)[SPLITS.index(split)]
    part_bars = bars.loc[part.index]
    kept = [c for fam, cols in families.items() if fam not in drop_families for c in cols]
    observed = normalizer.transform(part)[kept]
    return SplitData(
        fold=fold_number,
        split=split,
        index=part.index,
        open=part_bars["open"].to_numpy(float),
        high=part_bars["high"].to_numpy(float),
        low=part_bars["low"].to_numpy(float),
        close=part_bars["close"].to_numpy(float),
        gap_hours=part_bars["gap_hours"].to_numpy(float),
        vol_24=part["vol_24"].to_numpy(float),
        features=observed.to_numpy(np.float32),
        feature_names=tuple(kept),
    )


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """Entorno semi-Markov de BTCUSDT horario (una posición a la vez, sin apalancamiento).

    Observación (Box float32, recortada a [-10, 10]; 24 valores por defecto):
        0-20  las 21 variables de data.py normalizadas, en la barra t - data_delay_bars
              (incluye la familia creencia: trend_mean, trend_std, trend_snr);
              menos si load_split recibió drop_families.
        n+0   log(capital / capital_inicial)       componente física
        n+1   drawdown del capital en el episodio  componente física (<= 0)
        n+2   log1p(horas faltantes antes de la barra observada) / log1p(145)

    Acciones (Discrete(3)): 0 = quedarse fuera, 1 = abrir largo, 2 = abrir corto.

    Recompensa: 0 si se queda fuera; al cerrar una operación,
    ``reward.log_return_contribution`` (100 * log del cambio de capital, neto
    de costos).

    ``info`` en cada paso (contrato fijo para train.py y evaluate.py):
        tau_horas      horas reales entre esta decisión y la siguiente
        capital        capital después del paso (USDT)
        operacion      None si se quedó fuera; si operó, el registro de la
                       bitácora (mismas claves que ``trade_log_frame``):
                       entrada_ts, salida_ts, lado, motivo_salida
                       (stop | take_profit | horizonte | fin_split),
                       precio_entrada, precio_salida, stop, take_profit,
                       tamano_btc, nocional_usdt, pnl_bruto, costos,
                       pnl_neto, capital_antes, capital_despues,
                       duracion_horas, ambigua, cruza_hueco,
                       entrada_post_hueco

    Parámetros:
        split_data         SplitData de load_split.
        horizon_hours      horizonte máximo de la operación en horas de reloj (24).
        k_stop, k_take     distancia de stop y take-profit en múltiplos de
                           vol_24 * sqrt(horizon_hours) (1.0 y 1.0).
        cost_rate          costo por lado (0.00125).
        position_fraction  fracción del capital que entra en cada operación (1.0).
        initial_capital    capital al inicio de cada episodio (1,000,000 USDT).
        episode_hours      duración de la ventana del episodio en horas de
                           reloj; None = toda la partición.
        random_start       True = la ventana empieza en una barra al azar.
        intrabar_priority  'stop_first' (por defecto) o 'tp_first' cuando stop
                           y take-profit caen dentro de la misma barra.
        data_delay_bars    barras de retraso de la fuente de datos (0); la
                           observación y la volatilidad de las barreras usan
                           la barra t - data_delay_bars, el llenado no cambia.
        reward_scale       factor de la contribución (100).
        reward_timing      'terminal' (diseño: se paga al cierre) o 'per_bar'
                           (alternativa de la pregunta 3: mientras la operación
                           está abierta cada barra es un paso forzado cuya
                           acción se ignora y paga su cambio marcado a mercado;
                           la observación agrega lado de la posición y
                           fracción del horizonte transcurrida).

    Embargo: no se abren operaciones a menos de ``horizon_hours`` del final
    de la partición, para que ninguna barrera use precios del periodo
    siguiente. Si aun así no hay barra después del horizonte (hueco al
    final), la operación se cierra en el último close con motivo 'fin_split'.
    """

    metadata = {"render_modes": []}

    def __init__(self, split_data, horizon_hours=24, k_stop=1.0, k_take=1.0,
                 cost_rate=reward.COST_RATE, position_fraction=1.0, initial_capital=1_000_000.0,
                 episode_hours=None, random_start=False, intrabar_priority="stop_first",
                 data_delay_bars=0, reward_scale=100.0, reward_timing="terminal"):
        super().__init__()
        if intrabar_priority not in INTRABAR_PRIORITIES:
            raise ValueError(f"intrabar_priority debe ser uno de {INTRABAR_PRIORITIES}")
        if reward_timing not in REWARD_TIMINGS:
            raise ValueError(f"reward_timing debe ser uno de {REWARD_TIMINGS}")
        if not 0 < position_fraction <= 1:
            raise ValueError("position_fraction debe estar en (0, 1]: sin apalancamiento")
        self.d = split_data
        self.horizon = pd.Timedelta(hours=horizon_hours)
        self.horizon_hours = horizon_hours
        self.k_stop, self.k_take = k_stop, k_take
        self.cost_rate = cost_rate
        self.position_fraction = position_fraction
        self.initial_capital = initial_capital
        self.episode_hours = episode_hours
        self.random_start = random_start
        self.intrabar_priority = intrabar_priority
        self.data_delay_bars = data_delay_bars
        self.reward_scale = reward_scale
        self.reward_timing = reward_timing

        self.first_decision = data_delay_bars
        embargo_limit = self.d.index[-1] - self.horizon
        self.last_decision = int(np.searchsorted(self.d.index, embargo_limit, side="right")) - 1
        if self.last_decision < self.first_decision:
            raise ValueError("La partición es más corta que el horizonte más el retraso.")

        n_obs = self.d.features.shape[1] + 3 + (2 if reward_timing == "per_bar" else 0)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._t = None

    # -- ciclo de episodio -------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reinicia capital y bitácora. ``options={'start_index': i}`` fija la barra inicial."""
        super().reset(seed=seed)
        options = options or {}
        self._t = options.get("start_index", self._sample_start())
        if not self.first_decision <= self._t <= self.last_decision:
            raise ValueError("start_index fuera del rango de decisiones de la partición")
        self._window_end = (self.d.index[self._t] + pd.Timedelta(hours=self.episode_hours)
                            if self.episode_hours else None)
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trade_log = []
        self.equity = [(self.d.index[self._t], self.capital)]
        self._pending = []  # solo per_bar: (barra, contribución) de los pasos forzados
        self._open_trade = None
        return self._observation(), {"tau_horas": 0.0, "capital": self.capital, "operacion": None}

    def step(self, action):
        """Aplica la decisión en la barra actual y avanza hasta la siguiente decisión."""
        if self._t is None or (self._t > self.last_decision and not self._pending):
            raise RuntimeError("Llama reset() antes de step() o después de terminar el episodio.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Acción fuera del espacio: {action}")
        t = self._t
        record = None
        if self._pending:  # per_bar: paso forzado, la acción se ignora
            next_t, contribution = self._pending.pop(0)
        elif action == ACTION_FLAT:
            next_t, contribution = t + 1, 0.0
        else:
            trade, exit_bar, acct = self._run_trade(t, SIDE_BY_ACTION[action])
            if self.reward_timing == "terminal":
                next_t = exit_bar
                contribution = reward.log_return_contribution(
                    trade["capital_antes"], trade["pnl_neto"], self.reward_scale)
            else:
                interim = self.d.close[t + 1:exit_bar]
                rewards = reward.per_bar_contributions(
                    trade["capital_antes"], acct, SIDE_BY_ACTION[action], trade["precio_entrada"],
                    interim, self.reward_scale)
                self._pending = list(zip(range(t + 1, exit_bar + 1), rewards))
                next_t, contribution = self._pending.pop(0)
            self._open_trade = trade
        if self._open_trade is not None and not self._pending:
            record = self._close_trade()
        self._t = next_t
        self.equity.append((self.d.index[next_t], self.capital))
        tau = (self.d.index[next_t] - self.d.index[t]) / ONE_HOUR
        info = {"tau_horas": float(tau), "capital": self.capital, "operacion": record}
        return self._observation(), float(contribution), False, self._is_truncated(), info

    @property
    def in_trade(self):
        """True si hay una operación abierta con pasos forzados pendientes (solo per_bar)."""
        return bool(self._pending)

    def _close_trade(self):
        """Aplica al capital la operación abierta y la registra en la bitácora."""
        record, self._open_trade = self._open_trade, None
        self.capital = record["capital_despues"]
        self.peak_capital = max(self.peak_capital, self.capital)
        self.trade_log.append(record)
        return record

    def _sample_start(self):
        """Barra inicial: la primera, o una al azar con ventana completa dentro de la partición."""
        if not (self.random_start and self.episode_hours):
            return self.first_decision
        latest = self.d.index[self.last_decision] - pd.Timedelta(hours=self.episode_hours)
        stop = int(np.searchsorted(self.d.index, latest, side="right"))
        if stop <= self.first_decision:
            return self.first_decision
        return int(self.np_random.integers(self.first_decision, stop))

    def _is_truncated(self):
        """Fin de ventana o de partición: truncamiento (el mercado sigue), nunca terminación.

        Nunca se trunca con una operación abierta (pasos forzados de per_bar).
        """
        if self._pending:
            return False
        if self._t > self.last_decision:
            return True
        return self._window_end is not None and self.d.index[self._t] >= self._window_end

    # -- estado ------------------------------------------------------------

    def _observation(self):
        """Vector S_t observado: información (con retraso) + componente física."""
        seen = self._t - self.data_delay_bars
        physical = [
            np.log(self.capital / self.initial_capital),
            self.capital / self.peak_capital - 1.0,
            np.log1p(self.d.gap_hours[seen]) / MAX_GAP_LOG,
        ]
        if self.reward_timing == "per_bar":
            physical += self._position_state()
        obs = np.concatenate([self.d.features[seen], np.asarray(physical, np.float32)])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _position_state(self):
        """Solo per_bar: lado de la posición abierta y fracción del horizonte transcurrida."""
        if not self._pending:
            return [0.0, 0.0]
        trade = self._open_trade
        elapsed = (self.d.index[self._t] + ONE_HOUR - trade["entrada_ts"]) / self.horizon
        return [1.0 if trade["lado"] == "largo" else -1.0, float(elapsed)]

    # -- transición física: gestor de la posición ---------------------------

    def barrier_prices(self, t, side, entry_price):
        """Precios de stop y take-profit fijados con la volatilidad conocida en la decisión."""
        sigma = self.d.vol_24[t - self.data_delay_bars]
        distance = max(sigma, 1e-8) * np.sqrt(self.horizon_hours)
        stop = entry_price * np.exp(-side * self.k_stop * distance)
        take = entry_price * np.exp(side * self.k_take * distance)
        return stop, take

    def _run_trade(self, t, side):
        """Simula la operación abierta en t hasta su cierre.

        Devuelve (registro de bitácora, barra de la siguiente decisión, TradeAccounting).
        """
        d = self.d
        entry_bar = t + 1
        entry_price = d.open[entry_bar]
        entry_time = d.index[entry_bar]
        stop, take = self.barrier_prices(t, side, entry_price)
        deadline = entry_time + self.horizon
        exit_bar, exit_price, exit_time, reason, ambiguous = self._first_exit(
            entry_bar, side, stop, take, deadline)
        notional = self.position_fraction * self.capital
        acct = reward.trade_accounting(side, entry_price, exit_price, notional, self.cost_rate)
        record = {
            "entrada_ts": entry_time,
            "salida_ts": exit_time,
            "lado": SIDE_NAME[side],
            "motivo_salida": reason,
            "precio_entrada": entry_price,
            "precio_salida": exit_price,
            "stop": stop,
            "take_profit": take,
            "tamano_btc": acct.qty,
            "nocional_usdt": acct.notional,
            "pnl_bruto": acct.gross_pnl,
            "costos": acct.costs,
            "pnl_neto": acct.net_pnl,
            "capital_antes": self.capital,
            "capital_despues": self.capital + acct.net_pnl,
            "duracion_horas": (exit_time - entry_time) / ONE_HOUR,
            "ambigua": ambiguous,
            "cruza_hueco": bool((d.gap_hours[entry_bar + 1:exit_bar + 1] > 0).any()),
            "entrada_post_hueco": bool(d.gap_hours[entry_bar] > 0),
        }
        return record, exit_bar, acct

    def _first_exit(self, entry_bar, side, stop, take, deadline):
        """Recorre las barras desde la entrada y devuelve el primer evento de cierre.

        Orden dentro de cada barra j:
          1. Si la barra empieza en o después del horizonte (hubo un hueco),
             se sale en su open: 'horizonte'.
          2. Si el open ya atravesó una barrera (hueco de precio), se llena
             en el open, que puede ser peor que el stop.
          3. Si high/low tocan ambas barreras, la barra es ambigua y se
             resuelve con ``intrabar_priority``.
          4. Si la barra cierra en el horizonte, se sale en su close.
        Devuelve (barra, precio, marca de tiempo, motivo, ambigua).
        """
        d = self.d
        for j in range(entry_bar, len(d.index)):
            bar_open, bar_close = d.index[j], d.index[j] + ONE_HOUR
            if bar_open >= deadline:
                return j, d.open[j], bar_open, "horizonte", False
            gap_exit = self._gap_through(side, d.open[j], stop, take)
            if gap_exit:
                return j, d.open[j], bar_open, gap_exit, False
            hit_stop = d.low[j] <= stop if side == 1 else d.high[j] >= stop
            hit_take = d.high[j] >= take if side == 1 else d.low[j] <= take
            if hit_stop and hit_take:
                if self.intrabar_priority == "stop_first":
                    return j, stop, bar_close, "stop", True
                return j, take, bar_close, "take_profit", True
            if hit_stop:
                return j, stop, bar_close, "stop", False
            if hit_take:
                return j, take, bar_close, "take_profit", False
            if bar_close >= deadline:
                return j, d.close[j], bar_close, "horizonte", False
        last = len(d.index) - 1
        return last, d.close[last], d.index[last] + ONE_HOUR, "fin_split", False

    @staticmethod
    def _gap_through(side, price, stop, take):
        """Motivo de salida si el open ya está más allá de una barrera; None si no."""
        if (price <= stop) if side == 1 else (price >= stop):
            return "stop"
        if (price >= take) if side == 1 else (price <= take):
            return "take_profit"
        return None

    # -- salidas para evaluate.py --------------------------------------------

    def trade_log_frame(self):
        """Bitácora del episodio como DataFrame (una fila por operación)."""
        return pd.DataFrame(self.trade_log, columns=TRADE_LOG_COLUMNS)

    def equity_frame(self):
        """Capital después de cada decisión (cambia solo al cerrar operaciones)."""
        return pd.DataFrame(self.equity, columns=["fecha", "capital"]).set_index("fecha")


TRADE_LOG_COLUMNS = [
    "entrada_ts", "salida_ts", "lado", "motivo_salida", "precio_entrada", "precio_salida",
    "stop", "take_profit", "tamano_btc", "nocional_usdt", "pnl_bruto", "costos", "pnl_neto",
    "capital_antes", "capital_despues", "duracion_horas", "ambigua", "cruza_hueco",
    "entrada_post_hueco",
]


def make_env(fold_number, split, drop_families=(), **env_kwargs):
    """Crea el entorno de un fold y partición con los valores por defecto del diseño.

    En train el episodio es una ventana de 720 horas (30 días) con inicio al
    azar; en valid y test es la partición completa, empezando en su primera
    barra. ``drop_families`` pasa a load_split (ablación). Cualquier
    parámetro de TradingEnv se puede sobrescribir.
    """
    defaults = ({"episode_hours": 720, "random_start": True} if split == "train"
                else {"episode_hours": None, "random_start": False})
    return TradingEnv(load_split(fold_number, split, drop_families), **{**defaults, **env_kwargs})


# ---------------------------------------------------------------------------
# Políticas base y rollout
# ---------------------------------------------------------------------------

def always_flat(env):
    """Política base: nunca opera."""
    return lambda obs: ACTION_FLAT


def always_long(env):
    """Política base: abre un largo en cada decisión (con las mismas barreras)."""
    return lambda obs: ACTION_LONG


def random_policy(env):
    """Política base: acción uniforme al azar (usa el generador del action_space)."""
    return lambda obs: env.action_space.sample()


BASELINES = {"aleatoria": random_policy, "siempre_largo": always_long, "siempre_fuera": always_flat}


def rollout(env, policy, seed=data.SEED, options=None):
    """Corre un episodio completo con ``policy(obs) -> acción``.

    Para un modelo de SB3: ``policy = lambda obs: model.predict(obs, deterministic=True)[0]``.
    Devuelve un dict con capital inicial y final, número de decisiones,
    bitácora (DataFrame) y curva de capital (DataFrame).
    """
    obs, _ = env.reset(seed=seed, options=options)
    env.action_space.seed(seed)
    steps, done = 0, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        steps += 1
        done = terminated or truncated
    return {
        "capital_inicial": env.initial_capital,
        "capital_final": env.capital,
        "decisiones": steps,
        "bitacora": env.trade_log_frame(),
        "capital": env.equity_frame(),
    }


# ---------------------------------------------------------------------------
# Replay buffer semi-Markov para DQN
# ---------------------------------------------------------------------------

class SMDPReplayBuffer(ReplayBuffer):
    """Replay buffer que guarda la duración real tau de cada transición.

    Cada transición es (estado en la apertura, acción, recompensa al cierre,
    estado al cierre, tau_horas). En el muestreo devuelve
    ``discounts = gamma_per_hour ** tau``, que el DQN de SB3 (>= 2.6) usa en
    lugar de su gamma fijo: objetivo = r + (1 - done) * gamma_h**tau * max Q(s').
    Así una operación de 24 h se descuenta más que quedarse fuera 1 h.

    Uso: ``DQN(..., replay_buffer_class=SMDPReplayBuffer,
    replay_buffer_kwargs={"gamma_per_hour": 0.999})``.
    """

    def __init__(self, *args, gamma_per_hour=0.999, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError("SMDPReplayBuffer no soporta optimize_memory_usage=True")
        self.gamma_per_hour = gamma_per_hour
        self.tau_hours = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """Guarda tau antes de que la clase base avance la posición del buffer."""
        self.tau_hours[self.pos] = [info["tau_horas"] for info in infos]
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        """Igual que ReplayBuffer._get_samples, más el descuento gamma_h**tau por transición."""
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        dones = self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])
        discounts = self.gamma_per_hour ** self.tau_hours[batch_inds, env_indices]
        samples = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env),
            dones.reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            discounts.reshape(-1, 1),
        )
        return ReplayBufferSamples(*map(self.to_torch, samples))
