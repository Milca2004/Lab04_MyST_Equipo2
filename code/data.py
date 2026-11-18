"""Carga, limpieza, folds walk-forward, variables causales y normalización.

Dueña: Milca.

Supuestos y hallazgos del archivo de Canvas (verificados corriendo código):

* Zona horaria: la columna ``Date`` no trae zona; se asume UTC (es lo que
  publica Binance) y el índice queda con zona UTC explícita.
* Formato: primera línea con una URL (``skiprows=1``), orden descendente y
  640 fechas con fracción de segundo (597 ``.000`` y 43 ``.789``), por lo que
  se usa ``format="mixed"``.
* Solo se usa la columna ``Date``. La columna ``Unix`` no es confiable: en
  672 filas de 2025 viene en microsegundos (16 dígitos) en lugar de
  milisegundos.
* Filas: el archivo trae 75,588 filas; el enunciado dice 75,558.
* Duplicados: 506 filas repiten fecha y OHLCV de otra fila (todas en 2025;
  solo cambia la escala de ``Unix``). Se eliminan -> 75,082 barras únicas.
* 43 barras fuera de hora exacta (2018-02-09 09:28:14.789 a
  2018-02-11 03:28:14.789) se redondean HACIA ABAJO a la hora. Evidencia: la
  barra etiquetada 03:28:14 cierra en 7976.74 y la barra exacta de las 04:00
  abre en 7976.74, así que es la barra 03:00-04:00 con la etiqueta recorrida
  (hora de recolección). Con floor su cierre se conoce a las 04:00 (sin
  fuga); con ceil chocaría con la barra existente de las 04:00.
* Huecos: NO se rellenan. La columna booleana ``gap`` marca las barras cuya
  barra anterior (una hora antes) falta. Contra la malla horaria completa
  faltan 1,157 horas antes del floor y 1,114 después (las 43 horas de 2018
  quedan cubiertas), repartidas en 376 huecos. Desde abril de 2025 falta casi
  a diario la barra de 00:00 y hay huecos largos (145 h en julio de 2025).
* 4 barras con volumen cero (2019-06-07, 2020-12-21, 2021-02-11,
  2023-03-24); se conservan.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Binance_BTCUSDT_1h.csv"

RAW_TO_CLEAN_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume BTC": "volume",
    "Volume USDT": "volume_usdt",
    "tradecount": "trades",
}
OHLCV = list(RAW_TO_CLEAN_COLUMNS.values())
ONE_HOUR = pd.Timedelta(hours=1)


def read_raw(path=DATA_PATH):
    """Lee el CSV tal como viene (sin limpiar), con la fecha ya interpretada en UTC."""
    raw = pd.read_csv(path, skiprows=1, encoding="utf-8")
    raw["date"] = pd.to_datetime(raw["Date"].astype(str), format="mixed", utc=True)
    return raw


def clean_bars(raw):
    """Convierte el CSV crudo en barras horarias limpias.

    Pasos: redondea la fecha hacia abajo a la hora, elimina filas repetidas
    (misma fecha y OHLCV), ordena de forma ascendente y marca ``gap``.
    Falla si dos barras distintas caen en la misma hora (no se elige una en
    silencio).
    """
    bars = raw.rename(columns=RAW_TO_CLEAN_COLUMNS)[["date"] + OHLCV].copy()
    bars["date"] = bars["date"].dt.floor("h")
    bars = bars.drop_duplicates().sort_values("date")
    if bars["date"].duplicated().any():
        raise ValueError("Hay horas con dos barras de contenido distinto.")
    bars = bars.set_index("date")
    bars["gap"] = bars.index.to_series().diff().gt(ONE_HOUR).to_numpy()
    return bars


def load_clean(path=DATA_PATH):
    """Devuelve el DataFrame horario OHLCV limpio.

    Índice ``date`` (apertura de la barra, UTC) en orden ascendente y sin
    duplicados. Columnas: open, high, low, close, volume (BTC), volume_usdt,
    trades y ``gap`` (True si la barra anterior falta).
    """
    return clean_bars(read_raw(path))


@dataclass(frozen=True)
class Fold:
    """Un fold walk-forward. Cada periodo es (inicio, fin) en texto, ambos inclusivos.

    Los textos usan la selección parcial de pandas: ``("2022", "2022")`` es
    todo el año 2022 y ``("2017-08", "2021-12")`` va del primer día de
    agosto de 2017 al último de diciembre de 2021.
    """

    number: int
    train: tuple
    valid: tuple
    test: tuple

    def split(self, df):
        """Parte ``df`` (índice de fechas) en (train, valid, test) sin traslape."""
        return tuple(df.loc[start:end] for start, end in (self.train, self.valid, self.test))


def get_folds():
    """Los 4 folds de ventana expansiva del enunciado (sección 3.6)."""
    return [
        Fold(1, ("2017-08", "2021-12"), ("2022", "2022"), ("2023", "2023")),
        Fold(2, ("2017-08", "2022-12"), ("2023", "2023"), ("2024", "2024")),
        Fold(3, ("2017-08", "2023-12"), ("2024", "2024"), ("2025", "2025")),
        Fold(4, ("2017-08", "2024-12"), ("2025", "2025"), ("2026-01", "2026-04")),
    ]


SHORT_WINDOW = 24
LONG_WINDOW = 168
RSI_WINDOW = 14
TREND_HALF_LIFE_HOURS = 24


def log_return_features(close):
    """Familia retornos: log-retorno a 1, 4, 24 y 168 barras."""
    log_close = np.log(close)
    return pd.DataFrame(
        {f"ret_{k}": log_close.diff(k) for k in (1, 4, SHORT_WINDOW, LONG_WINDOW)}
    )


def volatility_features(bars):
    """Familia volatilidad: desviación móvil del retorno, rango de la barra y Parkinson."""
    ret_1 = np.log(bars["close"]).diff()
    log_range = np.log(bars["high"] / bars["low"])
    parkinson = np.sqrt((log_range**2).rolling(SHORT_WINDOW).mean() / (4 * np.log(2)))
    return pd.DataFrame(
        {
            "vol_24": ret_1.rolling(SHORT_WINDOW).std(),
            "vol_168": ret_1.rolling(LONG_WINDOW).std(),
            "range_1": log_range,
            "parkinson_24": parkinson,
        }
    )


def volume_features(bars):
    """Familia volumen: log-volumen y log-operaciones contra su media móvil.

    Se usa log1p porque hay 4 barras con volumen cero.
    """
    log_volume = np.log1p(bars["volume"])
    log_trades = np.log1p(bars["trades"])
    return pd.DataFrame(
        {
            "volume_ratio_24": log_volume - log_volume.rolling(SHORT_WINDOW).mean(),
            "volume_ratio_168": log_volume - log_volume.rolling(LONG_WINDOW).mean(),
            "trades_ratio_24": log_trades - log_trades.rolling(SHORT_WINDOW).mean(),
        }
    )


def ema(series, span):
    """Media móvil exponencial causal (recursiva, sin ajuste de sesgo inicial)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close, window=RSI_WINDOW):
    """RSI de Wilder entre 0 y 100 (suavizado exponencial con alpha = 1/window)."""
    change = close.diff()
    smooth = {"adjust": False, "alpha": 1 / window, "min_periods": window}
    gain = change.clip(lower=0).ewm(**smooth).mean()
    loss = (-change).clip(lower=0).ewm(**smooth).mean()
    return 100 - 100 / (1 + gain / loss)


def trend_momentum_features(close):
    """Familia tendencia/momentum: distancia del log-precio a su EMA y RSI."""
    log_close = np.log(close)
    return pd.DataFrame(
        {
            "dist_ema_24": log_close - ema(log_close, SHORT_WINDOW),
            "dist_ema_168": log_close - ema(log_close, LONG_WINDOW),
            "rsi_14": rsi(close),
        }
    )


def time_features(index):
    """Familia tiempo: hora del día y día de la semana (UTC) en seno y coseno."""
    hour = index.hour.to_numpy()
    day = index.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * day / 7),
            "dow_cos": np.cos(2 * np.pi * day / 7),
        },
        index=index,
    )


def kalman_trend(ret_1, noise_var, hours_elapsed, half_life=TREND_HALF_LIFE_HOURS):
    """Filtro de Kalman de la tendencia latente mu_t (log-retorno esperado por hora).

    Modelo (por hora de reloj):
        estado:      mu_t = mu_{t-1} + eta,   Var(eta) = q * R_t por hora
        observación: y_t = mu_t + eps,        Var(eps) = R_t
    donde y_t es el log-retorno horario y R_t = vol_24_t**2 la varianza
    horaria del ruido.

    Huecos: en una barra con hueco de dt horas solo se hace la predicción
    (la varianza crece q * R_t * dt y la media no cambia); su retorno, que
    abarca varias horas en las que mu pudo moverse, no se usa como medición.
    Así el hueco eleva la incertidumbre en lugar de rellenarse.

    q no se estima con datos: se fija para que la ganancia estacionaria sin
    huecos, k, olvide la mitad de la información en ``half_life`` horas
    (k = 1 - 0.5**(1/half_life) y q = k**2 / (1 - k)).

    Devuelve (media, desviación estándar) posteriores de mu_t; NaN hasta que
    existe R_t. Cada paso usa solo datos hasta t (causal).
    """
    gain = 1 - 0.5 ** (1 / half_life)
    q = gain**2 / (1 - gain)
    y, r, dt = (np.asarray(a, dtype=float) for a in (ret_1, noise_var, hours_elapsed))
    mean = np.full(len(y), np.nan)
    var = np.full(len(y), np.nan)
    mu, p = 0.0, np.nan
    for t in range(len(y)):
        if not (np.isfinite(r[t]) and np.isfinite(y[t])):
            continue
        p = r[t] if np.isnan(p) else p + q * r[t] * dt[t]
        if dt[t] == 1:
            k = p / (p + r[t])
            mu = mu + k * (y[t] - mu)
            p = (1 - k) * p
        mean[t], var[t] = mu, p
    return mean, np.sqrt(var)


def belief_features(bars):
    """Familia creencia: tendencia estimada por Kalman, su incertidumbre y su razón."""
    ret_1 = np.log(bars["close"]).diff()
    noise_var = ret_1.rolling(SHORT_WINDOW).var()
    hours = bars.index.to_series().diff() / pd.Timedelta(hours=1)
    mean, std = kalman_trend(ret_1, noise_var, hours)
    return pd.DataFrame(
        {"trend_mean": mean, "trend_std": std, "trend_snr": mean / std},
        index=bars.index,
    )


def build_features(bars):
    """Variables causales del estado de información y sus familias.

    La fila t usa solo datos hasta el cierre de la barra t (se verifica en
    tests/test_causality.py). Devuelve ``(features, families)``, con
    ``families = {familia: [columnas]}`` para la ablación por familia.

    Huecos: las ventanas móviles cuentan BARRAS OBSERVADAS, no horas de reloj.
    Una ventana que cruza un hueco cubre más tiempo del nominal, y el retorno
    de una barra con ``gap`` abarca varias horas. No se rellena ningún precio;
    ``gap_report`` cuenta cuántas filas tienen un hueco dentro de su ventana.
    El filtro de la familia creencia sí usa las horas reales transcurridas.

    Calentamiento: las primeras 168 barras (2017-08) quedan con NaN porque
    sus ventanas aún no están completas.
    """
    groups = {
        "retornos": log_return_features(bars["close"]),
        "volatilidad": volatility_features(bars),
        "volumen": volume_features(bars),
        "tendencia_momentum": trend_momentum_features(bars["close"]),
        "tiempo": time_features(bars.index),
        "creencia": belief_features(bars),
    }
    features = pd.concat(groups.values(), axis=1)
    families = {name: list(frame.columns) for name, frame in groups.items()}
    return features, families


class Normalizer:
    """Escalado robusto ajustado SOLO con el train de un fold (un objeto por fold).

    z = (x - mediana_train) / IQR_train, recortado a [-clip, clip].
    Se usan mediana e IQR (no media y desviación) porque los retornos tienen
    colas gordas y unos pocos extremos inflarían la desviación estándar.
    Solo se escalan ``columns``; las demás pasan sin cambio (p. ej. las de
    tiempo, que ya están en [-1, 1]). Los estadísticos se congelan en
    ``fit`` y se aplican sin modificación a validación y prueba.
    """

    def __init__(self, columns, clip=5.0):
        self.columns = list(columns)
        self.clip = clip
        self.median_ = None
        self.iqr_ = None

    def fit(self, train):
        """Calcula mediana e IQR de cada columna con el train del fold."""
        values = train[self.columns]
        self.median_ = values.median()
        self.iqr_ = values.quantile(0.75) - values.quantile(0.25)
        if (self.iqr_ <= 0).any():
            raise ValueError(f"IQR cero en: {list(self.iqr_[self.iqr_ <= 0].index)}")
        return self

    def transform(self, x):
        """Aplica los estadísticos congelados del train a cualquier periodo."""
        if self.median_ is None:
            raise RuntimeError("Normalizer.transform antes de fit.")
        scaled = x.copy()
        z = (x[self.columns] - self.median_) / self.iqr_
        scaled[self.columns] = z.clip(-self.clip, self.clip)
        return scaled


def cleaning_summary(raw, bars):
    """Cifras de la limpieza para el reporte (todo el archivo, sin partir en folds)."""
    timestamps = raw["date"]
    off_hour = (timestamps != timestamps.dt.floor("h")).sum()
    full_grid = pd.date_range(bars.index[0], bars.index[-1], freq="h")
    return {
        "filas_crudas": len(raw),
        "duplicados_eliminados": len(raw) - len(bars),
        "barras_redondeadas_a_la_hora": int(off_hour),
        "barras_unicas": len(bars),
        "horas_en_malla_completa": len(full_grid),
        "horas_faltantes": len(full_grid) - len(bars),
        "barras_con_gap": int(bars["gap"].sum()),
        "barras_volumen_cero": int((bars["volume"] == 0).sum()),
        "inicio": str(bars.index[0]),
        "fin": str(bars.index[-1]),
    }
