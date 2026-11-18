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
