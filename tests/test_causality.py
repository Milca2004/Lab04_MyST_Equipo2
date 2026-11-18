"""Prueba obligatoria 2 (sección 3.11): causalidad de la observación.

Reconstruir la observación con los datos hasta el índice i devuelve el mismo
vector que el cálculo sobre la serie completa. Se revisan cuatro índices:
una barra de 2018 (train de todos los folds), la barra que llega después del
hueco de 145 horas de julio de 2025, la barra siguiente a esa y la última
barra del archivo.
"""
import numpy as np
import pandas as pd
import pytest

import data

LONGEST_GAP_END = pd.Timestamp("2025-07-22 01:00", tz="UTC")


@pytest.fixture(scope="module")
def bars():
    """Barras limpias del archivo completo."""
    return data.load_clean()


@pytest.fixture(scope="module")
def full_features(bars):
    """Variables calculadas una sola vez sobre la serie completa."""
    return data.build_features(bars)


def check_indices(bars):
    """Índices a revisar, con un nombre que dice por qué se eligió cada uno."""
    gap_position = bars.index.get_loc(LONGEST_GAP_END)
    return {
        "barra_2018": 5000,
        "barra_con_hueco_145h": gap_position,
        "barra_despues_del_hueco": gap_position + 1,
        "ultima_barra": len(bars) - 1,
    }


CASES = ["barra_2018", "barra_con_hueco_145h", "barra_despues_del_hueco", "ultima_barra"]


def assert_same_vector(a, b):
    """Mismos valores (incluidos los NaN del calentamiento) con tolerancia de redondeo."""
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-12, atol=1e-15)


def test_el_hueco_existe(bars):
    """El índice 'cercano a un hueco' de verdad cae justo después del hueco de 145 h."""
    assert bars.loc[LONGEST_GAP_END, "gap_hours"] == 145


@pytest.mark.parametrize("case", CASES)
def test_variables_hasta_i_igual_a_serie_completa(bars, full_features, case):
    """build_features con datos hasta i da la misma fila i que con la serie completa."""
    i = check_indices(bars)[case]
    truncated, _ = data.build_features(bars.iloc[: i + 1])
    assert_same_vector(truncated.iloc[-1], full_features[0].iloc[i])


@pytest.mark.parametrize("case", CASES)
def test_observacion_normalizada_hasta_i(bars, full_features, case):
    """La observación normalizada (Normalizer congelado del train del fold 1) también coincide."""
    i = check_indices(bars)[case]
    features, families = full_features
    scaled_columns = [c for family, cols in families.items() if family != "tiempo" for c in cols]
    train = data.get_folds()[0].split(features.dropna())[0]
    normalizer = data.Normalizer(scaled_columns).fit(train)
    truncated, _ = data.build_features(bars.iloc[: i + 1])
    assert_same_vector(normalizer.transform(truncated.iloc[[-1]]).iloc[0],
                       normalizer.transform(features.iloc[[i]]).iloc[0])


@pytest.mark.parametrize("case", CASES[:-1])
def test_alterar_el_futuro_no_cambia_la_fila_i(bars, full_features, case):
    """Si se alteran los precios posteriores a i, la fila i no cambia (no hay fuga del futuro)."""
    i = check_indices(bars)[case]
    altered = bars.copy()
    future = altered.index[i + 1:]
    altered.loc[future, ["open", "high", "low", "close"]] *= 2.0
    altered.loc[future, ["volume", "trades"]] *= 3
    altered_features, _ = data.build_features(altered)
    assert_same_vector(altered_features.iloc[i], full_features[0].iloc[i])
    assert not np.allclose(altered_features.iloc[i + 1], full_features[0].iloc[i + 1], equal_nan=True)
