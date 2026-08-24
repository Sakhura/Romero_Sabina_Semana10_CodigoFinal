"""
test_unit.py
------------
Pruebas unitarias de funciones puras e individuales del sistema.
Complementan a `test_integration.py` (que prueba la interacción entre
módulos) con casos de borde a nivel de función, tal como pide la pauta
de la Semana 10 ("pruebas exhaustivas... que demuestren que el sistema
funciona correctamente" en distintos escenarios).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmms_connector import clasificar_riesgo, prioridad_desde_riesgo
from eda import calcular_correlaciones, seleccionar_predictoras
from reproducibility_audit import _coincide_a_n_decimales


class TestClasificarRiesgo:
    """Casos de borde de la regla de riesgo (WBS 4.1)."""

    def test_umbral_alto_exacto_es_alto(self):
        assert clasificar_riesgo(0.70) == "ALTO"

    def test_justo_bajo_el_umbral_alto_es_medio(self):
        assert clasificar_riesgo(0.6999) == "MEDIO"

    def test_umbral_medio_exacto_es_medio(self):
        assert clasificar_riesgo(0.40) == "MEDIO"

    def test_justo_bajo_el_umbral_medio_es_bajo(self):
        assert clasificar_riesgo(0.3999) == "BAJO"

    def test_probabilidad_cero_es_bajo(self):
        assert clasificar_riesgo(0.0) == "BAJO"

    def test_probabilidad_uno_es_alto(self):
        assert clasificar_riesgo(1.0) == "ALTO"

    @pytest.mark.parametrize(
        "nivel,prioridad_esperada",
        [("ALTO", "URGENTE"), ("MEDIO", "PROGRAMADA"), ("BAJO", "SIN ACCIÓN")],
    )
    def test_prioridad_desde_riesgo(self, nivel, prioridad_esperada):
        assert prioridad_desde_riesgo(nivel) == prioridad_esperada


class TestSeleccionPredictoras:
    """Casos de borde de la selección de predictoras por correlación (OE1, Sumativa 2)."""

    def _dataset_sintetico(self) -> pd.DataFrame:
        rng = np.random.RandomState(0)
        n = 500
        faulty = rng.binomial(1, 0.1, n)
        # variable con correlación fuerte (por construcción)
        fuerte = faulty * 5 + rng.normal(0, 1, n)
        # variable con correlación nula (ruido puro)
        ruido = rng.normal(0, 1, n)
        return pd.DataFrame(
            {
                "temperature": fuerte,
                "pressure": ruido,
                "vibration": fuerte,
                "humidity": ruido,
                "equipment": rng.choice(["Turbine", "Pump", "Compressor"], n),
                "location": rng.choice(["Atlanta", "Chicago"], n),
                "faulty": faulty,
            }
        )

    def test_variable_fuerte_se_incluye(self):
        df = self._dataset_sintetico()
        predictoras = seleccionar_predictoras(df, umbral=0.15)
        assert "temperature" in predictoras
        assert "vibration" in predictoras

    def test_variable_ruido_se_excluye(self):
        df = self._dataset_sintetico()
        predictoras = seleccionar_predictoras(df, umbral=0.15)
        assert "pressure" not in predictoras
        assert "humidity" not in predictoras

    def test_umbral_mas_estricto_selecciona_menos_variables(self):
        df = self._dataset_sintetico()
        predictoras_laxo = seleccionar_predictoras(df, umbral=0.05)
        predictoras_estricto = seleccionar_predictoras(df, umbral=0.9)
        assert len(predictoras_estricto) <= len(predictoras_laxo)

    def test_correlaciones_estan_entre_menos_uno_y_uno(self):
        df = self._dataset_sintetico()
        correlaciones = calcular_correlaciones(df)
        assert (correlaciones.abs() <= 1.0 + 1e-9).all()


class TestReproducibilidad:
    """Tolerancia de comparación de la auditoría de reproducibilidad (V4)."""

    def test_valores_identicos_coinciden(self):
        assert _coincide_a_n_decimales(0.974552, 0.974552, decimales=4)

    def test_diferencia_en_quinto_decimal_coincide_a_4_decimales(self):
        assert _coincide_a_n_decimales(0.97455, 0.97456, decimales=4)

    def test_diferencia_en_tercer_decimal_no_coincide_a_4_decimales(self):
        assert not _coincide_a_n_decimales(0.9740, 0.9750, decimales=4)
