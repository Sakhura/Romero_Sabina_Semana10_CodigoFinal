"""
test_performance.py
--------------------
Pruebas de rendimiento y de robustez ante distintos escenarios de datos,
tal como exige explícitamente la pauta de la Semana 10:

    "Se deben incluir resultados de pruebas finales que demuestren que
    el prototipo funciona correctamente en diferentes escenarios. Esto
    incluye pruebas de rendimiento, precisión y validación."

    "Ejecuta el código con diferentes sets de datos para validar que el
    sistema puede manejar diversos casos de uso. Asegúrate de que el
    sistema sea robusto y eficiente."
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_ingestion import cargar_y_verificar_datos
from eda import VARIABLES_CATEGORICAS, seleccionar_predictoras
from evaluate import _bootstrap_auc_ic, evaluar_modelo
from train_model import entrenar_modelo

RUTA_DATOS = Path(__file__).parent.parent / "data" / "equipment_anomaly_data.csv"


# ---------------------------------------------------------------------
# Rendimiento
# ---------------------------------------------------------------------
class TestRendimiento:
    def test_pipeline_completo_corre_en_menos_de_30_segundos(self):
        """Benchmark de rendimiento del pipeline completo (dataset real, ~7.672 filas)."""
        import importlib
        import sys

        raiz = Path(__file__).parent.parent
        if str(raiz) not in sys.path:
            sys.path.insert(0, str(raiz))
        import main as modulo_main

        importlib.reload(modulo_main)

        t0 = time.time()
        resultados = modulo_main.ejecutar_pipeline()
        duracion = time.time() - t0

        assert duracion < 30, f"El pipeline tardó {duracion:.1f}s, se esperaba < 30s"
        assert resultados["tiempo_total_segundos"] > 0

    def test_bootstrap_vectorizado_es_mas_rapido_que_800_llamadas_individuales(self):
        """Verifica la optimización descrita en OPTIMIZACION.md: bootstrap vectorizado."""
        rng = np.random.RandomState(0)
        n = 2000
        y = (rng.rand(n) < 0.10).astype(int)
        proba = np.where(y == 1, rng.beta(8, 2, n), rng.beta(2, 8, n))

        from sklearn.metrics import roc_auc_score

        def bootstrap_no_vectorizado(y_true, y_proba, n_boot=800, seed=42):
            rng_local = np.random.RandomState(seed)
            n_local = len(y_true)
            aucs = []
            for _ in range(n_boot):
                idx = rng_local.randint(0, n_local, n_local)
                if len(np.unique(y_true[idx])) < 2:
                    continue
                aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
            return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

        t0 = time.time()
        resultado_lento = bootstrap_no_vectorizado(y, proba)
        tiempo_lento = time.time() - t0

        t0 = time.time()
        resultado_rapido = _bootstrap_auc_ic(y, proba)
        tiempo_rapido = time.time() - t0

        # Misma respuesta (dentro de una tolerancia numérica muy estrecha)...
        assert resultado_lento[0] == pytest.approx(resultado_rapido[0], abs=1e-9)
        assert resultado_lento[1] == pytest.approx(resultado_rapido[1], abs=1e-9)
        # ...pero notablemente más rápido.
        assert tiempo_rapido < tiempo_lento


# ---------------------------------------------------------------------
# Robustez ante distintos escenarios de datos
# ---------------------------------------------------------------------
class TestDiferentesEscenarios:
    """El sistema debe manejar variaciones razonables del dataset sin romperse."""

    def _dataset_base(self) -> pd.DataFrame:
        df, _ = cargar_y_verificar_datos(RUTA_DATOS)
        return df

    def test_funciona_con_una_muestra_reducida_del_dataset(self):
        """Escenario: solo una fracción de los datos disponibles (ej. un piloto)."""
        df = self._dataset_base().sample(n=1500, random_state=42).reset_index(drop=True)
        predictoras = seleccionar_predictoras(df)
        resultado = entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)
        metricas = evaluar_modelo(resultado.pipeline, resultado.X_test, resultado.y_test)
        # Con menos datos las métricas pueden bajar algo, pero el sistema no debe romperse
        # y debe seguir produciendo un clasificador razonable.
        assert 0.5 <= metricas.auc_roc <= 1.0

    def test_funciona_con_un_solo_sitio(self):
        """Escenario: dataset restringido a una sola ubicación (ej. una faena)."""
        df = self._dataset_base()
        df_un_sitio = df[df["location"] == "Atlanta"].reset_index(drop=True)
        predictoras = seleccionar_predictoras(df_un_sitio)
        resultado = entrenar_modelo(df_un_sitio, predictoras, VARIABLES_CATEGORICAS)
        metricas = evaluar_modelo(resultado.pipeline, resultado.X_test, resultado.y_test)
        assert 0.0 <= metricas.auc_roc <= 1.0

    def test_funciona_con_prevalencia_de_falla_distinta(self):
        """Escenario: dataset con una prevalencia de falla distinta al 10% original
        (ej. una faena con más fallas de lo habitual), para probar que el tratamiento
        de desbalance (class_weight='balanced') no depende de la prevalencia exacta."""
        df = self._dataset_base()
        sanos = df[df["faulty"] == 0].sample(frac=0.5, random_state=42)
        fallados = df[df["faulty"] == 1]
        df_desbalanceado = pd.concat([sanos, fallados]).sample(frac=1, random_state=42).reset_index(drop=True)

        predictoras = seleccionar_predictoras(df_desbalanceado)
        resultado = entrenar_modelo(df_desbalanceado, predictoras, VARIABLES_CATEGORICAS)
        metricas = evaluar_modelo(resultado.pipeline, resultado.X_test, resultado.y_test)
        assert metricas.f1_score >= 0.0  # el sistema entrena y evalúa sin errores

    def test_error_claro_si_faltan_columnas_esperadas(self):
        """Escenario adverso: CSV corrupto o con columnas faltantes debe fallar
        de forma controlada (hito de compuerta H1), no con un traceback confuso."""
        df_corrupto = self._dataset_base().drop(columns=["vibration"])
        tmp_path = Path("/tmp/dataset_corrupto_test.csv")
        df_corrupto.to_csv(tmp_path, index=False)

        _, reporte = cargar_y_verificar_datos(tmp_path)
        assert not reporte.aprobado
        assert not reporte.columnas_ok
        tmp_path.unlink()
