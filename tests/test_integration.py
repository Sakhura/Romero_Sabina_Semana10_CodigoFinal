"""
test_integration.py
--------------------
Pruebas de integración del sistema completo, tal como exige la pauta
de la Semana 9 ("Pruebas de Integración" y "Pruebas y Resultados"):

    - Verifican que los módulos (ingesta, EDA, entrenamiento, evaluación,
      integración CMMS/tablero) interactúan correctamente end-to-end.
    - Verifican que el flujo de datos es correcto en cada frontera entre
      módulos (shape, tipos, ausencia de fuga de información).
    - Verifican que el sistema alcanza las métricas de rendimiento
      declaradas como objetivo en las Tareas Sumativas 1 y 2
      (F1 >= 0,85 en entrenamiento, AUC-ROC >= 0,90 en prueba).

Ejecutar con:
    pytest tests/ -v
desde la raíz del proyecto (con `src/` en el PYTHONPATH; ver README.md
o usar `python -m pytest`, que ya añade la raíz al path junto con el
`conftest.py` incluido).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cmms_connector import clasificar_riesgo, generar_ordenes_trabajo
from dashboard_prototype import calcular_riesgo_por_equipo, ejecutar_prototipo_tablero
from data_ingestion import N_REGISTROS_ESPERADOS, cargar_y_verificar_datos
from eda import VARIABLES_CATEGORICAS, seleccionar_predictoras
from evaluate import AUC_OBJETIVO, evaluar_modelo
from train_model import F1_OBJETIVO, entrenar_modelo

RUTA_DATOS = Path(__file__).parent.parent / "data" / "equipment_anomaly_data.csv"


# ---------------------------------------------------------------------
# Fixtures: se calculan una sola vez por sesión de pruebas porque
# entrenar el Random Forest en cada test sería lento e innecesario.
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def datos_cargados():
    df, reporte = cargar_y_verificar_datos(RUTA_DATOS)
    return df, reporte


@pytest.fixture(scope="session")
def predictoras(datos_cargados):
    df, _ = datos_cargados
    return seleccionar_predictoras(df)


@pytest.fixture(scope="session")
def resultado_entrenamiento(datos_cargados, predictoras):
    df, _ = datos_cargados
    return entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)


@pytest.fixture(scope="session")
def metricas_prueba(resultado_entrenamiento):
    return evaluar_modelo(resultado_entrenamiento.pipeline, resultado_entrenamiento.X_test, resultado_entrenamiento.y_test)


# ---------------------------------------------------------------------
# FASE 1 · Ingesta y EDA
# ---------------------------------------------------------------------
class TestIngestaYEDA:
    def test_carga_el_numero_de_filas_esperado(self, datos_cargados):
        df, _ = datos_cargados
        assert len(df) == N_REGISTROS_ESPERADOS

    def test_reporte_de_integridad_aprobado(self, datos_cargados):
        _, reporte = datos_cargados
        assert reporte.aprobado, reporte.resumen()

    def test_sin_valores_nulos(self, datos_cargados):
        df, _ = datos_cargados
        assert df.isnull().sum().sum() == 0

    def test_prevalencia_de_falla_cercana_al_10_por_ciento(self, datos_cargados):
        df, _ = datos_cargados
        assert 0.08 <= df["faulty"].mean() <= 0.12

    def test_seleccion_de_predictoras_respeta_el_umbral_de_correlacion(self, predictoras):
        # 'humidity' tiene r ~ 0.01 y debe quedar excluida (Objetivo Específico 1, Sumativa 2).
        assert "humidity" not in predictoras
        # Las tres variables de proceso con mayor señal deben estar presentes.
        for var in ["temperature", "pressure", "vibration"]:
            assert var in predictoras
        # Las categóricas siempre se incluyen para la codificación de la Fase 2.
        for var in VARIABLES_CATEGORICAS:
            assert var in predictoras


# ---------------------------------------------------------------------
# FASE 2 · Entrenamiento (interacción EDA -> train_model)
# ---------------------------------------------------------------------
class TestEntrenamiento:
    def test_particion_es_70_30_estratificada(self, resultado_entrenamiento, datos_cargados):
        df, _ = datos_cargados
        n_total = len(df)
        proporcion_test = len(resultado_entrenamiento.X_test) / n_total
        assert 0.29 <= proporcion_test <= 0.31

    def test_prevalencia_se_conserva_en_ambas_particiones(self, resultado_entrenamiento):
        prev_train = resultado_entrenamiento.y_train.mean()
        prev_test = resultado_entrenamiento.y_test.mean()
        # V2: ambas particiones deben conservar ~10% de prevalencia (±2 pp)
        assert abs(prev_train - prev_test) <= 0.03

    def test_no_hay_fuga_de_informacion_entre_train_y_test(self, resultado_entrenamiento):
        idx_train = set(resultado_entrenamiento.X_train.index)
        idx_test = set(resultado_entrenamiento.X_test.index)
        assert idx_train.isdisjoint(idx_test)

    def test_f1_entrenamiento_cumple_el_objetivo_H2(self, resultado_entrenamiento):
        assert resultado_entrenamiento.f1_train >= F1_OBJETIVO

    def test_pipeline_entrenado_predice_probabilidades_validas(self, resultado_entrenamiento):
        proba = resultado_entrenamiento.pipeline.predict_proba(resultado_entrenamiento.X_test)
        assert proba.shape[1] == 2
        assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------
# FASE 3 · Validación (interacción train_model -> evaluate)
# ---------------------------------------------------------------------
class TestValidacion:
    def test_auc_roc_cumple_el_objetivo_H3(self, metricas_prueba):
        assert metricas_prueba.auc_roc >= AUC_OBJETIVO

    def test_intervalo_de_confianza_bootstrap_es_coherente(self, metricas_prueba):
        assert metricas_prueba.auc_roc_ic95_bajo <= metricas_prueba.auc_roc <= metricas_prueba.auc_roc_ic95_alto

    def test_metricas_estan_en_rango_valido(self, metricas_prueba):
        for valor in [metricas_prueba.auc_roc, metricas_prueba.f1_score, metricas_prueba.precision, metricas_prueba.recall]:
            assert 0.0 <= valor <= 1.0


# ---------------------------------------------------------------------
# FASE 4 · Integración CMMS y tablero (interacción evaluate -> dashboard_prototype -> cmms_connector)
# ---------------------------------------------------------------------
class TestIntegracionCmmsYTablero:
    def test_clasificacion_de_riesgo_es_monotonica(self):
        assert clasificar_riesgo(0.95) == "ALTO"
        assert clasificar_riesgo(0.50) == "MEDIO"
        assert clasificar_riesgo(0.05) == "BAJO"

    def test_calculo_de_riesgo_agrega_columnas_esperadas(self, resultado_entrenamiento):
        df_riesgo = calcular_riesgo_por_equipo(resultado_entrenamiento.pipeline, resultado_entrenamiento.X_test)
        assert "probabilidad_falla" in df_riesgo.columns
        assert "nivel_riesgo" in df_riesgo.columns
        assert len(df_riesgo) == len(resultado_entrenamiento.X_test)

    def test_generacion_de_ordenes_de_trabajo_excluye_riesgo_bajo(self, resultado_entrenamiento):
        df_riesgo = calcular_riesgo_por_equipo(resultado_entrenamiento.pipeline, resultado_entrenamiento.X_test)
        ordenes = generar_ordenes_trabajo(df_riesgo)
        assert all(o.nivel_riesgo != "BAJO" for o in ordenes)
        assert len(ordenes) == int((df_riesgo["nivel_riesgo"] != "BAJO").sum())

    def test_pipeline_end_to_end_genera_todos_los_artefactos(self, resultado_entrenamiento, metricas_prueba, tmp_path):
        ruta_tablero = tmp_path / "tablero.html"
        ruta_ordenes = tmp_path / "ordenes_trabajo_cmms.json"

        df_riesgo = ejecutar_prototipo_tablero(
            resultado_entrenamiento.pipeline,
            resultado_entrenamiento.X_test,
            metricas_prueba.auc_roc,
            ruta_tablero,
            ruta_ordenes,
        )

        assert ruta_tablero.exists() and ruta_tablero.stat().st_size > 0
        assert ruta_ordenes.exists()

        with open(ruta_ordenes, encoding="utf-8") as f:
            ordenes_guardadas = json.load(f)
        assert isinstance(ordenes_guardadas, list)
        assert len(ordenes_guardadas) == int((df_riesgo["nivel_riesgo"] != "BAJO").sum())

        # El HTML debe contener el resumen de equipos evaluados y las clases de riesgo.
        contenido_html = ruta_tablero.read_text(encoding="utf-8")
        assert "Tablero de Riesgo" in contenido_html
        assert "AUC-ROC" in contenido_html


# ---------------------------------------------------------------------
# Prueba de humo (smoke test) del pipeline completo vía main.py
# ---------------------------------------------------------------------
def test_pipeline_completo_via_main(tmp_path, monkeypatch):
    """Ejecuta `ejecutar_pipeline()` de main.py apuntando a una carpeta de resultados temporal,
    confirmando que las cuatro fases se integran sin errores y cumplen los hitos H1-H4."""
    import importlib
    import sys

    raiz_proyecto = Path(__file__).parent.parent
    monkeypatch.chdir(raiz_proyecto)
    sys.path.insert(0, str(raiz_proyecto))

    import main as modulo_main

    importlib.reload(modulo_main)
    modulo_main.CARPETA_RESULTADOS = str(tmp_path)
    Path(modulo_main.CARPETA_RESULTADOS).mkdir(exist_ok=True)

    resultados = modulo_main.ejecutar_pipeline()

    assert resultados["reporte_integridad"].aprobado
    assert resultados["f1_entrenamiento"] >= F1_OBJETIVO
    assert resultados["metricas_prueba"].auc_roc >= AUC_OBJETIVO
    assert resultados["tiempo_total_segundos"] > 0
