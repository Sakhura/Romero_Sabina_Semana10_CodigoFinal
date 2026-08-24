"""
main.py
-------
Punto de entrada único del sistema. Ejecuta el pipeline analítico de
extremo a extremo descrito en la Carta Gantt (Semana 7) y completado en
el código final de la Semana 10 (Código Final y Demostración):

    Fase 1 - Ingesta y EDA           -> data_ingestion.py, eda.py
    Fase 2 - Entrenamiento           -> train_model.py
    Fase 3 - Validación              -> evaluate.py
    Fase 4 - Integración CMMS/tablero-> dashboard_prototype.py, cmms_connector.py
    Fase 5 - Cierre y reproducibilidad -> generar_informe_tecnico.py, reproducibility_audit.py

Uso:
    python main.py                        # pipeline completo con dataset por defecto
    python main.py --data otro.csv         # ejecuta con un dataset distinto (pruebas finales)
    python main.py --output resultados2    # cambia la carpeta de salida
    python main.py --skip-audit            # omite la auditoría de reproducibilidad (Fase 5)
    python main.py --quiet                 # solo advertencias/errores en consola

Todos los artefactos generados (figuras, métricas, modelo, tablero,
órdenes de trabajo, informe técnico) se escriben en la carpeta de
resultados indicada (por defecto `results/`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cmms_connector import UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO  # noqa: E402
from dashboard_prototype import ejecutar_prototipo_tablero  # noqa: E402
from data_ingestion import cargar_y_verificar_datos  # noqa: E402
from eda import VARIABLES_CATEGORICAS, generar_figuras_eda, seleccionar_predictoras  # noqa: E402
from evaluate import evaluar_modelo, generar_figuras_evaluacion, guardar_metricas_json  # noqa: E402
from generar_informe_tecnico import generar_informe  # noqa: E402
from train_model import F1_OBJETIVO, entrenar_modelo, guardar_modelo  # noqa: E402

# Variables de módulo (no constantes "congeladas"): reproducibility_audit.py las
# reasigna temporalmente para poder correr el pipeline dos veces sobre carpetas
# distintas sin duplicar código. CLI también las sobreescribe vía argparse.
RUTA_DATOS = "data/equipment_anomaly_data.csv"
CARPETA_RESULTADOS = "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("main")


def ejecutar_pipeline() -> dict:
    """Ejecuta el pipeline completo y retorna un diccionario con las métricas clave.

    Este diccionario también es usado por `tests/test_integration.py` para
    verificar, con datos reales, que el sistema entrega los resultados
    esperados por la rúbrica de la Semana 9 (Desarrollo e Integración,
    Pruebas de Integración, Resultados y Ejemplos de Pruebas).
    """
    t0 = time.time()
    Path(CARPETA_RESULTADOS).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # FASE 1 · Ingesta y EDA
    # ------------------------------------------------------------------
    logger.info("=== FASE 1 · Ingesta de datos y preparación (EDA) ===")
    df, reporte_integridad = cargar_y_verificar_datos(RUTA_DATOS)
    print(reporte_integridad.resumen())
    if not reporte_integridad.aprobado:
        raise RuntimeError("Hito H1 no superado: el histórico no pasó la verificación de integridad (V1).")

    predictoras = seleccionar_predictoras(df)
    generar_figuras_eda(df, f"{CARPETA_RESULTADOS}/figuras")
    logger.info("H1 cumplido. Predictoras seleccionadas: %s", predictoras)

    # ------------------------------------------------------------------
    # FASE 2 · Entrenamiento del modelo
    # ------------------------------------------------------------------
    logger.info("=== FASE 2 · Entrenamiento del modelo ===")
    resultado_entrenamiento = entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)
    guardar_modelo(resultado_entrenamiento, f"{CARPETA_RESULTADOS}/modelo_entrenado.joblib")

    if resultado_entrenamiento.f1_train < F1_OBJETIVO:
        raise RuntimeError(
            f"Hito H2 no superado: F1 en entrenamiento ({resultado_entrenamiento.f1_train:.4f}) "
            f"por debajo del objetivo ({F1_OBJETIVO})."
        )
    logger.info("H2 cumplido: F1 en entrenamiento = %.4f", resultado_entrenamiento.f1_train)

    # ------------------------------------------------------------------
    # FASE 3 · Validación del modelo
    # ------------------------------------------------------------------
    logger.info("=== FASE 3 · Validación del modelo ===")
    metricas = evaluar_modelo(
        resultado_entrenamiento.pipeline, resultado_entrenamiento.X_test, resultado_entrenamiento.y_test
    )
    generar_figuras_evaluacion(
        resultado_entrenamiento.pipeline,
        resultado_entrenamiento.X_test,
        resultado_entrenamiento.y_test,
        f"{CARPETA_RESULTADOS}/figuras",
    )
    guardar_metricas_json(
        metricas,
        f"{CARPETA_RESULTADOS}/metricas_prueba.json",
        extra={"f1_entrenamiento": resultado_entrenamiento.f1_train, "predictoras": predictoras},
    )

    if not metricas.cumple_objetivo_auc:
        logger.warning(
            "H3 no cumplido: AUC-ROC de prueba (%.4f) bajo el objetivo (%.2f). "
            "Ver Carta Gantt, sección 8 (cuellos de botella y respuesta).",
            metricas.auc_roc,
            metricas.umbral_objetivo_auc,
        )
    else:
        logger.info("H3 cumplido: AUC-ROC de prueba = %.4f", metricas.auc_roc)

    # ------------------------------------------------------------------
    # FASE 4 · Integración con el tablero y el CMMS
    # ------------------------------------------------------------------
    logger.info("=== FASE 4 · Integración con el tablero y el CMMS ===")
    df_riesgo = ejecutar_prototipo_tablero(
        resultado_entrenamiento.pipeline,
        resultado_entrenamiento.X_test,
        metricas.auc_roc,
        f"{CARPETA_RESULTADOS}/tablero.html",
        f"{CARPETA_RESULTADOS}/ordenes_trabajo_cmms.json",
    )
    n_alto = int((df_riesgo["nivel_riesgo"] == "ALTO").sum())
    n_medio = int((df_riesgo["nivel_riesgo"] == "MEDIO").sum())
    logger.info(
        "H4 cumplido: prototipo de tablero generado. Riesgo ALTO=%d (umbral>=%.0f%%), "
        "MEDIO=%d (umbral>=%.0f%%) de %d equipos evaluados.",
        n_alto,
        UMBRAL_RIESGO_ALTO * 100,
        n_medio,
        UMBRAL_RIESGO_MEDIO * 100,
        len(df_riesgo),
    )

    tiempo_total = time.time() - t0
    logger.info("=== Pipeline completo ejecutado en %.2f segundos ===", tiempo_total)

    return {
        "reporte_integridad": reporte_integridad,
        "predictoras": predictoras,
        "f1_entrenamiento": resultado_entrenamiento.f1_train,
        "metricas_prueba": metricas,
        "n_ordenes_alto": n_alto,
        "n_ordenes_medio": n_medio,
        "tiempo_total_segundos": tiempo_total,
    }


def ejecutar_fase5_cierre(carpeta_resultados: str, ejecutar_auditoria: bool = True) -> None:
    """Fase 5 · Cierre, reproducibilidad y transferencia (WBS 5.1, 5.2; hitos V4, H5).

    1. Genera el informe técnico del PoC a partir de las métricas reales
       de la ejecución que se acaba de completar (WBS 5.1).
    2. Si `ejecutar_auditoria` es True, corre el pipeline completo dos
       veces más (con la misma semilla) y certifica que el AUC-ROC y el
       F1 se reproducen al cuarto decimal (WBS 5.2, hito V4).
    """
    logger.info("=== FASE 5 · Cierre, reproducibilidad y transferencia ===")

    generar_informe(
        ruta_metricas=f"{carpeta_resultados}/metricas_prueba.json",
        ruta_salida=f"{carpeta_resultados}/informe_tecnico_poc.md",
    )

    if not ejecutar_auditoria:
        logger.info("Auditoría de reproducibilidad omitida (--skip-audit).")
        return

    from reproducibility_audit import auditar_reproducibilidad

    reporte_auditoria = auditar_reproducibilidad(RUTA_DATOS)
    ruta_evidencia = Path(carpeta_resultados) / "evidencia_auditoria_reproducibilidad.txt"
    ruta_evidencia.write_text(reporte_auditoria.resumen(), encoding="utf-8")

    if reporte_auditoria.coincide:
        logger.info("V4 certificada: la reproducibilidad se verificó al cuarto decimal. H5: PoC cerrado.")
    else:
        logger.warning("V4 NO certificada: revisar fuentes de no determinismo antes del cierre (H5).")


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline del Sistema de Monitoreo de Equipos Industriales (IDA300, Semana 10)."
    )
    parser.add_argument(
        "--data",
        default=RUTA_DATOS,
        help="Ruta al CSV de entrada (permite correr el sistema con distintos datasets de prueba).",
    )
    parser.add_argument(
        "--output",
        default=CARPETA_RESULTADOS,
        help="Carpeta donde se escriben todos los artefactos generados.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Omite la auditoría de reproducibilidad de la Fase 5 (pipeline se ejecuta una sola vez).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce el nivel de logging a WARNING (útil para demostraciones en vivo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parsear_argumentos()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    RUTA_DATOS = args.data
    CARPETA_RESULTADOS = args.output

    resultados = ejecutar_pipeline()
    ejecutar_fase5_cierre(CARPETA_RESULTADOS, ejecutar_auditoria=not args.skip_audit)

    print("\nResumen final del pipeline")
    print("---------------------------")
    print(f"Predictoras utilizadas : {resultados['predictoras']}")
    print(f"F1 (entrenamiento)     : {resultados['f1_entrenamiento']:.4f}")
    print(resultados["metricas_prueba"].resumen())
    print(f"Órdenes de trabajo ALTO/MEDIO generadas: {resultados['n_ordenes_alto']}/{resultados['n_ordenes_medio']}")
    print(f"Tiempo total de ejecución: {resultados['tiempo_total_segundos']:.2f} s")
    print(f"\nArtefactos disponibles en la carpeta '{CARPETA_RESULTADOS}/':")
    print("  - modelo_entrenado.joblib")
    print("  - metricas_prueba.json")
    print("  - tablero.html")
    print("  - ordenes_trabajo_cmms.json")
    print("  - informe_tecnico_poc.md")
    print("  - evidencia_auditoria_reproducibilidad.txt (si no se usó --skip-audit)")
    print("  - figuras/ (correlación, distribución, ROC, matriz de confusión, importancia)")
