"""
reproducibility_audit.py
-------------------------
Auditoría de reproducibilidad (Fase 5, WBS 5.2 de la Carta Gantt):
    "Auditoría de reproducibilidad: re-ejecución en limpio con semilla 42"
    -> V4 "Reproducibilidad certificada": las métricas se reproducen al
       cuarto decimal.

Esta pieza estaba planificada en la Carta Gantt de la Semana 7 pero
quedaba fuera del alcance de la pre-entrega de la Semana 9 (que cubría
las Fases 1-4). El código final de la Semana 10 la implementa: vuelve a
ejecutar el pipeline completo de principio a fin con la misma semilla y
compara, con `math.isclose` a la tolerancia declarada, que el AUC-ROC y
el F1 de ambas ejecuciones coinciden en las primeras 4 decimales.

No reentrena "a mano": reutiliza `main.ejecutar_pipeline()` dos veces
en carpetas de resultados independientes, exactamente como lo haría un
auditor externo que corre el mismo código dos veces.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from config import DECIMALES_REPRODUCIBILIDAD
except ImportError:  # pragma: no cover
    DECIMALES_REPRODUCIBILIDAD = 4

logger = logging.getLogger(__name__)


@dataclass
class ReporteReproducibilidad:
    auc_roc_corrida_1: float
    auc_roc_corrida_2: float
    f1_corrida_1: float
    f1_corrida_2: float
    decimales_exigidos: int
    coincide: bool

    def resumen(self) -> str:
        estado = "CERTIFICADA" if self.coincide else "NO CERTIFICADA"
        return (
            f"[V4 · Auditoría de reproducibilidad] {estado} "
            f"(tolerancia: {self.decimales_exigidos} decimales)\n"
            f"  - AUC-ROC  corrida 1 = {self.auc_roc_corrida_1:.6f} | corrida 2 = {self.auc_roc_corrida_2:.6f}\n"
            f"  - F1-score corrida 1 = {self.f1_corrida_1:.6f} | corrida 2 = {self.f1_corrida_2:.6f}\n"
        )


def _coincide_a_n_decimales(a: float, b: float, decimales: int) -> bool:
    tolerancia = 0.5 * 10 ** (-decimales)
    return math.isclose(a, b, abs_tol=tolerancia)


def auditar_reproducibilidad(ruta_datos: str | Path, decimales: int = DECIMALES_REPRODUCIBILIDAD) -> ReporteReproducibilidad:
    """Ejecuta el pipeline dos veces con la misma semilla y compara resultados.

    Reutiliza `main.ejecutar_pipeline()` para no duplicar la lógica de
    entrenamiento: una auditoría de reproducibilidad debe correr el mismo
    código de producción, no una copia paralela de él.
    """
    import importlib
    import sys

    raiz = Path(__file__).resolve().parent.parent
    if str(raiz) not in sys.path:
        sys.path.insert(0, str(raiz))

    import main as modulo_main

    importlib.reload(modulo_main)
    ruta_original = modulo_main.RUTA_DATOS
    carpeta_original = modulo_main.CARPETA_RESULTADOS

    try:
        modulo_main.RUTA_DATOS = str(ruta_datos)

        modulo_main.CARPETA_RESULTADOS = str(raiz / "results" / "_auditoria_corrida_1")
        Path(modulo_main.CARPETA_RESULTADOS).mkdir(parents=True, exist_ok=True)
        resultado_1 = modulo_main.ejecutar_pipeline()

        modulo_main.CARPETA_RESULTADOS = str(raiz / "results" / "_auditoria_corrida_2")
        Path(modulo_main.CARPETA_RESULTADOS).mkdir(parents=True, exist_ok=True)
        resultado_2 = modulo_main.ejecutar_pipeline()
    finally:
        modulo_main.RUTA_DATOS = ruta_original
        modulo_main.CARPETA_RESULTADOS = carpeta_original

    auc_1 = resultado_1["metricas_prueba"].auc_roc
    auc_2 = resultado_2["metricas_prueba"].auc_roc
    f1_1 = resultado_1["f1_entrenamiento"]
    f1_2 = resultado_2["f1_entrenamiento"]

    coincide = _coincide_a_n_decimales(auc_1, auc_2, decimales) and _coincide_a_n_decimales(f1_1, f1_2, decimales)

    reporte = ReporteReproducibilidad(
        auc_roc_corrida_1=auc_1,
        auc_roc_corrida_2=auc_2,
        f1_corrida_1=f1_1,
        f1_corrida_2=f1_2,
        decimales_exigidos=decimales,
        coincide=coincide,
    )
    logger.info(reporte.resumen())
    return reporte


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reporte = auditar_reproducibilidad("data/equipment_anomaly_data.csv")
    print(reporte.resumen())
    if not reporte.coincide:
        raise SystemExit("V4 no certificada: las métricas no se reprodujeron dentro de la tolerancia.")
