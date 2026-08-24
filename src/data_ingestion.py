"""
data_ingestion.py
------------------
Módulo de ingesta y verificación de integridad del histórico de monitoreo
de condición (Industrial Equipment Monitoring Dataset).

Responsabilidades (WBS 1.3 y V1 de la Carta Gantt, Semana 7):
    1. Cargar el archivo CSV con los 7.672 registros históricos.
    2. Verificar la integridad del archivo: número de filas, columnas
       esperadas, tipos de dato, ausencia de nulos y prevalencia de falla
       cercana al 10 % declarado en las Tareas Sumativas 1 y 2.
    3. Exponer una función única `cargar_y_verificar_datos()` que el resto
       del pipeline (main.py) puede invocar sin preocuparse por los
       detalles de la fuente de datos.

No se intenta "arreglar" datos corruptos aquí: si la verificación falla,
se lanza una excepción para detener el pipeline en el hito H1
(compuerta), tal como se documentó en la Carta Gantt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from config import (
        COLUMNAS_ESPERADAS,
        EQUIPOS_ESPERADOS,
        N_REGISTROS_ESPERADOS,
        PREVALENCIA_ESPERADA,
        TOLERANCIA_PREVALENCIA,
        UBICACIONES_ESPERADAS,
    )
except ImportError:  # pragma: no cover - fallback si se ejecuta fuera del paquete
    COLUMNAS_ESPERADAS = {
        "temperature": "numeric",
        "pressure": "numeric",
        "vibration": "numeric",
        "humidity": "numeric",
        "equipment": "categorical",
        "location": "categorical",
        "faulty": "numeric",
    }
    EQUIPOS_ESPERADOS = {"Turbine", "Compressor", "Pump"}
    UBICACIONES_ESPERADAS = {"Atlanta", "Chicago", "San Francisco", "New York", "Houston"}
    N_REGISTROS_ESPERADOS = 7672
    PREVALENCIA_ESPERADA = 0.10
    TOLERANCIA_PREVALENCIA = 0.02

logger = logging.getLogger(__name__)


@dataclass
class ReporteIntegridad:
    """Resultado de la verificación de integridad (hito V1)."""

    n_filas: int
    n_columnas: int
    columnas_ok: bool
    nulos_ok: bool
    tipos_de_equipo: set
    ubicaciones: set
    prevalencia_falla: float
    prevalencia_ok: bool

    @property
    def aprobado(self) -> bool:
        return all(
            [
                self.n_filas == N_REGISTROS_ESPERADOS,
                self.columnas_ok,
                self.nulos_ok,
                self.tipos_de_equipo == EQUIPOS_ESPERADOS,
                self.ubicaciones == UBICACIONES_ESPERADAS,
                self.prevalencia_ok,
            ]
        )

    def resumen(self) -> str:
        estado = "APROBADO" if self.aprobado else "RECHAZADO"
        return (
            f"[V1 · Verificación de integridad] {estado}\n"
            f"  - Filas: {self.n_filas} (esperado {N_REGISTROS_ESPERADOS})\n"
            f"  - Columnas esperadas presentes: {self.columnas_ok}\n"
            f"  - Sin valores nulos: {self.nulos_ok}\n"
            f"  - Tipos de equipo: {sorted(self.tipos_de_equipo)}\n"
            f"  - Ubicaciones: {sorted(self.ubicaciones)}\n"
            f"  - Prevalencia de falla: {self.prevalencia_falla:.4f} "
            f"(esperado {PREVALENCIA_ESPERADA} ± {TOLERANCIA_PREVALENCIA})\n"
        )


def _verificar_integridad(df: pd.DataFrame) -> ReporteIntegridad:
    """Construye el reporte de integridad (hito V1) a partir del DataFrame crudo."""

    columnas_ok = set(COLUMNAS_ESPERADAS).issubset(set(df.columns))
    nulos_ok = bool(df[list(COLUMNAS_ESPERADAS)].isnull().sum().sum() == 0) if columnas_ok else False

    tipos_de_equipo = set(df["equipment"].unique()) if "equipment" in df.columns else set()
    ubicaciones = set(df["location"].unique()) if "location" in df.columns else set()

    prevalencia = float(df["faulty"].mean()) if "faulty" in df.columns else float("nan")
    prevalencia_ok = abs(prevalencia - PREVALENCIA_ESPERADA) <= TOLERANCIA_PREVALENCIA

    return ReporteIntegridad(
        n_filas=len(df),
        n_columnas=df.shape[1],
        columnas_ok=columnas_ok,
        nulos_ok=nulos_ok,
        tipos_de_equipo=tipos_de_equipo,
        ubicaciones=ubicaciones,
        prevalencia_falla=prevalencia,
        prevalencia_ok=prevalencia_ok,
    )


def cargar_y_verificar_datos(ruta_csv: str | Path) -> tuple[pd.DataFrame, ReporteIntegridad]:
    """Carga el histórico y ejecuta la verificación de integridad (hito V1).

    Parameters
    ----------
    ruta_csv:
        Ruta al archivo `equipment_anomaly_data.csv`.

    Returns
    -------
    (df, reporte):
        El DataFrame crudo y el reporte de integridad. El DataFrame se
        retorna igualmente si la verificación falla, para permitir su
        inspección manual, pero el pipeline (main.py) debe detenerse si
        `reporte.aprobado` es False.
    """

    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_csv}")

    logger.info("Cargando histórico desde %s", ruta_csv)
    df = pd.read_csv(ruta_csv)

    # Normaliza la etiqueta a entero 0/1 (llega como float en el CSV original).
    if "faulty" in df.columns:
        df["faulty"] = df["faulty"].astype(int)

    reporte = _verificar_integridad(df)
    logger.info(reporte.resumen())

    return df, reporte


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df, reporte = cargar_y_verificar_datos("data/equipment_anomaly_data.csv")
    print(reporte.resumen())
    if not reporte.aprobado:
        raise SystemExit("V1 no aprobado: el histórico no pasó la verificación de integridad.")
