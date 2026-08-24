"""
cmms_connector.py
------------------
Conector al CMMS (Computerized Maintenance Management System), WBS 4.1.

Para esta prueba de concepto (alcance declarado en la Carta Gantt: "un
prototipo de tablero conectado al CMMS") no existe un CMMS real
disponible, de modo que este módulo simula la interfaz de creación de
órdenes de trabajo mediante un archivo JSON. La función pública
`crear_orden_trabajo()` es la que se reemplazaría por una llamada real
a la API del CMMS (por ejemplo REST o SOAP) en el despliegue
productivo (mes 4-5 del programa original, fuera de alcance del PoC).

Regla de orden de trabajo (WBS 4.1):
    Se genera una orden de trabajo automática para todo equipo cuya
    probabilidad de falla estimada por el modelo supere el umbral
    `UMBRAL_RIESGO_ALTO`. Los equipos con riesgo medio quedan en
    observación; los de riesgo bajo no generan acción.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from config import UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO
except ImportError:  # pragma: no cover
    UMBRAL_RIESGO_ALTO = 0.70
    UMBRAL_RIESGO_MEDIO = 0.40

logger = logging.getLogger(__name__)


@dataclass
class OrdenTrabajo:
    id_equipo: int
    tipo_equipo: str
    ubicacion: str
    probabilidad_falla: float
    nivel_riesgo: str
    fecha_creacion: str
    prioridad: str


def clasificar_riesgo(probabilidad: float) -> str:
    if probabilidad >= UMBRAL_RIESGO_ALTO:
        return "ALTO"
    if probabilidad >= UMBRAL_RIESGO_MEDIO:
        return "MEDIO"
    return "BAJO"


def prioridad_desde_riesgo(nivel_riesgo: str) -> str:
    return {"ALTO": "URGENTE", "MEDIO": "PROGRAMADA", "BAJO": "SIN ACCIÓN"}[nivel_riesgo]


def generar_ordenes_trabajo(df_riesgo: pd.DataFrame) -> list[OrdenTrabajo]:
    """Aplica la regla de orden de trabajo a un DataFrame con columna 'probabilidad_falla'.

    `df_riesgo` debe tener las columnas: equipment, location, probabilidad_falla,
    e idealmente un índice o columna identificadora del activo.
    """
    ordenes = []
    ahora = datetime.now(timezone.utc).isoformat()

    for idx, fila in df_riesgo.iterrows():
        nivel = clasificar_riesgo(fila["probabilidad_falla"])
        if nivel == "BAJO":
            continue  # sin acción, no se crea orden
        orden = OrdenTrabajo(
            id_equipo=int(idx),
            tipo_equipo=fila["equipment"],
            ubicacion=fila["location"],
            probabilidad_falla=round(float(fila["probabilidad_falla"]), 4),
            nivel_riesgo=nivel,
            fecha_creacion=ahora,
            prioridad=prioridad_desde_riesgo(nivel),
        )
        ordenes.append(orden)

    logger.info(
        "Órdenes de trabajo generadas: %d (ALTO=%d, MEDIO=%d)",
        len(ordenes),
        sum(1 for o in ordenes if o.nivel_riesgo == "ALTO"),
        sum(1 for o in ordenes if o.nivel_riesgo == "MEDIO"),
    )
    return ordenes


def crear_orden_trabajo(orden: OrdenTrabajo, ruta_salida: str | Path) -> None:
    """Simula la llamada a la API del CMMS: aquí, anexa la orden a un archivo JSON local.

    En un entorno productivo esta función se reemplaza por el request HTTP/SOAP
    real al CMMS (WBS 4.1: 'SW-03 - Licencia de tablero y conector al CMMS').
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    ordenes_existentes = []
    if ruta_salida.exists():
        with open(ruta_salida, "r", encoding="utf-8") as f:
            ordenes_existentes = json.load(f)

    ordenes_existentes.append(asdict(orden))
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(ordenes_existentes, f, indent=2, ensure_ascii=False)


def sincronizar_ordenes_con_cmms(ordenes: list[OrdenTrabajo], ruta_salida: str | Path) -> None:
    """Punto de entrada usado por el prototipo de tablero para volcar todas las órdenes."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump([asdict(o) for o in ordenes], f, indent=2, ensure_ascii=False)
    logger.info("Órdenes de trabajo sincronizadas con el CMMS (simulado) en %s", ruta_salida)
