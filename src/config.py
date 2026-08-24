"""
config.py
---------
Configuración centralizada del sistema.

OPTIMIZACIÓN (Semana 10): en la pre-entrega de la Semana 9, cada módulo
(`train_model.py`, `evaluate.py`, `cmms_connector.py`, `data_ingestion.py`)
declaraba sus propias constantes (semilla, umbrales, rutas). Duplicar la
semilla o el umbral de AUC-ROC en cuatro archivos distintos es una fuente
de errores: si alguien actualiza el objetivo de F1 en un solo lugar, los
demás quedan desincronizados sin que ningún test lo detecte.

Este módulo centraliza esas constantes. Los demás módulos las importan
desde aquí; ver `OPTIMIZACION.md` para el detalle de este cambio y su
justificación.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS_DEFECTO = RAIZ_PROYECTO / "data" / "equipment_anomaly_data.csv"
CARPETA_RESULTADOS_DEFECTO = RAIZ_PROYECTO / "results"

# ---------------------------------------------------------------------
# Reproducibilidad (Carta Gantt: semilla fija 42 en todas las fases)
# ---------------------------------------------------------------------
SEMILLA = 42

# ---------------------------------------------------------------------
# Fase 1 · Ingesta y EDA
# ---------------------------------------------------------------------
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

UMBRAL_CORRELACION = 0.15
VARIABLES_NUMERICAS = ["temperature", "pressure", "vibration", "humidity"]
VARIABLES_CATEGORICAS = ["equipment", "location"]

# ---------------------------------------------------------------------
# Fase 2 · Entrenamiento
# ---------------------------------------------------------------------
TEST_SIZE = 0.30
F1_OBJETIVO = 0.85
N_ESTIMATORS = 300

# ---------------------------------------------------------------------
# Fase 3 · Validación
# ---------------------------------------------------------------------
AUC_OBJETIVO = 0.90
N_BOOTSTRAP = 800

# ---------------------------------------------------------------------
# Fase 4 · Riesgo / CMMS
# ---------------------------------------------------------------------
UMBRAL_RIESGO_ALTO = 0.70
UMBRAL_RIESGO_MEDIO = 0.40

# ---------------------------------------------------------------------
# Fase 5 · Cierre y reproducibilidad (V4: 4 decimales, WBS 5.2)
# ---------------------------------------------------------------------
DECIMALES_REPRODUCIBILIDAD = 4
