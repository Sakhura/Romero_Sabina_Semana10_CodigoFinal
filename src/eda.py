"""
eda.py
------
Análisis exploratorio de datos (EDA) y selección de variables predictoras
por análisis de correlación (WBS 1.5 y 1.6 de la Carta Gantt).

Implementa el Objetivo Específico 1 de la Tarea Sumativa 2:
    "Seleccionar las variables predictoras de la falla mediante un
    análisis de correlación, conservando las que tengan |r| >= 0,15
    con la etiqueta de falla."

Genera además las figuras que respaldan el diagrama de Ishikawa de la
Tarea Sumativa 1 (distribución por tipo de equipo, ubicación y la
matriz de correlación con la variable objetivo).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin GUI, apto para ejecución en servidor/CI
import matplotlib.pyplot as plt
import pandas as pd

try:
    from config import UMBRAL_CORRELACION, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS
except ImportError:  # pragma: no cover
    UMBRAL_CORRELACION = 0.15
    VARIABLES_NUMERICAS = ["temperature", "pressure", "vibration", "humidity"]
    VARIABLES_CATEGORICAS = ["equipment", "location"]

logger = logging.getLogger(__name__)


def calcular_correlaciones(df: pd.DataFrame) -> pd.Series:
    """Correlación de Pearson de cada variable numérica con la etiqueta `faulty`."""
    return df[VARIABLES_NUMERICAS + ["faulty"]].corr()["faulty"].drop("faulty").sort_values()


def seleccionar_predictoras(df: pd.DataFrame, umbral: float = UMBRAL_CORRELACION) -> list[str]:
    """Aplica el criterio |r| >= umbral (Objetivo Específico 1) sobre las variables numéricas.

    Las variables categóricas (equipment, location) se conservan siempre,
    ya que se codifican por separado en la Fase 2 (WBS 2.1) y sustentan
    los Objetivos Secundarios 1 y 3 de la Tarea Sumativa 1 (firma por
    tipo de activo y transferencia entre sitios).
    """
    correlaciones = calcular_correlaciones(df)
    seleccionadas_numericas = correlaciones[correlaciones.abs() >= umbral].index.tolist()

    logger.info("Correlación de cada variable con 'faulty':\n%s", correlaciones)
    logger.info(
        "Variables numéricas seleccionadas (|r| >= %.2f): %s",
        umbral,
        seleccionadas_numericas,
    )
    descartadas = [v for v in VARIABLES_NUMERICAS if v not in seleccionadas_numericas]
    if descartadas:
        logger.info("Variables numéricas descartadas por baja correlación: %s", descartadas)

    return seleccionadas_numericas + VARIABLES_CATEGORICAS


def prevalencia_por_sitio(df: pd.DataFrame) -> pd.Series:
    """Prevalencia de falla por ubicación (respalda la rama entorno/sitio del Ishikawa)."""
    return df.groupby("location")["faulty"].mean().sort_values()


def generar_figuras_eda(df: pd.DataFrame, carpeta_salida: str | Path) -> None:
    """Genera y guarda las figuras de apoyo del EDA en `carpeta_salida`."""
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # 1) Matriz de correlación con la etiqueta
    correlaciones = calcular_correlaciones(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    correlaciones.plot(kind="barh", ax=ax, color="#2f5f8f")
    ax.axvline(UMBRAL_CORRELACION, color="red", linestyle="--", linewidth=1, label=f"Umbral |r|={UMBRAL_CORRELACION}")
    ax.axvline(-UMBRAL_CORRELACION, color="red", linestyle="--", linewidth=1)
    ax.set_title("Correlación de cada variable con 'faulty'")
    ax.set_xlabel("Coeficiente de correlación (r)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta_salida / "correlacion_variables.png", dpi=150)
    plt.close(fig)

    # 2) Distribución de vibración por estado de falla (mayor separador, d≈1.44)
    fig, ax = plt.subplots(figsize=(6, 4))
    for estado, etiqueta, color in [(0, "Sano", "#4c9a2a"), (1, "Fallado", "#c0392b")]:
        subset = df.loc[df["faulty"] == estado, "vibration"]
        ax.hist(subset, bins=40, alpha=0.6, label=etiqueta, color=color, density=True)
    ax.set_title("Distribución de vibración por estado del equipo")
    ax.set_xlabel("Vibración")
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta_salida / "distribucion_vibracion.png", dpi=150)
    plt.close(fig)

    # 3) Prevalencia de falla por ubicación (efecto de emplazamiento)
    prevalencia = prevalencia_por_sitio(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    prevalencia.plot(kind="bar", ax=ax, color="#8e5ea2")
    ax.axhline(df["faulty"].mean(), color="black", linestyle="--", linewidth=1, label="Prevalencia global")
    ax.set_title("Prevalencia de falla por ubicación")
    ax.set_ylabel("Prevalencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta_salida / "prevalencia_por_ubicacion.png", dpi=150)
    plt.close(fig)

    logger.info("Figuras de EDA guardadas en %s", carpeta_salida)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from data_ingestion import cargar_y_verificar_datos

    df, _ = cargar_y_verificar_datos("data/equipment_anomaly_data.csv")
    predictoras = seleccionar_predictoras(df)
    print("Predictoras seleccionadas:", predictoras)
    generar_figuras_eda(df, "results/figuras")
