"""
train_model.py
---------------
Preprocesamiento, partición, entrenamiento y validación del modelo de
clasificación de falla (Fases 2 y 3 de la Carta Gantt: WBS 2.1 a 3.2).

Implementa los Objetivos Específicos 2 y 3 de la Tarea Sumativa 2:
    OE2 - Entrenar un Random Forest con F1-score >= 0,85 (verificación
          en entrenamiento, hito H2).
    OE3 - Validar mediante partición hold-out estratificada 70/30 con
          AUC-ROC >= 0,90 en el conjunto de prueba (hito H3 / V3).

El desbalance de clases (prevalencia de falla ~10 %) se trata mediante
`class_weight="balanced"` en el propio Random Forest, siguiendo el
mismo espíritu de Chawla et al. (2002) referenciado en la Tarea
Sumativa 1, sin necesidad de sobremuestreo sintético para esta
prueba de concepto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from config import AUC_OBJETIVO, F1_OBJETIVO, N_ESTIMATORS, SEMILLA, TEST_SIZE
except ImportError:  # pragma: no cover
    SEMILLA = 42
    TEST_SIZE = 0.30
    F1_OBJETIVO = 0.85
    AUC_OBJETIVO = 0.90
    N_ESTIMATORS = 300

logger = logging.getLogger(__name__)


@dataclass
class ResultadoEntrenamiento:
    """Empaqueta el pipeline entrenado y los conjuntos usados, para reutilizar en evaluate.py."""

    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    f1_train: float
    metadata: dict = field(default_factory=dict)


def construir_pipeline(variables_categoricas: list[str]) -> Pipeline:
    """Construye el pipeline de preprocesamiento + Random Forest (WBS 2.1, 2.4)."""

    preprocesador = ColumnTransformer(
        transformers=[
            ("codificacion_categorica", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
        ],
        remainder="passthrough",  # deja pasar las variables numéricas tal cual
    )

    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",  # tratamiento del desbalance (10% de prevalencia)
        random_state=SEMILLA,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocesador", preprocesador), ("modelo", modelo)])


def particionar_datos(
    df: pd.DataFrame, predictoras: list[str], objetivo: str = "faulty"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición hold-out estratificada 70/30 (WBS 2.2), conservando la prevalencia (V2)."""

    X = df[predictoras]
    y = df[objetivo]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEMILLA)
    idx_train, idx_test = next(splitter.split(X, y))

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    prevalencia_train = y_train.mean()
    prevalencia_test = y_test.mean()
    logger.info(
        "[V2] Prevalencia train=%.4f | test=%.4f (original=%.4f)",
        prevalencia_train,
        prevalencia_test,
        y.mean(),
    )
    return X_train, X_test, y_train, y_test


def entrenar_modelo(
    df: pd.DataFrame, predictoras: list[str], variables_categoricas: list[str]
) -> ResultadoEntrenamiento:
    """Ejecuta partición + entrenamiento y verifica el hito H2 (F1 >= 0,85 en entrenamiento)."""

    X_train, X_test, y_train, y_test = particionar_datos(df, predictoras)

    pipeline = construir_pipeline(variables_categoricas)
    pipeline.fit(X_train, y_train)

    pred_train = pipeline.predict(X_train)
    f1_train = f1_score(y_train, pred_train)

    logger.info("[H2] F1-score en entrenamiento = %.4f (objetivo >= %.2f)", f1_train, F1_OBJETIVO)
    if f1_train < F1_OBJETIVO:
        logger.warning(
            "H2 no se cumple con los hiperparámetros actuales. "
            "Se sugiere ajustar el umbral de decisión o ponderar más la clase minoritaria "
            "antes de considerar cambiar de algoritmo (ver Carta Gantt, sección 8)."
        )

    return ResultadoEntrenamiento(
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        f1_train=f1_train,
        metadata={
            "predictoras": predictoras,
            "variables_categoricas": variables_categoricas,
            "semilla": SEMILLA,
            "test_size": TEST_SIZE,
        },
    )


def guardar_modelo(resultado: ResultadoEntrenamiento, ruta: str | Path) -> None:
    """Serializa el pipeline entrenado para su reutilización por el prototipo de tablero."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": resultado.pipeline, "metadata": resultado.metadata},
        ruta,
    )
    logger.info("Modelo guardado en %s", ruta)


def cargar_modelo(ruta: str | Path) -> tuple[Pipeline, dict]:
    """Carga un pipeline previamente entrenado (usado por dashboard_prototype.py)."""
    contenido = joblib.load(ruta)
    return contenido["pipeline"], contenido["metadata"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from data_ingestion import cargar_y_verificar_datos
    from eda import VARIABLES_CATEGORICAS, seleccionar_predictoras

    df, reporte = cargar_y_verificar_datos("data/equipment_anomaly_data.csv")
    if not reporte.aprobado:
        raise SystemExit("V1 no aprobado, no se puede continuar con el entrenamiento.")

    predictoras = seleccionar_predictoras(df)
    resultado = entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)
    guardar_modelo(resultado, "results/modelo_entrenado.joblib")
