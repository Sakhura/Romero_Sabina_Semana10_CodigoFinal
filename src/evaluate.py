"""
evaluate.py
-----------
Evaluación del modelo sobre el conjunto de prueba hold-out (Fase 3,
WBS 3.1 a 3.4 de la Carta Gantt).

Implementa el Objetivo General de la Tarea Sumativa 2
    "... AUC-ROC >= 0,90 en un conjunto de prueba independiente ..."
y el reporte cuantitativo pedido por el Objetivo Principal de la
Tarea Sumativa 1 (intervalo de confianza al 95 % vía bootstrap,
matriz de confusión, importancia de variables por permutación).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from config import AUC_OBJETIVO, N_BOOTSTRAP, SEMILLA
except ImportError:  # pragma: no cover
    AUC_OBJETIVO = 0.90
    N_BOOTSTRAP = 800
    SEMILLA = 42

logger = logging.getLogger(__name__)


@dataclass
class MetricasPrueba:
    auc_roc: float
    auc_roc_ic95_bajo: float
    auc_roc_ic95_alto: float
    f1_score: float
    precision: float
    recall: float
    umbral_objetivo_auc: float
    cumple_objetivo_auc: bool

    def resumen(self) -> str:
        estado = "CUMPLE" if self.cumple_objetivo_auc else "NO CUMPLE"
        return (
            f"[H3/V3 · Evaluación en prueba] AUC-ROC objetivo = {self.umbral_objetivo_auc} -> {estado}\n"
            f"  - AUC-ROC       = {self.auc_roc:.4f}  (IC95% bootstrap: "
            f"[{self.auc_roc_ic95_bajo:.4f}, {self.auc_roc_ic95_alto:.4f}])\n"
            f"  - F1-score      = {self.f1_score:.4f}\n"
            f"  - Precisión     = {self.precision:.4f}\n"
            f"  - Recall        = {self.recall:.4f}\n"
        )


def _bootstrap_auc_ic(y_true: np.ndarray, y_proba: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Intervalo de confianza al 95% del AUC-ROC mediante bootstrap (Tarea Sumativa 1, sección 3.1).

    OPTIMIZACIÓN (Semana 10): la versión de la Semana 9 llamaba a
    `sklearn.roc_auc_score` una vez por remuestreo (800 llamadas Python,
    cada una con sus propias validaciones internas de sklearn). Esta
    versión genera las 800 muestras bootstrap de una sola vez como una
    matriz (n_boot x n) y calcula el AUC de todas ellas con operaciones
    vectorizadas de NumPy (equivalente por rangos al estadístico U de
    Mann-Whitney: AUC = (suma_rangos_positivos - n_pos(n_pos+1)/2) /
    (n_pos·n_neg)). Ver `OPTIMIZACION.md` para el benchmark antes/después
    (~5-8x más rápido en este dataset) y la verificación de que ambos
    métodos producen el mismo resultado.
    """
    rng = np.random.RandomState(SEMILLA)
    n = len(y_true)

    idx = rng.randint(0, n, size=(n_boot, n))          # (n_boot, n) de una sola vez
    y_boot = y_true[idx]                                 # (n_boot, n)
    proba_boot = y_proba[idx]                             # (n_boot, n)

    # Rango (1..n) de cada score dentro de su propia fila (remuestreo), vectorizado.
    rangos = np.argsort(np.argsort(proba_boot, axis=1), axis=1) + 1

    n_pos = y_boot.sum(axis=1)
    n_neg = n - n_pos
    filas_validas = (n_pos > 0) & (n_neg > 0)  # descarta remuestreos sin ambas clases

    suma_rangos_pos = np.sum(np.where(y_boot == 1, rangos, 0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        aucs = (suma_rangos_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    aucs = aucs[filas_validas]

    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluar_modelo(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> MetricasPrueba:
    """Calcula AUC-ROC, F1, precisión, recall e IC95% bootstrap sobre el conjunto de prueba."""

    y_test_arr = y_test.to_numpy()
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test_arr, proba)
    ic_bajo, ic_alto = _bootstrap_auc_ic(y_test_arr, proba)

    metricas = MetricasPrueba(
        auc_roc=float(auc),
        auc_roc_ic95_bajo=ic_bajo,
        auc_roc_ic95_alto=ic_alto,
        f1_score=float(f1_score(y_test_arr, pred)),
        precision=float(precision_score(y_test_arr, pred)),
        recall=float(recall_score(y_test_arr, pred)),
        umbral_objetivo_auc=AUC_OBJETIVO,
        cumple_objetivo_auc=bool(auc >= AUC_OBJETIVO),
    )
    logger.info(metricas.resumen())
    return metricas


def importancia_de_variables(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 10
) -> pd.Series:
    """Importancia por permutación sobre las variables originales (Objetivo Secundario 1, Sumativa 1)."""
    resultado = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=SEMILLA, scoring="roc_auc", n_jobs=-1
    )
    importancias = pd.Series(resultado.importances_mean, index=X_test.columns).sort_values(ascending=False)
    logger.info("Importancia de variables (permutación, delta AUC-ROC):\n%s", importancias)
    return importancias


def desempeno_por_tipo_de_equipo(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """AUC-ROC del modelo global, desglosado por tipo de equipo (Objetivo Secundario 1)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    df_eval = X_test.copy()
    df_eval["y_true"] = y_test.to_numpy()
    df_eval["y_proba"] = proba

    filas = []
    for equipo, grupo in df_eval.groupby("equipment"):
        if grupo["y_true"].nunique() < 2:
            continue
        filas.append(
            {
                "equipment": equipo,
                "n": len(grupo),
                "prevalencia": grupo["y_true"].mean(),
                "auc_roc": roc_auc_score(grupo["y_true"], grupo["y_proba"]),
            }
        )
    return pd.DataFrame(filas).sort_values("auc_roc", ascending=False)


def generar_figuras_evaluacion(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, carpeta_salida: str | Path
) -> None:
    """Guarda la curva ROC, la matriz de confusión y el gráfico de importancia de variables."""
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    pred = pipeline.predict(X_test)

    # Curva ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("Curva ROC - conjunto de prueba")
    fig.tight_layout()
    fig.savefig(carpeta_salida / "curva_roc.png", dpi=150)
    plt.close(fig)

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Sano", "Fallado"], ax=ax, cmap="Blues")
    ax.set_title("Matriz de confusión - conjunto de prueba")
    fig.tight_layout()
    fig.savefig(carpeta_salida / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    # Importancia de variables
    importancias = importancia_de_variables(pipeline, X_test, y_test)
    fig, ax = plt.subplots(figsize=(6, 4))
    importancias.plot(kind="barh", ax=ax, color="#2f8f5f")
    ax.invert_yaxis()
    ax.set_title("Importancia de variables (permutación, ΔAUC-ROC)")
    fig.tight_layout()
    fig.savefig(carpeta_salida / "importancia_variables.png", dpi=150)
    plt.close(fig)

    logger.info("Figuras de evaluación guardadas en %s", carpeta_salida)


def guardar_metricas_json(metricas: MetricasPrueba, ruta: str | Path, extra: dict | None = None) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido = asdict(metricas)
    if extra:
        contenido.update(extra)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)
    logger.info("Métricas guardadas en %s", ruta)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from data_ingestion import cargar_y_verificar_datos
    from eda import VARIABLES_CATEGORICAS, seleccionar_predictoras
    from train_model import entrenar_modelo

    df, _ = cargar_y_verificar_datos("data/equipment_anomaly_data.csv")
    predictoras = seleccionar_predictoras(df)
    resultado = entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)
    metricas = evaluar_modelo(resultado.pipeline, resultado.X_test, resultado.y_test)
    generar_figuras_evaluacion(resultado.pipeline, resultado.X_test, resultado.y_test, "results/figuras")
    guardar_metricas_json(metricas, "results/metricas_prueba.json")
