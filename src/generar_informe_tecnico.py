"""
generar_informe_tecnico.py
----------------------------
Genera el informe técnico del PoC (Fase 5, WBS 5.1 de la Carta Gantt:
"Informe técnico del PoC, figuras y limitaciones declaradas") a partir
de los resultados reales de la última ejecución del pipeline, en lugar
de redactarlo a mano. Esto garantiza que las cifras del informe nunca
queden desactualizadas respecto del código: si el modelo cambia, el
informe se regenera con `python src/generar_informe_tecnico.py` y
refleja las métricas actuales.

Lee:
    results/metricas_prueba.json  (Fase 3, evaluate.py)
Escribe:
    results/informe_tecnico_poc.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PLANTILLA = """# Informe Técnico del PoC — Sistema de Monitoreo de Equipos Industriales

**Generado automáticamente el {fecha} a partir de `results/metricas_prueba.json`.**
IDA300 · Semana 10 · Autor: Sabina Romero

## 1. Resumen ejecutivo

El sistema clasifica el riesgo de falla de equipos rotativos (turbinas, compresores
y bombas) a partir de temperatura, presión, vibración, tipo de equipo y ubicación,
usando un Random Forest entrenado sobre el *Industrial Equipment Monitoring
Dataset* (7.672 registros, prevalencia de falla del 10 %).

## 2. Resultados obtenidos (última ejecución)

| Métrica | Valor | Objetivo (Sumativa 2) | Cumple |
|---|---|---|---|
| F1-score (entrenamiento) | {f1_entrenamiento:.4f} | ≥ 0,85 | {check_f1} |
| AUC-ROC (prueba) | {auc_roc:.4f} | ≥ 0,90 | {check_auc} |
| IC95% AUC-ROC (bootstrap) | [{auc_ic_bajo:.4f}, {auc_ic_alto:.4f}] | — | — |
| F1-score (prueba) | {f1_prueba:.4f} | — | — |
| Precisión (prueba) | {precision:.4f} | — | — |
| Recall (prueba) | {recall:.4f} | — | — |

## 3. Variables utilizadas

Predictoras seleccionadas por el criterio |r| ≥ 0,15 (Objetivo Específico 1,
Tarea Sumativa 2): {predictoras}.

`humidity` fue descartada por baja correlación con la falla (r ≈ 0,01),
consistente con el diagrama de Ishikawa de la Tarea Sumativa 1.

## 4. Figuras generadas

- `figuras/correlacion_variables.png` — correlación de cada variable con `faulty`
- `figuras/distribucion_vibracion.png` — distribución de vibración por estado
- `figuras/prevalencia_por_ubicacion.png` — prevalencia de falla por sitio
- `figuras/curva_roc.png` — curva ROC del conjunto de prueba
- `figuras/matriz_confusion.png` — matriz de confusión del conjunto de prueba
- `figuras/importancia_variables.png` — importancia por permutación

## 5. Limitaciones declaradas

- El conector al CMMS es simulado (no existe acceso a un CMMS real para este PoC).
- El tablero es un archivo HTML estático, no una aplicación con backend.
- Fuera de alcance: ingesta en streaming, despliegue productivo, predicción
  conformal con abstención, y validación *leave-one-site-out* (Objetivos
  Secundarios 2 y 3 de la Tarea Sumativa 1) — quedan como evolución futura.
- El modelo es específico para el corte transversal del dataset: no estima
  vida útil remanente (RUL), solo probabilidad instantánea de falla.

## 6. Cierre del proyecto (Fase 5)

Ver `evidencia_auditoria_reproducibilidad.txt` para la certificación V4
(reproducibilidad al cuarto decimal, hito de cierre H5).
"""


def generar_informe(
    ruta_metricas: str | Path = "results/metricas_prueba.json",
    ruta_salida: str | Path = "results/informe_tecnico_poc.md",
) -> Path:
    ruta_metricas = Path(ruta_metricas)
    with open(ruta_metricas, encoding="utf-8") as f:
        metricas = json.load(f)

    contenido = _PLANTILLA.format(
        fecha=datetime.now().strftime("%Y-%m-%d %H:%M"),
        f1_entrenamiento=metricas["f1_entrenamiento"],
        auc_roc=metricas["auc_roc"],
        check_f1="✅" if metricas["f1_entrenamiento"] >= 0.85 else "❌",
        check_auc="✅" if metricas["cumple_objetivo_auc"] else "❌",
        auc_ic_bajo=metricas["auc_roc_ic95_bajo"],
        auc_ic_alto=metricas["auc_roc_ic95_alto"],
        f1_prueba=metricas["f1_score"],
        precision=metricas["precision"],
        recall=metricas["recall"],
        predictoras=", ".join(metricas["predictoras"]),
    )

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(contenido, encoding="utf-8")
    logger.info("Informe técnico generado en %s", ruta_salida)
    return ruta_salida


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generar_informe()
