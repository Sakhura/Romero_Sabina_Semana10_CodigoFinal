"""
dashboard_prototype.py
-----------------------
Prototipo de tablero de riesgo por equipo (WBS 4.1) que integra:
    1. Las probabilidades de falla estimadas por el modelo entrenado
       (train_model.py / evaluate.py).
    2. La regla de orden de trabajo del conector CMMS (cmms_connector.py).

Genera un archivo HTML estático y autocontenido (results/tablero.html)
que puede abrirse en cualquier navegador sin backend, siguiendo el
alcance del PoC declarado en la Carta Gantt (mes 4: "prototipo de
tablero conectado al CMMS", sin la integración productiva con la
sensórica en streaming).

Este módulo es el punto de integración final del sistema: consume la
salida de todos los módulos anteriores (ingesta -> EDA -> entrenamiento
-> evaluación) y produce un artefacto único que demuestra que el
pipeline completo funciona de punta a punta.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from cmms_connector import clasificar_riesgo, generar_ordenes_trabajo, sincronizar_ordenes_con_cmms

logger = logging.getLogger(__name__)

_FILA_HTML = """
<tr class="riesgo-{nivel_css}">
  <td>{id_equipo}</td>
  <td>{tipo_equipo}</td>
  <td>{ubicacion}</td>
  <td>{probabilidad:.1%}</td>
  <td><span class="badge badge-{nivel_css}">{nivel_riesgo}</span></td>
  <td>{prioridad}</td>
</tr>
"""

_PLANTILLA_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Tablero de Riesgo · Sistema de Monitoreo de Equipos Industriales</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 2rem; color: #1f2d3d; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .subtitulo {{ color: #5b6b7a; margin-top: 0; }}
  .tarjetas {{ display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }}
  .tarjeta {{ background: white; border-radius: 8px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 160px; }}
  .tarjeta .valor {{ font-size: 1.8rem; font-weight: 700; }}
  .tarjeta .etiqueta {{ color: #5b6b7a; font-size: 0.85rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 0.6rem 0.9rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #22364f; color: white; position: sticky; top: 0; }}
  .badge {{ padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.78rem; font-weight: 600; color: white; }}
  .badge-alto {{ background: #c0392b; }}
  .badge-medio {{ background: #d68910; }}
  .badge-bajo {{ background: #4c9a2a; }}
  .riesgo-alto {{ background: #fdecea; }}
  .riesgo-medio {{ background: #fef5e6; }}
  .metricas {{ color: #5b6b7a; font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
  <h1>Tablero de Riesgo por Equipo</h1>
  <p class="subtitulo">Sistema de Monitoreo de Equipos Industriales · IDA300 · Prototipo integrado con el CMMS (simulado)</p>

  <div class="tarjetas">
    <div class="tarjeta"><div class="valor">{n_total}</div><div class="etiqueta">Equipos evaluados</div></div>
    <div class="tarjeta"><div class="valor">{n_alto}</div><div class="etiqueta">Riesgo ALTO (orden urgente)</div></div>
    <div class="tarjeta"><div class="valor">{n_medio}</div><div class="etiqueta">Riesgo MEDIO (programada)</div></div>
    <div class="tarjeta"><div class="valor">{auc_roc:.3f}</div><div class="etiqueta">AUC-ROC (conjunto de prueba)</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID equipo</th><th>Tipo</th><th>Ubicación</th><th>Prob. de falla</th><th>Riesgo</th><th>Orden de trabajo</th>
      </tr>
    </thead>
    <tbody>
      {filas}
    </tbody>
  </table>

  <p class="metricas">
    Umbral de riesgo alto: probabilidad &ge; {umbral_alto:.0%} · Umbral de riesgo medio: probabilidad &ge; {umbral_medio:.0%}.
    Mostrando los {n_mostrados} equipos de mayor riesgo del conjunto de prueba ({n_total} en total).
  </p>
</body>
</html>
"""


def calcular_riesgo_por_equipo(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Calcula la probabilidad de falla de cada equipo y la adjunta al DataFrame original."""
    df_riesgo = X.copy()
    df_riesgo["probabilidad_falla"] = pipeline.predict_proba(X)[:, 1]
    df_riesgo["nivel_riesgo"] = df_riesgo["probabilidad_falla"].apply(clasificar_riesgo)
    return df_riesgo.sort_values("probabilidad_falla", ascending=False)


def generar_tablero_html(
    df_riesgo: pd.DataFrame, auc_roc: float, ruta_salida: str | Path, top_n: int = 100
) -> None:
    """Renderiza el tablero HTML autocontenido con las N filas de mayor riesgo."""

    from cmms_connector import UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO, prioridad_desde_riesgo

    top = df_riesgo.head(top_n)
    filas_html = "".join(
        _FILA_HTML.format(
            id_equipo=idx,
            tipo_equipo=fila["equipment"],
            ubicacion=fila["location"],
            probabilidad=fila["probabilidad_falla"],
            nivel_riesgo=fila["nivel_riesgo"],
            nivel_css=fila["nivel_riesgo"].lower(),
            prioridad=prioridad_desde_riesgo(fila["nivel_riesgo"]),
        )
        for idx, fila in top.iterrows()
    )

    html = _PLANTILLA_HTML.format(
        n_total=len(df_riesgo),
        n_alto=int((df_riesgo["nivel_riesgo"] == "ALTO").sum()),
        n_medio=int((df_riesgo["nivel_riesgo"] == "MEDIO").sum()),
        auc_roc=auc_roc,
        filas=filas_html,
        umbral_alto=UMBRAL_RIESGO_ALTO,
        umbral_medio=UMBRAL_RIESGO_MEDIO,
        n_mostrados=len(top),
    )

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(html, encoding="utf-8")
    logger.info("Tablero HTML generado en %s", ruta_salida)


def ejecutar_prototipo_tablero(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    auc_roc: float,
    ruta_tablero: str | Path,
    ruta_ordenes_cmms: str | Path,
) -> pd.DataFrame:
    """Orquesta el prototipo completo de la Fase 4: riesgo -> órdenes CMMS -> tablero HTML."""

    df_riesgo = calcular_riesgo_por_equipo(pipeline, X_test)
    ordenes = generar_ordenes_trabajo(df_riesgo)
    sincronizar_ordenes_con_cmms(ordenes, ruta_ordenes_cmms)
    generar_tablero_html(df_riesgo, auc_roc, ruta_tablero)
    return df_riesgo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from data_ingestion import cargar_y_verificar_datos
    from eda import VARIABLES_CATEGORICAS, seleccionar_predictoras
    from evaluate import evaluar_modelo
    from train_model import entrenar_modelo

    df, _ = cargar_y_verificar_datos("data/equipment_anomaly_data.csv")
    predictoras = seleccionar_predictoras(df)
    resultado = entrenar_modelo(df, predictoras, VARIABLES_CATEGORICAS)
    metricas = evaluar_modelo(resultado.pipeline, resultado.X_test, resultado.y_test)

    ejecutar_prototipo_tablero(
        resultado.pipeline,
        resultado.X_test,
        metricas.auc_roc,
        "results/tablero.html",
        "results/ordenes_trabajo_cmms.json",
    )
