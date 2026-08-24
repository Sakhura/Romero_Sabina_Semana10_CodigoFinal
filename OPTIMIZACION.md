# Informe de Optimización y Ajustes Realizados

**IDA300 · Semana 10 · Código Final y Demostración**
Autor: Sabina Romero · Base: código entregado en la Semana 9
(`Romero_Sabina_Semana9_PreEntregaCodigo`)

Este informe describe, de forma breve y verificable, los cambios realizados
sobre el código de la pre-entrega de la Semana 9 para llegar al código final:
qué se optimizó, qué se agregó para completar las funcionalidades planificadas
en la Carta Gantt, y qué ajustes mejoraron el rendimiento o la robustez del
sistema.

## 1. Optimización de rendimiento: bootstrap del AUC-ROC vectorizado

**Problema.** `evaluate.py` calculaba el intervalo de confianza al 95 % del
AUC-ROC con 800 remuestreos bootstrap, llamando a `sklearn.roc_auc_score`
una vez por remuestreo dentro de un bucle Python. Cada llamada de sklearn
incluye validaciones de entrada que no aportan nada cuando se repiten 800
veces sobre datos ya validados.

**Cambio.** Se reemplazó el bucle por una versión vectorizada: las 800
muestras bootstrap se generan de una sola vez como una matriz `(800, n)`, y
el AUC de cada fila se calcula con el equivalente por rangos del estadístico
U de Mann-Whitney (`AUC = (suma_rangos_positivos − n_pos·(n_pos+1)/2) /
(n_pos·n_neg)`), usando `np.argsort` vectorizado en vez de 800 llamadas
independientes.

**Verificación de correctitud.** Se comparó el resultado exacto (percentiles
2.5 y 97.5 del AUC) de ambos métodos sobre los mismos datos: la diferencia
es del orden de `1e-16` (error de punto flotante), es decir, no hay pérdida
de precisión. Esta comparación está automatizada en
`tests/test_performance.py::TestRendimiento::test_bootstrap_vectorizado_es_mas_rapido_que_800_llamadas_individuales`.

**Medición.** Sobre un conjunto de prueba del tamaño real (~2.300 filas):

| Método | Tiempo |
|---|---|
| Bucle con `sklearn.roc_auc_score` (Semana 9) | ≈ 1.50 s |
| Vectorizado con NumPy (Semana 10) | ≈ 0.19 – 0.32 s |
| **Speedup** | **≈ 5x – 8x** |

Código de referencia: `src/evaluate.py`, función `_bootstrap_auc_ic`.

## 2. Configuración centralizada (`src/config.py`)

**Problema.** En la Semana 9, la semilla (`42`), los umbrales (`F1_OBJETIVO`,
`AUC_OBJETIVO`, `UMBRAL_CORRELACION`, `UMBRAL_RIESGO_ALTO/MEDIO`) y las rutas
por defecto estaban declarados de forma independiente en cuatro archivos
distintos (`train_model.py`, `evaluate.py`, `cmms_connector.py`,
`data_ingestion.py`). Esto es frágil: si se actualiza un objetivo del
proyecto en un solo lugar, los demás módulos quedan desincronizados sin que
ningún test lo detecte necesariamente a tiempo.

**Cambio.** Se creó `src/config.py` con todas las constantes del sistema en
un solo lugar, documentadas y agrupadas por fase de la Carta Gantt. Cada
módulo las importa desde `config` (con un `try/except ImportError` que
conserva el valor local como respaldo, para que cada módulo siga siendo
ejecutable de forma aislada durante el desarrollo).

## 3. Interfaz de línea de comandos en `main.py`

**Problema.** La Semana 9 solo permitía `python main.py` con el dataset y la
carpeta de salida fijados en el código. Para las pruebas finales de la
Semana 10 (que exigen ejecutar el sistema "con diferentes sets de datos")
era necesario poder variar la entrada sin editar el código fuente.

**Cambio.** Se agregó `argparse` con las opciones `--data`, `--output`,
`--skip-audit` y `--quiet`. Esto permitió, por ejemplo, ejecutar el pipeline
completo sobre un dataset reordenado (mismo tamaño, distinto orden de filas)
sin tocar el código, y comprobar que el sistema falla de forma controlada
—no con un traceback confuso— cuando el dataset no cumple el contrato de
integridad esperado (ver sección 5, "Pruebas Finales y Resultados", del
`README.md`).

## 4. Fase 5 completada: cierre y reproducibilidad

**Contexto.** La Carta Gantt de la Semana 7 planificó cinco fases para el
PoC. La pre-entrega de la Semana 9 cubrió explícitamente las Fases 1 a 4
("Quedan fuera del alcance la ingesta en streaming... y el despliegue
productivo", pero la Fase 5 — cierre, informe y auditoría de
reproducibilidad — sí estaba dentro del alcance planificado y quedaba
pendiente para esta entrega final.

**Cambio.** Se agregaron dos módulos nuevos:

- `src/generar_informe_tecnico.py` (WBS 5.1): genera
  `results/informe_tecnico_poc.md` leyendo las métricas reales de la última
  ejecución (`metricas_prueba.json`), en vez de redactar un informe estático
  que pudiera quedar desactualizado respecto al código.
- `src/reproducibility_audit.py` (WBS 5.2, hito V4): vuelve a ejecutar el
  pipeline completo dos veces con la misma semilla (42) y certifica que el
  AUC-ROC y el F1 se reproducen al cuarto decimal, tal como exige la Carta
  Gantt ("las métricas se reproducen al cuarto decimal"). El resultado
  queda en `results/evidencia_auditoria_reproducibilidad.txt`.

`main.py` ahora ejecuta esta Fase 5 automáticamente después del pipeline
principal (puede omitirse con `--skip-audit` para iteraciones rápidas
durante el desarrollo).

## 5. Pruebas adicionales

Se agregaron dos archivos de pruebas nuevos junto a los de integración de
la Semana 9:

- `tests/test_unit.py`: pruebas unitarias de funciones puras (reglas de
  riesgo en los umbrales exactos, selección de predictoras con variables
  sintéticas de correlación conocida, tolerancia de la auditoría de
  reproducibilidad).
- `tests/test_performance.py`: benchmark de tiempo de ejecución del
  pipeline completo, verificación del speedup del bootstrap vectorizado, y
  cuatro escenarios de datos distintos (muestra reducida, un solo sitio,
  prevalencia de falla alterada, dataset con columnas faltantes).

En total, la suite pasó de 18 pruebas (Semana 9) a **40 pruebas** (Semana
10), todas en verde (`results/evidencia_pruebas_pytest_final.txt`).

## 6. Resumen de impacto

| Aspecto | Semana 9 | Semana 10 |
|---|---|---|
| Cálculo del IC95% del AUC-ROC | ~1.5 s (bucle) | ~0.2–0.3 s (vectorizado) |
| Constantes del sistema | Duplicadas en 4 archivos | Centralizadas en `config.py` |
| Ejecución con distintos datasets | No soportada (rutas fijas en el código) | `--data` / `--output` por CLI |
| Fases de la Carta Gantt cubiertas | 1–4 | 1–5 (cierre y reproducibilidad) |
| Pruebas automatizadas | 18 | 40 |
| Auditoría de reproducibilidad (V4) | No implementada | Certificada al 4º decimal |
