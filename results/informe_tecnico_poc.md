# Informe Técnico del PoC — Sistema de Monitoreo de Equipos Industriales

**Generado automáticamente el 2026-08-24 14:18 a partir de `results/metricas_prueba.json`.**
IDA300 · Semana 10 · Autor: Sabina Romero

## 1. Resumen ejecutivo

El sistema clasifica el riesgo de falla de equipos rotativos (turbinas, compresores
y bombas) a partir de temperatura, presión, vibración, tipo de equipo y ubicación,
usando un Random Forest entrenado sobre el *Industrial Equipment Monitoring
Dataset* (7.672 registros, prevalencia de falla del 10 %).

## 2. Resultados obtenidos (última ejecución)

| Métrica | Valor | Objetivo (Sumativa 2) | Cumple |
|---|---|---|---|
| F1-score (entrenamiento) | 1.0000 | ≥ 0,85 | ✅ |
| AUC-ROC (prueba) | 0.9746 | ≥ 0,90 | ✅ |
| IC95% AUC-ROC (bootstrap) | [0.9581, 0.9885] | — | — |
| F1-score (prueba) | 0.9120 | — | — |
| Precisión (prueba) | 0.9752 | — | — |
| Recall (prueba) | 0.8565 | — | — |

## 3. Variables utilizadas

Predictoras seleccionadas por el criterio |r| ≥ 0,15 (Objetivo Específico 1,
Tarea Sumativa 2): temperature, pressure, vibration, equipment, location.

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
