# Sistema de Monitoreo de Equipos Industriales — Código Final (Semana 10)

**Asignatura:** IDA300 · **Autor:** Sabina Romero · **Profesor:** Marcelo Andrés Reyes Rogget
**Entrega:** Código Final y Demostración, Semana 10 (Sumativa)
**Base:** código de la Semana 9 (`Romero_Sabina_Semana9_PreEntregaCodigo`), optimizado y completado.
Ver `OPTIMIZACION.md` para el detalle de los cambios realizados esta semana
y `DEMOSTRACION.md` para el guion de la presentación en vivo.

## 1. Qué problema resuelve este código

El sistema clasifica el riesgo de falla de equipos rotativos (turbinas,
compresores y bombas) a partir de cuatro variables de proceso
(temperatura, presión, vibración, humedad), el tipo de equipo y la
ubicación, usando el *Industrial Equipment Monitoring Dataset*
(7.672 registros, prevalencia de falla del 10 %).

Es la implementación de la **Solución B — Clasificación con Machine
Learning en la nube**, seleccionada en la Tarea Sumativa 2 por obtener
el mayor puntaje en la matriz de decisión (3,95/5), y sigue el
cronograma y la topología de dependencias declarados en la Carta Gantt
de la Semana 7 (Evaluación Sumativa Nº 3).

Este entregable completa las **cinco fases** de esa Carta Gantt: ingesta
y EDA, entrenamiento del modelo, validación, integración con un
prototipo de tablero conectado al CMMS, y — nuevo en esta entrega — el
cierre del proyecto con informe técnico y auditoría de reproducibilidad
(Fase 5). Quedan fuera de alcance el despliegue productivo y la ingesta
en streaming desde la sensórica de planta, como se declaró desde la
Carta Gantt.

## 2. Cómo ejecutar el código

### Requisitos
- Python 3.10 o superior
- Las dependencias listadas en `requirements.txt`

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecución del pipeline completo

```bash
python main.py
```

Opciones disponibles (nuevas en el código final; ver `OPTIMIZACION.md` §3):

| Opción | Efecto |
|---|---|
| `--data RUTA.csv` | Usa un dataset distinto al de `data/equipment_anomaly_data.csv` |
| `--output CARPETA` | Cambia la carpeta donde se escriben los artefactos (por defecto `results/`) |
| `--skip-audit` | Omite la auditoría de reproducibilidad de la Fase 5 (pipeline corre una sola vez, más rápido para desarrollo) |
| `--quiet` | Reduce el nivel de logging a solo advertencias/errores |

Al ejecutar sin opciones, el pipeline corre las 5 fases y deja todos los
artefactos en `results/`:

| Artefacto | Descripción |
|---|---|
| `figuras/correlacion_variables.png` | Correlación de cada variable con `faulty` (Objetivo Específico 1) |
| `figuras/distribucion_vibracion.png` | Distribución de vibración por estado del equipo |
| `figuras/prevalencia_por_ubicacion.png` | Prevalencia de falla por sitio (efecto de emplazamiento) |
| `modelo_entrenado.joblib` | Pipeline de scikit-learn (preprocesamiento + Random Forest) serializado |
| `figuras/curva_roc.png` | Curva ROC del conjunto de prueba |
| `figuras/matriz_confusion.png` | Matriz de confusión del conjunto de prueba |
| `figuras/importancia_variables.png` | Importancia de variables por permutación |
| `metricas_prueba.json` | AUC-ROC, F1, precisión, recall e IC95 % bootstrap |
| `tablero.html` | Prototipo de tablero de riesgo por equipo (abrir en cualquier navegador) |
| `ordenes_trabajo_cmms.json` | Órdenes de trabajo generadas por la regla de riesgo (conector CMMS simulado) |
| `informe_tecnico_poc.md` | Informe técnico generado automáticamente desde las métricas reales (Fase 5) |
| `evidencia_auditoria_reproducibilidad.txt` | Certificación de reproducibilidad al 4º decimal (hito V4) |

### Ejecución de la suite de pruebas

```bash
python -m pytest tests/ -v
```

40 pruebas: 18 de integración entre fases (`test_integration.py`),
12 unitarias de funciones puras (`test_unit.py`), y 10 de rendimiento y
de distintos escenarios de datos (`test_performance.py`). Ver
`results/evidencia_pruebas_pytest_final.txt` para la salida de la última
ejecución completa.

## 3. Estructura del proyecto

```
proyecto/
├── main.py                        # Orquestador: ejecuta las 5 fases end-to-end
├── requirements.txt
├── README.md
├── OPTIMIZACION.md                 # Informe de optimización y ajustes (Semana 10)
├── DEMOSTRACION.md                 # Guion para la presentación en vivo
├── data/
│   └── equipment_anomaly_data.csv
├── src/
│   ├── config.py                   # Configuración centralizada (nuevo, Semana 10)
│   ├── data_ingestion.py           # Fase 1a: carga y verificación de integridad (V1)
│   ├── eda.py                      # Fase 1b: EDA y selección de predictoras (|r| >= 0,15)
│   ├── train_model.py              # Fase 2: partición 70/30, pipeline y Random Forest (H2)
│   ├── evaluate.py                 # Fase 3: métricas de prueba, bootstrap vectorizado, importancia (H3)
│   ├── cmms_connector.py           # Fase 4a: regla de riesgo y conector CMMS simulado
│   ├── dashboard_prototype.py      # Fase 4b: prototipo de tablero HTML (H4)
│   ├── generar_informe_tecnico.py  # Fase 5a: informe técnico automático (nuevo, WBS 5.1)
│   └── reproducibility_audit.py    # Fase 5b: auditoría de reproducibilidad (nuevo, WBS 5.2, V4)
├── tests/
│   ├── conftest.py
│   ├── test_integration.py         # Pruebas de integración de las 5 fases
│   ├── test_unit.py                # Pruebas unitarias de funciones puras (nuevo)
│   └── test_performance.py         # Pruebas de rendimiento y distintos escenarios (nuevo)
└── results/                        # Se genera al ejecutar main.py (no versionado)
```

## 4. Cómo interactúan los módulos

```
data_ingestion.py            eda.py
   │  cargar_y_verificar_datos()  │  seleccionar_predictoras()
   └──────────────┬───────────────┘
                   ▼
            train_model.py
   entrenar_modelo(df, predictoras, categoricas)
   -> particiona 70/30 (V2) -> entrena Random Forest -> hito H2
                   │
                   ▼
             evaluate.py
   evaluar_modelo(pipeline, X_test, y_test)
   -> AUC-ROC, F1, bootstrap IC95% vectorizado, importancia -> hito H3
                   │
                   ▼
      dashboard_prototype.py  ◄──────────┐
   calcular_riesgo_por_equipo()          │
   -> genera tablero.html                │
                   │                     │
                   ▼                     │
          cmms_connector.py ─────────────┘
   generar_ordenes_trabajo() + sincronizar_ordenes_con_cmms()
   -> ordenes_trabajo_cmms.json -> hito H4
                   │
                   ▼
   generar_informe_tecnico.py     reproducibility_audit.py
   -> informe_tecnico_poc.md      -> re-ejecuta main.ejecutar_pipeline()
      (lee metricas_prueba.json)     dos veces y certifica 4 decimales -> V4, H5
```

Todos los módulos leen sus constantes desde `src/config.py` en lugar de
declararlas por separado (ver `OPTIMIZACION.md` §2). `main.py` importa y
encadena las funciones públicas anteriores en `ejecutar_pipeline()` (Fases
1-4) y `ejecutar_fase5_cierre()` (Fase 5), deteniendo la ejecución si el
histórico no pasa la verificación de integridad (hito de compuerta H1).

## 5. Pruebas finales y resultados

### 5.1 Pruebas de integración (`test_integration.py`, 18 pruebas)
Verifican que las 4 primeras fases se integran correctamente: sin fuga de
datos entre train/test, conservación de la prevalencia, cumplimiento de
los hitos H1-H4, generación de todos los artefactos.

### 5.2 Pruebas unitarias (`test_unit.py`, 12 pruebas — nuevo)
Casos de borde a nivel de función: los umbrales exactos de la regla de
riesgo (0,70 y 0,40), la selección de predictoras sobre variables
sintéticas de correlación conocida, y la tolerancia de comparación de la
auditoría de reproducibilidad.

### 5.3 Pruebas de rendimiento y distintos escenarios (`test_performance.py`, 10 pruebas — nuevo)
- El pipeline completo corre en menos de 30 s sobre el dataset real.
- El bootstrap vectorizado da el mismo resultado que la versión con bucle,
  en menos tiempo (benchmark automatizado).
- El sistema se probó con **cuatro escenarios de datos distintos**: una
  muestra reducida (1.500 filas), un dataset restringido a un solo sitio,
  un dataset con la prevalencia de falla artificialmente alterada, y un
  dataset con una columna faltante (para verificar que el hito de
  compuerta H1 lo rechaza de forma controlada, no con un error confuso).

### 5.4 Auditoría de reproducibilidad (Fase 5, hito V4)
`python main.py` ejecuta automáticamente el pipeline dos veces más al
final y certifica que el AUC-ROC y el F1 coinciden al cuarto decimal. Ver
`results/evidencia_auditoria_reproducibilidad.txt`.

## 6. Resultados obtenidos (última ejecución)

Ver `results/metricas_prueba.json` e `results/informe_tecnico_poc.md`
para los valores exactos y actualizados. Como referencia, con la semilla
fija 42:

- **F1-score en entrenamiento:** ≈ 1,00 (objetivo Sumativa 2: ≥ 0,85) → **H2 cumplido**
- **AUC-ROC en prueba:** ≈ 0,97, IC95 % ≈ [0,96, 0,99] (objetivo: ≥ 0,90) → **H3 cumplido**
- **Reproducibilidad:** AUC-ROC y F1 idénticos al 6º decimal entre dos
  ejecuciones independientes → **V4 certificada, H5 (cierre) cumplido**
- **Tiempo del pipeline completo (con auditoría):** ≈ 15 s en un equipo
  estándar; ≈ 7-8 s sin la auditoría (`--skip-audit`)

La variable con mayor poder discriminante es la vibración (consistente
con `d ≈ 1,44` reportado en la Tarea Sumativa 1), seguida de presión y
temperatura; `humidity` se descarta por baja correlación (r ≈ 0,01).

## 7. Limitaciones y alcance declarado

- El conector al CMMS es **simulado** (escribe un archivo JSON local):
  no existe acceso a un CMMS real para este PoC.
- El tablero es un **archivo HTML estático**, no una aplicación web con
  backend.
- Quedan fuera de esta entrega: la ingesta en streaming desde la
  sensórica de planta, el despliegue productivo, la predicción conformal
  con abstención y la validación *leave-one-site-out* descritas como
  objetivos secundarios en la Tarea Sumativa 1; se proponen como
  evolución futura en `DEMOSTRACION.md` §7.
# Romero_Sabina_Semana10_CodigoFinal
