# PIM5 — Modelo de Riesgo Crediticio

Proyecto Integrador Módulo 5 (Henry, Data Science). El objetivo es construir y desplegar un modelo de machine learning que prediga si un cliente pagará a tiempo un crédito (`Pago_atiempo`: 1 = paga a tiempo, 0 = mora), y llevarlo a producción aplicando buenas prácticas de MLOps.

## Caso de negocio

Una entidad financiera necesita anticipar el riesgo de no pago de sus clientes de crédito, usando tanto información propia (monto, plazo, cuota, historial interno) como información del buró de crédito nacional DataCrédito (score, ingresos reportados, huella de consultas, créditos en otros sectores). Un modelo confiable permite decidir mejor a quién prestar y en qué condiciones, reduciendo pérdidas por cartera en mora.

## Estructura del repositorio

```
├── Base_de_datos.xlsx        # Dataset de créditos otorgados
├── requirements.txt
└── src/
    ├── carga_datos.py                  # Carga del dataset (v1.0.1)
    ├── comprension_eda.ipynb           # EDA: univariable, bivariable, multivariable (v1.0.1)
    ├── ft_engineering.py               # Pipeline de feature engineering (v1.1.0)
    ├── model_training_evaluation.py    # Entrenamiento y evaluación (v1.1.1)
    ├── modelo_riesgo_credito.pkl       # Pipeline + modelo + umbral serializados
    ├── curvas_evaluacion.png           # Curvas ROC y Precision-Recall del modelo elegido
    ├── model_deploy.py
    └── model_monitoring.py
```

## Ramas

- `developer`: desarrollo activo.
- `certification`: integración de avances cerrados.
- `master`: versión estable / entrega final.

## Estado del proyecto

**Avance 1 (en cierre):**
- Estructura de carpetas y entorno virtual configurados.
- `cargar_datos.py` funcional, lee `Base_de_datos.xlsx` desde una ruta relativa al proyecto.
- EDA completo en `comprension_eda.ipynb`, con hallazgos principales:
  - La variable objetivo (`Pago_atiempo`) está **desbalanceada** (~95% pagos a tiempo vs. ~5% en mora), lo que exige priorizar métricas como recall/F1/ROC-AUC sobre accuracy en el modelado.
  - `tendencia_ingresos` contiene valores fuera de las categorías esperadas (`Creciente`, `Decreciente`, `Estable`) — probable error de captura/exportación de la fuente original, tratado como dato faltante.

**Avance 2 (cerrado):**

### v1.1.0 — Ingeniería de características (`ft_engineering.py`)

Pipeline de scikit-learn ajustado **solo con train** y aplicado igual a train y test, para no filtrar información del set de prueba.

**Decisión principal: eliminación de variables con data leakage.**

Los dos pendientes del backlog del avance 1 se resolvieron, y ambos terminaron en el mismo lugar:

| Variable | Hallazgo | Decisión |
|---|---|---|
| `puntaje` | Correlación **0.923** con `Pago_atiempo`. Ninguna variable legítima de originación de crédito alcanza ese nivel: es un score calculado *después* de conocer el comportamiento de pago. | Eliminada |
| `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor` | Describen el estado de la deuda *durante* la vida del crédito, no al momento de otorgarlo. `saldo_mora > 0` aparece en 3.9% de los morosos vs. 0.34% de quienes pagaron: es consecuencia del impago, no causa. | Eliminadas |

El criterio fue: **el modelo solo puede usar información disponible en el momento real de la decisión de crédito.** Al aprobar un crédito nuevo, ninguna de estas variables existe todavía.

Para dejar constancia del costo de esa decisión, `model_training_evaluation.py` incluye un experimento de contraste que reentrena el mismo XGBoost devolviendo `puntaje` al set de features:

| Escenario | PR-AUC en test | ROC-AUC en test |
|---|---|---|
| Con `puntaje` (leakage) | **1.0000** | **1.0000** |
| Sin leakage (modelo real) | 0.1367 | 0.6760 |

Un modelo perfecto sobre datos de riesgo crediticio no es un buen modelo, es un síntoma. El resultado confirma que `puntaje` contiene el target, y que cualquier métrica obtenida con ella sería ficticia.

**Corrección de un bug de orden en el pipeline.** El `Winsorizer` corría *después* de `CrearRatiosFinancieros`, así que los ratios se calculaban con el salario sucio. Con `salario_cliente` llegando hasta 22.000 millones (errores de captura por ceros de más), esas filas obtenían ratios cercanos a cero que aparentaban riesgo bajísimo. Ahora la winsorización va antes.

**Otros ajustes de tratamiento de outliers:**
- Se cambió el capping de IQR 1.5 a **percentil 99 en la cola derecha**. Con IQR el tope de `salario_cliente` caía en ~9.2M y se recortaban 718 clientes (6.7%) con ingresos altos reales, cuando el problema son ~22 errores de captura.
- Se winsorizan también los ratios, porque la cola *izquierda* del salario también está sucia (24 clientes con salario 0) y producía ratios de hasta 840× el ingreso.

**Nueva variable derivada:** `ratio_ingresos_burodeclarado` (`promedio_ingresos_datacredito / salario_cliente`), que mide la discrepancia entre el ingreso que reporta DataCrédito y el que declara el cliente. Resultó ser la **4.ª variable más influyente** del modelo final.

**Otros cambios:** `tendencia_ingresos` pasó de un encoding ordinal arbitrario a un mapeo explícito que sí respeta el orden de negocio (Decreciente < Estable < Creciente); imputación diferenciada por tipo de variable (mediana + indicador de nulo donde el faltante es informativo, moda en la categórica).

### v1.1.1 — Modelamiento (`model_training_evaluation.py`)

**Definición de la clase positiva.** El target se invierte a `y_mora = 1 - Pago_atiempo`, de modo que la clase positiva sea el cliente que **no** paga. Así "recall" responde la pregunta de negocio real (*¿qué porcentaje de los morosos detectamos?*) en vez de medir qué tan bien identificamos a los buenos pagadores, que con 95% de la muestra es trivial.

**Métrica de selección: PR-AUC**, no accuracy ni ROC-AUC. Con 4.7% de morosos, un modelo que prediga "todos pagan" obtiene 95% de accuracy, y el ROC-AUC se vuelve optimista porque la tasa de falsos positivos se diluye contra 10.252 negativos.

Comparación con validación cruzada estratificada de 5 folds, solo sobre train:

| Modelo | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0.1492** | 0.6901 | 0.4302 | 0.1158 | 0.1823 |
| XGBoost | 0.1374 | 0.6653 | 0.2956 | 0.1307 | 0.1811 |
| Regresión Logística | 0.1176 | 0.6715 | 0.5940 | 0.0805 | 0.1417 |
| Árbol de Decisión | 0.0889 | 0.6188 | 0.5477 | 0.0714 | 0.1260 |
| Baseline (Dummy) | 0.0481 | 0.5030 | 0.0538 | 0.0530 | 0.0534 |

Todos los modelos manejan el desbalance explícitamente (`class_weight="balanced"`, o `scale_pos_weight` en XGBoost). El `DummyClassifier` no es un candidato: es la línea base contra la cual se verifica que los demás realmente aprenden algo.

**Modelo seleccionado: Random Forest**, con **PR-AUC 3.1× por encima del baseline**.

**Ajuste del umbral de decisión.** El corte por defecto en 0.5 no tiene nada de especial. El umbral se optimizó por F1 usando predicciones **fuera de fold** (`cross_val_predict`), no predicciones sobre train: Random Forest predice sus propios datos de entrenamiento casi perfecto, y un umbral ajustado sobre eso no sobrevive a datos nuevos. Umbral final: **0.5407** (F1 estimado 0.1907 fuera de fold, 0.1469 en test — sin salto artificial, señal de que la estimación era honesta).

**Resultado final en test** (2.153 registros, 102 morosos):

| Métrica | Valor |
|---|---|
| PR-AUC | 0.1367 |
| ROC-AUC | 0.6760 |
| Recall (morosos detectados) | 0.2549 |
| Precision | 0.1032 |
| F1 | 0.1469 |

Curvas ROC y Precision-Recall en `src/curvas_evaluacion.png`.

**Lectura honesta del resultado:** el modelo detecta 1 de cada 4 morosos y, de cada 10 clientes que marca como riesgosos, acierta en 1. Está claramente por encima del azar (PR-AUC 3.1× el baseline), pero no es un modelo listo para decidir aprobaciones por sí solo — sirve como señal de apoyo dentro de un proceso que incluya más criterios. Ese es el techo real que dan estos datos una vez retiradas las variables contaminadas, y es un resultado mucho más útil que un ROC-AUC de 0.99 que se derrumbaría en producción.

**Variables más influyentes:** `puntaje_datacredito`, `huella_consulta`, `edad_cliente`, `ratio_ingresos_burodeclarado`, `capital_prestado`. Los cuatro ratios creados en v1.1.0 aparecen en el top 10, lo que valida el trabajo de feature engineering.

**Backlog para los siguientes avances:**
- Tuning de hiperparámetros del Random Forest (`GridSearchCV` / `RandomizedSearchCV`), no explorado en este avance.
- Probar técnicas de remuestreo (SMOTE, undersampling) como alternativa al ponderado de clases.
- Definir con negocio el costo relativo de un falso negativo vs. un falso positivo, para reemplazar F1 por un criterio de umbral basado en costo real.

**Avances 3 y 4:** monitoreo de *data drift*, app en Streamlit, API con FastAPI y contenerización con Docker — pendientes.