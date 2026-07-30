# PIM5 — Modelo de Riesgo Crediticio

Proyecto Integrador Módulo 5 (Henry, Data Science). El objetivo es construir y desplegar un modelo de machine learning que prediga si un cliente pagará a tiempo un crédito (`Pago_atiempo`: 1 = paga a tiempo, 0 = mora), y llevarlo a producción aplicando buenas prácticas de MLOps.

## Caso de negocio

Una entidad financiera necesita anticipar el riesgo de no pago de sus clientes de crédito, usando tanto información propia (monto, plazo, cuota, historial interno) como información del buró de crédito nacional DataCrédito (score, ingresos reportados, huella de consultas, créditos en otros sectores). Un modelo confiable permite decidir mejor a quién prestar y en qué condiciones, reduciendo pérdidas por cartera en mora.

## Estructura del repositorio

```
├── Base_de_datos.xlsx          # Dataset de créditos otorgados
├── requirements.txt            # Dependencias de desarrollo (versiones fijadas)
├── requirements-docker.txt     # Dependencias de runtime para la imagen
├── Dockerfile                  # Imagen única para API y app (v1.3.0)
├── docker-compose.yml           # Orquestación de los dos servicios (v1.3.0)
├── .dockerignore
├── .github/workflows/ci.yml    # CI: build + verificación de los servicios (v1.3.1)
└── src/
    ├── cargar_datos.py                 # Carga del dataset (v1.0.1)
    ├── comprension_eda.ipynb           # EDA: univariable, bivariable, multivariable (v1.0.1)
    ├── ft_engineering.py               # Pipeline de feature engineering (v1.1.0)
    ├── model_training_evaluation.py    # Entrenamiento y evaluación (v1.1.1)
    ├── model_deploy.py                 # API REST con FastAPI (v1.2.0)
    ├── model_monitoring.py             # Monitoreo y data drift (v1.2.1)
    ├── app_streamlit.py                # App web: scoring + dashboard de drift (v1.2.2)
    ├── modelo_riesgo_credito.pkl       # Pipeline + modelo + umbral serializados
    ├── curvas_evaluacion.png           # Curvas ROC y Precision-Recall del modelo elegido
    ├── predicciones_historicas.csv     # Datos + pronósticos (insumo del monitoreo)
    ├── metricas_drift.csv              # Métricas de drift por variable
    ├── drift_temporal.csv              # Evolución del drift por periodo
    └── ejemplo_lote.csv                # Plantilla para la carga masiva de la app
```

## Ramas

- `developer`: desarrollo activo.
- `certification`: integración de avances cerrados.
- `main`: versión estable / entrega final.

## Cómo correr el proyecto

### Con Docker (recomendado)

Un solo comando levanta la API y la app:

```bash
docker compose up --build
```

- API con documentación interactiva: **http://localhost:8000/docs**
- App web: **http://localhost:8501**

Streamlit espera automáticamente a que el modelo termine de cargar, así que la app está lista cuando responde. Para apagar:

```bash
docker compose down
```

El log de predicciones vive en un volumen y sobrevive al `down`. Para borrarlo también: `docker compose down -v`.

### En local, sin Docker

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Los scripts se ejecutan **desde `src/`**, porque el artefacto serializado necesita que `ft_engineering` sea importable (ver el avance 3):

```bash
cd src

python ft_engineering.py                # pipeline + chequeos de sanidad
python model_training_evaluation.py     # entrena, evalúa y guarda el modelo
python model_monitoring.py              # reporte de data drift

uvicorn model_deploy:app --reload        # API + doc en http://localhost:8000/docs
streamlit run app_streamlit.py          # app web (requiere la API arriba)
```

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

**Avance 3 (cerrado):**

### Paso previo — arreglo de bloqueadores

El proyecto no corría. `ft_engineering.py` y `model_training_evaluation.py` importaban `from Mlops_Pipeline.src.cargar_datos import cargarDatos`, que falla con `ModuleNotFoundError` porque no existe ningún `__init__.py` ni el directorio padre está en `sys.path`. Se volvió al import plano.

Quedó documentada además una restricción que condiciona el despliegue: el `.pkl` guarda los transformadores personalizados como una **referencia al módulo** `ft_engineering`, no su código. Por eso `joblib.load` solo funciona si ese módulo es importable, y tanto la API como el monitoreo insertan su propio directorio en `sys.path` al arrancar. Al contenerizar (avance 4) esto se traduce en `PYTHONPATH=/app/src`.

### v1.2.0 — API con FastAPI (`src/model_deploy.py`)

El modelo se carga **una sola vez** al arranque vía el evento `lifespan`, no por request.

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/salud` | Healthcheck |
| GET | `/modelo` | Metadata: umbral, métricas de test, variables excluidas por leakage |
| POST | `/predecir` | Una solicitud → probabilidad, decisión y banda de riesgo |
| POST | `/predecir-lote` | Scoring masivo en una sola llamada |

**El contrato pide 17 variables, no 22.** Es consecuencia directa del avance 2: las 5 variables con data leakage no existen cuando hay que aprobar un crédito nuevo, así que pedirlas sería incoherente. La API solo acepta información disponible en el momento real de la decisión.

**La respuesta devuelve el umbral explícitamente.** La decisión se toma en 0.5407, no en 0.5, y quien consuma la API necesita poder auditar con qué criterio se marcó a un cliente. `puntaje_datacredito` y `promedio_ingresos_datacredito` son opcionales, porque su ausencia es un caso legítimo que el pipeline ya trata como señal de riesgo propia.

Cada predicción se registra en `src/registro_predicciones.csv`, que es el insumo del monitoreo.

### v1.2.1 — Monitoreo y data drift (`src/model_monitoring.py`)

Compara la población de **referencia** (nov 2024 – jun 2025, 8.378 créditos) contra la **actual** (jul 2025 – abr 2026, 2.385 créditos), con corte en `2025-07-01`.

**Hallazgo principal: hay drift real, no hubo que simular nada.**

| Variable | PSI | KS | Jensen-Shannon | Estado |
|---|---|---|---|---|
| `promedio_ingresos_datacredito` | 0.3272 | 0.2258 | 0.0562 | 🔴 severo |
| `total_otros_prestamos` | 0.1819 | 0.1401 | 0.0313 | 🟡 moderado |
| `plazo_meses` | 0.1129 | 0.1620 | 0.2249 | 🟡 moderado |
| `cuota_pactada` | 0.0779 | 0.1243 | 0.1131 | 🟢 estable |
| `capital_prestado` | 0.0526 | 0.0846 | 0.0834 | 🟢 estable |

Semáforo global: **1 severo, 2 moderados, 13 estables.** Umbrales estándar de PSI: `<0.10` verde, `0.10–0.25` amarillo, `>0.25` rojo.

**Por qué el KS no basta.** Cinco variables más dan p-valor < 0.05 pero PSI por debajo de 0.05. Con 8.378 registros de referencia, las pruebas de hipótesis detectan como significativas diferencias demasiado pequeñas para afectar al modelo. El monitoreo reporta ambas métricas y solo alerta cuando el PSI —que mide magnitud, no significancia— lo justifica.

**La tasa de mora cayó de 5.19% a 3.19%, y eso no es una buena noticia.** Significa que la población que el modelo está evaluando ya no es la que aprendió. Un modelo entrenado sobre clientes con 5.19% de mora aplicado a clientes con 3.19% sobreestima el riesgo de forma sistemática, que es exactamente lo que se observa: la mora predicha es de 10–14% por mes contra 3–6% real, una brecha promedio de **8.4 puntos**. Parte viene de `class_weight="balanced"`, que a cambio de detectar más morosos infla la predicción de la clase minoritaria.

**Dos correcciones metodológicas que hicieron falta:**

1. **El PSI se disparaba por muestras chicas, no por drift.** Con un épsilon fijo de `1e-6` para los bins vacíos, enero de 2026 (89 valores no nulos) daba PSI 3.27, de los cuales **3.08 venían de solo dos bins vacíos** — un falso positivo severo. Se reemplazó por suavizado de Laplace, `(conteo + 0.5) / (n + 0.5·bins)`, cuya corrección escala con el tamaño de muestra. El mismo mes pasó a 0.975, y las métricas globales apenas se movieron (0.3298 → 0.3272), que es la señal de que el arreglo solo toca lo que debía tocar. Se añadió además un mínimo de 50 observaciones no nulas por variable y periodo.

2. **La detección de tendencias era inútil y las alertas eran ruido.** Exigir que el PSI creciera en *cada* periodo es un criterio que ninguna serie real cumple; se cambió por correlación de Spearman ≥ 0.7. Y al implementar la detección de saltos abruptos, el sistema pasó a emitir 14 alertas, varias sobre variables con PSI de 0.06 y una marcando como "cambio abrupto" una *caída* de 0.434 a 0.187, que es una mejora. Se filtró a variables que cruzan el umbral moderado y a saltos solo hacia arriba: quedaron 11 alertas accionables. Un tablero que alarma por buenas noticias enseña al usuario a ignorarlo.

**Análisis temporal.** Cada periodo se compara contra la referencia fija, no contra el periodo anterior: comparando con el anterior, una deriva lenta y sostenida pasaría desapercibida porque cada mes se parece a su vecino mientras la población se aleja del modelo. Se distingue tendencia (deriva gradual → reentrenar) de salto abrupto (probable cambio operativo o dato roto → revisar la fuente antes de reentrenar).

### v1.2.2 — App Streamlit (`src/app_streamlit.py`)

Dos pantallas, consumiendo la API por HTTP con la URL en la variable de entorno `API_URL`.

**Scoring de solicitudes:** formulario con las 17 variables y carga masiva por CSV (`src/ejemplo_lote.csv` sirve de plantilla). El resultado se presenta como banda de riesgo con la probabilidad.

**Monitoreo de drift:** semáforo agregado, tabla de métricas con barras de riesgo, histogramas de referencia vs. actual, evolución del PSI con las bandas de alerta, comparación de mora observada vs. predicha, y las recomendaciones automáticas.

La app consume la API en vez de cargar el `.pkl` directamente para que exista **una sola implementación** de la lógica de scoring. Si la app cargara el modelo por su cuenta habría dos caminos de inferencia que mantener sincronizados.

`app_streamlit.py` no forma parte de la estructura de carpetas definida por el enunciado, pero Streamlit necesita su propio punto de entrada. No reemplaza a ninguno de los archivos exigidos.

**Bug encontrado al probar la carga masiva:** `df.where(pd.notna(df), None)` no convierte NaN en None sobre columnas float — pandas lo vuelve a coercer a NaN, que no es JSON válido, y el envío falla. Habría roto la carga de cualquier CSV con datos del buró faltantes, o sea el 27% de las filas reales. Se corrigió con `df.astype(object).where(...)`.

**Backlog del avance 3:**
- Reentrenar con datos recientes, que es lo que el propio monitoreo recomienda.
- Recalibrar la probabilidad (`CalibratedClassifierCV`) para cerrar la brecha de 8.4 puntos entre mora predicha y observada.
- Apuntar el monitoreo a `registro_predicciones.csv` cuando haya tráfico real, en vez de al histórico scoreado.

**Avance 4 (cerrado):**

### v1.3.0 — Contenerización

**Una sola imagen para los dos servicios.** La API y la app comparten código, dependencias y modelo; lo único que cambia es el comando de arranque. Mantener dos imágenes obligaría a construir dos veces las mismas librerías.

| Servicio | Puerto | Comando |
|---|---|---|
| `api` | 8000 | `uvicorn model_deploy:app --host 0.0.0.0 --port 8000` |
| `app` | 8501 | `streamlit run app_streamlit.py --server.address=0.0.0.0` |

**La imagen base está fijada al patch: `python:3.14.6-slim`.** No es exceso de celo. `modelo_riesgo_credito.pkl` se serializó con Python 3.14.6 y scikit-learn 1.9.0, y un pickle de sklearn cargado con otra versión emite `InconsistentVersionWarning` o falla al reconstruir los estimadores. Por la misma razón se fijaron **todas** las versiones en `requirements.txt`, que hasta este avance no tenía ninguna: dos instalaciones hechas en fechas distintas podían producir resultados distintos contra el mismo artefacto.

**`PYTHONPATH=/app/src` en vez de depender del directorio de trabajo.** El pickle guarda los transformadores personalizados como una *referencia al módulo* `ft_engineering`, no su código, así que `joblib.load` necesita poder importarlo. Hasta ahora eso obligaba a ejecutar todo desde `src/`; en el contenedor queda resuelto de forma explícita.

**`requirements-docker.txt` separado.** El runtime excluye `xgboost` y `matplotlib` — verificado que solo los importa `model_training_evaluation.py`, que no corre en el contenedor porque el modelo llega ya entrenado — más jupyter, notebook, ipykernel, ipywidgets, seaborn y db-dtypes. Son las dependencias más pesadas del proyecto y ninguna hace falta para servir.

**`.dockerignore` obligatorio.** El contexto de build sin filtrar pesa **1.6 GB, de los cuales `venv/` es prácticamente todo**. Docker lo copiaría íntegro al daemon en cada build.

Imagen final: **1.2 GB**.

**Bug encontrado al verificar el volumen.** El log de predicciones no se escribía: `[Errno 13] Permission denied: '/app/registro/registro_predicciones.csv'`. La causa es que Docker inicializa un volumen nombrado copiando permisos del directorio que exista **en la imagen**, y si no existe lo crea como `root` — pero el contenedor corre como `appuser` por seguridad. Se resolvió creando `/app/registro` en el Dockerfile con el dueño correcto antes de cambiar de usuario.

El fallo era silencioso: `registrar_predicciones` captura las excepciones de escritura a propósito, para que un problema en el log no tumbe una predicción ya calculada. Buen diseño defensivo, pero significa que sin revisar los logs el problema no se nota.

**`depends_on` con `condition: service_healthy`.** Sin la condición, Streamlit arranca antes de que el modelo termine de cargar y la barra lateral muestra "API no disponible" en el primer render, obligando a recargar. El healthcheck apunta a `/salud`, que confirma que el modelo quedó en memoria y no solo que el proceso vive.

### v1.3.1 — CI/CD con GitHub Actions

`.github/workflows/ci.yml` corre en cada push a `developer`, `certification` y `main`.

**No se limita a comprobar que la imagen compila.** Un build puede construirse perfectamente y devolver 500 al predecir — exactamente lo que pasaría si una versión de scikit-learn dejara de ser compatible con el pickle. El workflow levanta los servicios y verifica:

- El healthcheck de la API y `GET /salud` con `modelo_cargado: true`.
- `GET /modelo` devuelve el umbral.
- **`POST /predecir` contra un valor conocido:** la solicitud de `.github/workflows/solicitud_prueba.json` debe dar `probabilidad_mora: 0.657709`. Si cambia, alguna dependencia dejó de ser compatible con el modelo serializado.
- Que los campos del buró siguen siendo opcionales.
- Que la validación rechaza datos inválidos (`edad_cliente: 9` → 422).
- Que Streamlit responde en `/_stcore/health`.
- Que la app alcanza la API por su **nombre de servicio** dentro de la red de compose.
- Que no aparece ningún `InconsistentVersionWarning` en los logs.

### Verificación realizada

Sobre el stack levantado con `docker compose up`:

| Comprobación | Resultado |
|---|---|
| Ambos servicios | `healthy` |
| `POST /predecir` | `0.657709` — **idéntico a fuera del contenedor** |
| Avisos de versión de sklearn | 0 |
| Log de predicciones en el volumen | Escribe, y **persiste tras `down` + `up`** |
| Usuario del contenedor | `uid=1000(appuser)`, sin privilegios |
| App → API por nombre de servicio | `http://api:8000` responde |
| Ambas pantallas de Streamlit | Renderizan dentro del contenedor |
| `model_monitoring.py` en el contenedor | 1 rojo, 2 amarillo, 13 verde |

Que la probabilidad sea idéntica dentro y fuera del contenedor es el criterio que de verdad prueba que el fijado de versiones funcionó.

**Backlog del avance 4:**
- Publicar la imagen en un registro (GHCR o Docker Hub) desde el workflow, con tags por versión.
- Reducir la imagen con un build multi-etapa; buena parte del 1.2 GB son las librerías científicas, difíciles de recortar más sin sacrificar funcionalidad.
- Extender el CI a los pendientes del avance 3: reentrenamiento por el drift severo y recalibración de la probabilidad.