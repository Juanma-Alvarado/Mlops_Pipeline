"""
model_monitoring.py

Trabajo de monitoreo y detección de *data drift* del modelo de riesgo
crediticio (PIM5).

Correr:
    cd src
    python model_monitoring.py

QUÉ PROBLEMA RESUELVE
---------------------
Un modelo se entrena con una foto de la población de clientes. Si esa
población cambia —otro perfil de solicitante, otra mezcla de productos, otra
política comercial— el modelo sigue respondiendo con la misma confianza pero
sus probabilidades dejan de ser válidas, y nadie se entera hasta que la
cartera se deteriora. El monitoreo compara la población actual contra la de
referencia y avisa antes de que eso pase.

QUÉ ENCONTRAMOS EN ESTOS DATOS
------------------------------
El dataset ya tiene drift temporal real, no hubo que simular nada. Partiendo
en julio de 2025, 7 de 13 variables numéricas cambian de distribución de
forma significativa, y la tasa de mora cae de 5.19% a 3.19%. Esa caída no es
una buena noticia: significa que el modelo está evaluando clientes distintos
a los que aprendió.

CÓMO SE OBTIENE LA TABLA DE DATOS + PRONÓSTICOS
-----------------------------------------------
El enunciado pide monitorear una tabla con los datos junto con los pronósticos
entregados. En producción esa tabla la alimenta la API (`model_deploy.py`
escribe cada predicción en `registro_predicciones.csv`), pero mientras no haya
tráfico real se construye scoreando todo el histórico con el modelo en
producción: `predicciones_historicas.csv`. Es el mismo formato, así que el
monitoreo funciona igual cuando se le apunte al registro en vivo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

DIRECTORIO_SRC = Path(__file__).resolve().parent
if str(DIRECTORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_SRC))

import joblib

from cargar_datos import cargarDatos
from ft_engineering import CATEGORIAS_VALIDAS_TENDENCIA, COLUMNAS_LEAKAGE, TARGET

RUTA_ARTEFACTO = DIRECTORIO_SRC / "modelo_riesgo_credito.pkl"
RUTA_PREDICCIONES = DIRECTORIO_SRC / "predicciones_historicas.csv"
RUTA_METRICAS = DIRECTORIO_SRC / "metricas_drift.csv"
RUTA_DRIFT_TEMPORAL = DIRECTORIO_SRC / "drift_temporal.csv"

# Fecha que separa la población de referencia (con la que se validó el modelo)
# de la población actual (la que hay que vigilar).
FECHA_CORTE = "2025-07-01"

# Mínimo de registros para que un periodo entre al análisis temporal.
#
# No es un detalle cosmético: el volumen mensual se desploma de 1.917 créditos
# en enero de 2025 a 11 en abril de 2026. Con muestras de 11 registros, el KS
# y el PSI dan valores altísimos por puro ruido de muestreo, no por drift real.
# Sin este filtro el dashboard reportaría una crisis inexistente en la cola.
MIN_REGISTROS_PERIODO = 100

# Mínimo de observaciones NO NULAS para que el PSI de una variable sea creíble.
#
# El filtro anterior cuenta filas del periodo, pero una variable puede tener
# muchos nulos dentro de esas filas: `promedio_ingresos_datacredito` tiene ~27%
# de nulos, así que un mes de 127 créditos deja solo 89 valores para comparar
# contra 10 bins. Por debajo de este mínimo se reporta NaN en vez de un número
# que parece preciso y no lo es.
MIN_MUESTRA_PSI = 50

# Umbrales estándar de la industria para interpretar el PSI.
UMBRAL_PSI_MODERADO = 0.10
UMBRAL_PSI_CRITICO = 0.25

# Nivel de significancia para las pruebas de hipótesis (KS y chi-cuadrado).
ALFA = 0.05

# Variables categóricas. `tipo_credito` son códigos de producto: aunque sea
# entero, comparar sus percentiles no significa nada, hay que comparar la
# mezcla de categorías.
VARIABLES_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

VARIABLES_NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

# Etiqueta para los valores de tendencia_ingresos que no son una categoría de
# negocio válida (el dataset trae montos filtrados en esa columna). Se agrupan
# como una categoría propia en vez de descartarlos: si la proporción de basura
# cambia, eso es drift de calidad de dato y también hay que verlo.
ETIQUETA_SIN_DATO = "sin_dato"


# Construcción de la tabla de datos + pronósticos
def generar_tabla_predicciones(forzar: bool = False) -> pd.DataFrame:
    """
    Scorea todo el histórico con el modelo en producción y persiste la tabla
    de datos + pronósticos que consume el monitoreo.

    Se reutiliza el `pipeline_ft` serializado, no se reajusta: las medianas y
    los límites de winsorización deben ser los del entrenamiento, porque son
    justamente el punto de referencia contra el cual queremos medir el cambio.
    """
    if RUTA_PREDICCIONES.exists() and not forzar:
        return pd.read_csv(RUTA_PREDICCIONES, parse_dates=["fecha_prestamo"])

    artefacto = joblib.load(RUTA_ARTEFACTO)
    pipeline, modelo, umbral = (
        artefacto["pipeline_ft"],
        artefacto["modelo"],
        artefacto["umbral"],
    )

    df = cargarDatos()
    X = df.drop(columns=[TARGET])

    probabilidades = modelo.predict_proba(pipeline.transform(X))[:, 1]

    tabla = X.drop(columns=[c for c in COLUMNAS_LEAKAGE if c in X.columns]).copy()
    tabla["probabilidad_mora"] = probabilidades
    tabla["prediccion"] = (probabilidades >= umbral).astype(int)
    # `Pago_atiempo` es 1 = paga; la clase que el modelo predice es la mora.
    tabla["mora_real"] = 1 - df[TARGET]

    tabla.to_csv(RUTA_PREDICCIONES, index=False)
    print(f"Tabla de predicciones generada: {RUTA_PREDICCIONES.name} "
          f"({len(tabla)} registros)")
    return tabla


def normalizar_tendencia(serie: pd.Series) -> pd.Series:
    """Agrupa los valores inválidos de tendencia_ingresos en una categoría."""
    return serie.where(serie.isin(CATEGORIAS_VALIDAS_TENDENCIA), ETIQUETA_SIN_DATO)


# Métricas de drift
def proporciones_suavizadas(conteos: np.ndarray, alfa: float = 0.5) -> np.ndarray:
    """
    Convierte conteos por bin en proporciones con suavizado de Laplace.

    `(conteo + alfa) / (n + alfa * n_bins)` nunca da 0, así que el logaritmo
    del PSI está siempre definido, y la corrección que recibe un bin vacío es
    proporcional al tamaño de la muestra en vez de una constante arbitraria.
    """
    total = conteos.sum()
    return (conteos + alfa) / (total + alfa * len(conteos))


def calcular_psi(referencia: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """
    Population Stability Index entre dos distribuciones numéricas.

    Los bins se construyen con los **deciles de la referencia**, no con cortes
    de ancho fijo ni con los de la muestra actual. Es lo correcto: el PSI mide
    cuánto se movió la población respecto a la que el modelo conoce, así que
    la partición tiene que venir de esa población. Si los bins se recalcularan
    con los datos actuales, cada muestra se compararía contra una regla
    distinta y los valores no serían comparables entre periodos.

    Los bins vacíos se corrigen con suavizado de Laplace y no con un épsilon
    fijo. La diferencia importa mucho: con un épsilon de 1e-6, un solo bin
    vacío aporta `(0 - 0.15) * log(1e-6 / 0.15)` ≈ 1.7 al PSI, así que en
    muestras chicas el índice se dispara por falta de datos y no por drift.
    Midiendo enero de 2026 (89 valores no nulos) el PSI daba 3.27, de los
    cuales 3.08 venían de dos bins vacíos: un falso positivo severo.

    Con suavizado de Laplace la corrección escala con el tamaño de muestra
    (`(conteo + 0.5) / (n + 0.5 * n_bins)`), así que un bin vacío pesa en
    proporción a cuántos datos había realmente para llenarlo.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    if len(referencia) == 0 or len(actual) < MIN_MUESTRA_PSI:
        return np.nan

    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))

    # Variables muy concentradas (p. ej. un sector con 90% de ceros) pueden
    # tener menos deciles distintos que bins pedidos; con menos de 2 cortes no
    # hay partición posible.
    if len(cortes) < 3:
        return 0.0

    cortes[0], cortes[-1] = -np.inf, np.inf

    conteo_ref = np.histogram(referencia, bins=cortes)[0]
    conteo_act = np.histogram(actual, bins=cortes)[0]

    prop_ref = proporciones_suavizadas(conteo_ref)
    prop_act = proporciones_suavizadas(conteo_act)

    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))


def calcular_psi_categorico(referencia: pd.Series, actual: pd.Series) -> float:
    """
    PSI para variables categóricas: compara la proporción de cada categoría.

    Se usa la unión de categorías de ambas muestras, para que una categoría
    que aparece solo en la población actual (un producto nuevo, por ejemplo)
    cuente como drift en vez de pasar desapercibida.
    """
    categorias = sorted(set(referencia.dropna()) | set(actual.dropna()), key=str)

    if not categorias:
        return np.nan

    conteo_ref = referencia.value_counts().reindex(categorias, fill_value=0)
    conteo_act = actual.value_counts().reindex(categorias, fill_value=0)

    prop_ref = proporciones_suavizadas(conteo_ref.to_numpy())
    prop_act = proporciones_suavizadas(conteo_act.to_numpy())

    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))


def calcular_js(referencia: pd.Series, actual: pd.Series, n_bins: int = 20) -> float:
    """
    Divergencia de Jensen-Shannon entre dos distribuciones numéricas.

    `scipy.spatial.distance.jensenshannon` devuelve la *distancia* (la raíz de
    la divergencia). Con `base=2` queda acotada en [0, 1], donde 0 es
    distribuciones idénticas y 1 es sin solapamiento. Esa acotación es la
    ventaja sobre el PSI, que no tiene techo y es difícil de comparar entre
    variables.

    Los bordes de los bins se calculan sobre las dos muestras juntas para que
    ambos histogramas queden sobre la misma rejilla.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    if len(referencia) == 0 or len(actual) == 0:
        return np.nan

    bordes = np.histogram_bin_edges(
        np.concatenate([referencia, actual]), bins=n_bins
    )

    hist_ref = np.histogram(referencia, bins=bordes)[0]
    hist_act = np.histogram(actual, bins=bordes)[0]

    if hist_ref.sum() == 0 or hist_act.sum() == 0:
        return np.nan

    return float(jensenshannon(hist_ref, hist_act, base=2))


def calcular_ks(referencia: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """
    Prueba de Kolmogorov-Smirnov de dos muestras.

    Devuelve el estadístico (máxima distancia entre las funciones de
    distribución acumulada) y el p-valor. Un p-valor bajo dice que el cambio
    difícilmente sea azar; el estadístico dice qué tan grande es. Hay que leer
    los dos: con muestras grandes el KS detecta como significativos cambios
    ínfimos que no afectan al modelo.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    if len(referencia) < 2 or len(actual) < 2:
        return np.nan, np.nan

    estadistico, p_valor = ks_2samp(referencia, actual)
    return float(estadistico), float(p_valor)


def calcular_chi2(referencia: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """
    Prueba de chi-cuadrado de independencia para variables categóricas.

    Contrasta si la mezcla de categorías de la población actual es compatible
    con la de referencia.
    """
    tabla = pd.crosstab(
        pd.concat(
            [
                pd.Series("referencia", index=referencia.index),
                pd.Series("actual", index=actual.index),
            ]
        ),
        pd.concat([referencia, actual]),
    )

    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return np.nan, np.nan

    estadistico, p_valor, _, _ = chi2_contingency(tabla)
    return float(estadistico), float(p_valor)


def clasificar_severidad(psi: float) -> str:
    """
    Semáforo de drift según los umbrales estándar de PSI.

    - verde  (< 0.10): población estable, sin acción.
    - amarillo (0.10-0.25): cambio moderado, vigilar.
    - rojo   (> 0.25): cambio severo, el modelo puede estar degradado.
    """
    if pd.isna(psi):
        return "sin_dato"
    if psi > UMBRAL_PSI_CRITICO:
        return "rojo"
    if psi >= UMBRAL_PSI_MODERADO:
        return "amarillo"
    return "verde"


# Evaluación global: referencia vs actual
def evaluar_drift(df_referencia: pd.DataFrame, df_actual: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las métricas de drift por variable y devuelve la tabla
    ordenada de mayor a menor severidad.
    """
    filas = []

    for variable in VARIABLES_NUMERICAS:
        ref, act = df_referencia[variable], df_actual[variable]
        ks, p_ks = calcular_ks(ref, act)
        filas.append(
            {
                "variable": variable,
                "tipo": "numerica",
                "psi": calcular_psi(ref, act),
                "ks": ks,
                "p_valor": p_ks,
                "jensen_shannon": calcular_js(ref, act),
                "prueba": "Kolmogorov-Smirnov",
            }
        )

    for variable in VARIABLES_CATEGORICAS:
        ref, act = df_referencia[variable], df_actual[variable]
        if variable == "tendencia_ingresos":
            ref, act = normalizar_tendencia(ref), normalizar_tendencia(act)

        chi2, p_chi2 = calcular_chi2(ref.astype(str), act.astype(str))
        filas.append(
            {
                "variable": variable,
                "tipo": "categorica",
                "psi": calcular_psi_categorico(ref.astype(str), act.astype(str)),
                "ks": np.nan,
                "p_valor": p_chi2,
                "jensen_shannon": np.nan,
                "prueba": "Chi-cuadrado",
                "chi2": chi2,
            }
        )

    tabla = pd.DataFrame(filas)
    tabla["severidad"] = tabla["psi"].apply(clasificar_severidad)
    tabla["significativo"] = tabla["p_valor"] < ALFA

    return tabla.sort_values("psi", ascending=False).reset_index(drop=True)


# Muestreo periódico: evolución del drift en el tiempo
def drift_temporal(
    tabla: pd.DataFrame, periodicidad: str = "M", fecha_corte: str = FECHA_CORTE
) -> pd.DataFrame:
    """
    Recalcula el drift de cada periodo contra la referencia fija.

    La referencia se mantiene constante (la población con la que se validó el
    modelo) en vez de comparar cada periodo con el anterior. Si se comparara
    con el anterior, una deriva lenta y sostenida pasaría desapercibida:
    cada mes se parecería a su vecino mientras la población se aleja mes a mes
    del modelo.

    Los periodos con menos de `MIN_REGISTROS_PERIODO` registros se descartan
    (ver la nota de esa constante).
    """
    referencia = tabla[tabla["fecha_prestamo"] < fecha_corte]
    actual = tabla[tabla["fecha_prestamo"] >= fecha_corte].copy()
    actual["periodo"] = actual["fecha_prestamo"].dt.to_period(periodicidad)

    filas = []
    for periodo, grupo in actual.groupby("periodo", observed=True):
        if len(grupo) < MIN_REGISTROS_PERIODO:
            continue

        for variable in VARIABLES_NUMERICAS:
            psi = calcular_psi(referencia[variable], grupo[variable])
            ks, p_valor = calcular_ks(referencia[variable], grupo[variable])
            filas.append(
                {
                    "periodo": str(periodo),
                    "variable": variable,
                    "n_registros": len(grupo),
                    "psi": psi,
                    "ks": ks,
                    "p_valor": p_valor,
                    "severidad": clasificar_severidad(psi),
                }
            )

    return pd.DataFrame(filas)


def resumen_por_periodo(
    tabla: pd.DataFrame, periodicidad: str = "M"
) -> pd.DataFrame:
    """
    Métricas de negocio por periodo: volumen, mora real y mora predicha.

    Vigilar la brecha entre la mora real y la predicha es complementario al
    drift de features: puede haber drift en las variables sin que el modelo
    pierda calibración, y también lo contrario.
    """
    periodos = tabla["fecha_prestamo"].dt.to_period(periodicidad)

    resumen = tabla.groupby(periodos).agg(
        n_registros=("prediccion", "size"),
        mora_real=("mora_real", "mean"),
        mora_predicha=("prediccion", "mean"),
        probabilidad_media=("probabilidad_mora", "mean"),
    )
    resumen.index = resumen.index.astype(str)
    resumen.index.name = "periodo"

    return resumen.reset_index()


# Análisis temporal: tendencias y cambios abruptos
def detectar_patrones_temporales(
    temporal: pd.DataFrame,
    correlacion_minima: float = 0.7,
    factor_salto: float = 3.0,
) -> list[tuple[str, str, str]]:
    """
    Distingue deriva gradual de saltos puntuales en la serie de PSI.

    La distinción tiene consecuencias prácticas opuestas: una tendencia
    sostenida se resuelve reentrenando con datos recientes, mientras que un
    salto abrupto casi siempre es un cambio operativo o un dato roto en la
    fuente, y reentrenar sobre eso sería aprender el error.

    - **Tendencia**: correlación de Spearman entre el orden del periodo y el
      PSI por encima de `correlacion_minima`. Se usa Spearman y no una
      pendiente lineal porque solo interesa que el drift crezca de forma
      consistente, no que lo haga a ritmo constante. Antes esto se comprobaba
      exigiendo que la serie creciera en *cada* paso, criterio que ninguna
      serie real cumple: un único mes plano lo invalidaba.

    - **Cambio abrupto**: un salto entre periodos consecutivos mayor a
      `factor_salto` veces la variación típica (mediana de los saltos) de esa
      misma variable. El umbral es relativo a cada serie, porque una variable
      naturalmente ruidosa no debe alarmar con la misma magnitud que una
      estable. Solo se reportan saltos **hacia arriba**: que el PSI caiga de
      0.43 a 0.19 es la población acercándose otra vez a la de referencia, y
      alarmar por una mejora entrena al usuario a ignorar el tablero.

    Además, una variable solo entra al análisis si en algún periodo alcanza el
    umbral de drift moderado. Sin ese filtro se reportan tendencias
    impecablemente reales pero irrelevantes —un PSI que pasa de 0.018 a 0.067
    sigue estando muy por debajo de lo que afecta al modelo— y el tablero se
    llena de alertas que nadie puede accionar.
    """
    if temporal.empty:
        return []

    patrones = []

    for variable, grupo in temporal.groupby("variable"):
        serie = grupo.sort_values("periodo")
        psi = serie["psi"].to_numpy()
        periodos = serie["periodo"].to_numpy()

        if len(psi) < 3 or np.isnan(psi).any():
            continue

        if np.nanmax(psi) < UMBRAL_PSI_MODERADO:
            continue

        # Tendencia sostenida.
        correlacion = pd.Series(psi).corr(pd.Series(range(len(psi))), method="spearman")
        if correlacion is not None and correlacion >= correlacion_minima:
            patrones.append(
                (
                    variable,
                    "tendencia",
                    f"{periodos[0]}: {psi[0]:.3f} -> {periodos[-1]}: {psi[-1]:.3f}, "
                    f"correlación de Spearman {correlacion:.2f}",
                )
            )

        # Cambio abrupto (solo deterioros, ver docstring).
        diferencias = np.diff(psi)
        variacion_tipica = np.median(np.abs(diferencias))
        saltos = np.where(diferencias > 0, diferencias, 0.0)
        if variacion_tipica > 0 and saltos.max() > 0:
            indice = int(np.argmax(saltos))
            if saltos[indice] > factor_salto * variacion_tipica:
                patrones.append(
                    (
                        variable,
                        "salto",
                        f"de {psi[indice]:.3f} a {psi[indice + 1]:.3f} entre "
                        f"{periodos[indice]} y {periodos[indice + 1]}, "
                        f"{saltos[indice] / variacion_tipica:.1f}x su variación típica",
                    )
                )

    return patrones


# Recomendaciones automáticas
def generar_recomendaciones(
    metricas: pd.DataFrame, temporal: pd.DataFrame, resumen: pd.DataFrame
) -> list[str]:
    """
    Traduce las métricas en acciones concretas.

    El objetivo es que quien mire el dashboard no tenga que saber qué es un
    PSI para entender si hay que hacer algo.
    """
    recomendaciones = []

    criticas = metricas[metricas["severidad"] == "rojo"]["variable"].tolist()
    moderadas = metricas[metricas["severidad"] == "amarillo"]["variable"].tolist()

    if criticas:
        recomendaciones.append(
            f"CRÍTICO: {len(criticas)} variable(s) con drift severo "
            f"(PSI > {UMBRAL_PSI_CRITICO}): {', '.join(criticas)}. "
            "Se recomienda reentrenar el modelo con datos recientes."
        )

    if moderadas:
        recomendaciones.append(
            f"ATENCIÓN: {len(moderadas)} variable(s) con drift moderado "
            f"(PSI {UMBRAL_PSI_MODERADO}-{UMBRAL_PSI_CRITICO}): "
            f"{', '.join(moderadas)}. Vigilar en los próximos periodos."
        )

    significativas = metricas[
        metricas["significativo"] & (metricas["severidad"] == "verde")
    ]["variable"].tolist()
    if significativas:
        recomendaciones.append(
            f"Cambio estadísticamente significativo pero de magnitud pequeña en: "
            f"{', '.join(significativas)}. Con muestras grandes las pruebas "
            "detectan diferencias mínimas; el PSI bajo indica que no son "
            "relevantes para el modelo. No requiere acción."
        )

    for variable, tipo, detalle in detectar_patrones_temporales(temporal):
        if tipo == "tendencia":
            recomendaciones.append(
                f"TENDENCIA: el drift de `{variable}` crece de forma sostenida "
                f"({detalle}). Es deriva progresiva, no un evento puntual: "
                "conviene programar reentrenamiento periódico en vez de "
                "esperar a que cruce el umbral crítico."
            )
        else:
            recomendaciones.append(
                f"CAMBIO ABRUPTO: `{variable}` salta bruscamente ({detalle}). "
                "Un salto puntual suele venir de un cambio operativo o de un "
                "problema en la fuente de datos, no de deriva gradual: revisar "
                "primero la captura del dato antes de reentrenar."
            )

    # Brecha de calibración entre mora observada y predicha.
    validos = resumen[resumen["n_registros"] >= MIN_REGISTROS_PERIODO]
    if not validos.empty:
        brecha = (validos["mora_predicha"] - validos["mora_real"]).abs().mean()
        if brecha > 0.05:
            recomendaciones.append(
                f"CALIBRACIÓN: la tasa de mora predicha se desvía de la real en "
                f"{brecha:.1%} promedio por periodo. Revisar el umbral de "
                "decisión antes de reentrenar."
            )

    if not recomendaciones:
        recomendaciones.append(
            "Sin drift relevante. La población actual es compatible con la de "
            "entrenamiento; no se requiere acción."
        )

    return recomendaciones


def main():
    print("=" * 78)
    print("MONITOREO DE DATA DRIFT — Modelo de Riesgo Crediticio")
    print("=" * 78)

    tabla = generar_tabla_predicciones(forzar=True)

    referencia = tabla[tabla["fecha_prestamo"] < FECHA_CORTE]
    actual = tabla[tabla["fecha_prestamo"] >= FECHA_CORTE]

    print(f"\nCorte: {FECHA_CORTE}")
    print(f"  Referencia: {len(referencia):>6} registros "
          f"({referencia['fecha_prestamo'].min():%Y-%m} a "
          f"{referencia['fecha_prestamo'].max():%Y-%m}) | "
          f"mora real {referencia['mora_real'].mean():.2%}")
    print(f"  Actual    : {len(actual):>6} registros "
          f"({actual['fecha_prestamo'].min():%Y-%m} a "
          f"{actual['fecha_prestamo'].max():%Y-%m}) | "
          f"mora real {actual['mora_real'].mean():.2%}")

    metricas = evaluar_drift(referencia, actual)
    metricas.to_csv(RUTA_METRICAS, index=False)

    print("\n" + "-" * 78)
    print("MÉTRICAS DE DRIFT POR VARIABLE")
    print("-" * 78)
    columnas = ["variable", "tipo", "psi", "ks", "jensen_shannon", "p_valor", "severidad"]
    print(
        metricas[columnas].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
            na_rep="-",
        )
    )

    conteo = metricas["severidad"].value_counts()
    print(f"\nSemáforo: {conteo.get('rojo', 0)} rojo | "
          f"{conteo.get('amarillo', 0)} amarillo | {conteo.get('verde', 0)} verde")

    temporal = drift_temporal(tabla)
    temporal.to_csv(RUTA_DRIFT_TEMPORAL, index=False)

    print("\n" + "-" * 78)
    print(f"EVOLUCIÓN TEMPORAL (periodos con >= {MIN_REGISTROS_PERIODO} registros)")
    print("-" * 78)
    if temporal.empty:
        print("Ningún periodo alcanza el mínimo de registros.")
    else:
        pivote = temporal.pivot(index="periodo", columns="variable", values="psi")
        top = metricas.head(5)["variable"].tolist()
        print(pivote[[c for c in top if c in pivote.columns]].to_string(
            float_format=lambda v: f"{v:.4f}"
        ))
        print(f"\n(PSI de las 5 variables con más drift; "
              f"{temporal['periodo'].nunique()} periodos analizados)")

    resumen = resumen_por_periodo(tabla)
    print("\n" + "-" * 78)
    print("MORA REAL VS PREDICHA POR MES")
    print("-" * 78)
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("RECOMENDACIONES AUTOMÁTICAS")
    print("=" * 78)
    for i, recomendacion in enumerate(generar_recomendaciones(metricas, temporal, resumen), 1):
        print(f"\n{i}. {recomendacion}")

    print(f"\n\nArchivos generados: {RUTA_PREDICCIONES.name}, "
          f"{RUTA_METRICAS.name}, {RUTA_DRIFT_TEMPORAL.name}")


if __name__ == "__main__":
    main()
