"""
ft_engineering.py

Pipeline de Feature Engineering para el modelo de riesgo crediticio (PIM5).

El objetivo final del modelo es predecir `Pago_atiempo`
(1 = paga a tiempo, 0 = no paga a tiempo / mora), por lo que las decisiones
priorizan preservar la señal relacionada con capacidad y comportamiento de pago del cliente.

IMPORTANTE (evitar data leakage):
El pipeline se ajusta (`fit`) SOLO con el set de entrenamiento y se aplica
(`transform`) igual a train y test. Nunca se debe ajustar con el dataset
completo antes del split.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer,
    AddMissingIndicator,
)
from feature_engine.outliers import Winsorizer

from carga_datos import cargarDatos

TARGET = "Pago_atiempo"

# Categorías válidas de negocio para tendencia_ingresos.
# Cualquier otro valor es ruido de captura/exportación, no una categoría real.
CATEGORIAS_VALIDAS_TENDENCIA = ["Creciente", "Decreciente", "Estable"]

# Orden de negocio de tendencia_ingresos: un ingreso decreciente es peor que
# uno estable, y uno estable peor que uno creciente. Es una variable ordinal
# real, así que se codifica respetando esa jerarquía.
ORDEN_TENDENCIA = {"Decreciente": 0, "Estable": 1, "Creciente": 2}

# Ratios financieros creados por `CrearRatiosFinancieros`.
RATIOS = [
    "ratio_cuota_salario",
    "ratio_capital_salario",
    "ratio_ingresos_burodeclarado",
]

# ---------------------------------------------------------------------------
# DECISIÓN CLAVE DEL AVANCE 2: eliminación de variables con data leakage
# ---------------------------------------------------------------------------
# Al revisar la relación de cada variable con el target se encontró que:
#
#   * `puntaje` correlaciona 0.923 con `Pago_atiempo`. Ninguna variable
#     legítima de originación de crédito alcanza esa correlación. Se trata
#     casi con seguridad de un score calculado DESPUÉS de conocer el
#     comportamiento de pago del cliente, es decir, el target disfrazado.
#     Conservarla produce un modelo con ROC-AUC ~0.99 que en producción no
#     sirve, porque al momento de decidir si se aprueba un crédito nuevo esa
#     variable todavía no existe.
#
#   * `saldo_mora`, `saldo_total`, `saldo_principal` y `saldo_mora_codeudor`
#     describen el estado de la deuda DURANTE la vida del crédito, no al
#     momento de otorgarlo (saldo_mora > 0 aparece en 3.9% de los morosos vs.
#     0.34% de quienes pagaron a tiempo: es consecuencia del impago, no causa).
#     Son leakage temporal.
#
# Decisión tomada: se eliminan todas. El modelo pierde performance aparente
# pero gana validez: solo usa información disponible en el momento real de
# la decisión de crédito. El README documenta la comparación de métricas
# con y sin estas variables.
#
# Consecuencia en el pipeline: se retiran también las etapas que dependían
# de ellas (indicador de codeudor, imputación de saldos con 0 y el ratio
# `ratio_mora_saldo`), porque quedan sin insumo.
COLUMNAS_LEAKAGE = [
    "puntaje",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
]


# Transformadores personalizados
class TransformadorBase(BaseEstimator, TransformerMixin):
    """
    Base común de los transformadores propios de este pipeline.

    Ninguno de ellos necesita aprender parámetros de los datos (son reglas
    de negocio fijas: renombrar, mapear, dividir columnas), así que su `fit`
    no hace cálculos. Aun así debe dejar constancia de que fue ajustado
    (`fitted_`), porque scikit-learn valida con `check_is_fitted` antes de
    permitir `transform`, y un objeto sin ningún atributo terminado en `_`
    se considera no ajustado.
    """

    def fit(self, X, y=None):
        self.fitted_ = True
        return self


class EliminarLeakage(TransformadorBase):
    """
    Elimina las variables identificadas como data leakage (ver
    `COLUMNAS_LEAKAGE` y su justificación arriba).

    Va como PRIMERA etapa del pipeline para que ninguna etapa posterior
    (imputación, ratios, winsorización) llegue a usarlas por accidente.
    """

    def __init__(self, columnas):
        self.columnas = columnas

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=[c for c in self.columnas if c in X.columns])


class LimpiarTendenciaIngresos(TransformadorBase):
    """
    Convierte a NaN cualquier valor de `tendencia_ingresos` que no sea una
    de las 3 categorías de negocio esperadas.

    Justificación de negocio: valores como '8315' o '-566272' no son una
    tendencia de ingresos, son un error de formato (parecen montos que se
    filtraron a esta columna). Tratarlos como una cuarta categoría
    distorsionaría el análisis; lo correcto es tratarlos como dato faltante
    real y dejar que la imputación decida qué hacer con ellos.
    """

    def transform(self, X):
        X = X.copy()
        X["tendencia_ingresos"] = X["tendencia_ingresos"].where(
            X["tendencia_ingresos"].isin(CATEGORIAS_VALIDAS_TENDENCIA), np.nan
        )
        return X


class CrearRatiosFinancieros(TransformadorBase):
    """
    Crea variables de razón (ratios) que capturan capacidad de pago y
    consistencia de ingresos de forma relativa, no absoluta.

    Justificación de negocio:
    - `ratio_cuota_salario`: dos clientes con la misma cuota pactada pueden
      tener capacidad de pago muy distinta según su salario. Este ratio es
      un proxy directo de sobreendeudamiento, un factor clásico en scoring
      crediticio.
    - `ratio_capital_salario`: nivel de apalancamiento del cliente respecto
      a su ingreso.
    - `ratio_ingresos_burodeclarado`: compara el ingreso que reporta
      DataCrédito contra el salario que declara el cliente. Un valor muy por
      debajo de 1 indica que el cliente infla su ingreso frente a la entidad,
      lo cual es en sí mismo una señal de riesgo (y también una señal de
      informalidad si es al revés).

    IMPORTANTE: esta etapa debe ejecutarse DESPUÉS de winsorizar
    `salario_cliente`. Si se calcula con el salario sucio, los clientes con
    salario mal capturado (p. ej. 22.000 millones por ceros de más) obtienen
    ratios cercanos a cero que aparentan riesgo bajísimo cuando en realidad
    es un error de captura.

    Las divisiones por cero (salario en 0) se controlan explícitamente para
    no generar infinitos que rompan el modelo.
    """

    def transform(self, X):
        X = X.copy()

        salario_valido = X["salario_cliente"] > 0

        X["ratio_cuota_salario"] = np.where(
            salario_valido, X["cuota_pactada"] / X["salario_cliente"], np.nan
        )

        X["ratio_capital_salario"] = np.where(
            salario_valido, X["capital_prestado"] / X["salario_cliente"], np.nan
        )

        X["ratio_ingresos_burodeclarado"] = np.where(
            salario_valido,
            X["promedio_ingresos_datacredito"] / X["salario_cliente"],
            np.nan,
        )

        return X


class VariablesTemporales(TransformadorBase):
    """
    Extrae mes y año de `fecha_prestamo` y elimina la fecha cruda.

    Justificación de negocio: la fecha exacta no generaliza a créditos
    futuros (no es una variable que el modelo pueda usar en producción de
    forma directa), pero el mes/año sí puede capturar estacionalidad o
    efectos macroeconómicos del periodo en que se otorgó el crédito.
    """

    def transform(self, X):
        X = X.copy()
        X["mes_prestamo"] = X["fecha_prestamo"].dt.month
        X["anio_prestamo"] = X["fecha_prestamo"].dt.year
        X = X.drop(columns=["fecha_prestamo"])
        return X


class CodificarTipoLaboral(TransformadorBase):
    """
    Codifica `tipo_laboral` como binaria: 1 = Empleado, 0 = Independiente.

    Justificación de negocio: es una variable binaria real (solo 2 valores
    en los datos), por lo que un encoding ordinal/one-hot es innecesario;
    un mapeo directo es más simple e igual de efectivo.
    """

    def transform(self, X):
        X = X.copy()
        X["tipo_laboral"] = (X["tipo_laboral"] == "Empleado").astype(int)
        return X


class CodificarTendenciaIngresos(TransformadorBase):
    """
    Codifica `tendencia_ingresos` respetando su orden de negocio
    (Decreciente < Estable < Creciente), según `ORDEN_TENDENCIA`.

    Se usa un mapeo explícito en vez del OrdinalEncoder de feature-engine
    con `encoding_method="arbitrary"`, porque ese método asigna los números
    en el orden en que aparecen las categorías en los datos, lo cual NO
    garantiza la jerarquía de negocio. Para una variable ordinal como esta,
    el orden es justamente la información que se quiere transmitir al modelo.
    """

    def transform(self, X):
        X = X.copy()
        X["tendencia_ingresos"] = X["tendencia_ingresos"].map(ORDEN_TENDENCIA)
        return X


# Construcción del pipeline
def construir_pipeline():
    """
    Ensambla el pipeline completo de feature engineering.

    Orden de las etapas (importa, cada una depende de la anterior):
    1. Eliminar variables con data leakage (`puntaje` y saldos).
    2. Limpiar ruido de tendencia_ingresos -> NaN reales.
    3. Winsorizar montos SIN nulos (capital_prestado, salario_cliente,
       cuota_pactada). Va ANTES de los ratios: si se hiciera después, los
       ratios quedarían calculados con salarios mal capturados.
    4. Indicador de nulo + imputación con mediana para variables donde el
       nulo sí es un dato perdido genuino: puntaje_datacredito,
       promedio_ingresos_datacredito. `promedio_ingresos_datacredito` tiene
       ~27% de nulos, así que el indicador es informativo por sí mismo
       (no tener información en el buró es una señal de riesgo).
    5. Winsorizar promedio_ingresos_datacredito. Se hace en un segundo paso
       separado porque esta variable tenía nulos: winsorizar antes de imputar
       calcularía los límites sobre una muestra incompleta.
    6. Imputar tendencia_ingresos (categórica) con la moda.
    7. Crear ratios financieros, ya con todos los insumos limpios.
    8. Imputar con mediana los ratios que quedaron NaN por salario = 0.
    9. Winsorizar los ratios. La winsorización del paso 3 solo corrige la
       cola derecha del salario, pero el problema de captura también existe
       en la cola izquierda: hay 24 clientes con salario 0 y 8 más por debajo
       de medio salario mínimo, que producen ratios de hasta 840 veces el
       ingreso. No se recorta el salario por la izquierda (un salario bajo
       puede ser real y su nivel importa por sí mismo); se recorta el ratio,
       que es donde el error se vuelve un valor imposible de negocio.
    10. Extraer variables temporales de fecha_prestamo.
    11. Codificar tipo_laboral (binaria) y tendencia_ingresos (ordinal).

    Sobre el método de winsorización: se usa capping por percentil (p99) en
    la cola derecha y no el IQR 1.5 habitual. Con IQR 1.5 el tope de
    `salario_cliente` cae en ~9.2 millones y se recortarían 718 clientes
    (6.7%), muchos de ellos ingresos altos perfectamente reales. El problema
    detectado son ~22 registros con salarios de miles de millones (errores de
    captura por ceros de más), y el capping en p99 los corrige sin destruir
    la cola legítima de ingresos altos. Solo se recorta la cola derecha: los
    valores bajos son plausibles y no muestran patrón de error.
    """

    columnas_mediana = ["puntaje_datacredito", "promedio_ingresos_datacredito"]

    montos_sin_nulos = ["capital_prestado", "salario_cliente", "cuota_pactada"]

    pipeline = Pipeline(
        steps=[
            ("eliminar_leakage", EliminarLeakage(columnas=COLUMNAS_LEAKAGE)),
            ("limpiar_tendencia", LimpiarTendenciaIngresos()),
            (
                "winsorizar_montos",
                Winsorizer(
                    capping_method="quantiles",
                    tail="right",
                    fold=0.01,
                    variables=montos_sin_nulos,
                ),
            ),
            (
                "indicador_nulos_mediana",
                AddMissingIndicator(variables=columnas_mediana),
            ),
            (
                "imputar_mediana",
                MeanMedianImputer(
                    imputation_method="median", variables=columnas_mediana
                ),
            ),
            (
                "winsorizar_ingresos_buro",
                Winsorizer(
                    capping_method="quantiles",
                    tail="right",
                    fold=0.01,
                    variables=["promedio_ingresos_datacredito"],
                ),
            ),
            (
                "imputar_tendencia",
                CategoricalImputer(
                    imputation_method="frequent", variables=["tendencia_ingresos"]
                ),
            ),
            ("ratios_financieros", CrearRatiosFinancieros()),
            (
                "imputar_ratios",
                MeanMedianImputer(
                    imputation_method="median", variables=RATIOS
                ),
            ),
            (
                "winsorizar_ratios",
                Winsorizer(
                    capping_method="quantiles",
                    tail="right",
                    fold=0.01,
                    variables=RATIOS,
                ),
            ),
            ("variables_temporales", VariablesTemporales()),
            ("codificar_tipo_laboral", CodificarTipoLaboral()),
            ("codificar_tendencia", CodificarTendenciaIngresos()),
        ]
    )

    return pipeline


def dividir_y_transformar(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Divide en train/test ANTES de ajustar el pipeline, y lo ajusta solo con
    train, para evitar data leakage hacia el set de prueba.

    Retorna X_train, X_test, y_train, y_test ya transformados, y el pipeline
    ya ajustado (necesario para el avance de despliegue).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = construir_pipeline()
    X_train_t = pipeline.fit_transform(X_train, y_train)
    X_test_t = pipeline.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, pipeline


def validar_transformacion(X_train: pd.DataFrame):
    """
    Chequeos de sanidad sobre el set transformado, para confirmar que el
    resultado es apto para entrar a un modelo de scikit-learn.
    """
    print("\n1. Nulos restantes:", X_train.isnull().sum().sum())

    columnas_texto = X_train.select_dtypes(include="object").columns.tolist()
    print("2. Columnas de texto sin codificar (debe estar vacío):", columnas_texto)

    print("3. Distribución de tendencia_ingresos codificada:")
    print(X_train["tendencia_ingresos"].value_counts().sort_index())

    print("4. Infinitos en los ratios:", np.isinf(X_train[RATIOS]).sum().sum())
    print(X_train[RATIOS].describe())

    print("\n5. Variables finales del modelo:", list(X_train.columns))


if __name__ == "__main__":
    datos = cargarDatos()
    X_train, X_test, y_train, y_test, pipeline = dividir_y_transformar(datos)

    print("Shape X_train transformado:", X_train.shape)
    print("Shape X_test transformado:", X_test.shape)

    validar_transformacion(X_train)
