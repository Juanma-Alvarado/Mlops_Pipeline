"""
model_deploy.py

API REST que expone el modelo de riesgo crediticio (PIM5) con FastAPI.

Levantar en local:
    cd src
    uvicorn model_deploy:app --reload

Documentación interactiva: http://localhost:8000/docs

NOTA SOBRE EL ARTEFACTO SERIALIZADO
-----------------------------------
`modelo_riesgo_credito.pkl` contiene el pipeline de feature engineering, que a
su vez contiene instancias de los transformadores personalizados definidos en
`ft_engineering.py`. Pickle no guarda el código de esas clases, solo una
referencia al módulo donde viven, así que `ft_engineering` tiene que ser
importable en el momento de cargar el artefacto o `joblib.load` falla con
`ModuleNotFoundError`.

Por eso este módulo agrega su propio directorio a `sys.path`: así la API
funciona sin importar desde dónde se invoque a uvicorn.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DIRECTORIO_SRC = Path(__file__).resolve().parent
if str(DIRECTORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_SRC))

RUTA_ARTEFACTO = DIRECTORIO_SRC / "modelo_riesgo_credito.pkl"

# La ruta del registro es configurable para poder montarla en un volumen de
# Docker: así el log de predicciones sobrevive a `docker compose down` en vez
# de morir con el contenedor.
RUTA_REGISTRO = Path(
    os.getenv("RUTA_REGISTRO", DIRECTORIO_SRC / "registro_predicciones.csv")
)

# Métricas del modelo en el set de test, medidas en el avance 2 (v1.1.1).
# Se exponen en /modelo para que quien consuma la API sepa qué esperar y no
# sobreinterprete una probabilidad puntual.
METRICAS_TEST = {
    "pr_auc": 0.1367,
    "roc_auc": 0.6760,
    "recall": 0.2549,
    "precision": 0.1032,
    "f1": 0.1469,
}

# Estado del servicio. Se llena una sola vez en el arranque (ver `lifespan`):
# cargar un Random Forest de 300 árboles en cada request sería inaceptable.
artefacto: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar y lo libera al apagar."""
    if not RUTA_ARTEFACTO.exists():
        raise RuntimeError(
            f"No se encontró el artefacto en {RUTA_ARTEFACTO}. "
            "Ejecuta `python model_training_evaluation.py` para generarlo."
        )
    artefacto.update(joblib.load(RUTA_ARTEFACTO))
    print(f"Modelo cargado: {artefacto['nombre_modelo']} "
          f"(umbral {artefacto['umbral']:.4f})")
    yield
    artefacto.clear()


app = FastAPI(
    title="API de Riesgo Crediticio — PIM5",
    description=(
        "Estima la probabilidad de que un cliente **no pague a tiempo** un "
        "crédito, a partir de información disponible en el momento de la "
        "solicitud.\n\n"
        "El modelo fue entrenado excluyendo deliberadamente las variables con "
        "*data leakage* (`puntaje` y los saldos de la deuda), porque no existen "
        "todavía cuando hay que decidir si se aprueba un crédito nuevo. Por eso "
        "esta API pide 17 variables y no las 22 del dataset original."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# Esquemas de entrada y salida
class SolicitudCredito(BaseModel):
    """
    Datos de una solicitud de crédito.

    Son las 17 variables de originación: las 22 del dataset original menos las
    5 descartadas por data leakage y menos la variable objetivo.

    `puntaje_datacredito` y `promedio_ingresos_datacredito` son opcionales
    porque pueden faltar legítimamente (cliente sin historial suficiente en el
    buró). El pipeline los imputa con la mediana del entrenamiento y además
    marca la ausencia en una variable indicadora, porque no tener información
    en el buró es en sí mismo una señal de riesgo.
    """

    tipo_credito: int = Field(..., description="Código de la línea de crédito")
    fecha_prestamo: datetime = Field(..., description="Fecha de la solicitud")
    capital_prestado: float = Field(..., gt=0, description="Monto solicitado")
    plazo_meses: int = Field(..., gt=0, description="Plazo en meses")
    edad_cliente: int = Field(..., ge=18, le=100)
    tipo_laboral: str = Field(..., description="'Empleado' o 'Independiente'")
    salario_cliente: float = Field(..., ge=0, description="Salario declarado")
    total_otros_prestamos: float = Field(..., ge=0)
    cuota_pactada: float = Field(..., gt=0)
    cant_creditosvigentes: int = Field(..., ge=0)
    huella_consulta: int = Field(..., ge=0, description="Consultas al buró")
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    tendencia_ingresos: Optional[str] = Field(
        None, description="'Creciente', 'Decreciente' o 'Estable'"
    )
    puntaje_datacredito: Optional[float] = Field(None, description="Score del buró")
    promedio_ingresos_datacredito: Optional[float] = Field(
        None, description="Ingreso promedio reportado por el buró"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tipo_credito": 7,
                "fecha_prestamo": "2026-05-15T10:30:00",
                "capital_prestado": 3692160.0,
                "plazo_meses": 10,
                "edad_cliente": 42,
                "tipo_laboral": "Independiente",
                "salario_cliente": 8000000,
                "total_otros_prestamos": 2500000,
                "cuota_pactada": 341296,
                "cant_creditosvigentes": 10,
                "huella_consulta": 5,
                "creditos_sectorFinanciero": 5,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 0,
                "tendencia_ingresos": "Estable",
                "puntaje_datacredito": 695.0,
                "promedio_ingresos_datacredito": 908526.0,
            }
        }
    }


class LoteSolicitudes(BaseModel):
    """Varias solicitudes en una sola llamada, para scoring masivo."""

    solicitudes: list[SolicitudCredito]


class Prediccion(BaseModel):
    """
    Resultado del modelo para una solicitud.

    `umbral_aplicado` se devuelve explícitamente y no es un detalle interno:
    la decisión NO se toma en 0.5 sino en el umbral optimizado en el avance 2,
    y quien consuma la API necesita poder auditar con qué criterio se marcó
    a un cliente como riesgoso.
    """

    probabilidad_mora: float = Field(..., description="Probabilidad de no pago")
    prediccion: int = Field(..., description="1 = riesgo de mora, 0 = se espera pago")
    banda_riesgo: str = Field(..., description="bajo / medio / alto")
    umbral_aplicado: float


def clasificar_banda(probabilidad: float, umbral: float) -> str:
    """
    Traduce la probabilidad a una banda cualitativa.

    Las bandas se definen en relación al umbral del modelo, no con cortes
    inventados: 'alto' es exactamente lo que el modelo marca como mora, y
    'medio' es la zona que se queda cerca del umbral sin cruzarlo (clientes
    que conviene revisar manualmente en vez de aprobar sin más).

    Es una ayuda de presentación para el analista de crédito, no una salida
    del modelo.
    """
    if probabilidad >= umbral:
        return "alto"
    if probabilidad >= 0.75 * umbral:
        return "medio"
    return "bajo"


def predecir_dataframe(df: pd.DataFrame) -> list[Prediccion]:
    """
    Aplica el pipeline de feature engineering y el modelo a un DataFrame de
    solicitudes crudas.

    Es importante reutilizar el `pipeline_ft` serializado y no reimplementar
    las transformaciones: fue ajustado con el set de entrenamiento y lleva
    dentro las medianas, los límites de winsorización y la moda aprendidos
    ahí. Recalcularlos con los datos que llegan por la API daría resultados
    distintos a los del entrenamiento.
    """
    pipeline = artefacto["pipeline_ft"]
    modelo = artefacto["modelo"]
    umbral = artefacto["umbral"]

    X = pipeline.transform(df)
    probabilidades = modelo.predict_proba(X)[:, 1]

    return [
        Prediccion(
            probabilidad_mora=round(float(p), 6),
            prediccion=int(p >= umbral),
            banda_riesgo=clasificar_banda(float(p), umbral),
            umbral_aplicado=round(float(umbral), 6),
        )
        for p in probabilidades
    ]


def registrar_predicciones(df: pd.DataFrame, predicciones: list[Prediccion]) -> None:
    """
    Deja constancia de cada predicción en `registro_predicciones.csv`.

    Esta tabla —datos de entrada junto con el pronóstico entregado— es el
    insumo del trabajo de monitoreo (`model_monitoring.py`): sin un registro
    de lo que el modelo respondió en producción no hay forma de detectar que
    la población de solicitantes cambió respecto a la del entrenamiento.

    Un fallo al escribir el registro no debe tumbar la respuesta al cliente:
    la predicción ya se calculó bien y el monitoreo es un proceso secundario.
    """
    try:
        registro = df.copy()
        registro["momento_prediccion"] = datetime.now().isoformat(timespec="seconds")
        registro["probabilidad_mora"] = [p.probabilidad_mora for p in predicciones]
        registro["prediccion"] = [p.prediccion for p in predicciones]
        registro["banda_riesgo"] = [p.banda_riesgo for p in predicciones]

        registro.to_csv(
            RUTA_REGISTRO,
            mode="a",
            index=False,
            header=not RUTA_REGISTRO.exists(),
        )
    except Exception as error:  # noqa: BLE001
        print(f"[registro] No se pudo escribir la predicción: {error}")


# Endpoints
@app.get("/salud", tags=["Servicio"])
def salud():
    """Healthcheck: confirma que el servicio está arriba y el modelo cargado."""
    return {
        "estado": "ok",
        "modelo_cargado": bool(artefacto),
        "modelo": artefacto.get("nombre_modelo"),
    }


@app.get("/modelo", tags=["Servicio"])
def informacion_modelo():
    """
    Metadata del modelo en producción.

    Expone las métricas de test junto con el modelo para que nadie lea una
    probabilidad como una certeza: con PR-AUC 0.1367 el modelo es una señal
    de apoyo, no un decisor autónomo.
    """
    if not artefacto:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")

    return {
        "nombre": artefacto["nombre_modelo"],
        "version": app.version,
        "umbral_decision": round(artefacto["umbral"], 6),
        "variables_entrada": len(SolicitudCredito.model_fields),
        "variables_modelo": len(artefacto["columnas"]),
        "metricas_test": METRICAS_TEST,
        "clase_positiva": "mora (el cliente no paga a tiempo)",
        "variables_excluidas_por_leakage": [
            "puntaje",
            "saldo_mora",
            "saldo_total",
            "saldo_principal",
            "saldo_mora_codeudor",
        ],
    }


@app.post("/predecir", response_model=Prediccion, tags=["Predicción"])
def predecir(solicitud: SolicitudCredito):
    """Estima el riesgo de mora de una solicitud de crédito."""
    if not artefacto:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")

    df = pd.DataFrame([solicitud.model_dump()])

    try:
        predicciones = predecir_dataframe(df)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"No se pudo procesar la solicitud: {error}"
        ) from error

    registrar_predicciones(df, predicciones)
    return predicciones[0]


@app.post("/predecir-lote", response_model=list[Prediccion], tags=["Predicción"])
def predecir_lote(lote: LoteSolicitudes):
    """
    Estima el riesgo de un conjunto de solicitudes en una sola llamada.

    Se transforma todo el lote de una vez en lugar de fila por fila: el
    pipeline y el modelo son vectorizados, así que un lote de 500 solicitudes
    cuesta mucho menos que 500 llamadas a `/predecir`.
    """
    if not artefacto:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")

    if not lote.solicitudes:
        raise HTTPException(status_code=422, detail="El lote está vacío")

    df = pd.DataFrame([s.model_dump() for s in lote.solicitudes])

    try:
        predicciones = predecir_dataframe(df)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"No se pudo procesar el lote: {error}"
        ) from error

    registrar_predicciones(df, predicciones)
    return predicciones


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "model_deploy:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
