"""
app_streamlit.py

Aplicación web del modelo de riesgo crediticio (PIM5).

Correr (con la API ya levantada):
    cd src
    streamlit run app_streamlit.py

Tiene dos pantallas:
  1. Scoring de solicitudes — consulta el modelo a través de la API.
  2. Monitoreo de data drift — vigila si la población de clientes cambió.

POR QUÉ CONSUME LA API Y NO CARGA EL MODELO DIRECTO
---------------------------------------------------
La app podría hacer `joblib.load` del artefacto y ahorrarse el salto HTTP,
pero entonces habría dos caminos distintos de inferencia (el de la API y el de
la app) que tendrían que mantenerse sincronizados. Al consumir la API hay una
sola implementación de la lógica de scoring, y la app comprueba de paso que el
servicio funciona.

La URL se toma de la variable de entorno `API_URL` para que al contenerizar
(avance 4) solo haya que apuntarla al nombre del servicio.

NOTA: este archivo no está en la estructura de carpetas definida por el
enunciado, pero Streamlit necesita su propio punto de entrada y no hay forma
de evitarlo. No reemplaza a ninguno de los archivos exigidos.
"""

import os
import sys
from datetime import date, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

DIRECTORIO_SRC = Path(__file__).resolve().parent
if str(DIRECTORIO_SRC) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_SRC))

from model_monitoring import (  # noqa: E402
    FECHA_CORTE,
    MIN_MUESTRA_PSI,
    UMBRAL_PSI_CRITICO,
    UMBRAL_PSI_MODERADO,
    VARIABLES_CATEGORICAS,
    detectar_patrones_temporales,
    generar_recomendaciones,
    normalizar_tendencia,
    resumen_por_periodo,
)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

RUTA_PREDICCIONES = DIRECTORIO_SRC / "predicciones_historicas.csv"
RUTA_METRICAS = DIRECTORIO_SRC / "metricas_drift.csv"
RUTA_DRIFT_TEMPORAL = DIRECTORIO_SRC / "drift_temporal.csv"

EMOJI_SEMAFORO = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴", "sin_dato": "⚪"}

st.set_page_config(
    page_title="Riesgo Crediticio — PIM5", page_icon="🏦", layout="wide"
)


# Acceso a datos y a la API
@st.cache_data(ttl=60)
def consultar_api(ruta: str) -> dict | None:
    """Consulta un endpoint GET de la API; None si el servicio no responde."""
    try:
        respuesta = requests.get(f"{API_URL}{ruta}", timeout=5)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.RequestException:
        return None


@st.cache_data
def cargar_predicciones() -> pd.DataFrame | None:
    if not RUTA_PREDICCIONES.exists():
        return None
    return pd.read_csv(RUTA_PREDICCIONES, parse_dates=["fecha_prestamo"])


@st.cache_data
def cargar_metricas() -> pd.DataFrame | None:
    if not RUTA_METRICAS.exists():
        return None
    return pd.read_csv(RUTA_METRICAS)


@st.cache_data
def cargar_drift_temporal() -> pd.DataFrame | None:
    if not RUTA_DRIFT_TEMPORAL.exists():
        return None
    return pd.read_csv(RUTA_DRIFT_TEMPORAL)


def aviso_faltan_datos():
    st.warning(
        "Todavía no hay resultados de monitoreo. Genéralos con:\n\n"
        "```bash\ncd src\npython model_monitoring.py\n```"
    )


# Pantalla 1 — Scoring
def formulario_solicitud() -> dict:
    """Formulario con las 17 variables de originación que espera la API."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Crédito solicitado**")
        tipo_credito = st.selectbox("Tipo de crédito", [4, 6, 7, 9, 10, 68], index=2)
        capital_prestado = st.number_input(
            "Capital prestado", min_value=100_000, value=2_000_000, step=100_000
        )
        plazo_meses = st.number_input("Plazo (meses)", min_value=1, max_value=120, value=12)
        cuota_pactada = st.number_input(
            "Cuota pactada", min_value=10_000, value=200_000, step=10_000
        )
        fecha_prestamo = st.date_input("Fecha de solicitud", value=date.today())

    with col2:
        st.markdown("**Perfil del cliente**")
        edad_cliente = st.number_input("Edad", min_value=18, max_value=100, value=38)
        tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente"])
        salario_cliente = st.number_input(
            "Salario declarado", min_value=0, value=3_000_000, step=100_000
        )
        total_otros_prestamos = st.number_input(
            "Total otros préstamos", min_value=0, value=500_000, step=100_000
        )
        tendencia_ingresos = st.selectbox(
            "Tendencia de ingresos", ["Creciente", "Estable", "Decreciente"], index=1
        )

    with col3:
        st.markdown("**Información del buró (DataCrédito)**")
        sin_buro = st.checkbox(
            "Sin información del buró",
            help=(
                "El modelo trata la ausencia de datos del buró como una señal "
                "propia: no tener historial es en sí mismo un indicador de riesgo."
            ),
        )
        puntaje_datacredito = st.number_input(
            "Puntaje DataCrédito", min_value=0, max_value=1000, value=650,
            disabled=sin_buro,
        )
        promedio_ingresos_datacredito = st.number_input(
            "Ingreso promedio reportado", min_value=0, value=1_500_000, step=100_000,
            disabled=sin_buro,
        )
        cant_creditosvigentes = st.number_input("Créditos vigentes", min_value=0, value=2)
        huella_consulta = st.number_input("Huella de consulta", min_value=0, value=3)
        creditos_sectorFinanciero = st.number_input("Créditos sector financiero", min_value=0, value=2)
        creditos_sectorCooperativo = st.number_input("Créditos sector cooperativo", min_value=0, value=0)
        creditos_sectorReal = st.number_input("Créditos sector real", min_value=0, value=0)

    return {
        "tipo_credito": tipo_credito,
        "fecha_prestamo": datetime.combine(fecha_prestamo, datetime.min.time()).isoformat(),
        "capital_prestado": float(capital_prestado),
        "plazo_meses": int(plazo_meses),
        "edad_cliente": int(edad_cliente),
        "tipo_laboral": tipo_laboral,
        "salario_cliente": float(salario_cliente),
        "total_otros_prestamos": float(total_otros_prestamos),
        "cuota_pactada": float(cuota_pactada),
        "cant_creditosvigentes": int(cant_creditosvigentes),
        "huella_consulta": int(huella_consulta),
        "creditos_sectorFinanciero": int(creditos_sectorFinanciero),
        "creditos_sectorCooperativo": int(creditos_sectorCooperativo),
        "creditos_sectorReal": int(creditos_sectorReal),
        "tendencia_ingresos": tendencia_ingresos,
        "puntaje_datacredito": None if sin_buro else float(puntaje_datacredito),
        "promedio_ingresos_datacredito": (
            None if sin_buro else float(promedio_ingresos_datacredito)
        ),
    }


def mostrar_resultado(resultado: dict):
    """Presenta la predicción con su banda de riesgo."""
    probabilidad = resultado["probabilidad_mora"]
    banda = resultado["banda_riesgo"]
    umbral = resultado["umbral_aplicado"]

    colores = {"bajo": "#2e7d32", "medio": "#f9a825", "alto": "#c62828"}
    etiquetas = {
        "bajo": "Riesgo bajo",
        "medio": "Riesgo medio — revisión manual sugerida",
        "alto": "Riesgo alto",
    }

    st.markdown(
        f"""
        <div style="background:{colores[banda]};color:white;padding:1.2rem;
                    border-radius:8px;text-align:center;">
            <div style="font-size:1.3rem;font-weight:600;">{etiquetas[banda]}</div>
            <div style="font-size:2.4rem;font-weight:700;">{probabilidad:.1%}</div>
            <div style="opacity:.85;">probabilidad de no pago</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Decisión del modelo", "Mora" if resultado["prediccion"] else "Pago")
    col2.metric("Umbral aplicado", f"{umbral:.4f}")

    st.caption(
        f"La decisión se toma en el umbral {umbral:.4f}, optimizado por F1 con "
        "predicciones fuera de fold, y no en 0.5. El modelo es una señal de "
        "apoyo (PR-AUC 0.137 en test), no un decisor autónomo."
    )


def pantalla_scoring():
    st.title("Scoring de solicitudes de crédito")

    info = consultar_api("/modelo")
    if info is None:
        st.error(
            f"No se pudo contactar la API en `{API_URL}`. Levántala con:\n\n"
            "```bash\ncd src\nuvicorn model_deploy:app --reload\n```"
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo en producción", info["nombre"])
    col2.metric("Umbral de decisión", f"{info['umbral_decision']:.4f}")
    col3.metric("PR-AUC en test", f"{info['metricas_test']['pr_auc']:.4f}")

    st.divider()

    pestana_individual, pestana_lote = st.tabs(["Solicitud individual", "Carga masiva"])

    with pestana_individual:
        with st.form("solicitud"):
            solicitud = formulario_solicitud()
            enviar = st.form_submit_button("Evaluar riesgo", type="primary")

        if enviar:
            try:
                respuesta = requests.post(
                    f"{API_URL}/predecir", json=solicitud, timeout=15
                )
                respuesta.raise_for_status()
                mostrar_resultado(respuesta.json())
            except requests.HTTPError:
                st.error(f"La API rechazó la solicitud: {respuesta.text}")
            except requests.RequestException as error:
                st.error(f"Error de conexión con la API: {error}")

    with pestana_lote:
        st.markdown(
            "Sube un CSV con una fila por solicitud y las 17 columnas que espera "
            "la API. Se envía en una sola llamada a `/predecir-lote`."
        )
        archivo = st.file_uploader("Archivo CSV", type="csv")

        if archivo is not None:
            df = pd.read_csv(archivo)
            st.write(f"{len(df)} solicitudes leídas.")

            if st.button("Evaluar lote", type="primary"):
                # `astype(object)` es imprescindible: sobre una columna float,
                # `where(..., None)` vuelve a convertir el None en NaN, y NaN no
                # es JSON válido, así que el envío falla. Pasando la columna a
                # object el None sobrevive y la API lo recibe como null. Afecta
                # a los CSV con datos del buró faltantes, que son ~27% del real.
                registros = df.astype(object).where(pd.notna(df), None).to_dict("records")
                try:
                    respuesta = requests.post(
                        f"{API_URL}/predecir-lote",
                        json={"solicitudes": registros},
                        timeout=120,
                    )
                    respuesta.raise_for_status()
                    resultados = pd.DataFrame(respuesta.json())
                    salida = pd.concat([df, resultados], axis=1)

                    st.dataframe(salida, width="stretch")
                    st.download_button(
                        "Descargar resultados",
                        salida.to_csv(index=False).encode(),
                        "resultados_scoring.csv",
                        "text/csv",
                    )

                    marcados = int(resultados["prediccion"].sum())
                    st.info(
                        f"{marcados} de {len(resultados)} solicitudes "
                        f"({marcados / len(resultados):.1%}) quedan marcadas como "
                        "riesgo de mora."
                    )
                except requests.HTTPError:
                    st.error(f"La API rechazó el lote: {respuesta.text}")
                except requests.RequestException as error:
                    st.error(f"Error de conexión con la API: {error}")


# Pantalla 2 — Monitoreo de drift
def panel_semaforo(metricas: pd.DataFrame):
    """Indicadores agregados del estado de la población."""
    conteo = metricas["severidad"].value_counts()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🔴 Drift severo", int(conteo.get("rojo", 0)))
    col2.metric("🟡 Drift moderado", int(conteo.get("amarillo", 0)))
    col3.metric("🟢 Estables", int(conteo.get("verde", 0)))
    col4.metric("PSI máximo", f"{metricas['psi'].max():.4f}")


def tabla_metricas(metricas: pd.DataFrame):
    """Tabla de drift por variable con barras de riesgo."""
    vista = metricas.copy()
    vista["estado"] = vista["severidad"].map(EMOJI_SEMAFORO)

    columnas = ["estado", "variable", "tipo", "psi", "ks", "jensen_shannon", "p_valor", "prueba"]
    columnas = [c for c in columnas if c in vista.columns]

    st.dataframe(
        vista[columnas],
        width="stretch",
        hide_index=True,
        column_config={
            "psi": st.column_config.ProgressColumn(
                "PSI",
                help=f"< {UMBRAL_PSI_MODERADO} estable · "
                     f"{UMBRAL_PSI_MODERADO}–{UMBRAL_PSI_CRITICO} moderado · "
                     f"> {UMBRAL_PSI_CRITICO} severo",
                min_value=0.0,
                max_value=max(float(vista["psi"].max()), UMBRAL_PSI_CRITICO),
                format="%.4f",
            ),
            "ks": st.column_config.NumberColumn("KS", format="%.4f"),
            "jensen_shannon": st.column_config.NumberColumn("Jensen-Shannon", format="%.4f"),
            "p_valor": st.column_config.NumberColumn("p-valor", format="%.2e"),
        },
    )


def grafico_distribuciones(predicciones: pd.DataFrame, variable: str):
    """
    Histograma superpuesto de la referencia contra la población actual.

    Las frecuencias se normalizan a proporción porque las dos muestras tienen
    tamaños muy distintos (8.378 contra 2.385): con conteos absolutos la
    referencia taparía por completo a la actual y no se vería el cambio de
    forma, que es justo lo que interesa.
    """
    referencia = predicciones[predicciones["fecha_prestamo"] < FECHA_CORTE]
    actual = predicciones[predicciones["fecha_prestamo"] >= FECHA_CORTE]

    if variable in VARIABLES_CATEGORICAS:
        serie_ref, serie_act = referencia[variable], actual[variable]
        if variable == "tendencia_ingresos":
            serie_ref, serie_act = normalizar_tendencia(serie_ref), normalizar_tendencia(serie_act)

        datos = pd.concat(
            [
                serie_ref.astype(str).value_counts(normalize=True).rename("proporcion")
                .reset_index().assign(poblacion="Referencia"),
                serie_act.astype(str).value_counts(normalize=True).rename("proporcion")
                .reset_index().assign(poblacion="Actual"),
            ]
        ).rename(columns={variable: "categoria", "index": "categoria"})

        grafico = (
            alt.Chart(datos)
            .mark_bar()
            .encode(
                x=alt.X("categoria:N", title=variable),
                y=alt.Y("proporcion:Q", title="Proporción", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "poblacion:N",
                    title="Población",
                    scale=alt.Scale(
                        domain=["Referencia", "Actual"], range=["#5c6bc0", "#ef6c00"]
                    ),
                ),
                xOffset="poblacion:N",
                tooltip=["categoria", "poblacion", alt.Tooltip("proporcion:Q", format=".2%")],
            )
        )
    else:
        datos = pd.concat(
            [
                referencia[[variable]].dropna().assign(poblacion="Referencia"),
                actual[[variable]].dropna().assign(poblacion="Actual"),
            ]
        )

        grafico = (
            alt.Chart(datos)
            .transform_joinaggregate(total="count()", groupby=["poblacion"])
            .transform_bin("valor", field=variable, bin=alt.Bin(maxbins=40))
            .transform_aggregate(n="count()", groupby=["valor", "poblacion", "total"])
            .transform_calculate(proporcion="datum.n / datum.total")
            .mark_bar(opacity=0.6)
            .encode(
                x=alt.X("valor:Q", title=variable),
                y=alt.Y("proporcion:Q", stack=None, title="Proporción",
                        axis=alt.Axis(format="%")),
                color=alt.Color(
                    "poblacion:N",
                    title="Población",
                    scale=alt.Scale(
                        domain=["Referencia", "Actual"], range=["#5c6bc0", "#ef6c00"]
                    ),
                ),
                tooltip=["poblacion", alt.Tooltip("proporcion:Q", format=".2%")],
            )
        )

    st.altair_chart(grafico.properties(height=320), width="stretch")


def grafico_evolucion(temporal: pd.DataFrame, variables: list[str]):
    """Evolución del PSI por periodo, con las bandas de alerta de fondo."""
    datos = temporal[temporal["variable"].isin(variables)]

    if datos.empty:
        st.info("Selecciona al menos una variable.")
        return

    lineas = (
        alt.Chart(datos)
        .mark_line(point=True)
        .encode(
            x=alt.X("periodo:N", title="Periodo"),
            y=alt.Y("psi:Q", title="PSI"),
            color=alt.Color("variable:N", title="Variable"),
            tooltip=["periodo", "variable", alt.Tooltip("psi:Q", format=".4f"),
                     "n_registros"],
        )
    )

    umbrales = (
        alt.Chart(
            pd.DataFrame(
                {
                    "umbral": [UMBRAL_PSI_MODERADO, UMBRAL_PSI_CRITICO],
                    "etiqueta": ["Moderado", "Severo"],
                }
            )
        )
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="umbral:Q",
            color=alt.Color(
                "etiqueta:N",
                title="Umbral",
                scale=alt.Scale(domain=["Moderado", "Severo"],
                                range=["#f9a825", "#c62828"]),
            ),
        )
    )

    st.altair_chart((lineas + umbrales).properties(height=360), width="stretch")


def grafico_mora(resumen: pd.DataFrame):
    """Tasa de mora real contra la predicha por periodo."""
    datos = resumen.melt(
        id_vars=["periodo", "n_registros"],
        value_vars=["mora_real", "mora_predicha"],
        var_name="serie",
        value_name="tasa",
    )
    datos["serie"] = datos["serie"].map(
        {"mora_real": "Mora observada", "mora_predicha": "Mora predicha"}
    )

    grafico = (
        alt.Chart(datos)
        .mark_line(point=True)
        .encode(
            x=alt.X("periodo:N", title="Periodo"),
            y=alt.Y("tasa:Q", title="Tasa de mora", axis=alt.Axis(format="%")),
            color=alt.Color(
                "serie:N",
                title="",
                scale=alt.Scale(
                    domain=["Mora observada", "Mora predicha"],
                    range=["#2e7d32", "#ef6c00"],
                ),
            ),
            tooltip=["periodo", "serie", alt.Tooltip("tasa:Q", format=".2%"),
                     "n_registros"],
        )
    )

    st.altair_chart(grafico.properties(height=320), width="stretch")


def pantalla_monitoreo():
    st.title("Monitoreo de data drift")

    metricas = cargar_metricas()
    predicciones = cargar_predicciones()
    temporal = cargar_drift_temporal()

    if metricas is None or predicciones is None or temporal is None:
        aviso_faltan_datos()
        return

    referencia = predicciones[predicciones["fecha_prestamo"] < FECHA_CORTE]
    actual = predicciones[predicciones["fecha_prestamo"] >= FECHA_CORTE]

    st.caption(
        f"Referencia: {len(referencia):,} créditos hasta {FECHA_CORTE} · "
        f"Población actual: {len(actual):,} créditos desde {FECHA_CORTE}"
    )

    panel_semaforo(metricas)
    st.divider()

    st.subheader("Métricas de drift por variable")
    tabla_metricas(metricas)

    st.divider()
    st.subheader("Distribución histórica vs. actual")
    variable = st.selectbox(
        "Variable",
        metricas["variable"].tolist(),
        help="Ordenadas de mayor a menor drift.",
    )
    grafico_distribuciones(predicciones, variable)

    st.divider()
    st.subheader("Evolución del drift en el tiempo")
    st.caption(
        "Cada periodo se compara contra la misma referencia. Los periodos con "
        f"menos de {MIN_MUESTRA_PSI} observaciones se excluyen para no reportar "
        "drift que en realidad es ruido de muestreo."
    )
    por_defecto = metricas.head(3)["variable"].tolist()
    seleccion = st.multiselect(
        "Variables a graficar",
        sorted(temporal["variable"].unique()),
        default=[v for v in por_defecto if v in set(temporal["variable"])],
    )
    grafico_evolucion(temporal, seleccion)

    st.divider()
    st.subheader("Mora observada vs. predicha")
    resumen = resumen_por_periodo(predicciones)
    grafico_mora(resumen)

    st.divider()
    st.subheader("Alertas y recomendaciones")

    patrones = detectar_patrones_temporales(temporal)
    recomendaciones = generar_recomendaciones(metricas, temporal, resumen)

    for texto in recomendaciones:
        if texto.startswith("CRÍTICO"):
            st.error(texto)
        elif texto.startswith(("ATENCIÓN", "TENDENCIA", "CAMBIO ABRUPTO", "CALIBRACIÓN")):
            st.warning(texto)
        else:
            st.info(texto)

    if patrones:
        with st.expander("Detalle del análisis temporal"):
            st.dataframe(
                pd.DataFrame(patrones, columns=["variable", "patrón", "detalle"]),
                width="stretch",
                hide_index=True,
            )


# Navegación
def main():
    st.sidebar.title("🏦 Riesgo Crediticio")
    st.sidebar.caption("Proyecto Integrador Módulo 5 — PIM5")

    pantalla = st.sidebar.radio(
        "Pantalla", ["Scoring de solicitudes", "Monitoreo de data drift"]
    )

    st.sidebar.divider()
    salud = consultar_api("/salud")
    if salud is not None:
        st.sidebar.success(f"API conectada · {salud.get('modelo')}")
    else:
        st.sidebar.error("API no disponible")
    st.sidebar.caption(f"`{API_URL}`")

    if pantalla == "Scoring de solicitudes":
        pantalla_scoring()
    else:
        pantalla_monitoreo()


if __name__ == "__main__":
    main()
