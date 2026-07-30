# Imagen del modelo de riesgo crediticio (PIM5).
#
# Una sola imagen sirve los dos procesos: la API de FastAPI y la app de
# Streamlit. Comparten el mismo código y el mismo modelo; lo único que cambia
# es el comando de arranque (ver docker-compose.yml). Mantener dos imágenes
# separadas obligaría a construir dos veces las mismas dependencias.
#
# POR QUÉ LA VERSIÓN DE PYTHON ESTÁ FIJADA AL PATCH
# -------------------------------------------------
# `src/modelo_riesgo_credito.pkl` se serializó con Python 3.14.6 y
# scikit-learn 1.9.0. Un pickle de scikit-learn cargado con otra versión de la
# librería emite InconsistentVersionWarning y puede fallar al reconstruir los
# estimadores. Por eso tanto la imagen base como las dependencias de
# requirements-docker.txt están fijadas: el contenedor tiene que reproducir el
# entorno en el que se entrenó el modelo, no "algo parecido".
FROM python:3.14.6-slim

# PYTHONDONTWRITEBYTECODE: no escribir .pyc, que solo engordan la imagen.
# PYTHONUNBUFFERED: sin esto los logs de uvicorn se quedan en el buffer de
#   stdout y `docker logs` no muestra nada hasta que el proceso termina.
# PYTHONPATH: el pickle guarda los transformadores personalizados como una
#   referencia al módulo `ft_engineering`, no su código, así que joblib.load
#   necesita poder importarlo. Ponerlo en el PYTHONPATH resuelve esto sin
#   depender de que el directorio de trabajo sea /app/src.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Las dependencias se instalan ANTES de copiar el código para aprovechar la
# caché de capas de Docker: cambiar un .py no obliga a reinstalar pandas.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# El código y los artefactos ya generados (modelo entrenado y tablas de
# monitoreo), para que la app funcione en el primer arranque sin tener que
# ejecutar nada antes.
COPY src/ ./src/

# El dataset se incluye para que model_monitoring.py pueda regenerar la tabla
# de predicciones dentro del contenedor.
COPY Base_de_datos.xlsx .

# Usuario sin privilegios: un proceso que solo sirve predicciones no tiene
# ninguna razón para correr como root.
#
# El directorio del registro se crea AQUÍ, en la imagen, y no solo como punto
# de montaje en docker-compose. Docker inicializa un volumen nombrado vacío
# copiando el contenido y los permisos del directorio que exista en la imagen;
# si el directorio no existe, lo crea como root y el proceso (appuser) no puede
# escribir. Eso hacía que la API registrara
# "[registro] No se pudo escribir la predicción: [Errno 13] Permission denied"
# y perdiera el log en silencio, porque `registrar_predicciones` captura la
# excepción para no tumbar la predicción.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/registro \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/src

EXPOSE 8000

# El healthcheck usa el endpoint /salud, que confirma que el modelo quedó
# cargado en memoria y no solo que el proceso está vivo. docker-compose lo usa
# para no arrancar Streamlit antes de que la API esté lista.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/salud', timeout=4).status == 200 else 1)"

# Bind a 0.0.0.0 y no a 127.0.0.1: el default de model_deploy.py es localhost,
# que dentro del contenedor sería inalcanzable desde el host.
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
