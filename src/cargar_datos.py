import os
import pandas as pd

def cargarDatos():
    # 1. Ruta absoluta del directorio donde está este archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # 2. Subir un nivel para llegar a la carpeta donde está la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    # 3. Construir la ruta completa al Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")

    # 4. leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    print(df)
    return df

if __name__ == "__main__":
    # Si se ejecuta este script directamente, carga los datos y muestra las primeras filas

    datos = cargarDatos()
    print(datos.head())
    print(datos.columns)