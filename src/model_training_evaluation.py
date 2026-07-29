"""
model_training_evaluation.py

Entrenamiento y evaluación de modelos supervisados para el modelo de riesgo
crediticio (PIM5).

Consume el set ya transformado por `ft_engineering.py`, entrena varios
modelos candidatos, los compara con validación cruzada estratificada y
selecciona el de mejor performance.

DEFINICIÓN DE LA CLASE POSITIVA
-------------------------------
La variable original es `Pago_atiempo` (1 = paga a tiempo, 0 = mora), donde
la clase minoritaria es el 0 (511 de 10.763 registros, 4.7%).

Se invierte el target a `y_mora = 1 - Pago_atiempo`, de modo que la clase
positiva sea el cliente que NO paga. Esto no cambia el modelo, cambia cómo
se leen las métricas: con esta convención, "recall" significa "qué porcentaje
de los morosos logramos detectar", que es exactamente la pregunta de negocio.
Sin invertir, el recall mediría qué tan bien detectamos a los buenos
pagadores, que con 95% de la muestra es trivial y no aporta nada.

MÉTRICA DE SELECCIÓN
--------------------
Se usa PR-AUC (average precision) como criterio principal, no ROC-AUC ni
accuracy:
  - Accuracy es inútil aquí: un modelo que prediga "todos pagan" acierta 95%.
  - ROC-AUC es optimista con clases desbalanceadas, porque la tasa de falsos
    positivos se diluye contra los 10.252 negativos.
  - PR-AUC solo mira precision y recall de la clase minoritaria, que es la
    que importa.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para poder guardar los gráficos
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from xgboost import XGBClassifier

from carga_datos import cargarDatos
from ft_engineering import dividir_y_transformar

RANDOM_STATE = 42

# Métrica que decide qué modelo gana (ver docstring del módulo).
METRICA_SELECCION = "PR-AUC"


def construir_modelos(y_train):
    """
    Define los modelos candidatos.

    Todos manejan explícitamente el desbalance de clases (~4.7% de morosos):
    sin esto, los modelos aprenden a predecir siempre "paga a tiempo", que
    maximiza accuracy pero tiene recall 0 sobre la clase que interesa.

    - `class_weight="balanced"` en los modelos de scikit-learn: pondera cada
      clase de forma inversamente proporcional a su frecuencia.
    - `scale_pos_weight` en XGBoost: el equivalente, calculado como la razón
      entre negativos y positivos del set de entrenamiento.

    `DummyClassifier` no es un candidato real: es la línea base honesta contra
    la cual se mide si los demás aportan algo. Si un modelo no le gana, no
    está aprendiendo nada.

    La regresión logística va dentro de un `Pipeline` con `StandardScaler`
    porque es sensible a la escala de las variables (aquí conviven salarios
    en millones con ratios entre 0 y 3). Los modelos de árbol no lo necesitan:
    parten por umbrales, no por distancias.
    """
    razon_desbalance = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "Baseline (Dummy)": DummyClassifier(
            strategy="stratified", random_state=RANDOM_STATE
        ),
        "Regresión Logística": Pipeline(
            [
                ("escalador", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=razon_desbalance,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def comparar_modelos(modelos, X_train, y_train, n_splits=5):
    """
    Compara los modelos con validación cruzada estratificada sobre TRAIN.

    La validación cruzada se hace solo sobre train: el set de test se reserva
    intacto para una única evaluación final. Si se usara test para elegir el
    modelo, las métricas finales estarían infladas porque el test habría
    participado en la decisión.

    Se estratifica (`StratifiedKFold`) para que cada fold conserve la
    proporción de morosos. Con solo ~4.7% de positivos, un split aleatorio
    simple podría dejar folds con muy pocos casos de mora y volver las
    métricas inestables.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    metricas = {
        "PR-AUC": "average_precision",
        "ROC-AUC": "roc_auc",
        "Recall": "recall",
        "Precision": "precision",
        "F1": "f1",
    }

    resultados = []
    for nombre, modelo in modelos.items():
        scores = cross_validate(
            modelo, X_train, y_train, cv=cv, scoring=metricas, n_jobs=-1
        )
        # cross_validate nombra los resultados con las claves del diccionario
        # de scoring ("PR-AUC"), no con el nombre interno del scorer.
        fila = {"Modelo": nombre}
        for etiqueta in metricas:
            fila[etiqueta] = scores[f"test_{etiqueta}"].mean()
        fila[f"{METRICA_SELECCION} (std)"] = scores[f"test_{METRICA_SELECCION}"].std()
        resultados.append(fila)

    tabla = pd.DataFrame(resultados).sort_values(METRICA_SELECCION, ascending=False)
    return tabla.reset_index(drop=True)


def elegir_umbral(y_true, y_proba):
    """
    Elige el umbral de decisión que maximiza F1 sobre la clase "mora".

    Por defecto los clasificadores cortan en 0.5, pero ese valor no tiene
    nada de especial: es solo el punto medio de la probabilidad. Con clases
    desbalanceadas casi siempre existe un umbral mejor, y elegirlo es parte
    de ajustar el modelo al problema.

    Se optimiza F1 porque equilibra los dos errores que le cuestan a la
    entidad: aprobar un crédito que no se paga (falso negativo) y rechazar
    a un cliente bueno (falso positivo). Si el negocio definiera que uno de
    los dos es claramente más caro, aquí se cambiaría el criterio.
    """
    precision, recall, umbrales = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve devuelve un punto más que umbrales; se recorta.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0,
        )

    mejor = int(np.argmax(f1))
    return float(umbrales[mejor]), float(f1[mejor])


def evaluar_en_test(modelo, X_test, y_test, umbral):
    """
    Evaluación final sobre el set de test, una sola vez, con el umbral ya
    ajustado en train.
    """
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    print(f"\nUmbral de decisión aplicado: {umbral:.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, y_proba):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")

    matriz = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(
        pd.DataFrame(
            matriz,
            index=["Real: paga", "Real: mora"],
            columns=["Pred: paga", "Pred: mora"],
        )
    )

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test, y_pred, target_names=["paga a tiempo", "mora"], zero_division=0
        )
    )

    return y_proba


def graficar_curvas(y_test, y_proba, nombre_modelo, ruta_salida):
    """
    Guarda las curvas ROC y Precision-Recall del modelo seleccionado.

    Se incluye la línea base de cada curva para dar contexto: en ROC es la
    diagonal (azar), y en Precision-Recall es la proporción de morosos en el
    set, que con 4.7% deja ver de inmediato cuánto aporta el modelo por
    encima de adivinar.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title(f"Curva ROC — {nombre_modelo}")
    ax_roc.legend()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    tasa_base = y_test.mean()
    ax_pr.plot(
        recall,
        precision,
        label=f"PR-AUC = {average_precision_score(y_test, y_proba):.3f}",
    )
    ax_pr.axhline(
        tasa_base, ls="--", color="gray", label=f"Tasa base = {tasa_base:.3f}"
    )
    ax_pr.set_xlabel("Recall (morosos detectados)")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Curva Precision-Recall — {nombre_modelo}")
    ax_pr.legend()

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=120)
    plt.close(fig)
    print(f"\nCurvas guardadas en: {ruta_salida}")


def importancia_variables(modelo, columnas, top=15):
    """
    Muestra qué variables pesan más en el modelo seleccionado.

    Sirve como control de sanidad del avance de feature engineering: permite
    verificar si los ratios creados aportan y si no quedó ninguna variable
    con una importancia sospechosamente dominante (señal de leakage residual).
    """
    estimador = modelo[-1] if isinstance(modelo, Pipeline) else modelo

    if hasattr(estimador, "feature_importances_"):
        pesos = estimador.feature_importances_
    elif hasattr(estimador, "coef_"):
        pesos = np.abs(estimador.coef_[0])
    else:
        print("\nEl modelo seleccionado no expone importancia de variables.")
        return

    ranking = (
        pd.Series(pesos, index=columnas).sort_values(ascending=False).head(top)
    )
    print(f"\nTop {top} variables más influyentes:")
    print(ranking.to_string())


def experimento_contraste_leakage(df, X_train, X_test, y_train, y_test):
    """
    Experimento de contraste que documenta el costo de haber eliminado las
    variables con data leakage.

    Reentrena el mismo XGBoost agregando de vuelta `puntaje`, la variable con
    correlación 0.923 contra el target que se descartó en `ft_engineering.py`.
    El objetivo no es usar este modelo, sino dejar evidencia de por qué no se
    usa: si el PR-AUC salta a un valor casi perfecto, confirma que esa
    variable contiene el resultado que se quiere predecir y que cualquier
    métrica obtenida con ella sería ficticia.
    """
    puntaje = df["puntaje"]
    X_train_leak = X_train.assign(puntaje=puntaje.loc[X_train.index])
    X_test_leak = X_test.assign(puntaje=puntaje.loc[X_test.index])

    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    modelo.fit(X_train_leak, y_train)
    proba = modelo.predict_proba(X_test_leak)[:, 1]

    print("\n" + "=" * 70)
    print("EXPERIMENTO DE CONTRASTE: qué pasaría si NO elimináramos el leakage")
    print("=" * 70)
    print(f"XGBoost CON `puntaje`  -> PR-AUC: {average_precision_score(y_test, proba):.4f}"
          f" | ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print("Comparar contra el PR-AUC del modelo limpio reportado arriba.")


def main():
    datos = cargarDatos()

    X_train, X_test, y_train, y_test, pipeline_ft = dividir_y_transformar(datos)

    # Ver docstring del módulo: la clase positiva es el cliente que NO paga.
    y_train_mora = 1 - y_train
    y_test_mora = 1 - y_test

    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS — validación cruzada estratificada (5 folds)")
    print("=" * 70)
    print(f"Train: {X_train.shape[0]} registros, {y_train_mora.sum()} morosos "
          f"({y_train_mora.mean():.1%})")

    modelos = construir_modelos(y_train_mora)
    tabla = comparar_modelos(modelos, X_train, y_train_mora)
    print("\n" + tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    nombre_ganador = tabla.loc[0, "Modelo"]
    modelo_ganador = modelos[nombre_ganador]
    print(f"\nModelo seleccionado por {METRICA_SELECCION}: {nombre_ganador}")

    # Se reentrena con TODO el train (la CV solo servía para comparar).
    modelo_ganador.fit(X_train, y_train_mora)

    # El umbral se ajusta sobre train, nunca sobre test: el test debe quedar
    # libre de cualquier decisión tomada durante el entrenamiento.
    #
    # Pero no sirve usar `modelo.predict_proba(X_train)` directamente: el
    # modelo ya vio esas filas y las predice casi perfecto, así que el umbral
    # quedaría ajustado a un rendimiento que no se repite con datos nuevos.
    # Se usan predicciones fuera de fold (`cross_val_predict`): cada fila la
    # predice un modelo que no la tenía en su entrenamiento.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = cross_val_predict(
        modelos[nombre_ganador], X_train, y_train_mora, cv=cv, method="predict_proba"
    )[:, 1]
    umbral, f1_oof = elegir_umbral(y_train_mora, proba_oof)
    print(f"Umbral óptimo (fuera de fold): {umbral:.4f} (F1 estimado: {f1_oof:.4f})")

    print("\n" + "=" * 70)
    print(f"EVALUACIÓN FINAL EN TEST — {nombre_ganador}")
    print("=" * 70)
    proba_test = evaluar_en_test(modelo_ganador, X_test, y_test_mora, umbral)

    importancia_variables(modelo_ganador, X_train.columns)
    graficar_curvas(y_test_mora, proba_test, nombre_ganador, "curvas_evaluacion.png")

    experimento_contraste_leakage(datos, X_train, X_test, y_train_mora, y_test_mora)

    # Se guardan juntos el pipeline de transformación y el modelo: en
    # producción hay que aplicar exactamente las mismas transformaciones que
    # en entrenamiento, y el umbral no es 0.5.
    artefacto = {
        "pipeline_ft": pipeline_ft,
        "modelo": modelo_ganador,
        "umbral": umbral,
        "nombre_modelo": nombre_ganador,
        "columnas": list(X_train.columns),
    }
    joblib.dump(artefacto, "modelo_riesgo_credito.pkl")
    print("\nModelo guardado en: modelo_riesgo_credito.pkl")


if __name__ == "__main__":
    main()
