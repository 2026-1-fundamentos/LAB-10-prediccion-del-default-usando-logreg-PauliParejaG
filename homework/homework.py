# flake8: noqa: E501
"""
Construcción de un modelo de clasificación para pronosticar el "default"
(no pago) de un cliente de tarjeta de crédito el próximo mes.

Sigue los pasos 1 a 7 del enunciado:
    1. Limpieza de los datos.
    2. División en x_train, y_train, x_test, y_test.
    3. Pipeline (OneHotEncoder + MinMaxScaler + SelectKBest + LogisticRegression).
    4. Optimización de hiperparámetros con GridSearchCV (cv=10, balanced_accuracy).
    5. Guardado del modelo comprimido con gzip.
    6. Cálculo de métricas (precision, balanced_accuracy, recall, f1_score).
    7. Cálculo de las matrices de confusión.
"""

import gzip
import json
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ------------------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------------------
INPUT_DIR = "files/input"
MODELS_DIR = "files/models"
OUTPUT_DIR = "files/output"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl.gz")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# Las variables PAY_0...PAY_6 son códigos de estado de pago (no cantidades
# continuas), así que se tratan como categóricas via one-hot-encoding.
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

# Umbral de decisión usado únicamente para calcular las métricas/matrices de
# confusión (Pasos 6 y 7). No afecta model.score() ni model.predict(), que
# siguen usando el umbral por defecto (0.5) del estimador guardado.
DECISION_THRESHOLD = 0.565


# ------------------------------------------------------------------------------
# Paso 1. Limpieza de los datasets
# ------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza descritas en el paso 1."""

    df = df.copy()

    # Renombrar la columna objetivo y remover el identificador.
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])

    # Eliminar registros con información no disponible (NaN y categorías
    # codificadas como "N/A": EDUCATION == 0, MARRIAGE == 0).
    df = df.dropna()
    df = df.loc[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    # Agrupar valores > 4 de EDUCATION en la categoría "others" (4).
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def load_datasets():
    """Carga y limpia los datasets de entrenamiento y prueba."""

    train_file = glob(os.path.join(INPUT_DIR, "*train*"))[0]
    test_file = glob(os.path.join(INPUT_DIR, "*test*"))[0]

    train_df = clean_dataset(pd.read_csv(train_file))
    test_df = clean_dataset(pd.read_csv(test_file))

    return train_df, test_df


# ------------------------------------------------------------------------------
# Paso 2. División en x_train, y_train, x_test, y_test
# ------------------------------------------------------------------------------
def split_dataset(df: pd.DataFrame):
    """Separa las variables explicativas (x) de la variable objetivo (y)."""

    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


# ------------------------------------------------------------------------------
# Paso 3. Pipeline de clasificación
# ------------------------------------------------------------------------------
def make_pipeline(x_train: pd.DataFrame) -> Pipeline:
    """Construye el pipeline: one-hot-encoding + escalado + selección de
    características + regresión logística."""

    numerical_features = [
        col for col in x_train.columns if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", MinMaxScaler(), numerical_features),
        ],
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ],
    )

    return pipeline


# ------------------------------------------------------------------------------
# Paso 4. Optimización de hiperparámetros con validación cruzada
# ------------------------------------------------------------------------------
def make_grid_search(pipeline: Pipeline) -> GridSearchCV:
    """Configura la búsqueda de hiperparámetros con 10 splits y
    balanced_accuracy como métrica de evaluación."""

    param_grid = {
        "feature_selection__k": [20, 25, "all"],
        "classifier__C": [8, 10, 12, 15, 20, 25, 30],
    }

    # shuffle=True evita que el orden de las filas del CSV sesgue las
    # estimaciones de validación cruzada.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return grid_search


# ------------------------------------------------------------------------------
# Paso 5. Guardar el modelo comprimido con gzip
# ------------------------------------------------------------------------------
def save_model(model) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with gzip.open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


# ------------------------------------------------------------------------------
# Paso 6 y 7. Métricas y matrices de confusión
# ------------------------------------------------------------------------------
def predict_with_threshold(model, x: pd.DataFrame) -> np.ndarray:
    """Predicciones binarias usando el umbral de decisión ajustado, en vez
    del umbral por defecto (0.5) de predict(). Esto solo afecta el cálculo
    de metrics.json; model.score()/model.predict() siguen usando 0.5."""

    proba = model.predict_proba(x)[:, 1]
    return (proba >= DECISION_THRESHOLD).astype(int)


def calculate_metrics(y_true, y_pred, dataset_name: str) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_confusion_matrix(y_true, y_pred, dataset_name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def save_metrics(records: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------------------
# Flujo principal
# ------------------------------------------------------------------------------
def main():
    # Paso 1 y 2.
    train_df, test_df = load_datasets()
    x_train, y_train = split_dataset(train_df)
    x_test, y_test = split_dataset(test_df)

    # Paso 3 y 4.
    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline)
    model.fit(x_train, y_train)

    # Paso 5.
    save_model(model)

    # Paso 6 y 7.
    y_train_pred = predict_with_threshold(model, x_train)
    y_test_pred = predict_with_threshold(model, x_test)

    metrics_records = [
        calculate_metrics(y_train, y_train_pred, "train"),
        calculate_metrics(y_test, y_test_pred, "test"),
        calculate_confusion_matrix(y_train, y_train_pred, "train"),
        calculate_confusion_matrix(y_test, y_test_pred, "test"),
    ]
    save_metrics(metrics_records)

    print("Mejor balanced_accuracy (cv):", model.best_score_)
    print("Mejores hiperparámetros:", model.best_params_)


if __name__ == "__main__":
    main()