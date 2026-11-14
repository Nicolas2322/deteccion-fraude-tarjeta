"""Construcción de los modelos que el estudio compara.

La asignatura fija la comparación entre redes neuronales tabulares y Gradient
Boosting. Se añade una regresión logística como línea base mínima: cualquier
modelo que no la supere no justifica su complejidad.

LightGBM y PyTorch se importan dentro de la función que los necesita y no en la
cabecera del módulo. Así el resto del pipeline —datos, preprocesamiento, umbral
y métricas— se puede instalar, probar y ejecutar sin dependencias pesadas, y la
integración continua se mantiene ligera.
"""

from __future__ import annotations

from typing import Any

#: Modelos disponibles y familia a la que pertenecen.
CATALOGO = {
    "logistica": "lineal",
    "lightgbm": "arboles",
    "xgboost": "arboles",
    "mlp": "red",
    "ft_transformer": "red",
}


class ModeloNoDisponible(ImportError):
    """El modelo existe en el catálogo pero falta la biblioteca que lo implementa."""


def _logistica(semilla: int, **kw: Any):
    from sklearn.linear_model import LogisticRegression

    parametros = dict(max_iter=2000, class_weight="balanced", random_state=semilla)
    parametros.update(kw)
    return LogisticRegression(**parametros)


def _mlp(semilla: int, **kw: Any):
    from sklearn.neural_network import MLPClassifier

    parametros = dict(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
                      learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                      n_iter_no_change=8, random_state=semilla)
    parametros.update(kw)
    return MLPClassifier(**parametros)


def _lightgbm(semilla: int, **kw: Any):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ModeloNoDisponible("Falta lightgbm: pip install lightgbm") from exc

    parametros = dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, random_state=semilla,
                      is_unbalance=True, verbose=-1)
    parametros.update(kw)
    return LGBMClassifier(**parametros)


def _xgboost(semilla: int, **kw: Any):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ModeloNoDisponible("Falta xgboost: pip install xgboost") from exc

    parametros = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=semilla,
                      eval_metric="aucpr", tree_method="hist")
    parametros.update(kw)
    return XGBClassifier(**parametros)


def _ft_transformer(semilla: int, **kw: Any):  # pragma: no cover - fase siguiente
    raise ModeloNoDisponible(
        "FT-Transformer se incorpora en la fase de selección; requiere PyTorch. "
        "La interfaz ya está prevista para que el resto del pipeline no cambie."
    )


_FABRICAS = {
    "logistica": _logistica,
    "mlp": _mlp,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "ft_transformer": _ft_transformer,
}


def crear(nombre: str, semilla: int = 42, **kw: Any):
    """Devuelve un clasificador con la interfaz de scikit-learn.

    Todos exponen `fit` y `predict_proba`, de modo que el resto del pipeline
    los trata de forma idéntica y la comparación entre familias no depende de
    ninguna rama condicional repartida por el código.
    """
    if nombre not in _FABRICAS:
        raise KeyError(f"Modelo desconocido: {nombre}. Disponibles: {sorted(_FABRICAS)}")
    return _FABRICAS[nombre](semilla, **kw)


def familia(nombre: str) -> str:
    """Familia a la que pertenece un modelo del catálogo."""
    return CATALOGO[nombre]
