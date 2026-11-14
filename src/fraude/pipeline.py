"""Orquestación del pipeline completo.

Reúne las capas anteriores en el recorrido que describe el informe: partición,
preprocesamiento, entrenamiento, evaluación y elección del umbral. Se escribe
como función y no como script para que pueda probarse de extremo a extremo con
datos sintéticos, sin descargar el archivo original.

Los dos protocolos de validación se ejecutan en paralelo y se reportan por
separado, porque la distancia entre ambos mide el optimismo que introduce una
partición aleatoria sobre datos que en realidad son un flujo temporal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .datos import particion_temporal, pliegues_estratificados, resumir
from .metricas import informe
from .modelos import crear
from .preprocesamiento import preparar
from .umbral import Decision, curva_sensibilidad, umbral_optimo


@dataclass
class Resultado:
    """Todo lo que una ejecución del pipeline produce."""

    modelo: str
    costo_alerta_falsa: float
    resumen_datos: dict[str, Any]
    decision: dict[str, Any]
    validacion_cruzada: dict[str, float]
    prueba_temporal: dict[str, float]
    sensibilidad: list[tuple[float, float, float]] = field(default_factory=list)

    def a_diccionario(self) -> dict[str, Any]:
        return asdict(self)


def predicciones_fuera_de_particion(df: pd.DataFrame, modelo: str, n_pliegues: int,
                                    semilla: int) -> np.ndarray:
    """Devuelve una probabilidad por transacción, estimada siempre por un modelo
    que no la vio durante su entrenamiento."""
    y = df["Class"].to_numpy(dtype=int)
    fuera = np.zeros(len(df), dtype=float)
    for indice_ent, indice_val in pliegues_estratificados(y, n_pliegues, semilla):
        X_ent, y_ent, X_val, _, _ = preparar(df.iloc[indice_ent], df.iloc[indice_val])
        clasificador = crear(modelo, semilla=semilla).fit(X_ent, y_ent)
        fuera[indice_val] = clasificador.predict_proba(X_val)[:, 1]
    return fuera


def ejecutar(df: pd.DataFrame, modelo: str = "logistica", costo_alerta_falsa: float = 5.0,
             n_pliegues: int = 5, fraccion_prueba: float = 0.2, semilla: int = 42,
             rejilla_costos: np.ndarray | None = None) -> Resultado:
    """Recorre el pipeline entero y devuelve las métricas de ambos protocolos.

    El umbral se elige **sobre las predicciones fuera de partición** del tramo de
    desarrollo y después se aplica sin modificar al tramo de prueba. Elegirlo
    sobre la prueba produciría una estimación optimista de algo que en producción
    hay que fijar de antemano.
    """
    desarrollo, prueba = particion_temporal(df, fraccion_prueba)

    fuera = predicciones_fuera_de_particion(desarrollo, modelo, n_pliegues, semilla)
    importes_desarrollo = desarrollo["Amount"].to_numpy(dtype=float)
    y_desarrollo = desarrollo["Class"].to_numpy(dtype=int)
    decision: Decision = umbral_optimo(fuera, y_desarrollo, importes_desarrollo,
                                       costo_alerta_falsa)

    X_des, y_des, X_pru, y_pru, importes_prueba = preparar(desarrollo, prueba)
    final = crear(modelo, semilla=semilla).fit(X_des, y_des)
    p_prueba = final.predict_proba(X_pru)[:, 1]

    if rejilla_costos is None:
        rejilla_costos = np.array([1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0])

    return Resultado(
        modelo=modelo,
        costo_alerta_falsa=float(costo_alerta_falsa),
        resumen_datos={"desarrollo": str(resumir(desarrollo)), "prueba": str(resumir(prueba))},
        decision=asdict(decision),
        validacion_cruzada=informe(y_desarrollo, fuera, importes_desarrollo,
                                   decision.umbral, costo_alerta_falsa),
        prueba_temporal=informe(y_pru, p_prueba, importes_prueba,
                                decision.umbral, costo_alerta_falsa),
        sensibilidad=curva_sensibilidad(fuera, y_desarrollo, importes_desarrollo,
                                        rejilla_costos),
    )
