"""Transformación de las variables y escalado.

Solo dos de las treinta columnas necesitan tratamiento. Las V1 a V28 llegan ya
centradas y escaladas por el análisis de componentes principales que aplicaron
los autores del conjunto, de modo que tocarlas sería redundante. El trabajo está
en `Time` y `Amount`, y en impedir que la validación filtre información al
entrenamiento a través de los parámetros de escalado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .datos import COMPONENTES, SEGUNDOS_POR_DIA

#: Nombres de las columnas que produce `transformar`, en orden fijo.
VARIABLES = [*COMPONENTES, "importe_log", "hora_sen", "hora_cos"]


class NoAjustado(RuntimeError):
    """Se pidió transformar con un escalador que todavía no se ha ajustado."""


def hora_del_dia(segundos: np.ndarray | pd.Series) -> np.ndarray:
    """Convierte los segundos transcurridos en hora del día dentro de [0, 24)."""
    return np.asarray(segundos, dtype=float) % SEGUNDOS_POR_DIA / 3600.0


def transformar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las dos transformaciones de variable y devuelve la matriz de diseño.

    El importe se comprime con un logaritmo porque su distribución tiene una cola
    derecha muy larga; la forma log(1 + x) conserva el orden y admite el importe
    cero. La hora se codifica con seno y coseno para que las 23:00 y las 00:00
    queden próximas, cosa que el valor crudo en segundos no permite: ese valor
    además no se repetiría en datos futuros y por tanto no sirve como predictor.
    """
    hora = hora_del_dia(df["Time"])
    angulo = 2 * np.pi * hora / 24.0
    salida = df[COMPONENTES].astype(float).copy()
    salida["importe_log"] = np.log1p(df["Amount"].astype(float).to_numpy())
    salida["hora_sen"] = np.sin(angulo)
    salida["hora_cos"] = np.cos(angulo)
    return salida[VARIABLES]


class Escalador:
    """Estandariza las variables con los parámetros del conjunto de entrenamiento.

    Se implementa como una clase con ajuste explícito, en lugar de una función,
    para que resulte imposible escalar la validación con sus propios parámetros:
    `ajustar` solo se invoca sobre el entrenamiento y `aplicar` falla si no se ha
    ajustado antes.
    """

    def __init__(self) -> None:
        self.media_: np.ndarray | None = None
        self.desviacion_: np.ndarray | None = None

    @property
    def ajustado(self) -> bool:
        return self.media_ is not None

    def ajustar(self, X: pd.DataFrame) -> Escalador:
        valores = X[VARIABLES].to_numpy(dtype=float)
        self.media_ = valores.mean(axis=0)
        desviacion = valores.std(axis=0)
        # Una variable constante tendría desviación nula; se deja en 1 para no dividir por cero.
        self.desviacion_ = np.where(desviacion > 1e-12, desviacion, 1.0)
        return self

    def aplicar(self, X: pd.DataFrame) -> np.ndarray:
        if not self.ajustado:
            raise NoAjustado("Hay que llamar a `ajustar` con el entrenamiento antes de aplicar")
        return (X[VARIABLES].to_numpy(dtype=float) - self.media_) / self.desviacion_

    def ajustar_y_aplicar(self, X: pd.DataFrame) -> np.ndarray:
        return self.ajustar(X).aplicar(X)


def preparar(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve las matrices listas para entrenar y evaluar.

    Devuelve además los importes del conjunto de evaluación, porque el criterio
    de decisión del proyecto los necesita para calcular el costo esperado.
    """
    escalador = Escalador()
    X_ent = escalador.ajustar_y_aplicar(transformar(entrenamiento))
    X_eva = escalador.aplicar(transformar(evaluacion))
    y_ent = entrenamiento["Class"].to_numpy(dtype=int)
    y_eva = evaluacion["Class"].to_numpy(dtype=int)
    importes = evaluacion["Amount"].to_numpy(dtype=float)
    return X_ent, y_ent, X_eva, y_eva, importes
