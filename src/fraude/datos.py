"""Recolección, validación y particionamiento del conjunto de transacciones.

El módulo concentra todo lo que toca el archivo original. El resto del proyecto
trabaja sobre los objetos que esta capa devuelve y nunca lee el CSV directamente,
de modo que un cambio en la fuente se corrige en un solo sitio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Columnas que el archivo original debe traer. V1..V28 son los componentes
#: principales publicados por los autores; Time, Amount y Class van sin transformar.
COMPONENTES = [f"V{i}" for i in range(1, 29)]
COLUMNAS = ["Time", *COMPONENTES, "Amount", "Class"]

SEGUNDOS_POR_DIA = 24 * 60 * 60


class EsquemaInvalido(ValueError):
    """El archivo no tiene la forma que el proyecto espera."""


@dataclass(frozen=True)
class Resumen:
    """Cifras que describen un conjunto de transacciones."""

    filas: int
    fraudes: int
    prevalencia: float
    importe_total_fraude: float
    duracion_horas: float

    def __str__(self) -> str:  # pragma: no cover - solo presentación
        return (f"{self.filas:,} transacciones | {self.fraudes} fraudes "
                f"({self.prevalencia:.3%}) | {self.duracion_horas:.1f} h")


def validar_esquema(df: pd.DataFrame) -> None:
    """Comprueba la forma del conjunto antes de que llegue al resto del pipeline.

    Se valida de forma temprana y ruidosa porque un archivo con una columna
    renombrada produciría, si no, un error mucho más adelante y mucho menos claro.
    """
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise EsquemaInvalido(f"Faltan columnas: {faltan}")
    no_numericas = [c for c in COLUMNAS if not pd.api.types.is_numeric_dtype(df[c])]
    if no_numericas:
        raise EsquemaInvalido(f"Columnas no numéricas: {no_numericas}")
    etiquetas = set(pd.unique(df["Class"].dropna()))
    if not etiquetas <= {0, 1}:
        raise EsquemaInvalido(f"Class debe ser 0 o 1; se encontró {sorted(etiquetas)}")
    if df[COLUMNAS].isna().any().any():
        raise EsquemaInvalido("El conjunto contiene valores ausentes")
    if (df["Amount"] < 0).any():
        raise EsquemaInvalido("Hay importes negativos")


def cargar(ruta: str | Path, validar: bool = True) -> pd.DataFrame:
    """Lee el archivo original y devuelve las transacciones ordenadas en el tiempo.

    El orden temporal se impone aquí y no más tarde, porque la partición de
    prueba depende de él y un conjunto desordenado produciría una fuga silenciosa.
    """
    df = pd.read_csv(ruta)
    if validar:
        validar_esquema(df)
    return df.sort_values("Time", kind="mergesort").reset_index(drop=True)


def resumir(df: pd.DataFrame) -> Resumen:
    """Devuelve las cifras básicas del conjunto."""
    fraudes = int((df["Class"] == 1).sum())
    return Resumen(
        filas=len(df),
        fraudes=fraudes,
        prevalencia=fraudes / len(df) if len(df) else 0.0,
        importe_total_fraude=float(df.loc[df["Class"] == 1, "Amount"].sum()),
        duracion_horas=float(df["Time"].max() - df["Time"].min()) / 3600 if len(df) else 0.0,
    )


def particion_temporal(df: pd.DataFrame, fraccion_prueba: float = 0.2
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserva el último tramo del periodo como conjunto de prueba.

    Es la partición que reproduce la situación real descrita por Dal Pozzolo et al.
    (2018): un sistema en producción siempre predice hacia adelante. Una partición
    aleatoria permitiría entrenar con transacciones posteriores a las evaluadas.
    """
    if not 0 < fraccion_prueba < 1:
        raise ValueError("fraccion_prueba debe estar entre 0 y 1")
    corte = int(round(len(df) * (1 - fraccion_prueba)))
    # El corte se desplaza para no partir un grupo de transacciones simultáneas.
    tiempo_corte = df["Time"].iloc[corte]
    mascara = df["Time"] < tiempo_corte
    if mascara.sum() == 0 or mascara.sum() == len(df):
        mascara = np.arange(len(df)) < corte
    return df[mascara].reset_index(drop=True), df[~mascara].reset_index(drop=True)


def pliegues_estratificados(y: pd.Series | np.ndarray, n_pliegues: int = 5,
                            semilla: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reparte en pliegues conservando la proporción de fraudes de cada uno.

    Con una prevalencia del 0,17 % una partición no estratificada puede dejar
    pliegues con muy pocos positivos, lo que haría la métrica inestable.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    if n_pliegues < 2:
        raise ValueError("Se necesitan al menos dos pliegues")
    if int((y == 1).sum()) < n_pliegues:
        raise ValueError("Hay menos fraudes que pliegues; la estratificación no es posible")
    particionador = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    return list(particionador.split(np.zeros(len(y)), y))
