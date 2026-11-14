"""Servicio de puntuación: lo que el front end consume.

Separa deliberadamente tres cosas que en un cuaderno suelen ir mezcladas y que
desde la perspectiva del usuario final son distintas:

1. **Puntuar** una transacción devuelve una probabilidad. Es lo que el modelo
   sabe hacer y no es lo que el analista necesita.
2. **Decidir** convierte esa probabilidad en una acción —bloquear, revisar o
   dejar pasar— comparándola con un umbral que depende del costo de una alerta
   falsa y del importe en juego.
3. **Explicar** la decisión en los términos del negocio: cuánto dinero se
   arriesga, cuántas alertas implica ese ajuste.

El front end nunca ve un modelo ni un escalador: ve este servicio. Esa frontera
permite cambiar de arquitectura sin tocar la interfaz, que es justamente lo que
el proyecto hizo tres veces durante la fase anterior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .datos import COLUMNAS, COMPONENTES, validar_esquema
from .preprocesamiento import Escalador, transformar

#: Columnas que el usuario debe aportar: todas menos la etiqueta.
COLUMNAS_ENTRADA = [c for c in COLUMNAS if c != "Class"]

#: Franja intermedia: por debajo del umbral pero lo bastante cerca como para
#: que valga la pena una revisión humana en lugar de dejar pasar sin más.
FACTOR_REVISION = 0.25


class ModeloNoCargado(RuntimeError):
    """Se pidió una puntuación antes de cargar o entrenar el modelo."""


@dataclass(frozen=True)
class Veredicto:
    """Lo que el front end muestra para una transacción."""

    probabilidad: float
    umbral: float
    accion: str
    importe: float
    riesgo_esperado: float
    margen: float

    def a_diccionario(self) -> dict[str, Any]:
        return asdict(self)


def _accion(probabilidad: float, umbral: float) -> str:
    if probabilidad >= umbral:
        return "bloquear"
    if probabilidad >= umbral * (1 - FACTOR_REVISION):
        return "revisar"
    return "permitir"


class Servicio:
    """Modelo, escalador y umbral, cargados juntos y usados juntos.

    Mantenerlos en un solo objeto evita el error más fácil de cometer en
    producción: aplicar un escalador a un modelo que se entrenó con otro.
    """

    def __init__(self, modelo: Any, escalador: Escalador, umbral: float,
                 nombre: str = "desconocido", costo_alerta_falsa: float = 5.0,
                 curva: list[tuple[float, float, float]] | None = None) -> None:
        self.modelo = modelo
        self.escalador = escalador
        self.umbral = float(umbral)
        self.nombre = nombre
        self.costo_alerta_falsa = float(costo_alerta_falsa)
        #: (costo de la alerta falsa, umbral óptimo, costo total) sobre el
        #: tramo de desarrollo. Permite que el control de costo de la interfaz
        #: mueva el umbral de verdad y no de forma decorativa.
        self.curva = list(curva or [])

    def umbral_para(self, costo: float) -> float:
        """Umbral óptimo para un costo de alerta falsa dado.

        Se interpola sobre la curva calculada al exportar. Si el costo queda
        fuera del rango tabulado se devuelve el extremo más cercano, que es
        preferible a extrapolar una curva fuertemente no lineal.
        """
        if not self.curva:
            return self.umbral
        costos = np.array([c for c, _, _ in self.curva], dtype=float)
        umbrales = np.array([u for _, u, _ in self.curva], dtype=float)
        return float(np.interp(float(costo), costos, umbrales))

    # -- carga y persistencia ------------------------------------------------

    @classmethod
    def cargar(cls, ruta: str | Path) -> Servicio:
        """Lee un servicio previamente exportado."""
        import joblib

        ruta = Path(ruta)
        if not ruta.exists():
            raise ModeloNoCargado(
                f"No existe {ruta}. Ejecuta antes: python scripts/exportar.py datos/creditcard.csv"
            )
        guardado = joblib.load(ruta)
        return cls(**guardado)

    def guardar(self, ruta: str | Path) -> Path:
        import joblib

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": self.modelo, "escalador": self.escalador,
                     "umbral": self.umbral, "nombre": self.nombre,
                     "costo_alerta_falsa": self.costo_alerta_falsa,
                     "curva": self.curva}, ruta)
        return ruta

    # -- uso -----------------------------------------------------------------

    def probabilidades(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de fraude de cada fila, sin decidir nada todavía."""
        faltan = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
        if faltan:
            raise ValueError(
                f"Faltan {len(faltan)} columnas: {', '.join(faltan[:6])}"
                + (" …" if len(faltan) > 6 else "")
            )
        copia = df.copy()
        if "Class" not in copia.columns:
            copia["Class"] = 0  # el validador espera el esquema completo
        validar_esquema(copia[COLUMNAS])
        X = self.escalador.aplicar(transformar(copia))
        return self.modelo.predict_proba(X)[:, 1]

    def juzgar(self, df: pd.DataFrame, umbral: float | None = None) -> list[Veredicto]:
        """Puntúa y decide, devolviendo un veredicto por transacción."""
        u = self.umbral if umbral is None else float(umbral)
        p = self.probabilidades(df)
        importes = df["Amount"].to_numpy(dtype=float)
        return [
            Veredicto(
                probabilidad=float(pi),
                umbral=u,
                accion=_accion(float(pi), u),
                importe=float(ai),
                riesgo_esperado=float(pi * ai),
                margen=float(pi - u),
            )
            for pi, ai in zip(p, importes, strict=True)
        ]

    def resumen_lote(self, veredictos: list[Veredicto]) -> dict[str, Any]:
        """Agrega un lote en las cifras que un analista mira primero."""
        acciones = [v.accion for v in veredictos]
        return {
            "transacciones": len(veredictos),
            "bloquear": acciones.count("bloquear"),
            "revisar": acciones.count("revisar"),
            "permitir": acciones.count("permitir"),
            "importe_bloqueado": sum(v.importe for v in veredictos if v.accion == "bloquear"),
            "riesgo_que_pasa": sum(v.riesgo_esperado for v in veredictos
                                   if v.accion == "permitir"),
        }

    def estado(self) -> dict[str, Any]:
        return {
            "modelo": self.nombre,
            "umbral": self.umbral,
            "costo_alerta_falsa": self.costo_alerta_falsa,
            "columnas_requeridas": COLUMNAS_ENTRADA,
            "componentes": len(COMPONENTES),
            "curva": [{"costo": c, "umbral": u, "costo_total": t} for c, u, t in self.curva],
            "factor_revision": FACTOR_REVISION,
        }
