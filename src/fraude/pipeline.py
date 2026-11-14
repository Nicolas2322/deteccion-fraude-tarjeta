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
from pathlib import Path
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


def _ruta_cache(cache: Path | None, nombre: str) -> Path | None:
    if cache is None:
        return None
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{nombre}.npy"


def predicciones_fuera_de_particion(df: pd.DataFrame, modelo: str, n_pliegues: int,
                                    semilla: int, cache: Path | None = None,
                                    hiper: dict[str, Any] | None = None) -> np.ndarray:
    """Devuelve una probabilidad por transacción, estimada siempre por un modelo
    que no la vio durante su entrenamiento.

    Con `cache`, las predicciones de cada pliegue se guardan en disco y se
    reutilizan en ejecuciones posteriores. Eso importa para los modelos caros:
    el FT-Transformer tarda minutos por pliegue en una máquina sin unidad de
    procesamiento gráfico, y sin caché cualquier interrupción obliga a repetir
    el entrenamiento completo. La caché no altera ningún resultado, solo evita
    recalcular lo ya calculado.
    """
    y = df["Class"].to_numpy(dtype=int)
    fuera = np.zeros(len(df), dtype=float)
    for k, (indice_ent, indice_val) in enumerate(
        pliegues_estratificados(y, n_pliegues, semilla)
    ):
        ruta = _ruta_cache(cache, f"fuera_{modelo}_s{semilla}_p{k}de{n_pliegues}")
        if ruta is not None and ruta.exists():
            fuera[indice_val] = np.load(ruta)
            continue
        X_ent, y_ent, X_val, _, _ = preparar(df.iloc[indice_ent], df.iloc[indice_val])
        clasificador = crear(modelo, semilla=semilla, **(hiper or {})).fit(X_ent, y_ent)
        fuera[indice_val] = clasificador.predict_proba(X_val)[:, 1]
        if ruta is not None:
            np.save(ruta, fuera[indice_val])
    return fuera


def ejecutar(df: pd.DataFrame, modelo: str = "logistica", costo_alerta_falsa: float = 5.0,
             n_pliegues: int = 5, fraccion_prueba: float = 0.2, semilla: int = 42,
             rejilla_costos: np.ndarray | None = None, cache: Path | None = None,
             hiper: dict[str, Any] | None = None) -> Resultado:
    """Recorre el pipeline entero y devuelve las métricas de ambos protocolos.

    El umbral se elige **sobre las predicciones fuera de partición** del tramo de
    desarrollo y después se aplica sin modificar al tramo de prueba. Elegirlo
    sobre la prueba produciría una estimación optimista de algo que en producción
    hay que fijar de antemano.
    """
    desarrollo, prueba = particion_temporal(df, fraccion_prueba)

    fuera = predicciones_fuera_de_particion(desarrollo, modelo, n_pliegues, semilla,
                                            cache=cache, hiper=hiper)
    importes_desarrollo = desarrollo["Amount"].to_numpy(dtype=float)
    y_desarrollo = desarrollo["Class"].to_numpy(dtype=int)
    decision: Decision = umbral_optimo(fuera, y_desarrollo, importes_desarrollo,
                                       costo_alerta_falsa)

    X_des, y_des, X_pru, y_pru, importes_prueba = preparar(desarrollo, prueba)
    ruta = _ruta_cache(cache, f"prueba_{modelo}_s{semilla}")
    if ruta is not None and ruta.exists():
        p_prueba = np.load(ruta)
    else:
        final = crear(modelo, semilla=semilla, **(hiper or {})).fit(X_des, y_des)
        p_prueba = final.predict_proba(X_pru)[:, 1]
        if ruta is not None:
            np.save(ruta, p_prueba)

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
