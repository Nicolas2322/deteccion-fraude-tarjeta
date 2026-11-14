"""Elección del umbral de decisión por costo esperado.

El coeficiente AUPRC mide la calidad del modelo y no depende del umbral; el costo
esperado mide la calidad de la decisión y depende por completo de él. Este módulo
se ocupa de lo segundo.

El costo de operar en el umbral t es

    C(t) = A(t) + c * n(t)

donde A(t) es la suma de los importes de los fraudes que el sistema deja pasar y
n(t) el número de transacciones legítimas bloqueadas. La función solo cambia de
valor en los puntos donde alguna transacción cruza el umbral, de modo que no es
derivable y no procede un método de gradiente; en cambio admite solución exacta
evaluando los cortes entre transacciones consecutivas ordenadas por probabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    """Umbral elegido y lo que cuesta operar en él."""

    umbral: float
    costo: float
    fraudes_no_detectados: int
    importe_no_detectado: float
    alertas_falsas: int


def costo_total(p: np.ndarray, y: np.ndarray, importes: np.ndarray,
                c: float, umbral: float) -> float:
    """Costo de operar en un umbral concreto. Definición directa, sin optimizar.

    Sirve de referencia frente a la que se contrasta `umbral_optimo` en las pruebas.
    """
    p, y, importes = np.asarray(p), np.asarray(y), np.asarray(importes)
    bloqueada = p >= umbral
    no_detectados = (y == 1) & ~bloqueada
    falsas = (y == 0) & bloqueada
    return float(importes[no_detectados].sum() + c * falsas.sum())


def _acumulados(p: np.ndarray, y: np.ndarray, importes: np.ndarray):
    """Ordena por probabilidad descendente y acumula lo que el costo necesita.

    Devuelve, para cada corte i, el importe de fraude que todavía escapa y el
    número de legítimas ya bloqueadas si el umbral se sitúa en p[i].
    """
    orden = np.argsort(-np.asarray(p, dtype=float), kind="mergesort")
    p = np.asarray(p, dtype=float)[orden]
    y = np.asarray(y, dtype=int)[orden]
    importes = np.asarray(importes, dtype=float)[orden]

    if len(p) == 0:
        vacio = np.array([], dtype=float)
        return vacio, vacio, vacio, np.array([], dtype=bool), 0.0

    importe_fraude_total = float(importes[y == 1].sum())
    # Tras bloquear las i+1 primeras, el fraude que sigue escapando es el total
    # menos el ya capturado; las legítimas bloqueadas son las no fraudulentas vistas.
    capturado = np.cumsum(importes * (y == 1))
    escapa = importe_fraude_total - capturado
    falsas = np.cumsum(y == 0)
    # Solo son umbrales válidos los cortes donde cambia la probabilidad.
    valido = np.empty(len(p), dtype=bool)
    valido[:-1] = p[:-1] != p[1:]
    valido[-1] = True
    return p, escapa, falsas, valido, importe_fraude_total


def umbral_optimo(p: np.ndarray, y: np.ndarray, importes: np.ndarray,
                  c: float) -> Decision:
    """Devuelve el umbral que minimiza el costo esperado (Algoritmo 1).

    Forma vectorizada del recorrido descrito en el informe: una ordenación y una
    pasada de sumas acumuladas, de modo que el costo es O(n log n).
    """
    if c < 0:
        raise ValueError("El costo de una alerta falsa no puede ser negativo")
    p_ord, escapa, falsas, valido, total = _acumulados(p, y, importes)
    if len(p_ord) == 0:
        return Decision(1.0, 0.0, 0, 0.0, 0)

    costos = escapa + c * falsas
    # Candidato adicional: no bloquear nada, con umbral por encima de toda probabilidad.
    mejor_costo = total
    mejor_umbral = float(np.nextafter(p_ord[0], np.inf))
    indices = np.flatnonzero(valido)
    if len(indices):
        i = indices[np.argmin(costos[indices])]
        if costos[i] < mejor_costo:
            mejor_costo, mejor_umbral = float(costos[i]), float(p_ord[i])

    y_arr, imp = np.asarray(y, dtype=int), np.asarray(importes, dtype=float)
    bloqueada = np.asarray(p, dtype=float) >= mejor_umbral
    no_detectados = (y_arr == 1) & ~bloqueada
    return Decision(
        umbral=mejor_umbral,
        costo=mejor_costo,
        fraudes_no_detectados=int(no_detectados.sum()),
        importe_no_detectado=float(imp[no_detectados].sum()),
        alertas_falsas=int(((y_arr == 0) & bloqueada).sum()),
    )


def curva_sensibilidad(p: np.ndarray, y: np.ndarray, importes: np.ndarray,
                       costos_falsa: np.ndarray) -> list[tuple[float, float, float]]:
    """Recorre una rejilla de costos de alerta falsa (Algoritmo 2).

    Devuelve, para cada valor de c, el umbral óptimo y el costo mínimo. El orden
    por probabilidad se calcula una sola vez y se reutiliza en toda la rejilla.
    """
    p_ord, escapa, falsas, valido, total = _acumulados(p, y, importes)
    indices = np.flatnonzero(valido)
    salida: list[tuple[float, float, float]] = []
    for c in np.asarray(costos_falsa, dtype=float):
        costos = escapa[indices] + c * falsas[indices]
        j = int(np.argmin(costos))
        if costos[j] < total:
            salida.append((float(c), float(p_ord[indices[j]]), float(costos[j])))
        else:
            salida.append((float(c), float(np.nextafter(p_ord[0], np.inf)), float(total)))
    return salida


def umbral_por_importe(importes: np.ndarray, c: float) -> np.ndarray:
    """Umbral que depende del importe de cada transacción.

    Dejar pasar una transacción cuesta p * a y bloquearla cuesta (1 - p) * c, de
    modo que conviene bloquear cuando p * a > (1 - p) * c, es decir cuando

        p > c / (c + a)

    El umbral deja de ser constante y decrece con el importe: cuanto mayor es la
    cantidad en juego, menos evidencia se exige para bloquear. Es la
    especialización al caso del fraude del marco de Elkan (2001).
    """
    importes = np.asarray(importes, dtype=float)
    if c < 0:
        raise ValueError("El costo de una alerta falsa no puede ser negativo")
    with np.errstate(invalid="ignore"):
        return np.where(c + importes > 0, c / (c + importes), 1.0)


def decidir(p: np.ndarray, importes: np.ndarray, c: float) -> np.ndarray:
    """Aplica la política dependiente del importe y devuelve qué se bloquea."""
    return np.asarray(p, dtype=float) >= umbral_por_importe(importes, c)
