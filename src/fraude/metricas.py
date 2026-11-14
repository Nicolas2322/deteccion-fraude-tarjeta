"""Métricas de evaluación adecuadas a un conjunto con desbalance extremo.

Con una prevalencia del 0,17 %, un clasificador que responda siempre «no hay
fraude» alcanza una exactitud del 99,83 %. La exactitud queda por tanto
descartada y la métrica principal es el área bajo la curva de precisión y
exhaustividad, cuyo valor de referencia sin información es la propia prevalencia.
"""

from __future__ import annotations

import numpy as np

from .umbral import costo_total


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Área bajo la curva de precisión y exhaustividad (precisión media)."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y), np.asarray(p)))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Área bajo la curva ROC. Se reporta como referencia, no como criterio."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def linea_base_auprc(y: np.ndarray) -> float:
    """Valor que obtiene un modelo sin información: la prevalencia."""
    y = np.asarray(y)
    return float((y == 1).mean()) if len(y) else 0.0


def error_calibracion(y: np.ndarray, p: np.ndarray, cajas: int = 10) -> float:
    """Error de calibración esperado, por cajas de igual anchura.

    Importa porque el umbral del proyecto se elige sobre la probabilidad
    estimada: si el remuestreo la desplaza, como demuestran Dal Pozzolo et al.
    (2015), el umbral deja de tener sentido económico aunque el ordenamiento de
    las transacciones siga siendo correcto.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return 0.0
    bordes = np.linspace(0.0, 1.0, cajas + 1)
    indice = np.clip(np.digitize(p, bordes[1:-1], right=True), 0, cajas - 1)
    error = 0.0
    for caja in range(cajas):
        en_caja = indice == caja
        peso = en_caja.mean()
        if peso > 0:
            error += peso * abs(y[en_caja].mean() - p[en_caja].mean())
    return float(error)


def metricas_en_umbral(y: np.ndarray, p: np.ndarray, importes: np.ndarray,
                       umbral: float, c: float) -> dict[str, float]:
    """Indicadores operativos en el punto de corte elegido.

    Se reportan en números absolutos además de en proporciones: con 492 fraudes,
    decir «se escaparon 57» informa mucho más que decir «la exhaustividad fue
    del 88 %».
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    importes = np.asarray(importes, dtype=float)
    bloqueada = p >= umbral
    vp = int(((y == 1) & bloqueada).sum())
    fn = int(((y == 1) & ~bloqueada).sum())
    fp = int(((y == 0) & bloqueada).sum())
    vn = int(((y == 0) & ~bloqueada).sum())
    return {
        "umbral": float(umbral),
        "exhaustividad": vp / (vp + fn) if vp + fn else 0.0,
        "precision": vp / (vp + fp) if vp + fp else 0.0,
        "especificidad": vn / (vn + fp) if vn + fp else 0.0,
        "fraudes_detectados": float(vp),
        "fraudes_no_detectados": float(fn),
        "alertas_falsas": float(fp),
        "importe_no_detectado": float(importes[(y == 1) & ~bloqueada].sum()),
        "costo_total": costo_total(p, y, importes, c, umbral),
    }


def informe(y: np.ndarray, p: np.ndarray, importes: np.ndarray,
            umbral: float, c: float) -> dict[str, float]:
    """Reúne en un solo diccionario todo lo que el informe final debe reportar."""
    return {
        "auprc": auprc(y, p),
        "auprc_linea_base": linea_base_auprc(y),
        "roc_auc": roc_auc(y, p),
        "error_calibracion": error_calibracion(y, p),
        **metricas_en_umbral(y, p, importes, umbral, c),
    }
