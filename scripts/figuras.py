"""Genera las figuras del informe a partir de las predicciones en caché.

Uso:  python scripts/figuras.py datos/creditcard.csv --cache cache --salida figuras

No vuelve a entrenar nada: lee las predicciones que el pipeline guardó y
construye las curvas. Así las figuras del informe y las métricas de las tablas
provienen por construcción de la misma ejecución.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fraude.datos import cargar, particion_temporal  # noqa: E402
from fraude.metricas import auprc, error_calibracion, roc_auc  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

ETIQUETAS = {
    "logistica": "Regresión logística",
    "lightgbm": "LightGBM",
    "mlp": "Perceptrón multicapa",
    "ft_transformer": "FT-Transformer",
}
ESTILOS = {
    "logistica": dict(color="#999999", ls=":", lw=1.6),
    "lightgbm": dict(color="#444444", ls="--", lw=1.6),
    "mlp": dict(color="#777777", ls="-.", lw=1.6),
    "ft_transformer": dict(color="#111111", ls="-", lw=1.9),
}
COMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def curva_pr(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precisión y exhaustividad en cada umbral posible, sin usar scikit-learn."""
    orden = np.argsort(-p, kind="mergesort")
    y = y[orden]
    verdaderos = np.cumsum(y)
    alertas = np.arange(1, len(y) + 1)
    return verdaderos / y.sum(), verdaderos / alertas


def diagrama_calibracion(y: np.ndarray, p: np.ndarray, n: int = 12):
    """Probabilidad predicha media frente a frecuencia observada, por cubos.

    Los cubos se forman por cuantiles y no por cortes uniformes: con una
    prevalencia del 0,17 %, unos cortes uniformes dejarían casi todas las
    transacciones en el primer cubo y el resto vacíos.
    """
    bordes = np.quantile(p, np.linspace(0, 1, n + 1))
    bordes = np.unique(bordes)
    indice = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, len(bordes) - 2)
    xs, ys = [], []
    for k in range(len(bordes) - 1):
        m = indice == k
        if m.sum() >= 50:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description="Figuras del informe")
    ap.add_argument("datos", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--resultados", type=Path, default=Path("resultados"))
    ap.add_argument("--salida", type=Path, default=Path("figuras"))
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    _, prueba = particion_temporal(cargar(args.datos), 0.2)
    y = prueba["Class"].to_numpy(dtype=int)

    predicciones = {}
    for modelo in ETIQUETAS:
        ruta = args.cache / f"prueba_{modelo}_s{args.semilla}.npy"
        if ruta.exists():
            predicciones[modelo] = np.load(ruta)
    if not predicciones:
        raise SystemExit("No hay predicciones en la caché. Ejecuta antes scripts/evaluar.py.")

    # --- Figura: curvas de precisión y exhaustividad -----------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=200)
    base = y.mean()
    for modelo, p in predicciones.items():
        exh, pre = curva_pr(y, p)
        etq = f"{ETIQUETAS[modelo]} (AUPRC {auprc(y, p):.3f})".replace(".", ",")
        for a in ax:
            a.plot(exh, pre, label=etq, **ESTILOS[modelo])
    for a in ax:
        a.axhline(base, color="#bbbbbb", lw=1.0, ls=(0, (1, 2)))
        a.set_xlabel("Exhaustividad")
        a.spines[["top", "right"]].set_visible(False)
        a.xaxis.set_major_formatter(COMA)
        a.yaxis.set_major_formatter(COMA)
    ax[0].set_ylabel("Precisión")
    ax[0].set_title("a) Curva completa", fontsize=9.5)
    ax[0].text(0.52, base + 0.03, f"línea base: {base:.4f}".replace(".", ","),
               fontsize=6.6, color="#777777")
    ax[1].set_xlim(0, 0.9)
    ax[1].set_ylim(0.4, 1.02)
    ax[1].set_title("b) Zona de operación", fontsize=9.5)
    ax[0].legend(fontsize=6.8, loc="lower left", frameon=False)
    fig.tight_layout(pad=0.6)
    fig.savefig(args.salida / "curvas_pr.png")
    plt.close(fig)

    # --- Figura: calibración ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    for modelo, p in predicciones.items():
        xs, ys = diagrama_calibracion(y, p)
        ax[0].plot(xs, ys, marker="o", ms=3.2, **ESTILOS[modelo], label=ETIQUETAS[modelo])
    lim = [1e-5, 1.0]
    ax[0].plot(lim, lim, color="#bbbbbb", lw=1.0, ls=(0, (1, 2)))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Probabilidad predicha media del cubo")
    ax[0].set_ylabel("Frecuencia observada")
    ax[0].set_title("a) Diagrama de calibración", fontsize=9.5)
    ax[0].legend(fontsize=6.8, loc="upper left", frameon=False)

    nombres = list(predicciones)
    valores = [error_calibracion(y, predicciones[m]) for m in nombres]
    barras = ax[1].barh([ETIQUETAS[m] for m in nombres], valores,
                        color=["#dddddd" if v > 0.01 else "#555555" for v in valores],
                        ec="#333333", lw=0.9, height=0.55)
    ax[1].bar_label(barras, labels=[f"{v:.4f}".replace(".", ",") for v in valores],
                    fontsize=7.2, padding=3)
    ax[1].set_xlabel("Error de calibración esperado")
    ax[1].set_xlim(0, max(valores) * 1.35)
    ax[1].set_title("b) Error de calibración", fontsize=9.5)
    ax[1].xaxis.set_major_formatter(COMA)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    fig.savefig(args.salida / "calibracion.png")
    plt.close(fig)

    # --- Figura: sensibilidad al costo y comparación por protocolo ----------
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    for modelo in predicciones:
        ruta = args.resultados / f"metricas_{modelo}.json"
        if not ruta.exists():
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        c, _, costo = np.array(datos["sensibilidad"]).T
        ax[0].plot(c, costo / 1000, marker="o", ms=3.0, **ESTILOS[modelo],
                   label=ETIQUETAS[modelo])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Costo de una alerta falsa")
    ax[0].set_ylabel("Costo total en desarrollo (miles)")
    ax[0].set_title("a) Sensibilidad al costo de la alerta falsa", fontsize=9.5)
    ax[0].legend(fontsize=6.8, frameon=False)
    ax[0].yaxis.set_major_formatter(COMA)

    etq, cv, pt = [], [], []
    for modelo in predicciones:
        ruta = args.resultados / f"metricas_{modelo}.json"
        if not ruta.exists():
            continue
        d = json.loads(ruta.read_text(encoding="utf-8"))
        etq.append(ETIQUETAS[modelo])
        cv.append(d["validacion_cruzada"]["auprc"])
        pt.append(d["prueba_temporal"]["auprc"])
    pos = np.arange(len(etq))
    ax[1].bar(pos - 0.19, cv, 0.36, label="Validación cruzada", color="#cccccc",
              ec="#333333", lw=0.9)
    ax[1].bar(pos + 0.19, pt, 0.36, label="Prueba temporal", color="#555555",
              ec="#333333", lw=0.9)
    ax[1].set_xticks(pos)
    ax[1].set_xticklabels([e.replace(" ", "\n", 1) for e in etq], fontsize=7)
    ax[1].set_ylabel("AUPRC")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=6.8, frameon=False)
    ax[1].set_title("b) AUPRC bajo los dos protocolos", fontsize=9.5)
    ax[1].yaxis.set_major_formatter(COMA)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    fig.savefig(args.salida / "costo_y_protocolos.png")
    plt.close(fig)

    print(f"Escritas 3 figuras en {args.salida}/ a partir de "
          f"{len(predicciones)} modelos: {', '.join(predicciones)}")
    for modelo, p in predicciones.items():
        print(f"  {modelo:16s} AUPRC {auprc(y, p):.4f}  ROC {roc_auc(y, p):.4f}  "
              f"ECE {error_calibracion(y, p):.4f}")


if __name__ == "__main__":
    main()
