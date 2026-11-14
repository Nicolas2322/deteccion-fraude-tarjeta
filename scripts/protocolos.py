"""Descompone la diferencia entre los dos protocolos de validación.

Uso:  python scripts/protocolos.py datos/creditcard.csv --cache cache

El AUPRC de la validación cruzada se calcula agrupando en una sola lista las
predicciones de los cinco modelos de los cinco pliegues. Agrupar penaliza a los
modelos cuya escala de probabilidad varía entre pliegues: dos transacciones
igual de sospechosas pero evaluadas por modelos con escalas distintas quedan
mal ordenadas entre sí.

Este script separa las dos cosas. Calcula el AUPRC agrupado, que es el que el
pipeline reporta, y el promedio de los AUPRC que cada pliegue obtiene por
separado, que no sufre ese efecto. La diferencia entre ambos mide cuánto cuesta
agrupar. No entrena nada: lee las predicciones que el pipeline dejó en la caché.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraude.datos import cargar, particion_temporal, pliegues_estratificados
from fraude.metricas import auprc

MODELOS = ["logistica", "lightgbm", "mlp", "ft_transformer"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diferencia entre protocolos de validación")
    ap.add_argument("datos", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--resultados", type=Path, default=Path("resultados"))
    ap.add_argument("--pliegues", type=int, default=5)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    desarrollo, _ = particion_temporal(cargar(args.datos), 0.2)
    y = desarrollo["Class"].to_numpy(dtype=int)
    particiones = list(pliegues_estratificados(y, args.pliegues, args.semilla))

    print(f"{'modelo':16s} {'agrupado':>9s} {'media por pliegue':>18s} {'costo de agrupar':>17s}")
    salida = {}
    for modelo in MODELOS:
        rutas = [args.cache / f"fuera_{modelo}_s{args.semilla}_p{k}de{args.pliegues}.npy"
                 for k in range(args.pliegues)]
        if not all(r.exists() for r in rutas):
            continue
        agrupadas = np.zeros(len(y), dtype=float)
        por_pliegue = []
        for (_, indice_val), ruta in zip(particiones, rutas, strict=True):
            p = np.load(ruta)
            agrupadas[indice_val] = p
            por_pliegue.append(float(auprc(y[indice_val], p)))
        agrupado = float(auprc(y, agrupadas))
        media = float(np.mean(por_pliegue))
        salida[modelo] = {"agrupado": agrupado, "media_por_pliegue": media,
                          "por_pliegue": por_pliegue, "costo_de_agrupar": media - agrupado}
        print(f"{modelo:16s} {agrupado:9.4f} {media:18.4f} {media - agrupado:+17.4f}")
        print(f"{'':16s} por pliegue: " + "  ".join(f"{v:.4f}" for v in por_pliegue))

    if not salida:
        raise SystemExit("No hay predicciones en la caché. Ejecuta antes scripts/evaluar.py.")
    args.resultados.mkdir(parents=True, exist_ok=True)
    destino = args.resultados / "diagnostico_protocolos.json"
    destino.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito {destino}")


if __name__ == "__main__":
    main()
