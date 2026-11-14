"""Entrena el modelo elegido y lo exporta junto con su escalador y su umbral.

Uso:  python scripts/exportar.py datos/creditcard.csv --modelo mlp --costo 5

El artefacto que produce es lo que consume la interfaz web. Se guardan juntos
el modelo, el escalador y el umbral porque usarlos por separado es el error más
fácil de cometer al pasar a producción: un escalador ajustado con otros datos
invalida silenciosamente las probabilidades, y con ellas el umbral.

El umbral no se fija a 0,5 ni a ningún valor convencional: se calcula
minimizando el costo esperado sobre las predicciones fuera de partición del
tramo de desarrollo, igual que en el resto del proyecto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fraude.datos import cargar, particion_temporal
from fraude.modelos import CATALOGO, crear
from fraude.pipeline import predicciones_fuera_de_particion
from fraude.preprocesamiento import Escalador, transformar
from fraude.servicio import Servicio
from fraude.umbral import curva_sensibilidad, umbral_optimo


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta el servicio de puntuación")
    ap.add_argument("datos", type=Path)
    ap.add_argument("--modelo", choices=sorted(CATALOGO), default="mlp")
    ap.add_argument("--costo", type=float, default=5.0)
    ap.add_argument("--pliegues", type=int, default=5)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--salida", type=Path, default=Path("app/servicio.joblib"))
    args = ap.parse_args()

    df = cargar(args.datos)
    desarrollo, _ = particion_temporal(df, 0.2)

    print(f"Eligiendo el umbral con {args.pliegues} pliegues sobre "
          f"{len(desarrollo):,} transacciones…")
    fuera = predicciones_fuera_de_particion(desarrollo, args.modelo, args.pliegues,
                                            args.semilla, cache=args.cache)
    decision = umbral_optimo(fuera, desarrollo["Class"].to_numpy(dtype=int),
                             desarrollo["Amount"].to_numpy(dtype=float), args.costo)

    print("Reentrenando sobre el desarrollo completo…")
    escalador = Escalador().ajustar(transformar(desarrollo))
    X = escalador.aplicar(transformar(desarrollo))
    modelo = crear(args.modelo, semilla=args.semilla).fit(
        X, desarrollo["Class"].to_numpy(dtype=int))

    rejilla = np.unique(np.round(np.geomspace(0.5, 200, 28), 3))
    curva = curva_sensibilidad(fuera, desarrollo["Class"].to_numpy(dtype=int),
                               desarrollo["Amount"].to_numpy(dtype=float), rejilla)
    servicio = Servicio(modelo, escalador, decision.umbral, args.modelo, args.costo,
                        curva=curva)
    ruta = servicio.guardar(args.salida)

    print(f"\nModelo        : {args.modelo}")
    print(f"Umbral        : {decision.umbral:.6f}")
    print(f"Costo de alerta falsa: {args.costo}")
    print(f"Curva de costo: {len(curva)} puntos entre {rejilla[0]:g} y {rejilla[-1]:g}")
    print(f"Escrito       : {ruta} ({ruta.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
