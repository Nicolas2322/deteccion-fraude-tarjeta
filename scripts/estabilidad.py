"""Mide cuánto depende cada modelo de qué transacciones caen en el entrenamiento.

Uso:  python scripts/estabilidad.py datos/creditcard.csv --repeticiones 6

Una única cifra de AUPRC solo es informativa si el modelo da aproximadamente
la misma cifra cuando se le cambia un poco el conjunto de entrenamiento. Este
experimento comprueba exactamente eso: entrena varias veces sobre submuestras
aleatorias del tramo de desarrollo y evalúa siempre sobre el mismo tramo de
prueba, de modo que lo único que varía entre repeticiones es qué filas vio el
modelo.

El entrenamiento es determinista con datos y semilla fijos —hay una prueba que
lo comprueba—, así que toda la dispersión que aparezca procede de la submuestra
y no del azar del ajuste.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraude.datos import cargar, particion_temporal
from fraude.metricas import auprc
from fraude.modelos import crear
from fraude.preprocesamiento import preparar

#: Configuraciones a comparar: nombre legible, modelo e hiperparámetros.
CONFIGURACIONES = [
    ("logistica", "logistica", {}),
    ("lightgbm_profundo", "lightgbm",
     dict(n_estimators=500, num_leaves=31, min_child_samples=20)),
    ("lightgbm_regularizado", "lightgbm", {}),
    ("mlp", "mlp", {}),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Estabilidad frente a la submuestra")
    ap.add_argument("datos", type=Path)
    ap.add_argument("--repeticiones", type=int, default=6)
    ap.add_argument("--fraccion", type=float, default=0.8,
                    help="porción del desarrollo con la que se entrena cada vez")
    ap.add_argument("--fraccion-prueba", type=float, default=0.2)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--solo", nargs="*", default=None,
                    help="nombres de configuración a ejecutar; por defecto todas")
    ap.add_argument("--salida", type=Path, default=Path("resultados/estabilidad.json"))
    args = ap.parse_args()

    desarrollo, prueba = particion_temporal(cargar(args.datos), args.fraccion_prueba)
    X_des, y_des, X_pru, y_pru, _ = preparar(desarrollo, prueba)
    n = int(len(y_des) * args.fraccion)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    salida = {}
    if args.salida.exists():
        salida = json.loads(args.salida.read_text(encoding="utf-8"))

    print(f"Entrenando con {n:,} de {len(y_des):,} filas, {args.repeticiones} veces.")
    print(f"Evaluando siempre sobre las mismas {len(y_pru):,} filas del tramo de prueba.\n")

    for nombre, modelo, hiper in CONFIGURACIONES:
        if args.solo and nombre not in args.solo:
            continue
        valores = []
        for r in range(args.repeticiones):
            generador = np.random.default_rng(100 + r)
            indice = np.sort(generador.permutation(len(y_des))[:n])
            clasificador = crear(modelo, semilla=args.semilla, **hiper)
            clasificador.fit(X_des[indice], y_des[indice])
            valores.append(float(auprc(y_pru, clasificador.predict_proba(X_pru)[:, 1])))
        salida[nombre] = {
            "modelo": modelo,
            "hiperparametros": hiper,
            "auprc": valores,
            "media": float(np.mean(valores)),
            "desviacion": float(np.std(valores, ddof=1)),
            "minimo": float(min(valores)),
            "maximo": float(max(valores)),
        }
        v = salida[nombre]
        print(f"{nombre:24s} media {v['media']:.4f}  desviación {v['desviacion']:.4f}  "
              f"rango [{v['minimo']:.4f}, {v['maximo']:.4f}]")

    args.salida.write_text(json.dumps(salida, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nEscrito {args.salida}")


if __name__ == "__main__":
    main()
