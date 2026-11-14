"""Recolección y caracterización del conjunto.

Uso:  python scripts/explorar.py datos/creditcard.csv [--salida resultados]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraude.datos import cargar, particion_temporal, resumir


def main() -> None:
    p = argparse.ArgumentParser(description="Valida y describe el conjunto de transacciones")
    p.add_argument("datos", type=Path, help="ruta al archivo creditcard.csv")
    p.add_argument("--salida", type=Path, default=Path("resultados"))
    p.add_argument("--fraccion-prueba", type=float, default=0.2)
    args = p.parse_args()

    df = cargar(args.datos)
    total = resumir(df)
    desarrollo, prueba = particion_temporal(df, args.fraccion_prueba)

    print(f"Conjunto completo : {total}")
    print(f"Desarrollo        : {resumir(desarrollo)}")
    print(f"Prueba temporal   : {resumir(prueba)}")

    args.salida.mkdir(parents=True, exist_ok=True)
    destino = args.salida / "resumen_datos.json"
    destino.write_text(json.dumps({
        "completo": vars(total),
        "desarrollo": vars(resumir(desarrollo)),
        "prueba": vars(resumir(prueba)),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito {destino}")


if __name__ == "__main__":
    main()
