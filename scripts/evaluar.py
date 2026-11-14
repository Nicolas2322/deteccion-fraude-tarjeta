"""Ejecuta el pipeline completo y escribe las métricas.

Uso:  python scripts/evaluar.py datos/creditcard.csv --modelo lightgbm --costo 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraude.datos import cargar
from fraude.modelos import CATALOGO
from fraude.pipeline import ejecutar


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena, evalúa y elige el umbral por costo")
    p.add_argument("datos", type=Path)
    p.add_argument("--modelo", choices=sorted(CATALOGO), default="logistica")
    p.add_argument("--costo", type=float, default=5.0,
                   help="costo de bloquear una transacción legítima")
    p.add_argument("--pliegues", type=int, default=5)
    p.add_argument("--fraccion-prueba", type=float, default=0.2)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", type=Path, default=Path("resultados"))
    args = p.parse_args()

    resultado = ejecutar(cargar(args.datos), modelo=args.modelo,
                         costo_alerta_falsa=args.costo, n_pliegues=args.pliegues,
                         fraccion_prueba=args.fraccion_prueba, semilla=args.semilla)

    cv, pt = resultado.validacion_cruzada, resultado.prueba_temporal
    print(f"Modelo: {args.modelo}   costo de alerta falsa: {args.costo}")
    print(f"  Umbral elegido en desarrollo : {resultado.decision['umbral']:.6f}")
    print(f"  AUPRC  validación cruzada    : {cv['auprc']:.4f}  "
          f"(línea base {cv['auprc_linea_base']:.4f})")
    print(f"  AUPRC  prueba temporal       : {pt['auprc']:.4f}")
    print(f"  Diferencia entre protocolos  : {cv['auprc'] - pt['auprc']:+.4f}")
    print(f"  Exhaustividad en prueba      : {pt['exhaustividad']:.3f}")
    print(f"  Alertas falsas en prueba     : {int(pt['alertas_falsas'])}")
    print(f"  Costo total en prueba        : {pt['costo_total']:.2f}")

    args.salida.mkdir(parents=True, exist_ok=True)
    destino = args.salida / f"metricas_{args.modelo}.json"
    destino.write_text(json.dumps(resultado.a_diccionario(), indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nEscrito {destino}")


if __name__ == "__main__":
    main()
