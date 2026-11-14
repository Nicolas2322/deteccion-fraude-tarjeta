"""Exporta el modelo a JSON para que se ejecute en el navegador.

Uso:  python scripts/exportar_navegador.py --servicio app/servicio.joblib --salida docs

El despliegue público del proyecto es una página estática: no hay servidor que
mantener, el alojamiento es gratuito y permanente, y cualquiera puede probar el
sistema abriendo un enlace. A cambio, el modelo viaja al navegador.

Esa cesión es aceptable aquí y no lo sería en producción, y conviene decir por
qué: un detector de fraude cuyos pesos son públicos es un detector que un
defraudador puede estudiar para construir transacciones que lo esquiven. El
despliegue estático existe para que el trabajo sea comprobable por terceros,
no como arquitectura recomendada. La versión con servidor, en `app/`, es la
que un despliegue real usaría.

Solo se admite el perceptrón multicapa: su aritmética —tres productos de
matrices, ReLU y una sigmoide— cabe en treinta líneas de JavaScript sin
dependencias. Un ensamble de árboles o un transformador exigirían embarcar una
biblioteca entera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fraude.preprocesamiento import VARIABLES

DECIMALES = 7


def redondear(a: np.ndarray) -> list:
    """Reduce la precisión para que el archivo no pese de más.

    Siete decimales mantienen la predicción idéntica hasta la sexta cifra
    significativa, que es mucho más de lo que cualquier decisión necesita, y
    dejan el archivo en torno a un tercio de su tamaño original.
    """
    return np.round(a, DECIMALES).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta el modelo para el navegador")
    ap.add_argument("--servicio", type=Path, default=Path("app/servicio.joblib"))
    ap.add_argument("--salida", type=Path, default=Path("docs"))
    args = ap.parse_args()

    guardado = joblib.load(args.servicio)
    modelo = guardado["modelo"]
    if type(modelo).__name__ != "MLPClassifier":
        raise SystemExit(
            f"Solo se exporta el perceptrón multicapa; este servicio usa "
            f"{type(modelo).__name__}. Vuelve a exportarlo con --modelo mlp."
        )
    if modelo.activation != "relu" or modelo.out_activation_ != "logistic":
        raise SystemExit("La implementación en JavaScript asume ReLU y sigmoide.")

    escalador = guardado["escalador"]
    paquete = {
        "formato": 1,
        "modelo": guardado["nombre"],
        "umbral": float(guardado["umbral"]),
        "costo_alerta_falsa": float(guardado["costo_alerta_falsa"]),
        "variables": list(VARIABLES),
        "escalador": {
            "media": redondear(escalador.media_),
            "desviacion": redondear(escalador.desviacion_),
        },
        "capas": [
            {"pesos": redondear(W), "sesgos": redondear(b)}
            for W, b in zip(modelo.coefs_, modelo.intercepts_, strict=True)
        ],
        "curva": [{"costo": float(c), "umbral": float(u)}
                  for c, u, _ in guardado.get("curva", [])],
    }

    args.salida.mkdir(parents=True, exist_ok=True)
    destino = args.salida / "modelo.json"
    destino.write_text(json.dumps(paquete, separators=(",", ":")), encoding="utf-8")

    parametros = sum(len(c["sesgos"]) + len(c["pesos"]) * len(c["pesos"][0])
                     for c in paquete["capas"])
    print(f"Modelo      : {paquete['modelo']}")
    print(f"Arquitectura: {' → '.join(str(len(c['pesos'])) for c in paquete['capas'])} → 1")
    print(f"Parámetros  : {parametros:,}")
    print(f"Umbral      : {paquete['umbral']:.6f}")
    print(f"Escrito     : {destino} ({destino.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
