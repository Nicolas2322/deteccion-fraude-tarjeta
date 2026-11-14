"""Paridad entre la inferencia en Python y la del navegador.

El despliegue público ejecuta el modelo en JavaScript. Eso significa que hay
dos implementaciones del mismo cálculo —transformación de variables, escalado
y paso hacia adelante— y que pueden divergir sin que nada falle de forma
visible: la página seguiría dando un número, solo que el equivocado.

Esta prueba ejecuta las dos sobre las mismas transacciones y exige que
coincidan. Si alguien cambia el preprocesamiento en Python y olvida cambiarlo
en `docs/modelo.js`, aquí se nota.

Se omite si falta Node.js o si el modelo todavía no se ha exportado, de modo
que la integración continua no depende de ninguno de los dos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
MODELO_JSON = RAIZ / "docs" / "modelo.json"
MODELO_JS = RAIZ / "docs" / "modelo.js"
SERVICIO = RAIZ / "app" / "servicio.joblib"
MUESTRA = RAIZ / "datos_muestra" / "creditcard_muestra.csv"

pytestmark = pytest.mark.skipif(
    not (shutil.which("node") and MODELO_JSON.exists() and SERVICIO.exists()),
    reason="Requiere Node.js y el modelo exportado (scripts/exportar_navegador.py)",
)

GUION = """
import {{ ModeloLocal }} from "{js}";
import {{ readFileSync }} from "node:fs";
const paquete = JSON.parse(readFileSync("{json}", "utf8"));
const modelo = new ModeloLocal(paquete);
const filas = JSON.parse(readFileSync("{filas}", "utf8"));
console.log(JSON.stringify(filas.map((f) => modelo.juzgar(f))));
"""


@pytest.fixture(scope="module")
def transacciones() -> pd.DataFrame:
    from fraude.servicio import COLUMNAS_ENTRADA

    if not MUESTRA.exists():
        pytest.skip("Falta la muestra de datos")
    df = pd.read_csv(MUESTRA)
    # Se mezclan fraudes y legítimas, y se incluye el importe cero, que es el
    # caso límite de la transformación logarítmica.
    trozo = pd.concat([df[df.Class == 1].head(25), df[df.Class == 0].head(25)])
    if (df.Amount == 0).any():
        trozo = pd.concat([trozo, df[df.Amount == 0].head(5)])
    return trozo[COLUMNAS_ENTRADA].reset_index(drop=True)


@pytest.fixture(scope="module")
def veredictos_js(transacciones, tmp_path_factory) -> list[dict]:
    carpeta = tmp_path_factory.mktemp("js")
    filas = carpeta / "filas.json"
    filas.write_text(json.dumps(transacciones.to_dict(orient="records")), encoding="utf-8")
    guion = carpeta / "paridad.mjs"
    guion.write_text(GUION.format(js=MODELO_JS.as_uri(), json=MODELO_JSON,
                                  filas=filas), encoding="utf-8")
    salida = subprocess.run(["node", str(guion)], capture_output=True, text=True,
                            timeout=120)
    if salida.returncode:
        pytest.fail(f"Node falló:\n{salida.stderr[-2000:]}")
    return json.loads(salida.stdout)


@pytest.fixture(scope="module")
def veredictos_python(transacciones) -> list[dict]:
    from fraude.servicio import Servicio

    servicio = Servicio.cargar(SERVICIO)
    return [v.a_diccionario() for v in servicio.juzgar(transacciones)]


def test_las_probabilidades_coinciden(veredictos_js, veredictos_python):
    """Hasta la décima cifra decimal, que es muy por encima de lo necesario."""
    a = np.array([v["probabilidad"] for v in veredictos_js])
    b = np.array([v["probabilidad"] for v in veredictos_python])
    assert np.allclose(a, b, atol=1e-10), f"diferencia máxima {np.abs(a - b).max():.3e}"


def test_las_acciones_coinciden(veredictos_js, veredictos_python):
    assert [v["accion"] for v in veredictos_js] == [v["accion"] for v in veredictos_python]


def test_el_umbral_es_el_mismo(veredictos_js, veredictos_python):
    assert veredictos_js[0]["umbral"] == pytest.approx(veredictos_python[0]["umbral"])


def test_el_riesgo_esperado_coincide(veredictos_js, veredictos_python):
    a = np.array([v["riesgo_esperado"] for v in veredictos_js])
    b = np.array([v["riesgo_esperado"] for v in veredictos_python])
    assert np.allclose(a, b, atol=1e-8)


def test_hay_casos_de_las_dos_clases(veredictos_js):
    """La prueba no valdría si todas las transacciones cayeran del mismo lado."""
    assert len({v["accion"] for v in veredictos_js}) >= 2
