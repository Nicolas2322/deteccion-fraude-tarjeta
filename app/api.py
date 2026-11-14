"""API del detector de fraude.

Expone el servicio de puntuación por HTTP y sirve la interfaz web. Se mantiene
deliberadamente delgada: no contiene lógica de negocio ni de modelado, solo
traduce entre peticiones HTTP y el objeto `Servicio`. Toda decisión sobre qué
significa una probabilidad vive en `fraude.servicio`.

Arranque:
    uvicorn app.api:aplicacion --reload
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraude.datos import EsquemaInvalido
from fraude.servicio import COLUMNAS_ENTRADA, ModeloNoCargado, Servicio

RAIZ = Path(__file__).resolve().parent
WEB = RAIZ / "web"
ARTEFACTO = RAIZ / "servicio.joblib"
MAXIMO_FILAS = 20_000

aplicacion = FastAPI(
    title="Detector de transacciones fraudulentas",
    description="Puntúa transacciones y convierte la probabilidad en una decisión "
                "según el costo que la institución atribuye a una alerta falsa.",
    version="1.0",
)

_servicio: Servicio | None = None


def servicio() -> Servicio:
    """Carga el artefacto una sola vez y lo reutiliza entre peticiones."""
    global _servicio
    if _servicio is None:
        try:
            _servicio = Servicio.cargar(ARTEFACTO)
        except ModeloNoCargado as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _servicio


class Transaccion(BaseModel):
    """Una transacción tal como la envía el formulario.

    Se acepta un diccionario abierto en lugar de treinta campos declarados
    porque las componentes V1 a V28 no tienen significado propio y enumerarlas
    en el modelo de datos no añadiría ninguna validación útil; la validación
    real la hace el esquema del pipeline, que es el mismo que se usa al
    entrenar.
    """

    valores: dict[str, float] = Field(..., description="Time, V1…V28 y Amount")
    umbral: float | None = Field(None, ge=0, le=1,
                                 description="Umbral a aplicar; por defecto, el del modelo")


@aplicacion.get("/api/estado")
def estado() -> dict[str, Any]:
    """Configuración vigente: modelo, umbral, curva de costo y columnas requeridas."""
    return servicio().estado()


@aplicacion.get("/api/umbral")
def umbral_para(costo: float = 5.0) -> dict[str, float]:
    """Umbral óptimo para un costo de alerta falsa dado."""
    if costo <= 0:
        raise HTTPException(status_code=422, detail="El costo debe ser mayor que cero.")
    return {"costo": costo, "umbral": servicio().umbral_para(costo)}


@aplicacion.post("/api/transaccion")
def juzgar_una(entrada: Transaccion) -> dict[str, Any]:
    """Veredicto de una sola transacción."""
    faltan = [c for c in COLUMNAS_ENTRADA if c not in entrada.valores]
    if faltan:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan {len(faltan)} campos: {', '.join(faltan[:8])}"
                   + (" …" if len(faltan) > 8 else ""),
        )
    df = pd.DataFrame([entrada.valores])
    try:
        veredicto = servicio().juzgar(df, entrada.umbral)[0]
    except (EsquemaInvalido, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return veredicto.a_diccionario()


@aplicacion.post("/api/lote")
async def juzgar_lote(archivo: UploadFile = File(...),
                      umbral: float | None = None) -> dict[str, Any]:
    """Veredicto de un archivo CSV completo, con su resumen."""
    if not archivo.filename or not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422,
                            detail="El archivo debe ser un CSV con las columnas del esquema.")
    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es del usuario
        raise HTTPException(status_code=422,
                            detail=f"No se pudo leer el CSV: {exc}") from exc
    if len(df) > MAXIMO_FILAS:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo tiene {len(df):,} filas y el máximo por envío es "
                   f"{MAXIMO_FILAS:,}. Divídelo en partes.",
        )
    try:
        veredictos = servicio().juzgar(df, umbral)
    except (EsquemaInvalido, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ordenados = sorted(range(len(veredictos)),
                       key=lambda i: veredictos[i].riesgo_esperado, reverse=True)
    return {
        "resumen": servicio().resumen_lote(veredictos),
        "umbral": veredictos[0].umbral if veredictos else servicio().umbral,
        "filas": [{"fila": int(i) + 2, **veredictos[i].a_diccionario()}
                  for i in ordenados[:200]],
    }


@aplicacion.get("/")
def inicio() -> FileResponse:
    return FileResponse(WEB / "index.html")


if WEB.exists():
    aplicacion.mount("/web", StaticFiles(directory=WEB), name="web")
