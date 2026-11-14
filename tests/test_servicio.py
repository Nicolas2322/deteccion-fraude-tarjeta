"""Pruebas del servicio de puntuación y de la API.

Se comprueban las propiedades de las que depende la interfaz: que un esquema
incompleto produzca un error que nombre lo que falta, que la acción dependa
del umbral y no de un valor fijo, y que mover el costo mueva el umbral en la
dirección correcta.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from fraude.datos import COLUMNAS
from fraude.preprocesamiento import Escalador, transformar
from fraude.servicio import COLUMNAS_ENTRADA, Servicio


class ModeloFalso:
    """Devuelve una probabilidad fija, para aislar la lógica de decisión."""

    def __init__(self, p: float = 0.6) -> None:
        self.p = p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        columna = np.full(len(X), self.p)
        return np.column_stack([1 - columna, columna])


@pytest.fixture
def servicio(datos):
    escalador = Escalador().ajustar(transformar(datos))
    return Servicio(ModeloFalso(0.6), escalador, umbral=0.5, nombre="falso",
                    curva=[(1.0, 0.2, 900.0), (10.0, 0.6, 1200.0), (100.0, 0.9, 3000.0)])


@pytest.fixture
def entrada(datos):
    return datos.head(3)[COLUMNAS_ENTRADA].copy()


def test_columnas_de_entrada_son_el_esquema_sin_la_etiqueta():
    assert "Class" not in COLUMNAS_ENTRADA
    assert len(COLUMNAS_ENTRADA) == len(COLUMNAS) - 1


def test_falta_una_columna_y_el_error_la_nombra(servicio, entrada):
    with pytest.raises(ValueError, match="V17"):
        servicio.probabilidades(entrada.drop(columns=["V17"]))


def test_la_etiqueta_es_opcional(servicio, entrada):
    """El usuario final no conoce la etiqueta: pedirla sería absurdo."""
    con_etiqueta = entrada.copy()
    con_etiqueta["Class"] = 0
    assert np.allclose(servicio.probabilidades(entrada),
                       servicio.probabilidades(con_etiqueta))


def test_la_accion_depende_del_umbral(servicio, entrada):
    assert servicio.juzgar(entrada, umbral=0.5)[0].accion == "bloquear"
    assert servicio.juzgar(entrada, umbral=0.95)[0].accion == "permitir"


def test_existe_una_franja_de_revision(servicio, entrada):
    """Justo por debajo del umbral la acción es revisar, no permitir."""
    acciones = {servicio.juzgar(entrada, umbral=u)[0].accion
                for u in (0.5, 0.7, 0.95)}
    assert acciones == {"bloquear", "revisar", "permitir"}


def test_el_riesgo_esperado_es_probabilidad_por_importe(servicio, entrada):
    v = servicio.juzgar(entrada)[0]
    assert v.riesgo_esperado == pytest.approx(v.probabilidad * v.importe)


def test_el_umbral_crece_con_el_costo(servicio):
    """Si molestar a un cliente cuesta más, hay que estar más seguro."""
    umbrales = [servicio.umbral_para(c) for c in (1.0, 5.0, 20.0, 100.0)]
    assert umbrales == sorted(umbrales)


def test_el_costo_fuera_de_rango_no_extrapola(servicio):
    assert servicio.umbral_para(0.01) == pytest.approx(0.2)
    assert servicio.umbral_para(10_000) == pytest.approx(0.9)


def test_el_resumen_del_lote_cuadra(servicio, datos):
    veredictos = servicio.juzgar(datos.head(40)[COLUMNAS_ENTRADA])
    r = servicio.resumen_lote(veredictos)
    assert r["bloquear"] + r["revisar"] + r["permitir"] == r["transacciones"] == 40


def test_guardar_y_cargar_conserva_el_umbral(servicio, tmp_path):
    ruta = servicio.guardar(tmp_path / "s.joblib")
    recuperado = Servicio.cargar(ruta)
    assert recuperado.umbral == servicio.umbral
    assert recuperado.curva == servicio.curva


# ───────────── API ─────────────


@pytest.fixture
def cliente(servicio, monkeypatch):
    pytest.importorskip("fastapi")
    from app import api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "_servicio", servicio)
    return TestClient(api.aplicacion)


def test_estado_expone_lo_que_la_interfaz_necesita(cliente):
    cuerpo = cliente.get("/api/estado").json()
    assert cuerpo["columnas_requeridas"] == COLUMNAS_ENTRADA
    assert 0 <= cuerpo["umbral"] <= 1
    assert cuerpo["curva"]


def test_una_transaccion_devuelve_veredicto(cliente, entrada):
    r = cliente.post("/api/transaccion", json={"valores": entrada.iloc[0].to_dict()})
    assert r.status_code == 200
    assert r.json()["accion"] in {"bloquear", "revisar", "permitir"}


def test_faltan_campos_y_la_respuesta_dice_cuales(cliente, entrada):
    valores = entrada.iloc[0].to_dict()
    del valores["V3"]
    r = cliente.post("/api/transaccion", json={"valores": valores})
    assert r.status_code == 422
    assert "V3" in r.json()["detail"]


def test_lote_devuelve_resumen_y_filas(cliente, entrada):
    csv = io.BytesIO(entrada.to_csv(index=False).encode())
    r = cliente.post("/api/lote", files={"archivo": ("lote.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["resumen"]["transacciones"] == len(entrada)


def test_lote_rechaza_un_archivo_que_no_es_csv(cliente):
    r = cliente.post("/api/lote",
                     files={"archivo": ("notas.txt", io.BytesIO(b"hola"), "text/plain")})
    assert r.status_code == 422


def test_lote_rechaza_un_csv_sin_las_columnas(cliente):
    csv = io.BytesIO(pd.DataFrame({"a": [1, 2]}).to_csv(index=False).encode())
    r = cliente.post("/api/lote", files={"archivo": ("malo.csv", csv, "text/csv")})
    assert r.status_code == 422
    assert "Faltan" in r.json()["detail"]


def test_el_umbral_de_la_peticion_manda_sobre_el_del_modelo(cliente, entrada):
    valores = entrada.iloc[0].to_dict()
    bajo = cliente.post("/api/transaccion", json={"valores": valores, "umbral": 0.1})
    alto = cliente.post("/api/transaccion", json={"valores": valores, "umbral": 0.99})
    assert bajo.json()["accion"] == "bloquear"
    assert alto.json()["accion"] == "permitir"
