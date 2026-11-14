import numpy as np
import pytest

from conftest import conjunto_sintetico
from fraude.pipeline import ejecutar, predicciones_fuera_de_particion


@pytest.fixture(scope="module")
def resultado():
    df = conjunto_sintetico(n=3000, prevalencia=0.03, semilla=4)
    return ejecutar(df, modelo="logistica", costo_alerta_falsa=5.0, n_pliegues=3)


def test_el_pipeline_recorre_de_extremo_a_extremo(resultado):
    assert resultado.modelo == "logistica"
    assert set(resultado.validacion_cruzada) == set(resultado.prueba_temporal)
    assert np.isfinite(list(resultado.validacion_cruzada.values())).all()
    assert np.isfinite(list(resultado.prueba_temporal.values())).all()


def test_cada_transaccion_recibe_una_prediccion_de_un_modelo_que_no_la_vio():
    df = conjunto_sintetico(n=1500, prevalencia=0.04, semilla=9)
    fuera = predicciones_fuera_de_particion(df, "logistica", 3, 42)
    assert len(fuera) == len(df)
    assert np.all((fuera >= 0) & (fuera <= 1))
    assert len(np.unique(fuera)) > 100          # no quedó ningún pliegue sin rellenar


def test_el_modelo_supera_con_holgura_la_linea_base_sin_informacion(resultado):
    cv = resultado.validacion_cruzada
    assert cv["auprc"] > 3 * cv["auprc_linea_base"]


def test_el_umbral_se_fija_en_desarrollo_y_se_reutiliza_en_prueba(resultado):
    assert resultado.decision["umbral"] == pytest.approx(resultado.validacion_cruzada["umbral"])
    assert resultado.decision["umbral"] == pytest.approx(resultado.prueba_temporal["umbral"])


def test_el_costo_declarado_coincide_con_el_de_las_metricas(resultado):
    assert resultado.decision["costo"] == pytest.approx(
        resultado.validacion_cruzada["costo_total"])


def test_la_curva_de_sensibilidad_crece_con_el_costo(resultado):
    costos = [c for _, _, c in resultado.sensibilidad]
    assert costos == sorted(costos)


def test_el_resultado_se_serializa(resultado):
    import json
    texto = json.dumps(resultado.a_diccionario(), ensure_ascii=False)
    assert "validacion_cruzada" in texto
