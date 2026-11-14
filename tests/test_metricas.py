import numpy as np
import pytest

from fraude.metricas import (
    auprc,
    error_calibracion,
    informe,
    linea_base_auprc,
    metricas_en_umbral,
    roc_auc,
)


def test_la_linea_base_del_auprc_es_la_prevalencia(prediccion):
    _, y, _ = prediccion
    assert linea_base_auprc(y) == pytest.approx(y.mean())


def test_un_clasificador_sin_informacion_se_queda_en_la_prevalencia():
    """Con puntuaciones aleatorias el AUPRC debe rondar la prevalencia, no 0,5."""
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.02).astype(int)
    p = rng.random(20_000)
    assert auprc(y, p) == pytest.approx(y.mean(), abs=0.01)
    assert roc_auc(y, p) == pytest.approx(0.5, abs=0.03)


def test_un_ordenamiento_perfecto_llega_a_uno():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert auprc(y, p) == pytest.approx(1.0)
    assert roc_auc(y, p) == pytest.approx(1.0)


def test_la_calibracion_perfecta_da_error_nulo():
    """Si en cada caja la frecuencia observada iguala la probabilidad, el error es cero."""
    p = np.repeat([0.05, 0.25, 0.45, 0.65, 0.85], 2000)
    rng = np.random.default_rng(11)
    y = (rng.random(len(p)) < p).astype(int)
    assert error_calibracion(y, p) < 0.02


def test_la_calibracion_detecta_un_desplazamiento():
    """Un modelo que exagera la probabilidad debe acusar error de calibración."""
    rng = np.random.default_rng(5)
    y = (rng.random(5000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.6 * y + 0.3, 0, 1)
    assert error_calibracion(y, p) > 0.2


def test_las_cuentas_en_el_umbral_son_exactas():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    importes = np.array([100.0, 40.0, 7.0, 3.0, 1.0])
    m = metricas_en_umbral(y, p, importes, umbral=0.5, c=2.0)
    assert m["fraudes_detectados"] == 1          # solo el de 0,9
    assert m["fraudes_no_detectados"] == 1       # el de 0,2
    assert m["alertas_falsas"] == 1              # el legítimo de 0,8
    assert m["exhaustividad"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["importe_no_detectado"] == pytest.approx(40.0)
    assert m["costo_total"] == pytest.approx(40.0 + 2.0)


def test_el_informe_reune_todas_las_claves(prediccion):
    p, y, importes = prediccion
    r = informe(y, p, importes, umbral=0.5, c=5.0)
    esperadas = {"auprc", "auprc_linea_base", "roc_auc", "error_calibracion",
                 "umbral", "exhaustividad", "precision", "especificidad",
                 "fraudes_detectados", "fraudes_no_detectados", "alertas_falsas",
                 "importe_no_detectado", "costo_total"}
    assert esperadas <= set(r)
    assert all(np.isfinite(v) for v in r.values())
