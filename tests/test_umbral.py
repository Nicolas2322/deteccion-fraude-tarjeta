import numpy as np
import pytest

from fraude.umbral import (
    costo_total,
    curva_sensibilidad,
    decidir,
    umbral_optimo,
    umbral_por_importe,
)


def _optimo_por_fuerza_bruta(p, y, importes, c):
    """Evalúa el costo en todos los umbrales candidatos. Referencia lenta pero obvia."""
    candidatos = np.unique(np.concatenate([p, [np.nextafter(p.max(), np.inf)]]))
    costos = [costo_total(p, y, importes, c, t) for t in candidatos]
    return float(min(costos))


@pytest.mark.parametrize("c", [0.0, 1.0, 5.0, 25.0, 500.0, 1e9])
def test_el_algoritmo_coincide_con_la_fuerza_bruta(prediccion, c):
    """La versión vectorizada del Algoritmo 1 debe dar el mismo mínimo que la definición."""
    p, y, importes = prediccion
    assert umbral_optimo(p, y, importes, c).costo == pytest.approx(
        _optimo_por_fuerza_bruta(p, y, importes, c))


def test_el_costo_declarado_coincide_con_el_recalculado(prediccion):
    p, y, importes = prediccion
    d = umbral_optimo(p, y, importes, 10.0)
    assert costo_total(p, y, importes, 10.0, d.umbral) == pytest.approx(d.costo)
    assert d.fraudes_no_detectados >= 0 and d.alertas_falsas >= 0


def test_con_alerta_falsa_gratuita_conviene_bloquearlo_todo(prediccion):
    """Si c = 0 no hay razón para dejar pasar nada: el costo mínimo es cero."""
    p, y, importes = prediccion
    assert umbral_optimo(p, y, importes, 0.0).costo == pytest.approx(0.0)


def test_con_alerta_falsa_carisima_no_se_acepta_ninguna(prediccion):
    """Si bloquear una legítima es prohibitivo, el óptimo solo corta donde no hay ninguna.

    No implica dejarlo pasar todo: si las probabilidades más altas corresponden
    únicamente a fraudes, bloquear ese prefijo sale gratis en alertas falsas.
    """
    p, y, importes = prediccion
    d = umbral_optimo(p, y, importes, 1e9)
    assert d.alertas_falsas == 0
    assert d.costo <= importes[y == 1].sum() + 1e-9


def test_el_costo_crece_con_el_precio_de_la_alerta_falsa(prediccion):
    p, y, importes = prediccion
    costos = [umbral_optimo(p, y, importes, c).costo for c in (1, 5, 20, 100)]
    assert costos == sorted(costos)


def test_rechaza_un_costo_negativo(prediccion):
    p, y, importes = prediccion
    with pytest.raises(ValueError):
        umbral_optimo(p, y, importes, -1.0)


def test_la_curva_de_sensibilidad_coincide_punto_a_punto(prediccion):
    p, y, importes = prediccion
    rejilla = np.array([0.5, 2.0, 10.0, 50.0])
    curva = curva_sensibilidad(p, y, importes, rejilla)
    assert len(curva) == len(rejilla)
    for c, _, costo in curva:
        assert costo == pytest.approx(umbral_optimo(p, y, importes, c).costo)


def test_el_umbral_por_importe_decrece_con_el_importe():
    """A mayor cantidad en juego, menos evidencia se exige para bloquear."""
    importes = np.array([0.0, 1.0, 10.0, 100.0, 10_000.0])
    t = umbral_por_importe(importes, c=5.0)
    assert np.all(np.diff(t) < 0)
    assert t[0] == pytest.approx(1.0)
    assert t[-1] < 0.01


def test_el_umbral_por_importe_esta_entre_cero_y_uno():
    importes = np.array([0.0, 0.5, 3.0, 900.0])
    t = umbral_por_importe(importes, c=2.0)
    assert np.all((t >= 0) & (t <= 1))


def test_decidir_aplica_la_regla_dependiente_del_importe():
    p = np.array([0.30, 0.30])
    importes = np.array([1.0, 1000.0])          # umbrales 0,833 y 0,005
    assert decidir(p, importes, c=5.0).tolist() == [False, True]


def test_conjunto_vacio_no_rompe():
    vacio = np.array([])
    d = umbral_optimo(vacio, vacio, vacio, 1.0)
    assert d.costo == 0.0
