"""Pruebas del FT-Transformer.

Se comprueban propiedades de la arquitectura y del entrenamiento, no cifras de
rendimiento: el objetivo es que un fallo en el código se note, no medir cuánto
aprende el modelo. Todas las pruebas usan configuraciones diminutas para que la
integración continua siga tardando menos de un minuto.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fraude.ft_transformer import Configuracion, FTTransformer, Red, Tokenizador  # noqa: E402

MINIMA = dict(d_token=8, n_bloques=1, n_cabezas=2, epocas_max=2, tamano_lote=64)


@pytest.fixture
def separable():
    """Problema linealmente separable y desbalanceado, con pocas filas."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)).astype("float32")
    y = (X[:, 0] + X[:, 2] > 1.6).astype("float32")
    return X, y


def test_tokenizador_da_un_token_por_variable_mas_cls():
    tok = Tokenizador(n_variables=6, d_token=8)
    salida = tok(torch.zeros(5, 6))
    assert salida.shape == (5, 7, 8), "faltan tokens o sobra alguno"


def test_tokenizador_distingue_variables_con_el_mismo_valor():
    """El mismo número en dos variables debe producir tokens distintos."""
    tok = Tokenizador(n_variables=3, d_token=8)
    tokens = tok(torch.full((1, 3), 0.5))[0, 1:]
    assert not torch.allclose(tokens[0], tokens[1])


def test_la_red_devuelve_un_logit_por_fila():
    red = Red(6, Configuracion(**MINIMA))
    assert red(torch.zeros(5, 6)).shape == (5,)


def test_d_token_debe_ser_multiplo_de_las_cabezas():
    with pytest.raises(ValueError, match="múltiplo"):
        Configuracion(d_token=10, n_cabezas=4)


def test_predict_proba_devuelve_probabilidades_validas(separable):
    X, y = separable
    p = FTTransformer(**MINIMA).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_la_validacion_interna_recibe_positivos(separable):
    """Con prevalencia baja, un corte aleatorio podría dejarla sin ninguno.

    El reparto es estratificado justamente para impedirlo. Si no lo fuera, el
    AUPRC de la detección temprana no estaría definido y el historial saldría
    con valores nulos.
    """
    X, y = separable
    m = FTTransformer(**MINIMA).fit(X, y)
    assert len(m.historial_) >= 1
    assert all(np.isfinite(v) for v in m.historial_)


def test_la_misma_semilla_da_el_mismo_resultado(separable):
    X, y = separable
    a = FTTransformer(semilla=7, **MINIMA).fit(X, y).predict_proba(X)[:, 1]
    b = FTTransformer(semilla=7, **MINIMA).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(a, b)


def test_semillas_distintas_dan_resultados_distintos(separable):
    X, y = separable
    a = FTTransformer(semilla=1, **MINIMA).fit(X, y).predict_proba(X)[:, 1]
    b = FTTransformer(semilla=2, **MINIMA).fit(X, y).predict_proba(X)[:, 1]
    assert not np.allclose(a, b)


def test_la_deteccion_temprana_recupera_los_mejores_pesos(separable):
    """Tras entrenar, el modelo debe quedar en la época de mejor validación."""
    X, y = separable
    m = FTTransformer(**{**MINIMA, "epocas_max": 5, "paciencia": 1}).fit(X, y)
    assert m.epoca_mejor_ == int(np.argmax(m.historial_))


def test_aprende_un_problema_separable(separable):
    """Única prueba de rendimiento, con un umbral deliberadamente flojo.

    No mide calidad sino que descarta que la red esté desconectada: un modelo
    cuyo gradiente no llega a los pesos daría un AUPRC igual a la prevalencia.
    """
    from fraude.metricas import auprc

    X, y = separable
    m = FTTransformer(**{**MINIMA, "epocas_max": 30, "d_token": 16}).fit(X, y)
    assert auprc(y, m.predict_proba(X)[:, 1]) > 2 * y.mean()


def test_get_params_y_set_params(separable):
    m = FTTransformer(**MINIMA)
    assert m.get_params()["d_token"] == 8
    m.set_params(d_token=16)
    assert m.get_params()["d_token"] == 16
