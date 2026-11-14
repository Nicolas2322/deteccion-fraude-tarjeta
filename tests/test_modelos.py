import numpy as np
import pytest

from fraude.modelos import CATALOGO, ModeloNoDisponible, crear, familia


def test_el_catalogo_cubre_las_tres_familias():
    assert set(CATALOGO.values()) == {"lineal", "arboles", "red"}
    assert familia("lightgbm") == "arboles"
    assert familia("mlp") == "red"


def test_un_modelo_desconocido_falla_con_claridad():
    with pytest.raises(KeyError, match="Modelo desconocido"):
        crear("random_forest")


@pytest.mark.parametrize("nombre", ["logistica", "mlp"])
def test_los_modelos_sin_dependencias_pesadas_entrenan_y_predicen(nombre):
    """Contrato común: todos exponen fit y predict_proba y devuelven probabilidades."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8))
    y = (X[:, 0] + rng.normal(scale=0.4, size=600) > 1.2).astype(int)
    modelo = crear(nombre, semilla=42).fit(X, y)
    p = modelo.predict_proba(X)[:, 1]
    assert p.shape == (600,)
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("nombre", ["lightgbm", "xgboost", "ft_transformer"])
def test_los_modelos_pesados_avisan_si_falta_la_biblioteca(nombre):
    """La importación es diferida: el pipeline funciona sin estas dependencias."""
    try:
        crear(nombre)
    except ModeloNoDisponible as exc:
        assert "pip install" in str(exc) or "PyTorch" in str(exc)
