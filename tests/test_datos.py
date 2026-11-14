import numpy as np
import pytest

from conftest import conjunto_sintetico
from fraude.datos import (
    COLUMNAS,
    EsquemaInvalido,
    cargar,
    particion_temporal,
    pliegues_estratificados,
    resumir,
    validar_esquema,
)


def test_el_esquema_valido_pasa(datos):
    validar_esquema(datos)


@pytest.mark.parametrize("romper,mensaje", [
    (lambda d: d.drop(columns=["V5"]), "Faltan columnas"),
    (lambda d: d.assign(Amount=d.Amount.astype(str)), "no numéricas"),
    (lambda d: d.assign(Class=2), "Class debe ser"),
    (lambda d: d.assign(V1=np.nan), "valores ausentes"),
    (lambda d: d.assign(Amount=-1.0), "importes negativos"),
])
def test_el_esquema_invalido_falla_pronto(datos, romper, mensaje):
    """Un archivo mal formado debe detenerse aquí y no varios módulos después."""
    with pytest.raises(EsquemaInvalido, match=mensaje):
        validar_esquema(romper(datos))


def test_cargar_ordena_en_el_tiempo(tmp_path):
    desordenado = conjunto_sintetico(n=300).sample(frac=1, random_state=1)
    ruta = tmp_path / "t.csv"
    desordenado.to_csv(ruta, index=False)
    df = cargar(ruta)
    assert list(df.columns) == COLUMNAS or set(COLUMNAS) <= set(df.columns)
    assert df["Time"].is_monotonic_increasing


def test_resumen_cuenta_bien(datos):
    r = resumir(datos)
    assert r.filas == len(datos)
    assert r.fraudes == int(datos.Class.sum())
    assert r.prevalencia == pytest.approx(r.fraudes / r.filas)
    assert r.importe_total_fraude == pytest.approx(datos.loc[datos.Class == 1, 'Amount'].sum())
    assert r.duracion_horas > 0


def test_la_particion_temporal_no_mezcla_periodos(datos):
    ent, prueba = particion_temporal(datos, 0.25)
    assert len(ent) + len(prueba) == len(datos)
    assert ent["Time"].max() < prueba["Time"].min()
    assert len(prueba) / len(datos) == pytest.approx(0.25, abs=0.05)


def test_la_particion_temporal_rechaza_fracciones_absurdas(datos):
    for f in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            particion_temporal(datos, f)


def test_los_pliegues_conservan_la_proporcion(datos):
    y = datos["Class"].to_numpy()
    pliegues = pliegues_estratificados(y, 5)
    assert len(pliegues) == 5
    objetivo = y.mean()
    for _, val in pliegues:
        assert y[val].mean() == pytest.approx(objetivo, abs=0.02)


def test_los_pliegues_cubren_todo_sin_solaparse(datos):
    y = datos["Class"].to_numpy()
    vistos = np.concatenate([val for _, val in pliegues_estratificados(y, 5)])
    assert sorted(vistos.tolist()) == list(range(len(y)))


def test_estratificar_falla_si_hay_menos_fraudes_que_pliegues():
    y = np.array([0] * 50 + [1, 1])
    with pytest.raises(ValueError, match="menos fraudes que pliegues"):
        pliegues_estratificados(y, 5)
