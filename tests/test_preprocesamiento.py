import numpy as np
import pandas as pd
import pytest

from fraude.preprocesamiento import (
    VARIABLES,
    Escalador,
    NoAjustado,
    hora_del_dia,
    preparar,
    transformar,
)


def test_la_hora_se_envuelve_cada_dia():
    segundos = np.array([0, 3600, 23 * 3600, 24 * 3600, 25 * 3600])
    assert hora_del_dia(segundos) == pytest.approx([0, 1, 23, 0, 1])


def test_transformar_produce_las_variables_esperadas(datos):
    X = transformar(datos)
    assert list(X.columns) == VARIABLES
    assert len(X) == len(datos)
    assert np.isfinite(X.to_numpy()).all()


def test_el_logaritmo_admite_importe_cero():
    df = pd.DataFrame({**{c: [0.0] for c in VARIABLES[:28]},
                       "Time": [0.0], "Amount": [0.0], "Class": [0]})
    assert transformar(df)["importe_log"].iloc[0] == pytest.approx(0.0)


def test_la_codificacion_ciclica_acerca_medianoche():
    """Las 23:00 y las 00:00 distan una hora, y la codificación debe reflejarlo."""
    df = pd.DataFrame({**{c: [0.0, 0.0, 0.0] for c in VARIABLES[:28]},
                       "Time": [23 * 3600, 0.0, 12 * 3600],
                       "Amount": [1.0, 1.0, 1.0], "Class": [0, 0, 0]})
    X = transformar(df)[["hora_sen", "hora_cos"]].to_numpy()
    cerca = np.linalg.norm(X[0] - X[1])
    lejos = np.linalg.norm(X[1] - X[2])
    assert cerca < lejos


def test_el_escalador_exige_ajuste_previo(datos):
    with pytest.raises(NoAjustado):
        Escalador().aplicar(transformar(datos))


def test_el_escalador_estandariza_el_entrenamiento(datos):
    X = transformar(datos)
    Z = Escalador().ajustar_y_aplicar(X)
    assert Z.mean(axis=0) == pytest.approx(np.zeros(Z.shape[1]), abs=1e-8)
    assert Z.std(axis=0) == pytest.approx(np.ones(Z.shape[1]), abs=1e-8)


def test_una_variable_constante_no_divide_por_cero(datos):
    datos = datos.copy()
    datos["V7"] = 3.0
    Z = Escalador().ajustar_y_aplicar(transformar(datos))
    assert np.isfinite(Z).all()


def test_la_evaluacion_se_escala_con_los_parametros_del_entrenamiento(datos):
    """Si la evaluación se escalara con sus propios parámetros, su media sería cero."""
    ent, eva = datos.iloc[:1500], datos.iloc[1500:].copy()
    eva[VARIABLES[0]] = eva[VARIABLES[0]] + 50.0
    _, _, X_eva, _, _ = preparar(ent, eva)
    assert abs(X_eva[:, 0].mean()) > 5.0


def test_preparar_devuelve_formas_coherentes(datos):
    ent, eva = datos.iloc[:1500], datos.iloc[1500:]
    X_ent, y_ent, X_eva, y_eva, importes = preparar(ent, eva)
    assert X_ent.shape == (len(ent), len(VARIABLES))
    assert X_eva.shape == (len(eva), len(VARIABLES))
    assert len(y_ent) == len(ent) and len(y_eva) == len(eva)
    assert importes == pytest.approx(eva["Amount"].to_numpy())
