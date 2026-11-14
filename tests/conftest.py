"""Generador de conjuntos sintéticos con la misma forma que el archivo original.

Las pruebas no dependen del archivo real: comprueban propiedades del código, no
cifras concretas del conjunto. Eso permite que la integración continua se ejecute
sin descargar 150 MB y sin credenciales de Kaggle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraude.datos import COMPONENTES


def conjunto_sintetico(n: int = 2000, prevalencia: float = 0.02,
                       semilla: int = 0, horas: float = 48.0) -> pd.DataFrame:
    """Devuelve un conjunto con el esquema de Credit Card Fraud.

    Los fraudes reciben un desplazamiento en los primeros componentes para que
    exista señal y las métricas no queden degeneradas.
    """
    rng = np.random.default_rng(semilla)
    n_fraude = max(1, int(round(n * prevalencia)))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_fraude, replace=False)] = 1

    componentes = rng.normal(size=(n, len(COMPONENTES)))
    componentes[y == 1, :3] += 2.2

    df = pd.DataFrame(componentes, columns=COMPONENTES)
    df.insert(0, "Time", np.sort(rng.uniform(0, horas * 3600, size=n)))
    df["Amount"] = np.round(rng.lognormal(mean=3.0, sigma=1.2, size=n), 2)
    df["Class"] = y
    return df


@pytest.fixture
def datos() -> pd.DataFrame:
    return conjunto_sintetico()


@pytest.fixture
def prediccion():
    """Probabilidades, etiquetas e importes de un problema pequeño y controlable."""
    rng = np.random.default_rng(7)
    n = 400
    y = (rng.random(n) < 0.08).astype(int)
    p = np.clip(rng.beta(2, 8, size=n) + 0.45 * y, 0, 1)
    importes = np.round(rng.lognormal(3.0, 1.0, size=n), 2)
    return p, y, importes
