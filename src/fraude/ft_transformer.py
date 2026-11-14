"""FT-Transformer para datos tabulares (Gorishniy et al., 2021).

La arquitectura adapta el Transformer a una tabla. La idea central es que cada
variable de una fila se convierte en un vector —un *token*— igual que cada
palabra de una frase se convierte en un vector en procesamiento de lenguaje. A
esos tokens se les añade uno artificial, [CLS], que no corresponde a ninguna
variable y cuya única función es acumular información de todos los demás a
través de la atención. La predicción se lee de ese token al final.

El interés de la arquitectura para este proyecto es que la atención relaciona
cada variable con todas las demás sin privilegiar ninguna dirección del
espacio, mientras que un árbol solo puede partir según los ejes. Grinsztajn et
al. (2022) sostienen que esa diferencia explica buena parte de la ventaja de
los árboles en datos tabulares, y que la ventaja debería reducirse cuando los
datos ya están rotados. Las variables V1 a V28 de este conjunto son componentes
principales, es decir, una rotación. El proyecto pone esa predicción a prueba.

Solo se implementan variables numéricas: el conjunto no tiene categóricas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Configuracion:
    """Hiperparámetros del modelo y de su entrenamiento."""

    d_token: int = 64
    n_bloques: int = 3
    n_cabezas: int = 8
    factor_ffn: float = 4 / 3
    abandono_atencion: float = 0.2
    abandono_ffn: float = 0.1
    abandono_residual: float = 0.0
    tasa_aprendizaje: float = 1e-3
    decaimiento_pesos: float = 1e-5
    tamano_lote: int = 1024
    epocas_max: int = 30
    paciencia: int = 5
    fraccion_validacion: float = 0.15
    peso_positivo: float | None = None
    semilla: int = 42
    dispositivo: str = field(default="cpu")

    def __post_init__(self) -> None:
        if self.d_token % self.n_cabezas:
            raise ValueError(
                f"d_token ({self.d_token}) debe ser múltiplo de n_cabezas ({self.n_cabezas})"
            )


class Tokenizador(nn.Module):
    """Convierte cada variable numérica en un vector y antepone el token [CLS].

    Cada variable j tiene su propio vector de peso w_j y su sesgo b_j, de modo
    que el token de la variable j en una fila es x_j · w_j + b_j. Dos variables
    con el mismo valor numérico producen tokens distintos, que es justo lo que
    se quiere: el valor 0,5 no significa lo mismo en V3 que en el importe.
    """

    def __init__(self, n_variables: int, d_token: int) -> None:
        super().__init__()
        self.pesos = nn.Parameter(torch.empty(n_variables, d_token))
        self.sesgos = nn.Parameter(torch.empty(n_variables, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        # Inicialización uniforme con la escala de Gorishniy et al. (2021).
        limite = 1 / math.sqrt(d_token)
        for p in (self.pesos, self.sesgos, self.cls):
            nn.init.uniform_(p, -limite, limite)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (lote, n_variables) -> (lote, n_variables + 1, d_token)
        tokens = x.unsqueeze(-1) * self.pesos + self.sesgos
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class ReGLU(nn.Module):
    """Unidad lineal con compuerta rectificada; la que usa el artículo original."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class Bloque(nn.Module):
    """Un bloque Transformer en variante pre-norma.

    Se normaliza antes de cada subcapa y no después. Esa variante entrena de
    forma más estable sin necesidad de calentamiento de la tasa de aprendizaje,
    que es lo que conviene aquí: el presupuesto de cómputo es limitado y no
    sobra para ajustar un programa de calentamiento.
    """

    def __init__(self, cfg: Configuracion, primero: bool) -> None:
        super().__init__()
        self.primero = primero
        self.norma_atencion = nn.LayerNorm(cfg.d_token)
        self.atencion = nn.MultiheadAttention(
            cfg.d_token, cfg.n_cabezas, dropout=cfg.abandono_atencion, batch_first=True
        )
        self.norma_ffn = nn.LayerNorm(cfg.d_token)
        d_oculto = int(cfg.d_token * cfg.factor_ffn)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_token, d_oculto * 2),
            ReGLU(),
            nn.Dropout(cfg.abandono_ffn),
            nn.Linear(d_oculto, cfg.d_token),
        )
        self.abandono_residual = nn.Dropout(cfg.abandono_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # El primer bloque no normaliza la entrada: ya viene del tokenizador.
        h = x if self.primero else self.norma_atencion(x)
        atendido, _ = self.atencion(h, h, h, need_weights=False)
        x = x + self.abandono_residual(atendido)
        h = self.norma_ffn(x)
        return x + self.abandono_residual(self.ffn(h))


class Red(nn.Module):
    """Tokenizador, pila de bloques y cabeza de clasificación."""

    def __init__(self, n_variables: int, cfg: Configuracion) -> None:
        super().__init__()
        self.tokenizador = Tokenizador(n_variables, cfg.d_token)
        self.bloques = nn.ModuleList(
            Bloque(cfg, primero=(i == 0)) for i in range(cfg.n_bloques)
        )
        self.norma_final = nn.LayerNorm(cfg.d_token)
        self.cabeza = nn.Linear(cfg.d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tokenizador(x)
        for bloque in self.bloques:
            h = bloque(h)
        # Solo se lee el token [CLS], que ocupa la posición cero.
        return self.cabeza(F.relu(self.norma_final(h[:, 0]))).squeeze(-1)


class FTTransformer:
    """Envoltorio con la interfaz de scikit-learn: `fit` y `predict_proba`.

    El resto del pipeline trata a este modelo igual que a la regresión
    logística o a LightGBM, de modo que la comparación entre familias no
    depende de ninguna rama condicional repartida por el código.

    El entrenamiento reserva una fracción de las filas para detenerse pronto.
    Esa fracción sale del conjunto de entrenamiento que recibe, nunca del
    pliegue de evaluación, de modo que la detección temprana no ve los datos
    sobre los que se mide.
    """

    def __init__(self, **kw: object) -> None:
        self.cfg = Configuracion(**kw)  # type: ignore[arg-type]
        self.red_: Red | None = None
        self.historial_: list[float] = []
        self.epoca_mejor_: int = -1

    # -- utilidades internas -------------------------------------------------

    def _semilla(self) -> None:
        torch.manual_seed(self.cfg.semilla)
        np.random.seed(self.cfg.semilla)

    def _auprc(self, y: np.ndarray, p: np.ndarray) -> float:
        from .metricas import auprc

        return auprc(y, p)

    def _predecir_logits(self, X: np.ndarray) -> np.ndarray:
        assert self.red_ is not None
        self.red_.eval()
        salidas = []
        with torch.no_grad():
            for (lote,) in DataLoader(
                TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
                batch_size=4096,
            ):
                salidas.append(self.red_(lote.to(self.cfg.dispositivo)).cpu().numpy())
        return np.concatenate(salidas)

    # -- interfaz pública ----------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> FTTransformer:
        self._semilla()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Reparto interno estratificado: la detección temprana necesita
        # positivos, y con una prevalencia del 0,17 % un corte aleatorio podría
        # dejar la validación interna sin ninguno.
        generador = np.random.default_rng(self.cfg.semilla)
        indices_val = []
        for clase in (0.0, 1.0):
            idx = np.flatnonzero(y == clase)
            generador.shuffle(idx)
            corte = max(1, int(len(idx) * self.cfg.fraccion_validacion))
            indices_val.append(idx[:corte])
        val = np.concatenate(indices_val)
        mascara = np.ones(len(y), dtype=bool)
        mascara[val] = False
        X_tr, y_tr, X_val, y_val = X[mascara], y[mascara], X[val], y[val]

        self.red_ = Red(X.shape[1], self.cfg).to(self.cfg.dispositivo)
        optimizador = torch.optim.AdamW(
            self.red_.parameters(),
            lr=self.cfg.tasa_aprendizaje,
            weight_decay=self.cfg.decaimiento_pesos,
        )
        peso = (
            None
            if self.cfg.peso_positivo is None
            else torch.tensor(self.cfg.peso_positivo, device=self.cfg.dispositivo)
        )
        perdida = nn.BCEWithLogitsLoss(pos_weight=peso)

        cargador = DataLoader(
            TensorDataset(
                torch.as_tensor(X_tr, dtype=torch.float32),
                torch.as_tensor(y_tr, dtype=torch.float32),
            ),
            batch_size=self.cfg.tamano_lote,
            shuffle=True,
            drop_last=False,
        )

        mejor, sin_mejora, mejores_pesos = -np.inf, 0, None
        for epoca in range(self.cfg.epocas_max):
            self.red_.train()
            for xb, yb in cargador:
                xb = xb.to(self.cfg.dispositivo)
                yb = yb.to(self.cfg.dispositivo)
                optimizador.zero_grad(set_to_none=True)
                perdida(self.red_(xb), yb).backward()
                optimizador.step()

            puntuacion = self._auprc(y_val, self._predecir_logits(X_val))
            self.historial_.append(puntuacion)
            if puntuacion > mejor:
                mejor, sin_mejora, self.epoca_mejor_ = puntuacion, 0, epoca
                mejores_pesos = {
                    k: v.detach().clone() for k, v in self.red_.state_dict().items()
                }
            else:
                sin_mejora += 1
                if sin_mejora >= self.cfg.paciencia:
                    break

        if mejores_pesos is not None:
            self.red_.load_state_dict(mejores_pesos)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._predecir_logits(np.asarray(X, dtype=np.float32))
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self.cfg, k) for k in self.cfg.__dataclass_fields__}

    def set_params(self, **kw: object) -> FTTransformer:
        for k, v in kw.items():
            setattr(self.cfg, k, v)
        return self
