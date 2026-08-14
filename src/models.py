"""
Definición de las líneas base y de los modelos candidatos.

Incluye un clasificador de red neuronal implementado en PyTorch con interfaz
compatible con scikit-learn, para que participe de la misma validación cruzada
y las mismas métricas que el resto de modelos y la comparación sea justa.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .preprocessing import construir_preprocesador


# ===========================================================================
# Línea base 1: regla de negocio vigente
# ===========================================================================
class ReglaDeNegocio(ClassifierMixin, BaseEstimator):
    """
    Formaliza el procedimiento que EduTech ya aplica hoy: marcar como en riesgo
    a quien lleva `dias_umbral` o más sin ingresar a la plataforma.

    Es la línea base que de verdad importa. Un modelo solo justifica su costo
    de desarrollo y mantenimiento si supera lo que la empresa ya hace sin él.
    Como score continuo se usan los días de inactividad, lo que permite calcular
    AUC y comparar la regla en el mismo plano que los modelos.
    """

    def __init__(self, dias_umbral: int = 10):
        self.dias_umbral = dias_umbral

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self._max_dias = float(np.nanmax(np.asarray(X["DiasUltimaSesion"], dtype=float)))
        return self

    def _dias(self, X):
        return np.nan_to_num(np.asarray(X["DiasUltimaSesion"], dtype=float), nan=0.0)

    def predict(self, X):
        return (self._dias(X) >= self.dias_umbral).astype(int)

    def predict_proba(self, X):
        p = np.clip(self._dias(X) / max(self._max_dias, 1.0), 0, 1)
        return np.column_stack([1 - p, p])


# ===========================================================================
# Red neuronal (MLP) en PyTorch
# ===========================================================================
class _RedMLP(nn.Module):
    """
    Arquitectura: entrada -> [Linear -> BatchNorm -> ReLU -> Dropout] x k -> Linear(1)

    BatchNorm estabiliza el entrenamiento con variables de escalas muy distintas
    y Dropout es la principal medida contra el sobreajuste, dado que la red tiene
    más parámetros que el número de variables del problema.
    """

    def __init__(self, n_entradas: int, capas=(64, 32), dropout: float = 0.30):
        super().__init__()
        bloques: list[nn.Module] = []
        prev = n_entradas
        for h in capas:
            bloques += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        bloques.append(nn.Linear(prev, 1))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x).squeeze(-1)


class MLPTorch(ClassifierMixin, BaseEstimator):
    """
    Clasificador de red neuronal con interfaz scikit-learn.

    Decisiones de entrenamiento, todas exigidas por la rúbrica:

    - Función de pérdida: entropía cruzada binaria sobre logits
      (BCEWithLogitsLoss), con `pos_weight` para compensar que solo ~20% de los
      estudiantes deserta. Es la pérdida natural para clasificación binaria
      probabilística y la que hace que la salida sea interpretable como
      probabilidad.
    - Optimizador: Adam con weight decay (regularización L2 sobre los pesos).
    - Contra el sobreajuste: Dropout 0.30, BatchNorm, weight decay y
      early stopping sobre una partición interna de validación (15% del
      entrenamiento), restaurando los pesos de la mejor época.
    - Reproducibilidad: semilla fija para pesos iniciales y orden de los lotes.
    """

    def __init__(
        self,
        capas=(64, 32),
        dropout: float = 0.30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epocas: int = 200,
        paciencia: int = 20,
        usar_pos_weight: bool = True,
        seed: int = C.SEED,
    ):
        self.capas = capas
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epocas = max_epocas
        self.paciencia = paciencia
        self.usar_pos_weight = usar_pos_weight
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Partición interna para early stopping (estratificada)
        idx = rng.permutation(len(X))
        n_val = max(int(0.15 * len(X)), 1)
        idx_val, idx_tr = idx[:n_val], idx[n_val:]

        Xtr = torch.tensor(X[idx_tr])
        ytr = torch.tensor(y[idx_tr])
        Xva = torch.tensor(X[idx_val])
        yva = torch.tensor(y[idx_val])

        self.modelo_ = _RedMLP(self.n_features_in_, tuple(self.capas), self.dropout)

        pos_weight = None
        if self.usar_pos_weight:
            n_pos = float(ytr.sum())
            n_neg = float(len(ytr) - n_pos)
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        opt = torch.optim.Adam(
            self.modelo_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        mejor_perdida = np.inf
        mejores_pesos = None
        sin_mejora = 0
        self.historial_ = {"train": [], "val": []}

        for _ in range(self.max_epocas):
            self.modelo_.train()
            orden = torch.randperm(len(Xtr))
            perdidas = []
            for i in range(0, len(Xtr), self.batch_size):
                lote = orden[i : i + self.batch_size]
                if len(lote) < 2:  # BatchNorm necesita al menos 2 observaciones
                    continue
                opt.zero_grad()
                perdida = criterio(self.modelo_(Xtr[lote]), ytr[lote])
                perdida.backward()
                opt.step()
                perdidas.append(perdida.item())

            self.modelo_.eval()
            with torch.no_grad():
                perdida_val = criterio(self.modelo_(Xva), yva).item()
            self.historial_["train"].append(float(np.mean(perdidas)))
            self.historial_["val"].append(perdida_val)

            if perdida_val < mejor_perdida - 1e-5:
                mejor_perdida = perdida_val
                mejores_pesos = {
                    k: v.clone() for k, v in self.modelo_.state_dict().items()
                }
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= self.paciencia:
                    break

        if mejores_pesos is not None:
            self.modelo_.load_state_dict(mejores_pesos)
        self.epocas_entrenadas_ = len(self.historial_["train"])
        return self

    def predict_proba(self, X):
        self.modelo_.eval()
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(self.modelo_(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ===========================================================================
# Registro de modelos
# ===========================================================================
def construir_modelos(scale_pos_weight: float) -> dict[str, Pipeline]:
    """
    Devuelve el diccionario de modelos, cada uno como un Pipeline que incluye
    su propio preprocesamiento. Así el preprocesamiento se reajusta dentro de
    cada fold de validación cruzada y no hay fuga de información.
    """
    return {
        # ---------------- Líneas base ----------------
        "Línea base: prevalencia": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=False)),
                ("modelo", DummyClassifier(strategy="prior", random_state=C.SEED)),
            ]
        ),
        # ---------------- Modelos lineales ----------------
        "Regresión Logística": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=True)),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=C.SEED,
                    ),
                ),
            ]
        ),
        # ---------------- Modelos de árboles ----------------
        "Árbol de Decisión": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=False)),
                (
                    "modelo",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        random_state=C.SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=False)),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=12,
                        min_samples_leaf=15,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=C.SEED,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=False)),
                (
                    "modelo",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.5,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                        random_state=C.SEED,
                    ),
                ),
            ]
        ),
        # ---------------- Deep Learning ----------------
        "Red Neuronal (MLP)": Pipeline(
            [
                ("prep", construir_preprocesador(escalar=True)),
                ("modelo", MLPTorch()),
            ]
        ),
    }


# Espacios de búsqueda de hiperparámetros para la selección por validación cruzada
ESPACIOS_BUSQUEDA: dict[str, dict] = {
    "Regresión Logística": {
        "modelo__C": [0.01, 0.1, 0.5, 1.0, 5.0],
    },
    "Árbol de Decisión": {
        "modelo__max_depth": [3, 4, 5, 6, 8],
        "modelo__min_samples_leaf": [20, 50, 100],
    },
    "Random Forest": {
        "modelo__n_estimators": [300, 500],
        "modelo__max_depth": [8, 12, 16, None],
        "modelo__min_samples_leaf": [5, 15, 30],
    },
    "XGBoost": {
        "modelo__n_estimators": [300, 500, 700],
        "modelo__learning_rate": [0.03, 0.05, 0.1],
        "modelo__max_depth": [3, 4, 5, 6],
        "modelo__min_child_weight": [1, 5, 10],
        "modelo__subsample": [0.7, 0.85, 1.0],
        "modelo__colsample_bytree": [0.7, 0.85, 1.0],
        "modelo__reg_lambda": [0.5, 1.5, 5.0],
    },
    "Red Neuronal (MLP)": {
        "modelo__capas": [(32,), (64, 32), (128, 64), (128, 64, 32)],
        "modelo__dropout": [0.15, 0.30, 0.45],
        "modelo__lr": [3e-4, 1e-3, 3e-3],
        "modelo__weight_decay": [1e-5, 1e-4, 1e-3],
    },
}
