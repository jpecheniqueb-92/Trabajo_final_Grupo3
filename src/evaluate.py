"""
Métricas, validación cruzada, intervalos de confianza y calibración.

Por qué estas métricas y no accuracy
------------------------------------
Con 20% de deserción, un clasificador que diga "nadie deserta" acierta el 80%.
Cualquier accuracy por debajo de 0.80 es peor que no hacer nada, y una por
encima puede lograrse sin detectar a un solo desertor. Por eso el reporte se
apoya en:

  ROC-AUC      capacidad de ordenar; independiente del umbral y de la prevalencia
  PR-AUC       promedio de precisión; es la métrica sensible al desbalance y la
               que refleja el trabajo real del equipo (de los que contacto,
               cuántos iban a desertar)
  Brier        calidad de la probabilidad en sí misma, no solo del orden. Importa
               porque el umbral de decisión se calcula sobre la probabilidad
  Recall@k     de los desertores reales, cuántos caen en los k contactos que el
               equipo puede hacer
  Lift@k       cuántas veces mejor que contactar al azar

Y sobre todas ellas, el beneficio neto en soles de src/costs.py, que es el
criterio con el que finalmente se elige el modelo y el umbral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config as C


# ---------------------------------------------------------------------------
# Métricas puntuales
# ---------------------------------------------------------------------------
def metricas(y_true, y_prob, umbral: float = 0.5, k: int | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= umbral).astype(int)

    res = {
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "LogLoss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    if k is not None and k > 0:
        orden = np.argsort(-y_prob)[:k]
        tp_k = int(y_true[orden].sum())
        res[f"Precision@{k}"] = tp_k / k
        res[f"Recall@{k}"] = tp_k / max(y_true.sum(), 1)
        res[f"Lift@{k}"] = (tp_k / k) / max(y_true.mean(), 1e-9)

    return res


def ic_bootstrap(
    y_true,
    y_prob,
    metrica=roc_auc_score,
    n_muestras: int = 1000,
    seed: int = C.SEED,
) -> tuple[float, float, float]:
    """
    Intervalo de confianza al 95% por bootstrap.

    Sin esto no se puede afirmar que un modelo es mejor que otro: con ~2000
    observaciones de prueba, dos AUC que difieren en 0.01 son indistinguibles.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    valores = []
    for _ in range(n_muestras):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        valores.append(metrica(y_true[idx], y_prob[idx]))

    valores = np.array(valores)
    return (
        float(metrica(y_true, y_prob)),
        float(np.percentile(valores, 2.5)),
        float(np.percentile(valores, 97.5)),
    )


# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
def cv_evaluar(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = C.CV_FOLDS,
    seed: int = C.SEED,
) -> dict:
    """
    Validación cruzada estratificada de 5 particiones sobre entrenamiento.

    El Pipeline completo (imputación, codificación, escalado y modelo) se
    reajusta dentro de cada fold, de modo que ningún estadístico de la
    partición de evaluación entra al entrenamiento.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    auc, pr_auc, brier = [], [], []

    for idx_tr, idx_va in skf.split(X, y):
        from sklearn.base import clone

        m = clone(modelo)
        m.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        p = m.predict_proba(X.iloc[idx_va])[:, 1]
        auc.append(roc_auc_score(y.iloc[idx_va], p))
        pr_auc.append(average_precision_score(y.iloc[idx_va], p))
        brier.append(brier_score_loss(y.iloc[idx_va], p))

    return {
        "cv_roc_auc_media": float(np.mean(auc)),
        "cv_roc_auc_std": float(np.std(auc)),
        "cv_pr_auc_media": float(np.mean(pr_auc)),
        "cv_pr_auc_std": float(np.std(pr_auc)),
        "cv_brier_media": float(np.mean(brier)),
        "cv_folds": [round(a, 4) for a in auc],
    }


# ---------------------------------------------------------------------------
# Sobreajuste
# ---------------------------------------------------------------------------
def brecha_sobreajuste(modelo, X_tr, y_tr, X_va, y_va) -> dict:
    """Diferencia de AUC entre entrenamiento y validación."""
    p_tr = modelo.predict_proba(X_tr)[:, 1]
    p_va = modelo.predict_proba(X_va)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_va = roc_auc_score(y_va, p_va)
    return {
        "auc_train": float(auc_tr),
        "auc_valid": float(auc_va),
        "brecha": float(auc_tr - auc_va),
    }


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------
def datos_calibracion(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """
    Curva de fiabilidad: de los alumnos a los que el modelo asigna 30% de riesgo,
    ¿cuántos desertaron realmente?

    Es indispensable en este proyecto porque el umbral de decisión se deriva de
    una comparación económica sobre la probabilidad. Si el modelo dice 0.30 pero
    la frecuencia real es 0.55, el umbral óptimo calculado está mal.
    """
    frac_pos, media_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return pd.DataFrame(
        {"prob_predicha": media_pred, "frecuencia_observada": frac_pos}
    )


def error_calibracion_esperado(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: desviación promedio entre probabilidad predicha y frecuencia real."""
    df = datos_calibracion(y_true, y_prob, n_bins)
    return float(np.mean(np.abs(df["prob_predicha"] - df["frecuencia_observada"])))


# ---------------------------------------------------------------------------
# Equidad
# ---------------------------------------------------------------------------
def desempeno_por_subgrupo(
    df: pd.DataFrame, y_true, y_prob, columnas: list[str], umbral: float
) -> pd.DataFrame:
    """
    Desempeño desagregado por subgrupos sensibles.

    Un modelo con buen AUC global puede funcionar mucho peor para un subgrupo,
    y en un caso educativo eso significa que un tipo de estudiante recibe
    sistemáticamente menos acompañamiento. La rúbrica pide aspectos éticos y
    condiciones para una implementación responsable: esto es lo que los sustenta.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= umbral).astype(int)

    filas = []
    for col in columnas:
        for valor, idx in df.groupby(col, observed=True).groups.items():
            pos = df.index.get_indexer(idx)
            if len(pos) < 40 or len(np.unique(y_true[pos])) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(
                y_true[pos], y_pred[pos], labels=[0, 1]
            ).ravel()
            filas.append(
                {
                    "variable": col,
                    "grupo": str(valor),
                    "n": len(pos),
                    "tasa_desercion_real": round(float(y_true[pos].mean()), 4),
                    "tasa_contacto": round(float(y_pred[pos].mean()), 4),
                    "ROC_AUC": round(float(roc_auc_score(y_true[pos], y_prob[pos])), 4),
                    "Recall": round(float(tp / max(tp + fn, 1)), 4),
                    "Precision": round(float(tp / max(tp + fp, 1)), 4),
                }
            )
    return pd.DataFrame(filas)
