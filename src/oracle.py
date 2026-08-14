"""
Diagnóstico con el oráculo: ¿cuánta señal había realmente para capturar?

Este módulo es el que convierte la principal debilidad del proyecto -trabajar
con datos sintéticos- en una ventaja analítica.

En cualquier problema real hay una parte del resultado que ningún modelo puede
predecir, porque depende de variables que nadie observó: que al estudiante le
cambien el turno en el trabajo, que se enferme un familiar, que simplemente
pierda el interés. Esa es la incertidumbre irreducible del problema y fija un
techo al desempeño alcanzable. En un proyecto con datos reales ese techo es
desconocido, y por eso las presentaciones suelen quedarse en "obtuvimos AUC de
0.79", sin poder decir si eso está cerca o lejos del máximo posible.

Como acá conocemos la probabilidad verdadera con la que se generó cada etiqueta,
podemos construir el clasificador óptimo de Bayes -el que usa exactamente esa
probabilidad- y medir su desempeño. Su AUC es el techo teórico: ningún modelo,
por sofisticado que sea, puede superarlo de forma sistemática.

Con eso se responden dos preguntas que de otro modo quedan abiertas:

  ¿Vale la pena seguir invirtiendo en modelado?
      Si el mejor modelo captura el 96% de la señal recuperable, el margen que
      queda es de 4 puntos y probablemente rinda menos que invertir en conseguir
      nuevas variables.

  ¿Cuánto del error es culpa del modelo y cuánto del problema?
      Permite separar el error de estimación (mejorable con mejores modelos) del
      error irreducible (solo mejorable con mejores datos).

Las probabilidades verdaderas NUNCA entran al entrenamiento ni al conjunto de
variables predictoras. Se usan solo acá, después de que todos los modelos ya
fueron entrenados y evaluados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C
from .costs import curva_capacidad


def cargar_oraculo() -> pd.DataFrame:
    return pd.read_csv(C.DATA_RAW / "oracle_probabilities.csv")


def techo_bayes(df_particion: pd.DataFrame, oraculo: pd.DataFrame) -> dict:
    """Desempeño del clasificador óptimo de Bayes sobre una partición."""
    unido = df_particion.merge(oraculo, on="EstudianteID", how="left")
    y = unido[C.TARGET].to_numpy()
    p = unido["ProbabilidadVerdadera"].to_numpy()
    return {
        "n": int(len(y)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def fraccion_senal_capturada(auc_modelo: float, auc_oraculo: float) -> float:
    """
    Porcentaje del poder discriminante alcanzable que logra el modelo.

    Se mide sobre el exceso respecto del azar (AUC = 0.5), que es el punto de
    partida de cualquier clasificador sin información.
    """
    return (auc_modelo - 0.5) / (auc_oraculo - 0.5)


def brecha_economica(
    df_particion: pd.DataFrame,
    oraculo: pd.DataFrame,
    y_prob_modelo: np.ndarray,
    k: int,
) -> dict:
    """
    Cuánto dinero deja sobre la mesa el modelo frente al ranking perfecto posible.

    Compara tres políticas de contacto sobre los mismos k cupos: la del modelo,
    la del oráculo (el mejor ranking alcanzable con la información disponible)
    y la aleatoria (contactar sin modelo).
    """
    unido = df_particion.merge(oraculo, on="EstudianteID", how="left")
    y = unido[C.TARGET].to_numpy()
    p_or = unido["ProbabilidadVerdadera"].to_numpy()

    # Se compara en VALOR ESPERADO, no sobre el resultado realizado. El oráculo
    # ordena por probabilidad verdadera y por lo tanto maximiza los desertores
    # esperados dentro de los k cupos; el resultado realizado de una sola
    # muestra tiene ruido binomial y puede hacer que un modelo peor "gane" por
    # azar cuando k es chico. Usar la esperanza aísla la calidad del ranking.
    def _beneficio_esperado(scores):
        top = np.argsort(-np.asarray(scores))[:k]
        desertores_esperados = float(p_or[top].sum())
        ingreso = desertores_esperados * C.EFECTIVIDAD_INTERVENCION * C.VALOR_ESTUDIANTE_PERDIDO
        return ingreso - k * C.COSTO_CONTACTO, desertores_esperados

    b_mod, tp_mod = _beneficio_esperado(y_prob_modelo)
    b_or, tp_or = _beneficio_esperado(p_or)

    rng = np.random.default_rng(C.SEED)
    azar = [_beneficio_esperado(rng.random(len(y)))[0] for _ in range(50)]
    b_azar = float(np.mean(azar))

    # Beneficio realizado (el que efectivamente ocurrió en esta muestra)
    realizado = curva_capacidad(y, y_prob_modelo)
    fila = realizado.iloc[(realizado["k"] - k).abs().argmin()]

    return {
        "k": int(k),
        "beneficio_esperado_modelo": round(b_mod, 2),
        "beneficio_esperado_oraculo": round(b_or, 2),
        "beneficio_esperado_azar": round(b_azar, 2),
        "captura_del_maximo": round((b_mod - b_azar) / (b_or - b_azar), 4)
        if b_or != b_azar else np.nan,
        "desertores_esperados_modelo": round(tp_mod, 1),
        "desertores_esperados_oraculo": round(tp_or, 1),
        "beneficio_realizado_modelo": round(float(fila["beneficio_neto"]), 2),
        "TP_realizado_modelo": int(fila["TP"]),
    }
