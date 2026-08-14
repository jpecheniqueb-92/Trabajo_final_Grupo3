"""
Economía del error: qué cuesta equivocarse y dónde conviene poner el umbral.

La rúbrica pide explícitamente métricas coherentes con el costo de los errores.
Este módulo traduce la matriz de confusión a soles y responde tres preguntas que
el equipo académico sí puede accionar:

  1. ¿A partir de qué probabilidad conviene contactar a un estudiante?
  2. Si el equipo solo puede hacer k contactos por semana, ¿a quiénes llama?
  3. ¿Qué tan sensible es la respuesta a los supuestos que asumimos?

MARCO DE DECISIÓN
-----------------
La decisión no es "clasificar bien" sino "gastar media hora de tutor o no".
Para cada estudiante hay dos acciones posibles y cuatro desenlaces:

                          | El alumno iba a desertar | Iba a quedarse
    --------------------- | ------------------------ | ---------------
    Se le contacta        | se recupera con prob. e  | se gasta el costo
                          | beneficio  e·V - c       | costo -c
    No se le contacta     | se pierde el alumno      | nada
                          | 0 (referencia)           | 0

donde V es el valor económico del alumno perdido, e la efectividad de la
intervención y c el costo de un contacto. Fijamos el desenlace "no contactar"
como referencia con valor 0, de modo que todo se mide como beneficio
incremental de la campaña de retención frente a no hacer nada.

Contactar conviene cuando  p·e·V > c, es decir cuando  p > c / (e·V).
Ese cociente -y no 0.5- es el umbral correcto de decisión.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Parámetros económicos
# ---------------------------------------------------------------------------
def beneficio_por_recuperado(efectividad: float = C.EFECTIVIDAD_INTERVENCION) -> float:
    """Valor esperado de contactar a un estudiante que sí iba a desertar."""
    return efectividad * C.VALOR_ESTUDIANTE_PERDIDO


def umbral_optimo(
    efectividad: float = C.EFECTIVIDAD_INTERVENCION,
    costo_contacto: float = C.COSTO_CONTACTO,
) -> float:
    """Probabilidad a partir de la cual el contacto tiene valor esperado positivo."""
    return costo_contacto / beneficio_por_recuperado(efectividad)


# ---------------------------------------------------------------------------
# Evaluación económica de una política de contacto
# ---------------------------------------------------------------------------
def beneficio_neto(
    y_true: np.ndarray,
    contactar: np.ndarray,
    efectividad: float = C.EFECTIVIDAD_INTERVENCION,
    costo_contacto: float = C.COSTO_CONTACTO,
) -> dict[str, float]:
    """Beneficio incremental en soles de una política binaria de contacto."""
    y_true = np.asarray(y_true).astype(int)
    contactar = np.asarray(contactar).astype(int)

    tp = int(((contactar == 1) & (y_true == 1)).sum())
    fp = int(((contactar == 1) & (y_true == 0)).sum())
    fn = int(((contactar == 0) & (y_true == 1)).sum())
    tn = int(((contactar == 0) & (y_true == 0)).sum())

    recuperados = tp * efectividad
    ingreso = recuperados * C.VALOR_ESTUDIANTE_PERDIDO
    costo = (tp + fp) * costo_contacto

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "contactados": tp + fp,
        "estudiantes_recuperados": round(recuperados, 1),
        "ingreso_retenido": round(ingreso, 2),
        "costo_operativo": round(costo, 2),
        "beneficio_neto": round(ingreso - costo, 2),
        "roi": round((ingreso - costo) / costo, 3) if costo > 0 else np.nan,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else np.nan,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else np.nan,
    }


def curva_umbral(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    efectividad: float = C.EFECTIVIDAD_INTERVENCION,
    costo_contacto: float = C.COSTO_CONTACTO,
    n_puntos: int = 200,
) -> pd.DataFrame:
    """Beneficio neto en soles para cada umbral posible de decisión."""
    umbrales = np.linspace(0.01, 0.95, n_puntos)
    filas = []
    for u in umbrales:
        r = beneficio_neto(y_true, (y_prob >= u).astype(int), efectividad, costo_contacto)
        r["umbral"] = round(float(u), 4)
        filas.append(r)
    return pd.DataFrame(filas)


def curva_capacidad(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    efectividad: float = C.EFECTIVIDAD_INTERVENCION,
    costo_contacto: float = C.COSTO_CONTACTO,
    pasos: int = 60,
) -> pd.DataFrame:
    """
    Beneficio neto cuando el equipo contacta a los k estudiantes de mayor riesgo.

    Esta es la forma en que realmente se usa el modelo: el equipo de retención
    no puede llamar a todos los que superan un umbral, tiene una capacidad fija.
    La pregunta operativa es hasta qué posición del ranking conviene bajar.
    """
    y_true = np.asarray(y_true).astype(int)
    orden = np.argsort(-np.asarray(y_prob))
    y_ord = y_true[orden]
    n = len(y_true)

    ks = np.unique(np.linspace(1, n, pasos).astype(int))
    filas = []
    for k in ks:
        tp = int(y_ord[:k].sum())
        ingreso = tp * efectividad * C.VALOR_ESTUDIANTE_PERDIDO
        costo = k * costo_contacto
        filas.append(
            {
                "k": int(k),
                "pct_base_contactada": round(k / n, 4),
                "TP": tp,
                "precision_en_k": round(tp / k, 4),
                "recall_en_k": round(tp / max(y_true.sum(), 1), 4),
                "lift": round((tp / k) / y_true.mean(), 3),
                "beneficio_neto": round(ingreso - costo, 2),
                "beneficio_por_contacto": round((ingreso - costo) / k, 2),
            }
        )
    return pd.DataFrame(filas)


def sensibilidad(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    efectividades: tuple[float, ...] = (0.15, 0.20, 0.28, 0.35, 0.40),
    costos: tuple[float, ...] = (12.0, 18.0, 25.0, 36.0),
) -> pd.DataFrame:
    """
    ¿Cambia la decisión si nos equivocamos en los supuestos?

    Recorre la grilla de efectividad de la intervención y costo por contacto,
    y para cada combinación reporta el umbral óptimo y el beneficio que se
    obtendría aplicándolo. Sirve para saber cuáles supuestos hay que medir
    con cuidado antes de poner el modelo en producción.
    """
    filas = []
    for e in efectividades:
        for c in costos:
            u = umbral_optimo(e, c)
            r = beneficio_neto(y_true, (y_prob >= u).astype(int), e, c)
            filas.append(
                {
                    "efectividad": e,
                    "costo_contacto": c,
                    "umbral_optimo": round(u, 4),
                    "contactados": r["contactados"],
                    "beneficio_neto": r["beneficio_neto"],
                    "roi": r["roi"],
                    "rentable": r["beneficio_neto"] > 0,
                }
            )
    return pd.DataFrame(filas)


def resumen_supuestos() -> pd.DataFrame:
    """Tabla de supuestos económicos para el anexo del informe."""
    return pd.DataFrame(
        [
            ("Precio promedio del curso", f"S/ {C.PRECIO_CURSO:,.0f}",
             "Ticket de curso técnico online en el mercado peruano"),
            ("Margen de contribución", f"{C.MARGEN_CONTRIBUCION:.0%}",
             "Precio menos costos variables de plataforma y docencia"),
            ("Fracción del curso no dictada al desertar", f"{C.FRACCION_NO_DEVENGADA:.0%}",
             "El abandono ocurre en promedio cerca de la mitad del programa"),
            ("Costo de adquirir un estudiante de reemplazo", f"S/ {C.CAC:,.0f}",
             "CAC de marketing digital por matrícula"),
            ("Valor económico de un estudiante perdido",
             f"S/ {C.VALOR_ESTUDIANTE_PERDIDO:,.0f}",
             "Margen no devengado más costo de reposición"),
            ("Efectividad de la intervención de retención",
             f"{C.EFECTIVIDAD_INTERVENCION:.0%}",
             "Supuesto central; se somete a sensibilidad entre 15% y 40%"),
            ("Costo de un contacto de retención", f"S/ {C.COSTO_CONTACTO:,.0f}",
             f"{C.HORAS_POR_CONTACTO:.1f} h de tutor a S/ {C.COSTO_HORA_TUTOR:,.0f}/h"),
            ("Capacidad del equipo", f"{C.CONTACTOS_POR_SEMANA} contactos/semana",
             f"{C.TUTORES_RETENCION} tutores x {C.HORAS_SEMANA_POR_TUTOR} h/semana"),
            ("Umbral óptimo de decisión", f"{umbral_optimo():.3f}",
             "Costo del contacto dividido entre el beneficio esperado de recuperar"),
        ],
        columns=["Supuesto", "Valor", "Justificación"],
    )
