"""
Limpieza, ingeniería de variables, partición temporal y preprocesamiento.

REGLA QUE ORGANIZA ESTE MÓDULO
------------------------------
Se distingue entre dos tipos de transformación:

(a) Transformaciones fila a fila, que no usan ningún estadístico del conjunto
    (normalizar mayúsculas, marcar una edad de 999 como faltante, calcular un
    ratio entre dos columnas de la misma fila). Estas se pueden aplicar antes
    de partir los datos, porque el resultado de una fila no depende de las demás.

(b) Transformaciones que estiman parámetros a partir de los datos (imputar por
    la mediana, estandarizar, codificar categorías). Estas SOLO pueden estimarse
    sobre entrenamiento. Si se ajustan sobre el dataset completo antes de partir,
    la información del conjunto de prueba se filtra al modelo y las métricas
    quedan optimistas.

Por eso todo lo del tipo (b) vive dentro de un Pipeline de scikit-learn, que se
reajusta en cada fold de validación cruzada y sobre entrenamiento antes de
evaluar en test. Nada del tipo (b) se ejecuta fuera del Pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

# Variables derivadas creadas en `agregar_features` (todas fila a fila)
DERIVADAS_NUM = [
    "RatioForoTenure",
    "HorasPorSemanaEfectivas",
    "BrechaTareas",
]
DERIVADAS_BIN = [
    "InicioReciente",
    "SinComunidad",
    "NotaBajoAprobatorio",
    "NotaFaltante",
]


# ---------------------------------------------------------------------------
# 1. Limpieza (fila a fila, sin estadísticos del conjunto)
# ---------------------------------------------------------------------------
def limpiar(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Corrige los defectos conocidos de la extracción."""
    n0 = len(df)
    reporte: dict[str, int] = {}

    # (a) Duplicados exactos por doble carga del ETL
    df = df.drop_duplicates(subset="EstudianteID", keep="first").copy()
    reporte["duplicados_eliminados"] = n0 - len(df)

    # (b) Inconsistencias de formato en categóricas
    df["Modalidad"] = (
        df["Modalidad"].str.strip().str.lower().map(
            {"autoestudio": "Autoestudio", "con mentor en vivo": "Con mentor en vivo"}
        )
    )

    # (c) Edades imposibles -> se marcan como faltantes en vez de borrar la fila.
    #     Borrar la fila descartaría información válida de las demás variables.
    edad_invalida = ~df["Edad"].between(15, 80)
    reporte["edades_invalidas"] = int(edad_invalida.sum())
    df.loc[edad_invalida, "Edad"] = np.nan

    # (d) Rangos imposibles en variables de comportamiento
    for col, (lo, hi) in {
        "PctTareasATiempo": (0, 100),
        "PromedioNotas": (0, 20),
        "DiasUltimaSesion": (0, 365),
        "HorasUsoSemana": (0, 168),
    }.items():
        fuera = ~df[col].between(lo, hi) & df[col].notna()
        reporte[f"fuera_de_rango_{col}"] = int(fuera.sum())
        df.loc[fuera, col] = np.nan

    reporte["nulos_totales"] = int(df.isna().sum().sum())
    reporte["filas_finales"] = len(df)

    if verbose:
        print("Limpieza:")
        for k, v in reporte.items():
            print(f"  {k:32s} {v}")

    df.attrs["reporte_limpieza"] = reporte
    return df


# ---------------------------------------------------------------------------
# 2. Ingeniería de variables (fila a fila)
# ---------------------------------------------------------------------------
def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas con hipótesis de negocio explícita."""
    df = df.copy()

    # Intensidad de participación en comunidad relativa al tiempo en el curso.
    # Hipótesis: 3 mensajes en la semana 2 valen más que 3 mensajes en la 30.
    df["RatioForoTenure"] = df["ParticipacionForo"] / df["TenureSemanas"].clip(lower=1)

    # Horas de uso ajustadas por inactividad reciente: un alumno con 5 h/semana
    # promedio pero 20 días sin entrar no está realmente usando la plataforma.
    df["HorasPorSemanaEfectivas"] = df["HorasUsoSemana"] / (
        1 + df["DiasUltimaSesion"] / 7
    )

    # Distancia al umbral operativo de alerta que hoy usa el equipo académico.
    df["BrechaTareas"] = 60 - df["PctTareasATiempo"]

    # Ventana crítica de onboarding
    df["InicioReciente"] = (df["TenureSemanas"] <= 3).astype(int)

    # Aislamiento social en la plataforma
    df["SinComunidad"] = (df["ParticipacionForo"] == 0).astype(int)

    # Rendimiento por debajo de la nota aprobatoria peruana
    df["NotaBajoAprobatorio"] = (df["PromedioNotas"] < 13).fillna(False).astype(int)

    # INDICADOR DE FALTANTE: la ausencia de nota no es aleatoria, ocurre sobre
    # todo en alumnos que nunca rindieron evaluaciones. El hecho de que falte
    # es en sí mismo una señal de riesgo y hay que dejar que el modelo la use.
    df["NotaFaltante"] = df["PromedioNotas"].isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Partición temporal (out-of-time)
# ---------------------------------------------------------------------------
def particion_temporal(
    df: pd.DataFrame,
    cohortes_test: int = C.COHORTES_TEST,
    cohortes_val: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parte los datos por cohorte de matrícula, no al azar.

    Un split aleatorio mezcla alumnos de 2025 y 2026 en entrenamiento y prueba,
    lo que sobreestima el desempeño: en producción el modelo se entrena con el
    pasado y predice sobre alumnos que todavía no existían. Además el proceso
    tiene drift (la mezcla de modalidades cambia y la tasa base sube), así que
    el split aleatorio esconde exactamente el riesgo que importa medir.
    """
    cohortes = sorted(df[C.COL_COHORTE].unique())
    c_test = cohortes[-cohortes_test:]
    c_val = cohortes[-(cohortes_test + cohortes_val) : -cohortes_test]
    c_train = cohortes[: -(cohortes_test + cohortes_val)]

    train = df[df[C.COL_COHORTE].isin(c_train)].copy()
    val = df[df[C.COL_COHORTE].isin(c_val)].copy()
    test = df[df[C.COL_COHORTE].isin(c_test)].copy()

    for nombre, parte, coh in (
        ("train", train, c_train),
        ("valid", val, c_val),
        ("test ", test, c_test),
    ):
        print(
            f"  {nombre}: {len(parte):5d} filas | cohortes {coh[0]} a {coh[-1]} "
            f"| deserción {parte[C.TARGET].mean():.1%}"
        )

    return train, val, test


# ---------------------------------------------------------------------------
# 4. Preprocesador (se ajusta solo dentro del Pipeline)
# ---------------------------------------------------------------------------
def columnas_modelo() -> tuple[list[str], list[str], list[str]]:
    numericas = C.NUMERICAS + DERIVADAS_NUM
    binarias = C.BINARIAS + DERIVADAS_BIN
    categoricas = C.CATEGORICAS
    return numericas, binarias, categoricas


def construir_preprocesador(escalar: bool = True) -> ColumnTransformer:
    """
    Imputación + codificación + escalado, todo estimado en entrenamiento.

    `escalar` se desactiva para los modelos de árboles, que son invariantes a
    transformaciones monótonas de las variables y no lo necesitan.
    """
    numericas, binarias, categoricas = columnas_modelo()

    pasos_num: list = [("imputar", SimpleImputer(strategy="median", add_indicator=False))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(pasos_num), numericas),
            ("bin", "passthrough", binarias),
            (
                "cat",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                categoricas,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preparar(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Limpieza + features en un solo paso."""
    return agregar_features(limpiar(df, verbose=verbose))
