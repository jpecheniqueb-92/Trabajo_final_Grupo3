"""
Generación del dataset sintético de EduTech Perú.

POR QUÉ DATOS SINTÉTICOS
------------------------
No tenemos acceso a la base transaccional de una academia online real. La
rúbrica lo contempla explícitamente ("se podrá emplear una muestra anonimizada,
datos sintéticos o instrucciones reproducibles"). Trabajar con datos sintéticos
tiene un costo -no es evidencia sobre el mundo real- pero también una ventaja
metodológica poco aprovechada: conocemos el proceso generador, así que podemos
calcular el TECHO TEÓRICO de desempeño (el clasificador óptimo de Bayes) y
medir qué fracción de la señal recuperable captura cada modelo. Esa es la
diferencia entre decir "nuestro AUC es 0.79" y decir "nuestro AUC es 0.79 sobre
un máximo alcanzable de 0.82, es decir capturamos el 96% de la señal".

DISEÑO DEL PROCESO GENERADOR
----------------------------
Una versión anterior de este proyecto generaba el target como un logit lineal
de las variables. Eso hacía que la Regresión Logística ganara por construcción
-el modelo comparado era exactamente el modelo generador- y volvía vacía toda
la comparación de alternativas. Este generador introduce estructura que sí
existe en el comportamiento real de plataformas de e-learning y que un modelo
lineal no puede representar:

1. EFECTO UMBRAL en la inactividad: no es lo mismo pasar de 2 a 4 días sin
   entrar que de 12 a 14. El riesgo es plano al inicio y se dispara alrededor
   de los 12 días (curva sigmoide, no lineal).
2. EFECTO EN U de la edad: los estudiantes muy jóvenes (menor autorregulación)
   y los mayores (más carga laboral y familiar) desertan más que el grupo medio.
3. RENDIMIENTOS DECRECIENTES en las horas de uso: pasar de 0 a 3 horas semanales
   protege mucho; de 10 a 13 casi no agrega nada (saturación exponencial).
4. INTERACCIONES: el autoestudio sin participación en la comunidad es mucho más
   riesgoso que cualquiera de los dos factores por separado; empezar el curso
   sin experiencia previa en la plataforma multiplica el riesgo del onboarding.
5. EFECTO PISO en las notas: la nota solo predice deserción cuando cae por
   debajo del umbral aprobatorio peruano (13); arriba de eso es irrelevante.
6. NULOS NO ALEATORIOS (MNAR): el promedio de notas falta con más frecuencia
   justamente en los estudiantes inactivos, porque no llegaron a rendir
   evaluaciones. La ausencia del dato es en sí misma informativa.
7. DRIFT TEMPORAL: a lo largo de 18 cohortes mensuales la mezcla de modalidades
   cambia y la tasa base de deserción sube levemente, lo que obliga a validar
   fuera de tiempo y no con un split aleatorio.

Ninguno de estos mecanismos favorece a priori a un modelo específico: son
hechos del dominio. Cuál familia de modelos los captura mejor es precisamente
la pregunta empírica que responde el experimento.

Salidas:
    data/raw/edutech_desercion_raw.csv    dataset observable (lo que vería la empresa)
    data/raw/oracle_probabilities.csv     probabilidad verdadera de cada alumno

IMPORTANTE: oracle_probabilities.csv NO se usa nunca como variable predictora.
Se emplea únicamente en src/oracle.py para calcular el techo de Bayes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrar_intercepto(riesgo: np.ndarray, prevalencia_objetivo: float) -> float:
    """Encuentra por bisección el intercepto que produce la prevalencia deseada."""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoide(riesgo + mid).mean() < prevalencia_objetivo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generar(
    n: int = C.N_ESTUDIANTES,
    seed: int = C.SEED,
    prevalencia_objetivo: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera el dataset observable y las probabilidades verdaderas (oráculo)."""
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------
    # 1. Estructura temporal: 18 cohortes mensuales
    # ----------------------------------------------------------------------
    cohorte = rng.integers(0, C.N_COHORTES, size=n)
    meses = pd.period_range("2025-01", periods=C.N_COHORTES, freq="M")
    cohorte_mes = meses[cohorte].astype(str)
    t = cohorte / (C.N_COHORTES - 1)  # 0 a 1, avance temporal normalizado

    # ----------------------------------------------------------------------
    # 2. Variables demográficas y de contratación
    # ----------------------------------------------------------------------
    edad = rng.normal(27, 7, n).clip(16, 60).round(0)

    categoria_curso = rng.choice(
        ["Idiomas", "Programación", "Marketing Digital", "Finanzas", "Diseño"],
        size=n,
        p=[0.32, 0.24, 0.18, 0.14, 0.12],
    )

    # DRIFT: la modalidad autoestudio gana participación con el tiempo porque
    # la empresa la empuja por márgenes (55% al inicio, 70% al final)
    p_autoestudio = 0.55 + 0.15 * t
    modalidad = np.where(
        rng.random(n) < p_autoestudio, "Autoestudio", "Con mentor en vivo"
    )

    # DRIFT: Yape/Plin desplaza a la tarjeta a lo largo del periodo
    u = rng.random(n)
    p_yape = 0.22 + 0.16 * t
    p_tarjeta = 0.40 - 0.14 * t
    p_debito = 0.24 - 0.02 * t
    metodo_pago = np.where(
        u < p_yape,
        "Yape/Plin",
        np.where(
            u < p_yape + p_tarjeta,
            "Tarjeta de crédito",
            np.where(u < p_yape + p_tarjeta + p_debito, "Débito", "Transferencia"),
        ),
    )

    tiene_beca = rng.choice([0, 1], size=n, p=[0.72, 0.28])
    cursos_previos = rng.poisson(0.8, n)
    tenure = rng.gamma(2.2, 4.5, n).clip(1, 40).round(0)

    # ----------------------------------------------------------------------
    # 3. Variables de comportamiento
    #    Se generan con dependencias entre sí (un alumno comprometido usa más
    #    la plataforma Y entrega más tareas), lo que produce multicolinealidad
    #    realista y hace no trivial la selección de variables.
    # ----------------------------------------------------------------------
    compromiso = rng.normal(0, 1, n) + 0.35 * (modalidad == "Con mentor en vivo")
    compromiso += 0.25 * np.minimum(cursos_previos, 3)

    horas_uso = (rng.gamma(2.0, 1.6, n) * np.exp(0.28 * compromiso)).clip(0, 25)
    participacion_foro = rng.poisson(np.exp(0.55 + 0.42 * compromiso).clip(0.1, 12))

    pct_tareas = (
        _sigmoide(0.9 * compromiso + rng.normal(0, 0.7, n)) * 96 + 3
    ).clip(0, 100)

    promedio_notas = (
        7.5 + 0.085 * pct_tareas + 0.35 * compromiso + rng.normal(0, 1.5, n)
    ).clip(0, 20)

    quejas_soporte = rng.poisson(np.exp(-1.2 - 0.25 * compromiso).clip(0.02, 3))

    # La inactividad depende del compromiso: los desconectados acumulan días
    dias_ultima_sesion = (
        rng.exponential(np.exp(1.9 - 0.45 * compromiso).clip(0.5, 40))
    ).clip(0, 60).round(0)

    # ----------------------------------------------------------------------
    # 4. RIESGO LATENTE — aquí vive toda la no linealidad del dominio
    # ----------------------------------------------------------------------
    inicio_reciente = (tenure <= 3).astype(int)
    autoestudio = (modalidad == "Autoestudio").astype(int)
    sin_comunidad = (participacion_foro == 0).astype(int)
    sin_experiencia = (cursos_previos == 0).astype(int)

    prog_autoestudio = ((categoria_curso == "Programación") & (modalidad == "Autoestudio")).astype(int)

    riesgo = (
        # (1) efecto umbral: plano hasta ~10 días, se dispara después
        3.10 * _sigmoide((dias_ultima_sesion - 12.0) / 3.0)
        # (2) efecto en U de la edad, mínimo alrededor de los 30 años.
        #     Los más jóvenes tienen menor autorregulación y los mayores más
        #     carga laboral y familiar.
        + 0.55 * ((edad - 30.0) / 10.0) ** 2
        # (2b) CURVA DE BAÑERA en el tiempo de permanencia. El riesgo de
        #      abandono es alto en las primeras semanas (onboarding), cae en la
        #      zona media donde queda la cohorte comprometida, y vuelve a subir
        #      cerca del final por fatiga y por la exigencia del proyecto de
        #      cierre. Es no monótona: ningún modelo lineal en TenureSemanas
        #      puede representarla sin que alguien la codifique a mano.
        + 1.40 * np.exp(-tenure / 3.5)
        + 0.95 * _sigmoide((tenure - 22.0) / 2.5)
        # (2c) interacción entre exigencia del curso y falta de acompañamiento:
        #      programación en autoestudio es la combinación más frágil
        + 0.70 * prog_autoestudio
        # (3) rendimientos decrecientes de las horas de uso
        - 1.55 * (1.0 - np.exp(-horas_uso / 3.0))
        # (4) efecto sigmoide del cumplimiento de tareas, con quiebre en 55%
        - 1.90 * _sigmoide((pct_tareas - 55.0) / 8.0)
        # (5) efecto piso de las notas: solo pesa por debajo de la nota aprobatoria
        + 0.34 * np.clip(13.0 - promedio_notas, 0, None)
        # (6) interacción: autoestudio sin comunidad
        + 0.75 * autoestudio
        + 1.05 * autoestudio * sin_comunidad
        # (7) interacción: onboarding sin experiencia previa en la plataforma
        + 1.15 * inicio_reciente * sin_experiencia
        # (8) las quejas pesan más cuando no hay un mentor que las canalice
        + 0.42 * quejas_soporte
        + 0.40 * quejas_soporte * autoestudio
        # (9) la beca genera compromiso (costo hundido percibido / compromiso mutuo)
        - 0.38 * tiene_beca
        - 0.22 * np.minimum(cursos_previos, 4)
        # (10) drift: la tasa base sube a lo largo del periodo
        + 0.45 * t
        # (11) heterogeneidad no observada: motivación, situación laboral,
        #      salud, entorno familiar. Es el error irreducible del problema.
        + rng.normal(0, 0.85, n)
    )

    intercepto = _calibrar_intercepto(riesgo, prevalencia_objetivo)
    p_verdadera = _sigmoide(riesgo + intercepto)
    desercion = rng.binomial(1, p_verdadera)

    # ----------------------------------------------------------------------
    # 5. Ensamblado del dataset observable
    # ----------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "EstudianteID": np.arange(1, n + 1),
            "CohorteMes": cohorte_mes,
            "TenureSemanas": tenure.astype(int),
            "Edad": edad.astype(int),
            "CategoriaCurso": categoria_curso,
            "Modalidad": modalidad,
            "MetodoPago": metodo_pago,
            "HorasUsoSemana": horas_uso.round(1),
            "PctTareasATiempo": pct_tareas.round(1),
            "PromedioNotas": promedio_notas.round(1),
            "ParticipacionForo": participacion_foro,
            "QuejasSoporte": quejas_soporte,
            "DiasUltimaSesion": dias_ultima_sesion.astype(int),
            "TieneBecaDescuento": tiene_beca,
            "CursosPreviosCompletados": cursos_previos,
            "Desercion": desercion,
        }
    )

    # ----------------------------------------------------------------------
    # 6. Imperfecciones realistas de los datos
    # ----------------------------------------------------------------------
    # (a) NULOS NO ALEATORIOS: el promedio de notas falta cuando el estudiante
    #     nunca llegó a rendir evaluaciones, algo mucho más probable en los
    #     inactivos. La ausencia del dato es informativa (MNAR).
    p_falta_nota = _sigmoide((dias_ultima_sesion - 22.0) / 5.0) * 0.55 + 0.01
    df.loc[rng.random(n) < p_falta_nota, "PromedioNotas"] = np.nan

    # (b) Nulos aleatorios en horas de uso por fallas de telemetría (MCAR)
    df.loc[rng.random(n) < 0.018, "HorasUsoSemana"] = np.nan

    # (c) Registros duplicados por doble carga del ETL
    duplicados = df.sample(28, random_state=seed)
    df = pd.concat([df, duplicados], ignore_index=True)

    # (d) Inconsistencias de formato en una variable categórica, como en
    #     cualquier base alimentada por formularios distintos
    idx_formato = df.sample(frac=0.04, random_state=seed + 1).index
    df.loc[idx_formato, "Modalidad"] = df.loc[idx_formato, "Modalidad"].str.upper()

    # (e) Edades imposibles por error de digitación
    idx_error = df.sample(15, random_state=seed + 2).index
    df.loc[idx_error, "Edad"] = rng.choice([1, 2, 120, 999], size=len(idx_error))

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 7. Oráculo: probabilidad verdadera, solo para diagnóstico
    # ----------------------------------------------------------------------
    oraculo = pd.DataFrame(
        {"EstudianteID": np.arange(1, n + 1), "ProbabilidadVerdadera": p_verdadera}
    )

    return df, oraculo


def main() -> None:
    df, oraculo = generar()
    ruta_df = C.DATA_RAW / "edutech_desercion_raw.csv"
    ruta_or = C.DATA_RAW / "oracle_probabilities.csv"
    df.to_csv(ruta_df, index=False)
    oraculo.to_csv(ruta_or, index=False)

    print(f"Dataset generado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"  -> {ruta_df.relative_to(C.ROOT)}")
    print(f"  -> {ruta_or.relative_to(C.ROOT)}")
    print(f"\nTasa de deserción global: {df['Desercion'].mean():.1%}")
    print(f"Nulos: {int(df.isna().sum().sum())}  |  Duplicados: {int(df.duplicated().sum())}")
    print("\nTasa de deserción por cohorte (drift temporal):")
    print(
        df.groupby("CohorteMes")["Desercion"].agg(["size", "mean"]).round(3).to_string()
    )


if __name__ == "__main__":
    main()
