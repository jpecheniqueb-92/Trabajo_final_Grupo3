"""
Estudios de ablación: dos preguntas metodológicas que sostienen la elección
del modelo.

ESTUDIO 1 — ¿La comparación de modelos es informativa?
------------------------------------------------------
Un riesgo silencioso de trabajar con datos sintéticos es generar el target con
la misma familia de modelos que después se va a comparar. Si el proceso
generador es un logit lineal, la regresión logística gana por construcción, y
la "comparación de alternativas" no prueba nada: solo confirma que el modelo
que escribimos es el modelo que escribimos.

Este estudio corre exactamente los mismos modelos sobre dos escenarios:

  Escenario A: proceso generador logit lineal (el diseño ingenuo)
  Escenario B: proceso generador con umbrales, curva de bañera e interacciones

Si el orden de los modelos cambia entre escenarios, queda demostrado que la
conclusión del proyecto depende de la estructura de los datos y no de una
preferencia arbitraria, y que el Escenario B es el único que permite una
comparación con contenido empírico.

ESTUDIO 2 — ¿Cuánto aportan las variables derivadas del dominio?
----------------------------------------------------------------
Compara cada modelo con y sin las variables construidas a partir de hipótesis
de negocio. La hipótesis a contrastar es que un modelo lineal necesita que
alguien le codifique a mano la no linealidad, mientras que los modelos de
árboles la recuperan solos. Si es así, el trabajo de ingeniería de variables es
sustituible por capacidad del modelo, y conviene saberlo antes de invertir
semanas en construir variables.

Uso:
    python -m src.ablacion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import preprocessing as P
from .data_generation import _calibrar_intercepto, _sigmoide, generar
from .models import construir_modelos
from .preprocessing import particion_temporal, preparar

MODELOS_COMPARADOS = [
    "Regresión Logística",
    "Árbol de Decisión",
    "Random Forest",
    "XGBoost",
    "Red Neuronal (MLP)",
]


# ---------------------------------------------------------------------------
# Escenario A: el proceso generador ingenuo (logit lineal)
# ---------------------------------------------------------------------------
def generar_lineal(n: int = C.N_ESTUDIANTES, seed: int = C.SEED) -> pd.DataFrame:
    """
    Reproduce el enfoque ingenuo: mismas variables, pero el riesgo es una
    combinación lineal de ellas pasada por una sigmoide. Es decir, el proceso
    generador ES una regresión logística.
    """
    df, _ = generar(n=n, seed=seed)
    df = df.dropna(subset=["PromedioNotas", "HorasUsoSemana"]).copy()
    rng = np.random.default_rng(seed + 99)

    riesgo = (
        0.16 * df["DiasUltimaSesion"]
        - 0.14 * df["HorasUsoSemana"]
        - 0.045 * df["PctTareasATiempo"]
        - 0.10 * df["PromedioNotas"]
        - 0.22 * df["ParticipacionForo"]
        + 0.85 * df["QuejasSoporte"]
        - 0.55 * df["CursosPreviosCompletados"]
        - 0.55 * df["TieneBecaDescuento"]
        + 0.90 * (df["TenureSemanas"] < 3).astype(int)
        + np.where(df["Modalidad"] == "Autoestudio", 0.45, -0.15)
        + rng.normal(0, 0.55, len(df))
    ).to_numpy()

    intercepto = _calibrar_intercepto(riesgo, 0.20)
    df["Desercion"] = rng.binomial(1, _sigmoide(riesgo + intercepto))
    return df


# ---------------------------------------------------------------------------
# Motor común
# ---------------------------------------------------------------------------
def _evaluar(df: pd.DataFrame, etiqueta: str) -> pd.DataFrame:
    df = preparar(df, verbose=False)
    train, valid, _ = particion_temporal(df)
    quitar = [C.TARGET, "EstudianteID", C.COL_COHORTE]
    Xtr, ytr = train.drop(columns=quitar), train[C.TARGET]
    Xva, yva = valid.drop(columns=quitar), valid[C.TARGET]
    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    filas = []
    for nombre, pipe in construir_modelos(spw).items():
        if nombre not in MODELOS_COMPARADOS:
            continue
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        filas.append({
            "escenario": etiqueta,
            "Modelo": nombre,
            "ROC_AUC": round(roc_auc_score(yva, p), 4),
            "PR_AUC": round(average_precision_score(yva, p), 4),
        })
    d = pd.DataFrame(filas)
    d["puesto"] = d["PR_AUC"].rank(ascending=False).astype(int)
    return d


def estudio_proceso_generador() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ESTUDIO 1 — ¿Cambia el ganador según el proceso generador?")
    print("=" * 72)
    a = _evaluar(generar_lineal(), "A: generador logit lineal")
    b = _evaluar(generar()[0], "B: generador con umbrales e interacciones")
    res = pd.concat([a, b], ignore_index=True)

    piv = res.pivot(index="Modelo", columns="escenario", values="PR_AUC")
    puestos = res.pivot(index="Modelo", columns="escenario", values="puesto")
    print("\nPR-AUC en validación:")
    print(piv.round(4).to_string())
    print("\nPuesto en cada escenario:")
    print(puestos.to_string())

    ganador_a = a.sort_values("PR_AUC", ascending=False).iloc[0]["Modelo"]
    ganador_b = b.sort_values("PR_AUC", ascending=False).iloc[0]["Modelo"]
    print(f"\n  Escenario A gana: {ganador_a}")
    print(f"  Escenario B gana: {ganador_b}")
    if ganador_a != ganador_b:
        print("  -> El ganador depende del proceso generador. Comparar modelos sobre")
        print("     el Escenario A no habría probado nada sobre el problema real.")
    return res


def estudio_features_dominio() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("ESTUDIO 2 — ¿Cuánto aportan las variables derivadas del dominio?")
    print("=" * 72)
    df_base = generar()[0]
    original_num, original_bin = P.DERIVADAS_NUM.copy(), P.DERIVADAS_BIN.copy()

    filas = []
    for etiqueta, dn, db in [
        ("sin variables de dominio", [], []),
        ("con variables de dominio", original_num, original_bin),
    ]:
        P.DERIVADAS_NUM[:], P.DERIVADAS_BIN[:] = dn, db
        d = _evaluar(df_base.copy(), etiqueta)
        filas.append(d)
    P.DERIVADAS_NUM[:], P.DERIVADAS_BIN[:] = original_num, original_bin

    res = pd.concat(filas, ignore_index=True)
    piv = res.pivot(index="Modelo", columns="escenario", values="PR_AUC")
    piv["ganancia"] = (
        piv["con variables de dominio"] - piv["sin variables de dominio"]
    ).round(4)
    print("\nPR-AUC en validación:")
    print(piv.round(4).to_string())
    return res


def main() -> None:
    e1 = estudio_proceso_generador()
    e2 = estudio_features_dominio()
    e1.to_csv(C.METRICS / "11_ablacion_proceso_generador.csv", index=False)
    e2.to_csv(C.METRICS / "12_ablacion_features_dominio.csv", index=False)
    print(f"\nGuardado en {C.METRICS.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
