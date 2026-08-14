"""
Orquestador del experimento completo.

Ejecuta, en orden y de forma reproducible:

  1. Carga, limpieza y creación de variables
  2. Partición temporal (out-of-time)
  3. Búsqueda de hiperparámetros con validación cruzada sobre entrenamiento
  4. Comparación de líneas base y modelos en validación
  5. Elección del umbral por criterio económico
  6. Evaluación final en el conjunto de prueba temporal
  7. Diagnóstico con el oráculo, calibración, interpretabilidad y equidad
  8. Guardado de métricas y figuras

Uso:
    python -m src.run_experiment            # experimento completo
    python -m src.run_experiment --rapido   # sin búsqueda de hiperparámetros
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from . import costs, evaluate, oracle, plots
from .data_generation import main as generar_datos
from .models import ESPACIOS_BUSQUEDA, ReglaDeNegocio, construir_modelos
from .preprocessing import particion_temporal, preparar

warnings.filterwarnings("ignore", category=UserWarning)

N_ITER_BUSQUEDA = {
    "Regresión Logística": 5,
    "Árbol de Decisión": 15,
    "Random Forest": 12,
    "XGBoost": 25,
    "Red Neuronal (MLP)": 8,
}


def _json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"No serializable: {type(obj)}")


def main(rapido: bool = False) -> dict:
    t0 = time.time()
    resultados: dict = {}

    # ======================================================================
    print("\n" + "=" * 72)
    print("1. DATOS")
    print("=" * 72)
    ruta = C.DATA_RAW / "edutech_desercion_raw.csv"
    if not ruta.exists():
        generar_datos()
    crudo = pd.read_csv(ruta)
    df = preparar(crudo)
    df.to_csv(C.DATA_PROCESSED / "edutech_desercion_clean.csv", index=False)
    resultados["datos"] = {
        "filas_crudas": len(crudo),
        "filas_limpias": len(df),
        "tasa_desercion": float(df[C.TARGET].mean()),
        **crudo.attrs.get("reporte_limpieza", {}),
        **df.attrs.get("reporte_limpieza", {}),
    }

    print("\nPartición temporal:")
    train, valid, test = particion_temporal(df)
    quitar = [C.TARGET, "EstudianteID", C.COL_COHORTE]
    Xtr, ytr = train.drop(columns=quitar), train[C.TARGET]
    Xva, yva = valid.drop(columns=quitar), valid[C.TARGET]
    Xte, yte = test.drop(columns=quitar), test[C.TARGET]
    resultados["particion"] = {
        n: {"n": len(p), "tasa": float(p[C.TARGET].mean()),
            "cohortes": [p[C.COL_COHORTE].min(), p[C.COL_COHORTE].max()]}
        for n, p in (("train", train), ("valid", valid), ("test", test))
    }

    scale_pos = float((ytr == 0).sum() / (ytr == 1).sum())
    modelos = construir_modelos(scale_pos)

    # ======================================================================
    print("\n" + "=" * 72)
    print("2. BÚSQUEDA DE HIPERPARÁMETROS Y VALIDACIÓN CRUZADA")
    print("=" * 72)
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    ajustados: dict = {}
    filas_cv = []
    mejores_params: dict = {}

    for nombre, pipe in modelos.items():
        t = time.time()
        if nombre in ESPACIOS_BUSQUEDA and not rapido:
            busqueda = RandomizedSearchCV(
                pipe,
                ESPACIOS_BUSQUEDA[nombre],
                n_iter=N_ITER_BUSQUEDA[nombre],
                scoring="average_precision",  # PR-AUC: la métrica del desbalance
                cv=skf,
                random_state=C.SEED,
                n_jobs=1,
                refit=True,
            )
            busqueda.fit(Xtr, ytr)
            mejor = busqueda.best_estimator_
            mejores_params[nombre] = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in busqueda.best_params_.items()
            }
            cv_pr_media = float(busqueda.best_score_)
            idx = busqueda.best_index_
            cv_pr_std = float(busqueda.cv_results_["std_test_score"][idx])
        else:
            mejor = clone(pipe).fit(Xtr, ytr)
            mejores_params[nombre] = "valores por defecto"
            cv = evaluate.cv_evaluar(pipe, Xtr, ytr)
            cv_pr_media, cv_pr_std = cv["cv_pr_auc_media"], cv["cv_pr_auc_std"]

        ajustados[nombre] = mejor
        p_va = mejor.predict_proba(Xva)[:, 1]
        brecha = evaluate.brecha_sobreajuste(mejor, Xtr, ytr, Xva, yva)

        filas_cv.append({
            "Modelo": nombre,
            "CV_PR_AUC_media": round(cv_pr_media, 4),
            "CV_PR_AUC_std": round(cv_pr_std, 4),
            "Val_ROC_AUC": round(roc_auc_score(yva, p_va), 4),
            "Val_PR_AUC": round(average_precision_score(yva, p_va), 4),
            "AUC_train": round(brecha["auc_train"], 4),
            "Brecha_train_val": round(brecha["brecha"], 4),
            "segundos": round(time.time() - t, 1),
        })
        print(f"  {nombre:26s} CV PR-AUC {cv_pr_media:.4f} ± {cv_pr_std:.4f}"
              f" | val ROC {filas_cv[-1]['Val_ROC_AUC']:.4f}"
              f" | brecha {brecha['brecha']:+.3f}  ({time.time()-t:.0f}s)")

    # Línea base adicional: la regla de negocio que ya existe
    regla = ReglaDeNegocio(dias_umbral=10).fit(Xtr)
    ajustados["Línea base: regla de negocio"] = regla
    p_regla_va = regla.predict_proba(Xva)[:, 1]
    filas_cv.append({
        "Modelo": "Línea base: regla de negocio",
        "CV_PR_AUC_media": np.nan, "CV_PR_AUC_std": np.nan,
        "Val_ROC_AUC": round(roc_auc_score(yva, p_regla_va), 4),
        "Val_PR_AUC": round(average_precision_score(yva, p_regla_va), 4),
        "AUC_train": np.nan, "Brecha_train_val": np.nan, "segundos": 0.0,
    })

    tabla_cv = pd.DataFrame(filas_cv).sort_values("Val_PR_AUC", ascending=False)
    tabla_cv.to_csv(C.METRICS / "01_comparacion_validacion.csv", index=False)
    with open(C.METRICS / "02_mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(mejores_params, f, indent=2, ensure_ascii=False, default=_json)
    print("\n" + tabla_cv.to_string(index=False))

    # ======================================================================
    print("\n" + "=" * 72)
    print("3. ELECCIÓN DEL MODELO POR BENEFICIO ECONÓMICO EN VALIDACIÓN")
    print("=" * 72)
    u_opt = costs.umbral_optimo()
    print(f"  Umbral económico = costo contacto / beneficio esperado = "
          f"{C.COSTO_CONTACTO:.0f} / {costs.beneficio_por_recuperado():.1f} = {u_opt:.3f}")

    filas_eco = []
    for nombre, m in ajustados.items():
        p = m.predict_proba(Xva)[:, 1]
        r = costs.beneficio_neto(yva, (p >= u_opt).astype(int))
        filas_eco.append({"Modelo": nombre, "umbral": round(u_opt, 3), **r})
    tabla_eco = pd.DataFrame(filas_eco).sort_values("beneficio_neto", ascending=False)
    tabla_eco.to_csv(C.METRICS / "03_beneficio_validacion.csv", index=False)
    print(tabla_eco[["Modelo", "contactados", "precision", "recall",
                     "beneficio_neto", "roi"]].to_string(index=False))

    candidatos = tabla_cv[~tabla_cv["Modelo"].str.startswith("Línea base")]
    ganador = candidatos.iloc[0]["Modelo"]
    modelo_base = ajustados[ganador]
    print(f"\n  Modelo seleccionado: {ganador}")

    # ------------------------------------------------------------------
    # CALIBRACIÓN DE PROBABILIDADES
    # ------------------------------------------------------------------
    # Los modelos se entrenaron con class_weight / scale_pos_weight para
    # compensar el desbalance. Eso mejora la capacidad de ordenar, pero
    # DEFORMA la probabilidad: el modelo entrena como si el 50% desertara y
    # devuelve valores sistemáticamente altos. Como el umbral de decisión sale
    # de una comparación económica sobre la probabilidad (contactar si
    # p·e·V > c), usar la probabilidad deformada da un umbral equivocado.
    # Se corrige con regresión isotónica ajustada sobre validación, que es
    # monótona y por lo tanto no altera el ranking ni el AUC.
    try:
        from sklearn.frozen import FrozenEstimator

        modelo_final = CalibratedClassifierCV(
            FrozenEstimator(modelo_base), method="isotonic"
        ).fit(Xva, yva)
    except ImportError:  # scikit-learn < 1.6
        modelo_final = CalibratedClassifierCV(
            modelo_base, method="isotonic", cv="prefit"
        ).fit(Xva, yva)

    ece_antes = evaluate.error_calibracion_esperado(yte, modelo_base.predict_proba(Xte)[:, 1])
    ece_despues = evaluate.error_calibracion_esperado(yte, modelo_final.predict_proba(Xte)[:, 1])
    print(f"  Calibración isotónica sobre validación: "
          f"ECE {ece_antes:.4f} -> {ece_despues:.4f} en test")
    resultados["calibracion"] = {"ece_antes": ece_antes, "ece_despues": ece_despues}
    ajustados[f"{ganador} (calibrado)"] = modelo_final

    # ======================================================================
    print("\n" + "=" * 72)
    print("4. EVALUACIÓN FINAL EN TEST OUT-OF-TIME")
    print("=" * 72)
    probas_test = {n: m.predict_proba(Xte)[:, 1] for n, m in ajustados.items()}
    # Para medir capacidad de ordenar se usa el modelo tal cual; para toda
    # decisión económica se usan las probabilidades calibradas, que son las
    # únicas comparables contra un umbral derivado de soles.
    p_test_ranking = probas_test[ganador]
    p_test = probas_test[f"{ganador} (calibrado)"]

    filas_test = []
    for nombre, p in probas_test.items():
        auc_v, lo, hi = evaluate.ic_bootstrap(yte, p, roc_auc_score)
        pr_v, plo, phi = evaluate.ic_bootstrap(yte, p, average_precision_score)
        m_ = evaluate.metricas(yte, p, umbral=u_opt, k=C.CAPACIDAD_PERIODO_TEST)
        filas_test.append({
            "Modelo": nombre,
            "ROC_AUC": round(auc_v, 4), "ROC_AUC_IC95": f"[{lo:.3f}, {hi:.3f}]",
            "PR_AUC": round(pr_v, 4), "PR_AUC_IC95": f"[{plo:.3f}, {phi:.3f}]",
            "Brier": round(m_["Brier"], 4),
            "ECE": round(evaluate.error_calibracion_esperado(yte, p), 4),
            "Precision_umbral": round(m_["Precision"], 4),
            "Recall_umbral": round(m_["Recall"], 4),
            f"Lift@{C.CAPACIDAD_PERIODO_TEST}": round(m_[f"Lift@{C.CAPACIDAD_PERIODO_TEST}"], 2),
        })
    tabla_test = pd.DataFrame(filas_test).sort_values("PR_AUC", ascending=False)
    tabla_test.to_csv(C.METRICS / "04_resultados_test.csv", index=False)
    print(tabla_test.to_string(index=False))

    # ======================================================================
    print("\n" + "=" * 72)
    print("5. DIAGNÓSTICO CON EL ORÁCULO")
    print("=" * 72)
    orac = oracle.cargar_oraculo()
    techo = oracle.techo_bayes(test, orac)
    p_oraculo = test.merge(orac, on="EstudianteID", how="left")["ProbabilidadVerdadera"].to_numpy()
    print(f"  Techo de Bayes en test: ROC-AUC {techo['ROC_AUC']:.4f} | "
          f"PR-AUC {techo['PR_AUC']:.4f}")

    senal = {}
    for nombre, p in probas_test.items():
        if nombre.startswith("Línea base: prevalencia"):
            continue
        a = roc_auc_score(yte, p)
        senal[nombre] = {
            "ROC_AUC": round(a, 4),
            "fraccion_senal": round(oracle.fraccion_senal_capturada(a, techo["ROC_AUC"]), 4),
        }
        print(f"  {nombre:30s} captura {senal[nombre]['fraccion_senal']:.1%} de la señal recuperable")

    brecha_eco = oracle.brecha_economica(test, orac, p_test, C.CAPACIDAD_PERIODO_TEST)
    print(f"  Beneficio esperado con {C.CAPACIDAD_PERIODO_TEST} contactos (capacidad del periodo) -> "
          f"modelo S/ {brecha_eco['beneficio_esperado_modelo']:,.0f} | "
          f"oráculo S/ {brecha_eco['beneficio_esperado_oraculo']:,.0f} | "
          f"azar S/ {brecha_eco['beneficio_esperado_azar']:,.0f}")
    print(f"  El modelo captura el {brecha_eco['captura_del_maximo']:.1%} del máximo alcanzable")

    # ======================================================================
    print("\n" + "=" * 72)
    print("6. DECISIÓN ECONÓMICA SOBRE TEST")
    print("=" * 72)
    curva_u = costs.curva_umbral(yte, p_test)
    curva_k = costs.curva_capacidad(yte, p_test)
    sens = costs.sensibilidad(yte, p_test)
    curva_u.to_csv(C.METRICS / "05_curva_umbral.csv", index=False)
    curva_k.to_csv(C.METRICS / "06_curva_capacidad.csv", index=False)
    sens.to_csv(C.METRICS / "07_sensibilidad.csv", index=False)
    costs.resumen_supuestos().to_csv(C.METRICS / "08_supuestos_economicos.csv", index=False)

    eco_opt = costs.beneficio_neto(yte, (p_test >= u_opt).astype(int))
    eco_05 = costs.beneficio_neto(yte, (p_test >= 0.5).astype(int))
    eco_regla = costs.beneficio_neto(yte, regla.predict(Xte))
    fila_cap = curva_k.iloc[(curva_k["k"] - C.CAPACIDAD_PERIODO_TEST).abs().argmin()]

    print(f"  Umbral económico ({u_opt:.3f}): contacta {eco_opt['contactados']}, "
          f"beneficio S/ {eco_opt['beneficio_neto']:,.0f}, ROI {eco_opt['roi']:.1%}")
    print(f"  Umbral por defecto (0.500): contacta {eco_05['contactados']}, "
          f"beneficio S/ {eco_05['beneficio_neto']:,.0f}")
    print(f"  Regla de negocio actual:     contacta {eco_regla['contactados']}, "
          f"beneficio S/ {eco_regla['beneficio_neto']:,.0f}")

    # ======================================================================
    print("\n" + "=" * 72)
    print("7. INTERPRETABILIDAD, CALIBRACIÓN Y EQUIDAD")
    print("=" * 72)
    imp = permutation_importance(
        modelo_base, Xte, yte, scoring="average_precision",
        n_repeats=10, random_state=C.SEED, n_jobs=1,
    )
    tabla_imp = (
        pd.DataFrame({"variable": Xte.columns,
                      "importancia": imp.importances_mean,
                      "std": imp.importances_std})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    tabla_imp.to_csv(C.METRICS / "09_importancia_permutacion.csv", index=False)
    print(tabla_imp.head(8).to_string(index=False))

    equidad = evaluate.desempeno_por_subgrupo(
        test.reset_index(drop=True), yte.to_numpy(), p_test,
        ["Modalidad", "CategoriaCurso", "TieneBecaDescuento"], u_opt,
    )
    # Rango etario como variable sensible adicional
    test_edad = test.reset_index(drop=True).copy()
    test_edad["RangoEdad"] = pd.cut(
        test_edad["Edad"], [0, 24, 34, 100], labels=["16-24", "25-34", "35+"]
    )
    equidad = pd.concat([
        equidad,
        evaluate.desempeno_por_subgrupo(test_edad, yte.to_numpy(), p_test, ["RangoEdad"], u_opt),
    ], ignore_index=True)
    equidad.to_csv(C.METRICS / "10_equidad_subgrupos.csv", index=False)
    print("\n" + equidad.to_string(index=False))

    # ======================================================================
    print("\n" + "=" * 72)
    print("8. FIGURAS")
    print("=" * 72)
    figuras = []
    figuras.append(plots.tasa_por_variable(
        df, ["DiasUltimaSesion", "TenureSemanas", "Edad", "PctTareasATiempo",
             "HorasUsoSemana", "PromedioNotas"]))
    figuras.append(plots.drift_cohortes(df))
    figuras.append(plots.correlaciones(df, C.NUMERICAS))

    principales = [n for n in ["Regresión Logística", "Árbol de Decisión",
                               "Random Forest", "XGBoost", "Red Neuronal (MLP)",
                               "Línea base: regla de negocio"]
                   if n in probas_test]
    figuras.append(plots.curvas_roc_pr(
        yte, {n: probas_test[n] for n in principales}, p_oraculo))
    figuras.append(plots.techo_bayes_barras(
        {n: roc_auc_score(yte, probas_test[n]) for n in principales}, techo["ROC_AUC"]))
    figuras.append(plots.calibracion({
        n: evaluate.datos_calibracion(yte, probas_test[n])
        for n in principales if not n.startswith("Línea base")}))

    mlp = ajustados.get("Red Neuronal (MLP)")
    if mlp is not None and hasattr(mlp.named_steps["modelo"], "historial_"):
        figuras.append(plots.curva_aprendizaje_mlp(mlp.named_steps["modelo"].historial_))

    figuras.append(plots.beneficio_vs_umbral(curva_u, u_opt))
    # La política se ejecuta a capacidad: el umbral marca 856 candidatos, pero
    # el equipo contacta a los ~831 de mayor riesgo. Se anota ese punto (no el
    # umbral puro) para que el gráfico sea coherente con la tabla de impacto.
    cap = fila_cap  # con_capacidad_actual: k≈829, TP=367, beneficio=34.761
    figuras.append(plots.beneficio_vs_capacidad(
        curva_k, C.CAPACIDAD_PERIODO_TEST, len(yte),
        beneficio_operacion=cap["beneficio_neto"],
        k_operacion=C.CAPACIDAD_PERIODO_TEST,
        tp_operacion=cap["TP"]))
    figuras.append(plots.lift_por_decil(curva_k))
    figuras.append(plots.sensibilidad_heatmap(sens))
    figuras.append(plots.matriz_confusion(
        confusion_matrix(yte, (p_test >= u_opt).astype(int)),
        f"{ganador} — umbral económico {u_opt:.2f}", "decision_matriz_economica.png"))
    figuras.append(plots.matriz_confusion(
        confusion_matrix(yte, (p_test >= 0.5).astype(int)),
        f"{ganador} — umbral por defecto 0.50", "decision_matriz_default.png"))
    figuras.append(plots.importancia_permutacion(tabla_imp))
    figuras.append(plots.equidad(equidad))

    # Dependencia parcial: qué forma le atribuye cada familia de modelos
    for variable, archivo in [("DiasUltimaSesion", "interpret_pdp_dias.png"),
                              ("TenureSemanas", "interpret_pdp_tenure.png")]:
        grid = np.linspace(Xte[variable].quantile(0.01), Xte[variable].quantile(0.99), 30)
        curvas = {}
        for nombre in ["Regresión Logística", "XGBoost", "Red Neuronal (MLP)"]:
            if nombre not in ajustados:
                continue
            ys = []
            for v in grid:
                Xmod = Xte.copy()
                Xmod[variable] = v
                ys.append(ajustados[nombre].predict_proba(Xmod)[:, 1].mean())
            curvas[nombre] = np.array(ys)
        figuras.append(plots.dependencia_parcial(grid, curvas, variable, archivo))

    print("  " + "\n  ".join(figuras))

    # ======================================================================
    resumen = {
        "modelo_ganador": ganador,
        "umbral_economico": round(u_opt, 4),
        "calibracion": resultados["calibracion"],
        "n_variables_modelo": int(Xtr.shape[1]),
        "datos": resultados["datos"],
        "particion": resultados["particion"],
        "hiperparametros": mejores_params,
        "techo_bayes_test": techo,
        "senal_capturada": senal,
        "brecha_economica": brecha_eco,
        "test_ganador": tabla_test[tabla_test["Modelo"] == ganador].to_dict("records")[0],
        "economia_test": {
            "umbral_economico": eco_opt,
            "umbral_defecto": eco_05,
            "regla_negocio": eco_regla,
            "con_capacidad_actual": fila_cap.to_dict(),
            "beneficio_maximo_curva": curva_u.loc[curva_u["beneficio_neto"].idxmax()].to_dict(),
        },
        "top_variables": tabla_imp.head(10).to_dict("records"),
        "segundos_totales": round(time.time() - t0, 1),
    }
    with open(C.METRICS / "00_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=_json)

    print(f"\nListo en {resumen['segundos_totales']:.0f}s. "
          f"Métricas en reports/metrics/, figuras en reports/figures/")
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="omite la búsqueda de hiperparámetros")
    main(**vars(ap.parse_args()))
