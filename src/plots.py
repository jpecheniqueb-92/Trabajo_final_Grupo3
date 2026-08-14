"""Figuras del informe y de la presentación. Cada función guarda un PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from . import config as C

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.titleweight": "bold",
        "figure.autolayout": True,
    }
)


def _guardar(fig, nombre: str) -> str:
    ruta = C.FIGURES / nombre
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(ruta.relative_to(C.ROOT))


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def tasa_por_variable(df: pd.DataFrame, variables: list[str], bins: int = 12) -> str:
    """Tasa de deserción observada por tramo de cada variable continua."""
    n = len(variables)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.1 * ((n + 1) // 2), 5.6))
    for ax, var in zip(np.ravel(axes), variables):
        d = df[[var, C.TARGET]].dropna()
        try:
            tramos = pd.qcut(d[var], bins, duplicates="drop")
        except ValueError:
            tramos = pd.cut(d[var], bins)
        agg = d.groupby(tramos, observed=True)[C.TARGET].agg(["mean", "size"])
        centros = [iv.mid for iv in agg.index]
        ax.plot(centros, agg["mean"], "o-", color=C.PALETA["acento"], ms=3.5, lw=1.6)
        ax.axhline(df[C.TARGET].mean(), color=C.PALETA["gris"], ls="--", lw=1)
        ax.set_title(var, fontsize=8.5)
        ax.set_ylabel("tasa de deserción", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(
        "Relación entre comportamiento y deserción: las formas no son líneas rectas",
        fontsize=10.5,
        y=1.02,
        fontweight="bold",
    )
    return _guardar(fig, "eda_tasa_por_variable.png")


def drift_cohortes(df: pd.DataFrame) -> str:
    agg = df.groupby(C.COL_COHORTE)[C.TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(agg.index, agg["mean"], "o-", color=C.PALETA["primario"], lw=1.8, ms=4)
    z = np.polyfit(np.arange(len(agg)), agg["mean"], 1)
    ax.plot(
        agg.index,
        np.poly1d(z)(np.arange(len(agg))),
        "--",
        color=C.PALETA["acento"],
        lw=1.4,
        label=f"tendencia: +{z[0]*100:.2f} p.p. por mes",
    )
    ax.set_title("La tasa de deserción no es estable en el tiempo (drift)")
    ax.set_ylabel("tasa de deserción")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend(fontsize=8)
    return _guardar(fig, "eda_drift_cohortes.png")


def correlaciones(df: pd.DataFrame, columnas: list[str]) -> str:
    corr = df[columnas + [C.TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            if abs(v) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
    ax.grid(False)
    ax.set_title("Correlaciones entre variables")
    fig.colorbar(im, ax=ax, shrink=0.75)
    return _guardar(fig, "eda_correlaciones.png")


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------
def curvas_roc_pr(y_true, probas: dict[str, np.ndarray], p_oraculo=None) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    for nombre, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, lw=1.7, label=f"{nombre} ({auc(fpr, tpr):.3f})",
                color=C.MODELO_COLORES.get(nombre))
        pr, rc, _ = precision_recall_curve(y_true, p)
        a2.plot(rc, pr, lw=1.7, label=f"{nombre} ({auc(rc, pr):.3f})",
                color=C.MODELO_COLORES.get(nombre))

    if p_oraculo is not None:
        fpr, tpr, _ = roc_curve(y_true, p_oraculo)
        a1.plot(fpr, tpr, "k:", lw=2, label=f"Techo de Bayes ({auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p_oraculo)
        a2.plot(rc, pr, "k:", lw=2, label=f"Techo de Bayes ({auc(rc, pr):.3f})")

    a1.plot([0, 1], [0, 1], color=C.PALETA["gris"], ls="--", lw=1)
    a1.set_xlabel("Falsos positivos")
    a1.set_ylabel("Verdaderos positivos")
    a1.set_title("Curva ROC (test out-of-time)")
    a1.legend(fontsize=7, loc="lower right")

    a2.axhline(np.mean(y_true), color=C.PALETA["gris"], ls="--", lw=1)
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precisión")
    a2.set_title("Curva Precisión-Recall (test out-of-time)")
    a2.legend(fontsize=7, loc="upper right")
    return _guardar(fig, "modelos_roc_pr.png")


def techo_bayes_barras(auc_modelos: dict[str, float], auc_oraculo: float) -> str:
    nombres = list(auc_modelos)
    valores = [auc_modelos[n] for n in nombres]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(nombres))
    ax.barh(y, valores, color=[C.MODELO_COLORES.get(n, C.PALETA["primario"]) for n in nombres], height=0.6)
    ax.axvline(auc_oraculo, color="black", ls=":", lw=2)
    ax.axvline(0.5, color=C.PALETA["gris"], ls="--", lw=1)
    for i, v in enumerate(valores):
        pct = (v - 0.5) / (auc_oraculo - 0.5)
        ax.text(v + 0.004, i, f"{v:.3f}  ({pct:.0%} de la señal)", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=8)
    ax.set_xlim(0.45, auc_oraculo + 0.115)
    ax.set_xlabel("ROC-AUC en test")
    ax.set_title("Cuánta de la señal recuperable captura cada modelo", pad=22)
    ax.invert_yaxis()
    ax.set_ylim(len(nombres) - 0.45, -1.15)
    ax.text(auc_oraculo, -0.95, f"techo de Bayes  {auc_oraculo:.3f}", fontsize=8.5,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
    ax.text(0.5, -0.95, "azar", fontsize=8, ha="center", va="center",
            color=C.PALETA["gris"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C.PALETA["gris_claro"], lw=0.8))
    return _guardar(fig, "modelos_techo_bayes.png")


def calibracion(datos: dict[str, pd.DataFrame]) -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibración perfecta")
    for nombre, d in datos.items():
        ax.plot(d["prob_predicha"], d["frecuencia_observada"], "o-", ms=4, lw=1.5,
                label=nombre, color=C.MODELO_COLORES.get(nombre))
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada de deserción")
    ax.set_title("Calibración")
    ax.legend(fontsize=7)
    return _guardar(fig, "modelos_calibracion.png")


def curva_aprendizaje_mlp(historial: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(historial["train"], lw=1.6, color=C.PALETA["primario"], label="entrenamiento")
    ax.plot(historial["val"], lw=1.6, color=C.PALETA["acento"], label="validación interna")
    mejor = int(np.argmin(historial["val"]))
    ax.axvline(mejor, color=C.PALETA["gris"], ls="--", lw=1.2)
    ax.text(mejor, max(historial["train"]), f" early stopping (época {mejor})",
            fontsize=7.5, va="top")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada binaria")
    ax.set_title("Entrenamiento de la red neuronal")
    ax.legend(fontsize=8)
    return _guardar(fig, "mlp_curva_aprendizaje.png")


# ---------------------------------------------------------------------------
# Decisión económica
# ---------------------------------------------------------------------------
def beneficio_vs_umbral(curva: pd.DataFrame, umbral_opt: float) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(curva["umbral"], curva["beneficio_neto"], lw=2, color=C.PALETA["primario"])
    ax.axhline(0, color="black", lw=0.9)

    def _marcar(u, color, etiqueta, desplazamiento):
        fila = curva.iloc[(curva["umbral"] - u).abs().argmin()]
        ax.plot(fila["umbral"], fila["beneficio_neto"], "o", ms=8, color=color)
        ax.annotate(
            f"{etiqueta}\numbral {fila['umbral']:.2f}  ·  S/ {fila['beneficio_neto']:,.0f}".replace(",", "."),
            (fila["umbral"], fila["beneficio_neto"]),
            textcoords="offset points", xytext=desplazamiento, fontsize=8,
            color=color, fontweight="bold",
        )
        return fila

    f_opt = _marcar(umbral_opt, C.PALETA["verde"], "Umbral económico", (14, 8))
    f_05 = _marcar(0.5, C.PALETA["acento"], "Umbral por defecto", (-30, -34))
    diferencia = f_opt["beneficio_neto"] - f_05["beneficio_neto"]

    ax.set_xlabel("Umbral de probabilidad para contactar")
    ax.set_ylabel("Beneficio neto (S/)")
    ax.set_ylim(0, f_opt["beneficio_neto"] * 1.22)
    ax.set_title(
        f"Elegir el umbral por costo y no por defecto vale S/ {diferencia:,.0f}".replace(",", ".")
    )
    return _guardar(fig, "decision_beneficio_umbral.png")


def beneficio_vs_capacidad(curva: pd.DataFrame, capacidad: int, n_total: int,
                           beneficio_operacion: float | None = None,
                           k_operacion: int | None = None,
                           tp_operacion: int | None = None) -> str:
    """
    Curva de beneficio neto según cuántos estudiantes se contactan (k).

    La curva tiene forma de meseta: entre ~700 y ~860 contactos el beneficio
    varía menos de S/ 40, así que no existe un óptimo puntiagudo. Por eso el
    gráfico NO etiqueta el vértice matemático como "óptimo" (eso sugeriría, de
    forma engañosa, que la política debería contactar a ~696). En su lugar:

    - sombrea la meseta donde el beneficio está a menos de S/ 100 del máximo,
    - marca el punto de operación real de la política (`k_operacion`, 856
      contactos, umbral 0,133), que es el que se reporta en la tabla de impacto,
    - y muestra la capacidad del equipo (831) cayendo dentro de la meseta.

    `beneficio_operacion` / `tp_operacion` son el beneficio y los desertores
    detectados en ese punto de operación (S/ 34.952 y 372), coherentes con la
    tabla y el resto de la presentación.
    """
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(curva["k"], curva["beneficio_neto"], lw=2, color=C.PALETA["primario"],
            label="beneficio acumulado")
    ax.axhline(0, color="black", lw=0.9)
    tope = float(curva["beneficio_neto"].max())
    ax.set_ylim(0, tope * 1.30)

    # Meseta: rango de k donde el beneficio está a menos de S/ 100 del máximo.
    meseta = curva[curva["beneficio_neto"] >= tope - 100]
    if len(meseta) > 1:
        k_lo, k_hi = int(meseta["k"].min()), int(meseta["k"].max())
        ax.axvspan(k_lo, k_hi, color=C.PALETA["verde"], alpha=0.12)
        ax.text(k_lo - 40, tope * 0.42,
                f"meseta plana\n{k_lo}–{k_hi} contactos\n(beneficio ± S/ 40)",
                fontsize=7.5, ha="right", va="center",
                color=C.PALETA["verde"], fontweight="bold")

    # Capacidad del equipo.
    ax.axvline(capacidad, color=C.PALETA["acento"], ls="--", lw=1.5)
    ax.text(capacidad + 20, tope * 1.24,
            f"capacidad del equipo\n{capacidad} contactos", fontsize=8,
            color=C.PALETA["acento"], va="top", fontweight="bold")

    # Punto de operación real de la política (no un "óptimo" puntiagudo).
    if k_operacion is not None:
        fila_op = curva.iloc[(curva["k"] - k_operacion).abs().argsort().iloc[0]]
        y_op = beneficio_operacion if beneficio_operacion is not None else fila_op["beneficio_neto"]
        ax.plot(fila_op["k"], y_op, "o", ms=9, color=C.PALETA["primario"])
        etiqueta = f"política: {int(k_operacion)} contactos\nS/ {y_op:,.0f}".replace(",", ".")
        if tp_operacion is not None:
            etiqueta += f"  ·  {int(tp_operacion)} rescatados"
        ax.annotate(
            etiqueta, (fila_op["k"], y_op),
            textcoords="offset points", xytext=(14, -34), fontsize=8.5,
            color=C.PALETA["primario"], fontweight="bold",
        )
    ax.set_xlabel("Estudiantes contactados, ordenados por riesgo (k)")
    ax.set_ylabel("Beneficio neto (S/)")
    ax.set_title(
        f"Cuántos contactar de una base de {n_total:,} estudiantes".replace(",", ".")
    )
    return _guardar(fig, "decision_beneficio_capacidad.png")


def lift_por_decil(curva: pd.DataFrame) -> str:
    d = curva[curva["pct_base_contactada"] <= 1.0].copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(d["pct_base_contactada"] * 100, d["lift"], lw=2, color=C.PALETA["primario"])
    ax.axhline(1, color=C.PALETA["gris"], ls="--", lw=1.2)
    ax.text(60, 1.06, "contactar al azar", fontsize=8, color=C.PALETA["gris"])
    ax.set_xlabel("% de la base contactada (ordenada por riesgo)")
    ax.set_ylabel("Lift")
    ax.set_title("Cuántas veces mejor que contactar al azar")
    return _guardar(fig, "decision_lift.png")


def sensibilidad_heatmap(sens: pd.DataFrame) -> str:
    piv = sens.pivot(index="efectividad", columns="costo_contacto", values="beneficio_neto")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"S/ {c:.0f}" for c in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{e:.0%}" for e in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:,.0f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    ax.set_xlabel("Costo por contacto")
    ax.set_ylabel("Efectividad de la intervención")
    ax.set_title("Beneficio neto según los supuestos (S/)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _guardar(fig, "decision_sensibilidad.png")


def matriz_confusion(cm: np.ndarray, titulo: str, nombre: str) -> str:
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No contactar", "Contactar"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Se queda", "Deserta"], fontsize=8)
    ax.set_xlabel("Decisión del modelo"); ax.set_ylabel("Resultado real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.grid(False)
    ax.set_title(titulo, fontsize=9.5)
    return _guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Interpretabilidad
# ---------------------------------------------------------------------------
def importancia_permutacion(imp: pd.DataFrame, top: int = 12) -> str:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(d["variable"], d["importancia"], xerr=d["std"], color=C.PALETA["primario"], height=0.65)
    ax.set_xlabel("Caída de PR-AUC al permutar la variable")
    ax.set_title("Importancia por permutación (medida sobre test)")
    ax.tick_params(labelsize=8)
    return _guardar(fig, "interpret_importancia.png")


def dependencia_parcial(
    valores: np.ndarray,
    curvas: dict[str, np.ndarray],
    variable: str,
    nombre_archivo: str,
    verdad: np.ndarray | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for nombre, y in curvas.items():
        ax.plot(valores, y, lw=1.9, label=nombre, color=C.MODELO_COLORES.get(nombre))
    if verdad is not None:
        ax.plot(valores, verdad, "k:", lw=2.2, label="riesgo real (oráculo)")
    ax.set_xlabel(variable)
    ax.set_ylabel("Riesgo predicho promedio")
    ax.set_title(f"Qué forma le atribuye cada modelo a {variable}")
    ax.legend(fontsize=7.5)
    return _guardar(fig, nombre_archivo)


def equidad(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["Recall"], height=0.38, color=C.PALETA["primario"], label="Recall")
    ax.barh(y + 0.2, df["Precision"], height=0.38, color=C.PALETA["naranja"], label="Precisión")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.variable}: {r.grupo} (n={r.n})" for r in df.itertuples()], fontsize=7.5)
    ax.set_title("Desempeño por subgrupo al umbral de decisión")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    return _guardar(fig, "equidad_subgrupos.png")
