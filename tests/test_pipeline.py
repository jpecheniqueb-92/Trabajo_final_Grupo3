"""
Pruebas que protegen las decisiones metodológicas del proyecto.

No verifican que el modelo sea bueno -eso lo dicen las métricas- sino que el
experimento sea válido: que no haya fuga de información, que la partición sea
temporal, que la limpieza haga lo que dice y que los resultados se repitan con
la misma semilla. Son las condiciones sin las cuales las métricas no
significan nada.

Ejecutar:  python -m pytest tests/ -v      (o: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.costs import beneficio_neto, umbral_optimo  # noqa: E402
from src.data_generation import generar  # noqa: E402
from src.models import ReglaDeNegocio, construir_modelos  # noqa: E402
from src.preprocessing import (  # noqa: E402
    construir_preprocesador,
    limpiar,
    particion_temporal,
    preparar,
)


def test_generador_es_reproducible():
    """La misma semilla debe producir exactamente el mismo dataset."""
    a, _ = generar(n=500, seed=7)
    b, _ = generar(n=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generador_respeta_la_prevalencia_objetivo():
    df, _ = generar(n=6000, seed=1, prevalencia_objetivo=0.20)
    assert 0.17 < df[C.TARGET].mean() < 0.23


def test_limpieza_elimina_duplicados_y_marca_edades_imposibles():
    df, _ = generar(n=1500, seed=3)
    limpio = limpiar(df, verbose=False)
    assert limpio["EstudianteID"].is_unique
    assert limpio["Edad"].dropna().between(15, 80).all()
    assert set(limpio["Modalidad"].dropna()) <= {"Autoestudio", "Con mentor en vivo"}


def test_particion_es_temporal_y_sin_solapamiento():
    """Ninguna cohorte puede aparecer en dos particiones a la vez."""
    df, _ = generar(n=4000, seed=5)
    df = preparar(df, verbose=False)
    train, valid, test = particion_temporal(df)

    c_tr = set(train[C.COL_COHORTE])
    c_va = set(valid[C.COL_COHORTE])
    c_te = set(test[C.COL_COHORTE])
    assert not (c_tr & c_va) and not (c_va & c_te) and not (c_tr & c_te)
    # y el orden temporal debe respetarse
    assert max(c_tr) < min(c_va) and max(c_va) < min(c_te)
    # ningún estudiante puede estar en dos particiones
    ids = pd.concat([train, valid, test])["EstudianteID"]
    assert ids.is_unique


def test_preprocesador_no_usa_informacion_de_la_particion_de_evaluacion():
    """
    La mediana de imputación debe calcularse solo con entrenamiento.

    Se comprueba ajustando el preprocesador sobre entrenamiento y verificando
    que el valor imputado coincide con la mediana de entrenamiento y no con la
    del conjunto completo.
    """
    df, _ = generar(n=4000, seed=11)
    df = preparar(df, verbose=False)
    train, _, test = particion_temporal(df)

    prep = construir_preprocesador(escalar=False).fit(train)
    imputador = prep.named_transformers_["num"].named_steps["imputar"]
    idx = C.NUMERICAS.index("PromedioNotas")

    mediana_train = train["PromedioNotas"].median()
    mediana_todo = df["PromedioNotas"].median()
    assert np.isclose(imputador.statistics_[idx], mediana_train)
    if not np.isclose(mediana_train, mediana_todo):
        assert not np.isclose(imputador.statistics_[idx], mediana_todo)


def test_transformacion_no_deja_nulos_ni_columnas_del_target():
    df, _ = generar(n=2000, seed=13)
    df = preparar(df, verbose=False)
    train, _, test = particion_temporal(df)
    quitar = [C.TARGET, "EstudianteID", C.COL_COHORTE]
    prep = construir_preprocesador().fit(train.drop(columns=quitar))
    salida = prep.transform(test.drop(columns=quitar))
    assert not np.isnan(salida).any()
    assert C.TARGET not in prep.get_feature_names_out()


def test_modelos_entrenan_y_devuelven_probabilidades_validas():
    df, _ = generar(n=2500, seed=17)
    df = preparar(df, verbose=False)
    train, valid, _ = particion_temporal(df)
    quitar = [C.TARGET, "EstudianteID", C.COL_COHORTE]
    Xtr, ytr = train.drop(columns=quitar), train[C.TARGET]
    Xva = valid.drop(columns=quitar)

    for nombre, pipe in construir_modelos(4.0).items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        assert len(p) == len(Xva), nombre
        assert ((p >= 0) & (p <= 1)).all(), nombre


def test_umbral_economico_es_el_punto_de_indiferencia():
    """
    En el umbral óptimo, contactar a un estudiante con esa probabilidad exacta
    debe tener valor esperado cero: p·e·V = c.
    """
    u = umbral_optimo()
    valor_esperado = u * C.EFECTIVIDAD_INTERVENCION * C.VALOR_ESTUDIANTE_PERDIDO
    assert np.isclose(valor_esperado, C.COSTO_CONTACTO)


def test_beneficio_neto_cuadra_con_la_matriz_de_confusion():
    y = np.array([1, 1, 0, 0, 1, 0])
    contactar = np.array([1, 0, 1, 0, 1, 1])
    r = beneficio_neto(y, contactar)
    assert r["TP"] == 2 and r["FP"] == 2 and r["FN"] == 1 and r["TN"] == 1
    esperado = 2 * C.EFECTIVIDAD_INTERVENCION * C.VALOR_ESTUDIANTE_PERDIDO - 4 * C.COSTO_CONTACTO
    assert np.isclose(r["beneficio_neto"], round(esperado, 2))


def test_regla_de_negocio_marca_exactamente_lo_que_dice():
    df, _ = generar(n=800, seed=19)
    df = preparar(df, verbose=False)
    regla = ReglaDeNegocio(dias_umbral=10).fit(df)
    esperado = (df["DiasUltimaSesion"] >= 10).astype(int).to_numpy()
    assert (regla.predict(df) == esperado).all()


def test_el_oraculo_no_entra_al_conjunto_de_variables():
    """La probabilidad verdadera nunca puede estar entre las predictoras."""
    df, oraculo = generar(n=600, seed=23)
    assert "ProbabilidadVerdadera" not in df.columns
    df = preparar(df, verbose=False)
    assert "ProbabilidadVerdadera" not in df.columns
    assert "ProbabilidadVerdadera" in oraculo.columns


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK   {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"  FALLA {nombre}: {e}")
    print(f"\n{'Todas las pruebas pasaron' if not fallos else f'{fallos} prueba(s) fallaron'}")
    sys.exit(1 if fallos else 0)
