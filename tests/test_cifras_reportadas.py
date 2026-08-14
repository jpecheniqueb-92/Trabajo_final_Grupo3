"""
Verifica que cada cifra citada en la presentación y en el informe coincida con
lo que produjo el experimento.

Es fácil que una lámina quede desactualizada después de cambiar un supuesto o
volver a correr el pipeline. Esta prueba cierra ese hueco: si alguien modifica
`src/config.py` y regenera los resultados sin actualizar los documentos, la
prueba falla y dice exactamente qué número quedó viejo.

Ejecutar:  python tests/test_cifras_reportadas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402

RESUMEN = C.METRICS / "00_resumen.json"
TOLERANCIA = 0.005  # 0,5% de diferencia relativa admitida por redondeo


def _cargar():
    if not RESUMEN.exists():
        print("No existe reports/metrics/00_resumen.json.")
        print("Corré primero:  python -m src.run_experiment")
        sys.exit(2)
    with open(RESUMEN, encoding="utf-8") as f:
        return json.load(f)


def _comparar(nombre, esperado, real, entero=False):
    if entero:
        ok = int(esperado) == int(round(float(real)))
    else:
        ok = abs(float(esperado) - float(real)) <= TOLERANCIA * max(abs(float(esperado)), 1.0)
    estado = "OK  " if ok else "MAL "
    print(f"  {estado} {nombre:52s} documento={esperado:<12} experimento={real}")
    return ok


def main() -> int:
    r = _cargar()
    eco = r["economia_test"]
    fallos = 0

    print("\nCifras del resumen ejecutivo (láminas 1, 2 y 16)")
    fallos += not _comparar("Contactos con la regla actual", 414, eco["regla_negocio"]["contactados"], True)
    fallos += not _comparar("Desertores detectados por la regla actual", 200, eco["regla_negocio"]["TP"], True)
    fallos += not _comparar("Beneficio de la regla actual (S/)", 19623, eco["regla_negocio"]["beneficio_neto"], True)
    fallos += not _comparar("Contactos con umbral económico", 856, eco["umbral_economico"]["contactados"], True)
    fallos += not _comparar("Desertores detectados por el modelo", 372, eco["umbral_economico"]["TP"], True)
    fallos += not _comparar("Beneficio con umbral económico (S/)", 34952, eco["umbral_economico"]["beneficio_neto"], True)
    fallos += not _comparar("Beneficio con umbral 0,5 (S/)", 24256, eco["umbral_defecto"]["beneficio_neto"], True)

    ganancia = eco["umbral_economico"]["beneficio_neto"] - eco["regla_negocio"]["beneficio_neto"]
    fallos += not _comparar("Ganancia sobre la práctica actual (S/)", 15329, ganancia, True)
    fallos += not _comparar(
        "Ganancia porcentual sobre la práctica actual",
        78, round(100 * ganancia / eco["regla_negocio"]["beneficio_neto"]), True
    )
    costo_umbral_defecto = eco["umbral_economico"]["beneficio_neto"] - eco["umbral_defecto"]["beneficio_neto"]
    fallos += not _comparar("Costo de usar el umbral por defecto (S/)", 10696, costo_umbral_defecto, True)

    print("\nDecisión y supuestos (lámina 15)")
    fallos += not _comparar("Umbral económico", 0.133, r["umbral_economico"])
    fallos += not _comparar("Valor de un estudiante perdido (S/)", 483.5, round(C.VALOR_ESTUDIANTE_PERDIDO, 1))
    fallos += not _comparar("Costo por contacto (S/)", 18, C.COSTO_CONTACTO, True)
    fallos += not _comparar("Efectividad de la intervención", 0.28, C.EFECTIVIDAD_INTERVENCION)
    fallos += not _comparar("Capacidad del equipo en el periodo", 831, C.CAPACIDAD_PERIODO_TEST, True)
    fallos += not _comparar("k óptimo sin restricción de capacidad", 696, eco["con_capacidad_actual"]["k"] and
                            int(pd.read_csv(C.METRICS / "06_curva_capacidad.csv")
                                .sort_values("beneficio_neto", ascending=False).iloc[0]["k"]), True)

    print("\nResultados y techo de Bayes (láminas 11 y 12)")
    fallos += not _comparar("Techo de Bayes, ROC-AUC en test", 0.923, r["techo_bayes_test"]["ROC_AUC"])
    fallos += not _comparar("Señal capturada por el mejor modelo", 0.92,
                            max(v["fraccion_senal"] for v in r["senal_capturada"].values()))
    fallos += not _comparar("Captura del beneficio máximo alcanzable", 0.921, r["brecha_economica"]["captura_del_maximo"])
    fallos += not _comparar("Señal capturada por la regla de negocio", 0.531,
                            r["senal_capturada"]["Línea base: regla de negocio"]["fraccion_senal"])

    print("\nCalibración (lámina 14)")
    fallos += not _comparar("ECE antes de calibrar", 0.123, r["calibracion"]["ece_antes"])
    fallos += not _comparar("ECE después de calibrar", 0.025, r["calibracion"]["ece_despues"])

    print("\nDatos y partición (láminas 5, 6 y 7)")
    d, p = r["datos"], r["particion"]
    fallos += not _comparar("Registros crudos", 9028, d["filas_crudas"], True)
    fallos += not _comparar("Registros tras limpieza", 9000, d["filas_limpias"], True)
    fallos += not _comparar("Tasa de deserción global", 0.196, d["tasa_desercion"])
    fallos += not _comparar("Duplicados eliminados", 28, d["duplicados_eliminados"], True)
    fallos += not _comparar("Edades imposibles corregidas", 15, d["edades_invalidas"], True)
    fallos += not _comparar("n de entrenamiento", 5584, p["train"]["n"], True)
    fallos += not _comparar("n de validación", 1460, p["valid"]["n"], True)
    fallos += not _comparar("n de prueba", 1956, p["test"]["n"], True)

    print("\nModelo seleccionado")
    ok = r["modelo_ganador"] == "XGBoost"
    print(f"  {'OK  ' if ok else 'MAL '} {'Modelo ganador':52s} documento=XGBoost     experimento={r['modelo_ganador']}")
    fallos += not ok

    print()
    if fallos:
        print(f"{fallos} cifra(s) del documento no coinciden con el experimento.")
        print("Actualizá la presentación y el informe, o revisá qué cambió en src/config.py.")
        return 1
    print("Todas las cifras de la presentación y el informe coinciden con reports/metrics/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
