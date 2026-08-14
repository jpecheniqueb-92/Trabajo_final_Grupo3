"""
Configuración central del proyecto EduTech Perú.

Todas las rutas son relativas a la raíz del repositorio, de modo que el
proyecto se pueda clonar y ejecutar en cualquier máquina sin editar código.
Todas las semillas y supuestos económicos viven acá para que un solo archivo
documente las decisiones que afectan los resultados.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

for _p in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, METRICS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
SEED = 42
N_ESTUDIANTES = 9000
N_COHORTES = 18          # 18 cohortes mensuales (ene-2025 a jun-2026)
COHORTES_TEST = 4        # las últimas 4 cohortes se reservan como test temporal
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Variable objetivo y esquema de datos
# --------------------------------------------------------------------------
TARGET = "Desercion"
COL_COHORTE = "CohorteMes"

NUMERICAS = [
    "TenureSemanas",
    "Edad",
    "HorasUsoSemana",
    "PctTareasATiempo",
    "PromedioNotas",
    "ParticipacionForo",
    "QuejasSoporte",
    "DiasUltimaSesion",
    "CursosPreviosCompletados",
]

BINARIAS = [
    "TieneBecaDescuento",
]

CATEGORICAS = [
    "CategoriaCurso",
    "Modalidad",
    "MetodoPago",
]

# --------------------------------------------------------------------------
# Supuestos económicos del caso de negocio
# --------------------------------------------------------------------------
# Fuente de los supuestos: ver docs/supuestos_economicos.md. Todos los valores
# son declarados y sometidos a análisis de sensibilidad en src/costs.py.

PRECIO_CURSO = 890.0          # S/ ticket promedio de un curso técnico online
MARGEN_CONTRIBUCION = 0.62    # % del precio que es margen (sin costo de adquisición)
CAC = 180.0                   # S/ costo de adquirir un estudiante de reemplazo
FRACCION_NO_DEVENGADA = 0.55  # % del curso que queda sin dictar al momento del abandono

# Valor económico que se pierde cuando un estudiante deserta
VALOR_ESTUDIANTE_PERDIDO = (
    PRECIO_CURSO * MARGEN_CONTRIBUCION * FRACCION_NO_DEVENGADA + CAC
)  # ≈ S/ 483.5

# Efectividad de la intervención de retención (uplift). Supuesto central y
# rango para análisis de sensibilidad.
EFECTIVIDAD_INTERVENCION = 0.28
EFECTIVIDAD_RANGO = (0.15, 0.40)

# Costo operativo de contactar a un estudiante
HORAS_POR_CONTACTO = 0.5
COSTO_HORA_TUTOR = 36.0
COSTO_CONTACTO = HORAS_POR_CONTACTO * COSTO_HORA_TUTOR  # S/ 18

# Capacidad operativa del equipo de retención
TUTORES_RETENCION = 3
HORAS_SEMANA_POR_TUTOR = 8
CONTACTOS_POR_SEMANA = int(
    TUTORES_RETENCION * HORAS_SEMANA_POR_TUTOR / HORAS_POR_CONTACTO
)  # 48 contactos por semana

# El conjunto de prueba cubre COHORTES_TEST meses. Para comparar la capacidad
# del equipo contra el tamaño de la base evaluada hay que expresar ambas en las
# mismas unidades: cuántos contactos alcanza a hacer el equipo en ese periodo.
SEMANAS_PERIODO_TEST = COHORTES_TEST * 4.33
CAPACIDAD_PERIODO_TEST = int(CONTACTOS_POR_SEMANA * SEMANAS_PERIODO_TEST)  # ≈ 831

# --------------------------------------------------------------------------
# Estética de gráficos
# --------------------------------------------------------------------------
PALETA = {
    "primario": "#1B3A5C",
    "acento": "#E4002B",
    "verde": "#2A9D5C",
    "naranja": "#E8871A",
    "gris": "#8A94A6",
    "gris_claro": "#D8DDE5",
    "morado": "#6A4C93",
}

MODELO_COLORES = {
    "Línea base: prevalencia": PALETA["gris"],
    "Línea base: regla de negocio": PALETA["gris_claro"],
    "Regresión Logística": PALETA["primario"],
    "Árbol de Decisión": PALETA["naranja"],
    "Random Forest": PALETA["verde"],
    "XGBoost": PALETA["acento"],
    "Red Neuronal (MLP)": PALETA["morado"],
}
