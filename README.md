# EduTech Perú — Predicción de deserción de estudiantes

Proyecto final del curso **Machine Learning & Deep Learning** — ESAN — Grupo 3

Modelo de alerta temprana que identifica qué estudiantes de una academia online
van a abandonar su curso, para que el equipo académico intervenga antes de
perderlos. La pregunta que responde el proyecto no es "quién va a desertar" sino
**"a quién conviene llamar esta semana con los tutores que tenemos"**.

---

## Resultado en una línea

Sobre un periodo de prueba de cuatro meses y 1.956 estudiantes, el modelo con
umbral de decisión económico genera **S/ 34.952** de beneficio neto frente a
los **S/ 19.623** de la regla que la academia usa hoy: **S/ 15.329 adicionales
(+78%)**, con un ROI de 227% y sin necesidad de contratar más tutores.

| Política de contacto | Contactados | Desertores detectados | Beneficio neto |
|---|---:|---:|---:|
| Regla actual (≥10 días sin ingresar) | 414 | 200 | S/ 19.623 |
| Modelo con umbral por defecto (0,5) | 292 | 218 | S/ 24.256 |
| **Modelo con umbral económico (0,133)** | **856** | **372** | **S/ 34.952** |

---

## Qué distingue a este trabajo

**1. El umbral de decisión se calcula con dinero, no con F1.**
Contactar a un estudiante cuesta S/ 18 y recuperarlo vale S/ 483 por la
probabilidad de que la intervención funcione. De ahí sale el umbral óptimo
(0,133) y no del 0,5 que traen los modelos por defecto. Esa sola decisión vale
S/ 10.696 en el periodo evaluado.

**2. Se mide el techo teórico del problema.**
Como el proceso generador de los datos es conocido, se puede construir el
clasificador óptimo de Bayes y saber cuál es el máximo alcanzable: ROC-AUC
0,923. El mejor modelo llega a 0,884, es decir **captura el 91% de la señal
recuperable**. Eso permite afirmar algo que casi nunca se puede afirmar: seguir
puliendo modelos ya no rinde, lo que falta son variables nuevas.

**3. Las probabilidades se calibran antes de decidir.**
Entrenar con `scale_pos_weight` mejora el ranking pero deforma la probabilidad.
Como el umbral se deriva de una comparación económica sobre esa probabilidad,
usarla sin calibrar da un umbral equivocado. La calibración isotónica baja el
error de calibración de 0,123 a 0,025.

**4. La validación es temporal, no aleatoria.**
Los datos tienen drift: la tasa de deserción sube de 18% a 23% a lo largo de 18
cohortes mensuales. Un split aleatorio esconde exactamente ese riesgo. Acá el
modelo se entrena con las 11 primeras cohortes y se evalúa sobre las 4 últimas,
igual que ocurriría en producción.

**5. Se probó Deep Learning y se reporta que no compensa.**
La red neuronal (0,888) no supera de forma distinguible al boosting (0,884) ni
a la regresión logística (0,885): los intervalos de confianza al 95% se
solapan por completo. La conclusión no es "usamos el modelo más sofisticado"
sino "medimos si valía la pena y no lo valía".

---

## Cómo reproducir

Probado con **Python 3.11**. Todas las rutas son relativas al repositorio: no
hay que editar ningún archivo para ejecutarlo en otra máquina.

```bash
git clone <url-del-repositorio>
cd edutech-desercion

python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generar el dataset sintético (semilla fija: siempre da el mismo resultado)
python -m src.data_generation

# 2. Experimento completo: búsqueda de hiperparámetros, comparación,
#    calibración, evaluación temporal, análisis económico y figuras
python -m src.run_experiment          # ~6 min
python -m src.run_experiment --rapido # ~1 min, sin búsqueda de hiperparámetros

# 3. Estudios de ablación que sustentan las decisiones metodológicas
python -m src.ablacion

# 4. Pruebas de validez del experimento (fuga de datos, partición, economía)
python tests/test_pipeline.py

# 5. Verifica que cada cifra de la presentación y el informe coincida
#    con lo que produjo el experimento
python tests/test_cifras_reportadas.py
```

La quinta prueba es la que mantiene honesta a la presentación: contrasta las 30
cifras citadas en las láminas y en el informe contra
`reports/metrics/00_resumen.json`. Si alguien cambia un supuesto en
`src/config.py` y regenera los resultados sin actualizar los documentos, la
prueba falla y nombra el número que quedó viejo.

Todo corre en CPU. No se requiere GPU ni conexión a internet después de
instalar las dependencias.

### Qué se genera

| Ruta | Contenido |
|---|---|
| `data/raw/` | Dataset sintético y probabilidades verdaderas (oráculo) |
| `data/processed/` | Dataset limpio con variables derivadas |
| `reports/metrics/` | 12 archivos con todas las métricas y tablas del informe |
| `reports/figures/` | 17 figuras usadas en el informe y la presentación |

El archivo `reports/metrics/00_resumen.json` concentra todos los números que
aparecen en la presentación, para que cualquiera pueda verificar que la lámina
y el código dicen lo mismo.

---

## Dónde está cada cosa que pide la rúbrica

| Requisito de la rúbrica | Dónde se cumple |
|---|---|
| Preprocesamiento de los datos | `src/preprocessing.py`, `notebooks/01_exploracion.ipynb` |
| Construcción de la línea base | `src/models.py` → `ReglaDeNegocio` y `DummyClassifier`; resultados en `reports/metrics/01_comparacion_validacion.csv` |
| Entrenamiento de los modelos | `src/models.py` + `src/run_experiment.py` (secciones 1 a 3) |
| Evaluación | `src/evaluate.py`, `reports/metrics/04_resultados_test.csv` |
| Comparación de alternativas | `reports/metrics/01_comparacion_validacion.csv` y `04_resultados_test.csv`; ablaciones en `src/ablacion.py` |
| Prueba del objetivo principal | `src/costs.py` y `reports/metrics/03_beneficio_validacion.csv`, `05_curva_umbral.csv`, `06_curva_capacidad.csv` |
| Instrucciones de reproducción | Sección "Cómo reproducir" de este archivo |
| Métricas coherentes con el costo de los errores | `src/costs.py`, `docs/supuestos_economicos.md` |
| Justificación de ML frente a DL | `docs/informe.md` §4.2 y `src/models.py` → `MLPTorch` |
| Arquitectura, pérdida y medidas contra el sobreajuste del DL | `src/models.py` → `MLPTorch` (docstring) y `reports/figures/mlp_curva_aprendizaje.png` |
| Fuentes, herramientas y uso de IA declarados | Sección "Créditos y fuentes" de este archivo y `docs/informe.md` §9 |

## Estructura del repositorio

```
edutech-desercion/
├── src/
│   ├── config.py            Rutas, semillas y supuestos económicos en un solo lugar
│   ├── data_generation.py   Proceso generador sintético y documentado
│   ├── preprocessing.py     Limpieza, variables derivadas, partición temporal
│   ├── models.py            Líneas base, modelos de ML y red neuronal en PyTorch
│   ├── costs.py             Economía del error, umbral óptimo y sensibilidad
│   ├── evaluate.py          Métricas, validación cruzada, bootstrap, calibración
│   ├── oracle.py            Techo de Bayes y fracción de señal capturada
│   ├── plots.py             Figuras
│   ├── ablacion.py          Estudios de ablación metodológica
│   └── run_experiment.py    Orquestador de punta a punta
├── notebooks/
│   └── 01_exploracion.ipynb Análisis exploratorio comentado
├── tests/
│   ├── test_pipeline.py     Pruebas de fuga de datos, partición y economía
│   └── test_cifras_reportadas.py  Contrasta las láminas contra los resultados
├── data/                    Dataset sintético (versionado: pesa 2 MB y no es confidencial)
├── reports/
│   ├── metrics/             12 archivos con todas las métricas del informe
│   └── figures/             17 figuras
├── docs/
│   ├── informe.md                       Informe completo (fuente)
│   ├── supuestos_economicos.md
│   ├── guion_sustentacion.md
│   ├── EduTech_Desercion_Presentacion.pdf   ENTREGABLE
│   ├── EduTech_Desercion_Presentacion.pptx  (editable)
│   ├── EduTech_Desercion_Informe.pdf
│   └── EduTech_Desercion_Guion_Sustentacion.pdf
├── requirements.txt
└── README.md
```

---

## Decisiones metodológicas y por qué

**Por qué datos sintéticos.** No hay acceso a la base transaccional de una
academia real. La rúbrica lo contempla. El costo es que los resultados no son
evidencia sobre el mundo; el beneficio es que conocemos el proceso generador y
podemos medir el techo teórico, cosa imposible con datos reales.

**Por qué el proceso generador tiene esta forma.** Un generador que produce el
target como un logit lineal hace que la regresión logística gane por
construcción y vuelve vacía la comparación de modelos. El generador de este
proyecto incorpora mecanismos documentados del dominio: efecto umbral en la
inactividad, curva de bañera del riesgo a lo largo del curso, rendimientos
decrecientes del uso de plataforma, interacciones entre modalidad y comunidad,
y nulos no aleatorios. `src/ablacion.py` demuestra que el modelo ganador cambia
según cuál de los dos generadores se use, que es justamente lo que hace que la
comparación tenga contenido.

**Por qué PR-AUC y no accuracy.** Con 20% de deserción, decir "nadie deserta"
acierta el 80% de las veces. Cualquier accuracy por debajo de 0,80 es peor que
no hacer nada y cualquiera por encima puede lograrse sin detectar a un solo
desertor.

**Por qué XGBoost.** Fue el mejor en validación cruzada sobre entrenamiento
(PR-AUC 0,691 ± 0,031), que es el único criterio disponible antes de tocar el
conjunto de prueba. En prueba quedó estadísticamente empatado con la red
neuronal y la regresión logística; ese resultado se reporta tal cual en el
informe en lugar de cambiar la elección a posteriori.

**Limitación principal.** El modelo predice *quién va a desertar*, no *a quién
le sirve la intervención*. Un estudiante puede tener riesgo altísimo y ser
irrecuperable, y otro con riesgo medio puede responder muy bien a una llamada.
Medir eso requiere un experimento aleatorizado y un modelo de uplift; está
descrito en la sección de trabajo futuro del informe.

---

## Créditos y fuentes

- Datos: sintéticos, generados por `src/data_generation.py` con semilla fija.
  Los supuestos de comportamiento se basan en patrones documentados en
  literatura de deserción en educación en línea (ver informe, sección de
  referencias).
- Bibliotecas: scikit-learn, XGBoost, PyTorch, pandas, NumPy, Matplotlib, SHAP.
- Herramientas de IA: se utilizó un asistente de IA generativa como apoyo en la
  redacción de documentación y en la revisión de código. Todas las decisiones
  metodológicas, los supuestos económicos y la interpretación de resultados son
  del equipo, que se hace responsable de sustentarlos.

**Grupo 3** — Botta Loyola · Echenique Bejarano · Gonzales Lozada · Laura Lazo
