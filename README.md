# where-is-my-protein 🧬📍

**Un práctico de Machine Learning aplicado a Bioinformática:** ¿se puede predecir en qué
parte de la célula termina una proteína (núcleo, citoplasma o membrana) mirando *únicamente*
su secuencia de aminoácidos?

Este repositorio contiene el material completo de una clase práctica de 4 horas para
estudiantes en etapa inicial de formación en Bioinformática, pensada para usarse después de
una clase teórica introductoria a Machine Learning. Combina **Python** (para construir el
dataset y explorar los datos con ayuda de IA) y **Orange Data Mining** (para clasificar y
comparar modelos de forma visual, sin necesidad de programar).

## El problema

Las proteínas cumplen funciones distintas según dónde se ubiquen dentro de la célula. La
pregunta central del práctico es simple de enunciar y nada trivial de responder:

> ¿Es posible predecir la localización subcelular de una proteína usando solo información
> derivada de su secuencia de aminoácidos?

Para responderla, los estudiantes construyen y comparan dos representaciones numéricas
distintas de una misma secuencia, entrenan varios modelos sobre cada una, y discuten **qué
se gana y qué se pierde** en cada representación — la idea pedagógica central es que el
desempeño de un sistema de ML depende tanto de cómo se representan los datos como del
algoritmo elegido.

## El dataset

- **Fuente:** [UniProt](https://www.uniprot.org/) (Swiss-Prot, entradas revisadas), proteínas humanas.
- **3 clases** mutuamente excluyentes: `Nucleus`, `Cytoplasm`, `Membrane`.
- **~250 proteínas**, balanceadas entre clases, longitud entre 50 y 500 aminoácidos.
- Descargado automáticamente vía la API REST de UniProt, filtrando por keywords controladas
  (`KW-0539`, `KW-0963`, `KW-0472`) y descartando proteínas con localización ambigua (más de
  una de las tres keywords a la vez).

## Las dos representaciones

| Representación | Analogía | Nº de features |
|---|---|---|
| **Bag of aminoácidos** | *bag of words* en NLP: se ignora el orden, solo importa la proporción de cada uno de los 20 aminoácidos | 20 + 10 fisicoquímicos |
| **Dipéptidos** | frecuencia de cada par de aminoácidos consecutivos — conserva algo de información local | 400 + 10 fisicoquímicos |

Ambas representaciones se completan con **10 features fisicoquímicos globales**
(peso molecular, punto isoeléctrico, hidrofobicidad promedio, aromaticidad, índice de
inestabilidad, carga neta a pH 7, y fracciones de estructura secundaria), calculados con
[Biopython](https://biopython.org/) (`Bio.SeqUtils.ProtParam`).

## El pipeline

```
UniProt (API REST)
       │
       ▼
data/01_construir_dataset.py  ──►  features_composicion_fisicoquimicos.csv
       │                            features_dipeptidos_fisicoquimicos.csv
       ▼
notebooks/exploracion_*.ipynb  (exploración visual, asistida por IA)
       │
       ▼
orange/*.ows  (clasificación: Logistic Regression, Random Forest, SVM
               → Test and Score → Confusion Matrix)
       │
       ▼
   discusión y comparación de resultados
```

## Resultado

Los dipéptidos **no** mejoran el desempeño respecto a la composición simple — y no es un
error del práctico, es el resultado más interesante de toda la actividad. Con ~250 proteínas
y ~410 features, cualquier modelo tiende a sobreajustar (*curse of dimensionality*). Pero hay
algo más profundo: ni reducir la dimensionalidad, ni agrupar aminoácidos en un alfabeto
reducido, ni agregar composición del extremo N-terminal lograron superar a la composición
simple. Ninguna representación de tipo "bolsa de fragmentos" —por más grande o chica que
sea— codifica **posición**, y eso es justamente lo que distingue a una señal de localización
nuclear real. El detalle completo de este análisis está en `docs/guia_docentes.docx`.

## Estructura del repositorio

```
where-is-my-protein/
├── data/
│   ├── 01_construir_dataset.py   # descarga UniProt + calcula todas las features
│   └── README.md
├── notebooks/
│   ├── exploracion_alumnos.ipynb            # con prompts sugeridos para IA, sin resolver
│   └── exploracion_docentes_completa.ipynb  # resuelto, con gráficos de referencia
├── orange/
│   ├── bag_of_aminoacidos.ows    # File → LR/RF/SVM → Test and Score → Confusion Matrix
│   └── dipeptidos.ows            # mismo esqueleto, para la representación de dipéptidos
├── docs/
│   ├── guia_alumnos.docx         # agenda, instrucciones paso a paso, preguntas de informe
│   └── guia_docentes.docx        # + setup técnico, resultados esperados, troubleshooting
├── requirements.txt
└── LICENSE
```

## Cómo usarlo

1. `pip install -r requirements.txt`
2. Correr `data/01_construir_dataset.py` (requiere conexión a internet) para generar los dos CSV.
3. Abrir `notebooks/exploracion_docentes_completa.ipynb` y correr **Kernel → Restart & Run
   All** para reemplazar los gráficos de prueba por los reales.
4. Instalar [Orange Data Mining](https://orangedatamining.com/download/) (gratuito) y abrir
   los workflows de `orange/` para confirmar que cargan bien.
5. Repartir a los estudiantes: los 2 CSV, `notebooks/exploracion_alumnos.ipynb`, los `.ows`,
   y `docs/guia_alumnos.docx`.

Instrucciones detalladas, agenda de la clase y guía de discusión: `docs/guia_docentes.docx`.

## Stack

Python · pandas · Biopython · scikit-learn (solo para validaciones internas) · Orange Data
Mining · UniProt REST API

## Motivación

Este práctico fue diseñado para introducir Machine Learning a estudiantes de Bioinformática
a partir de un problema biológico real, integrando manejo de datos públicos, ingeniería de
representaciones, y uso crítico de asistentes de IA como herramienta de trabajo (documentado
explícitamente, no evitado). El diseño pasó por varias iteraciones — incluyendo intentos que
no funcionaron (selección de features, alfabeto reducido, composición N-terminal) y que
terminaron siendo, en sí mismos, parte de la lección.

## Licencia

MIT — ver [`LICENSE`](LICENSE). Usalo, adaptalo y compartilo libremente citando la fuente.
