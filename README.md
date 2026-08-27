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
y ~410 features, cualquier modelo tiende a sobreajustar (*curse of dimensionality*). El
detalle completo de este análisis está en `docs/guia_docentes.docx`.

### Qué probamos para romper ese techo, y no funcionó

Antes de aceptar el resultado como definitivo, probamos 4 formas distintas de recuperar
información posicional sin salir de Orange / sin entrenar una red neuronal desde cero.
Ninguna superó a la composición simple:

| # | Intento | Idea | Resultado |
|---|---|---|---|
| 1 | Selección de features (Rank) | Quedarse con los ~30-50 dipéptidos más informativos, para reducir sobreajuste | Sin mejora |
| 2 | Alfabeto reducido | Agrupar los 20 aminoácidos en 6 categorías fisicoquímicas y contar pares de grupos (36 features en vez de 400) | Sin mejora |
| 3 | Composición N-terminal | Agregar la composición calculada solo sobre los primeros 30 residuos, donde suelen concentrarse señales de localización | Sin mejora |
| 4 | Ventana deslizante de residuos básicos | Buscar, en cualquier posición de la secuencia, el tramo de 7 o 12 residuos con mayor proporción de K/R (aproximando una NLS) | Sin mejora significativa |

Los primeros dos intentos apuntaban a resolver *sobreajuste* (demasiadas features para pocas
muestras); los últimos dos apuntaban a resolver la falta de *información posicional*. Que
ninguno de los cuatro caminos funcione es una evidencia razonablemente sólida de que el techo
real combina dos cosas: (a) **el dataset es chico** (250 proteínas) para que cualquier señal
posicional, manual o aprendida, se muestre con solidez estadística, y (b) **ninguna
representación de tipo "bolsa de fragmentos"** —por más grande, chica o reagrupada que
sea— codifica *posición* de forma explícita, que es justamente lo que distingue a una señal
de localización nuclear real.

### Próximos pasos (no implementado en este repo)

- [ ] **Embeddings de un modelo de lenguaje de proteínas (ESM-2) en vez de bag-of-aminoácidos.**
  Es la opción de mejor costo/beneficio para superar el techo actual sin reentrenar un modelo
  de lenguaje propio: hay versiones chicas de ESM-2 (pocos millones de parámetros) que corren
  sin GPU. La idea es sacar el embedding de cada proteína del dataset y entrenar
  un clasificador simple (Logistic Regression o una MLP chica) sobre esos embeddings, en vez
  de sobre composición de aminoácidos. El embedding ya trae información posicional y
  evolutiva aprendida de haber visto millones de secuencias, sin que haya que diseñarla a
  mano — muy probablemente superaría a todo lo probado en este repo.
- [ ] One-hot encoding + CNN sobre la secuencia completa (con un dataset más grande, dado que
  con 250 proteínas es poco probable que rinda mejor que los intentos ya descartados).
- [ ] Agregar una cuarta clase (`Mitochondria`, keyword `KW-0496`) para acercarse más al
  problema real de clasificación multi-clase.

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
│   ├── bag_of_aminoacidos.ows              # File → LR/RF/SVM → Test and Score → Confusion Matrix
│   ├── dipeptidos.ows                      # mismo esqueleto, para la representación de dipéptidos
│   
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
explícitamente, no evitado). El diseño pasó por varias iteraciones — incluyendo los 4 intentos
fallidos documentados arriba — que terminaron siendo, en sí mismos, parte de la lección.

## Licencia

MIT — ver [`LICENSE`](LICENSE). Usalo, adaptalo y compartilo libremente citando la fuente.
