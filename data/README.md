# data/

## `02_comparar_representaciones.py`

Validación interna (scikit-learn) que compara, con las **mismas particiones** y los mismos
tres modelos de Orange, distintas representaciones a dos tamaños de dataset (~250 y 750
proteínas):

- composición, composición + fisicoquímicos (método original)
- sólo dipéptidos, dipéptidos + fisicoquímicos
- sólo señales, composición + fisicoquímicos + señales (método nuevo), dipéptidos + señales

```bash
python 02_comparar_representaciones.py
```

Genera `comparacion_representaciones.csv` (en este directorio) y
`../img/comparacion_representaciones.png`, e imprime una tabla resumen. Requiere que
`01_construir_dataset.py` ya haya corrido.

## `03_experimentos.py`

Experimentos controlados (sólo las 3 clases del práctico) con un **conjunto de test reservado
que el entrenamiento nunca ve**:

- Descarga todo lo disponible con el filtro del práctico (revisadas, humanas, evidencia a nivel
  de proteína, largo 50-500, keywords mutuamente excluyentes).
- Deja hasta 1000 proteínas por clase para entrenar (3000) y el resto es el test. El sobrante es
  muy favorable a Membrana, así que se evalúa sobre su subconjunto **balanceado** más grande
  (375 = 125 por clase) con **balanced accuracy**.

Los tres experimentos que se usan en clase (el resto se recortó para que entren en 4 h):

1. **Overfitting:** F1 en train (resustitución) vs F1 en test; la brecha es el overfitting.
2. **Curse of dimensionality:** con n fija se agregan features; F1_train sube, F1_test se estanca.
3. **Ablación de señales:** ¿qué familia hace el trabajo pesado? Péptido señal, NLS, TMD
   alfa-hélice y TMD beta, una por vez.

```bash
python 03_experimentos.py
```

Descarga el pool la primera vez (lo cachea en `_exp_pool_full.csv`, que no se versiona) y
reutiliza las funciones de features de `01_construir_dataset.py`. Genera
`experimentos_resultados.csv` y las figuras `../img/overfitting.png`,
`../img/curse_dimensionality.png` y `../img/ablacion_senales.png`.

## `04_esm2_embeddings.py`

Representación alternativa con **embeddings de ESM-2** (modelo de lenguaje de proteínas) + un
clasificador lineal, evaluada con el mismo train/test que `03`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python 04_esm2_embeddings.py
```

Usa `facebook/esm2_t6_8M_UR50D` (8M parámetros). Corre sin GPU: ~1-2 min para 750 proteínas.
Cachea los embeddings en `_esm2_<modelo>.npz` y genera `esm2_resultados.csv`. Para estudiantes
hay un notebook listo para Colab en `notebooks/esm2_embeddings_colab.ipynb`.

Resultado de referencia (test balanceado 375, balanced accuracy): composición 0.749, señales
0.816, **ESM-2 8M 0.853**.

## `01_construir_dataset.py`

Descarga proteínas humanas revisadas de UniProt (Swiss-Prot) para las 3 clases del práctico
y calcula directamente las representaciones numéricas usadas en Orange.

```bash
pip install requests pandas biopython
python 01_construir_dataset.py
```

- **750 proteínas** (250 por clase), longitud 50–500 aminoácidos.
- Solo entradas con **evidencia experimental a nivel de proteína** (`existence:1`), para que la
  localización anotada no sea una inferencia por homología. Se controla con
  `SOLO_EVIDENCIA_EXPERIMENTAL`.
- Se descartan secuencias con residuos no estándar (U, O, B, Z, X) para que `ProtParam` y las
  heurísticas de señales funcionen.
- La descarga está paginada (la API devuelve como máximo 500 filas por página).

### Salidas (se generan en este mismo directorio, no están versionadas)

| Archivo | Contenido |
|---|---|
| `features_composicion_fisicoquimicos.csv` | accession, sequence, length, 20 features de composición de aminoácidos, 13 fisicoquímicos, localization |
| `features_dipeptidos_fisicoquimicos.csv` | accession, sequence, length, 400 features de dipéptidos, 13 fisicoquímicos, localization |
| `features_composicion_senales_fisicoquimicos.csv` | accession, sequence, length, 20 de composición, 13 fisicoquímicos, 11 señales posicionales, localization |

Las columnas `accession` y `sequence` no son features — al cargar los CSV en Orange, marcarlas
como **meta** (o *skip*) en el editor de dominio del widget `File`.

### Features fisicoquímicos globales (los 3 CSV)

`longitud`, `peso_molecular`, `punto_isoelectrico`, `hidrofobicidad_promedio` (GRAVY),
`aromaticidad`, `indice_inestabilidad`, `carga_neta_pH7`, `fraccion_helice`, `fraccion_turn`,
`fraccion_hoja`, y tres agregados en esta versión:

- `indice_alifatico` (Ikai, 1980): termoestabilidad relativa; alto = rico en A/V/I/L.
- `fraccion_cargados`: fracción de K+R+H+D+E.
- `coef_extincion_molar`: coeficiente de extinción a 280 nm (Biopython).

### Señales posicionales (solo el tercer CSV)

Calculadas con heurísticas transparentes y deliberadamente simples, para mostrar el mecanismo
biológico. **No** son predictores entrenados.

| Grupo | Columnas | Idea |
|---|---|---|
| Péptido señal | `senal_peptido`, `senal_hidrofobicidad_max`, `senal_n_basicos` | Regiones n (básica), h (hidrofóbica) y c (residuos pequeños) en el N-terminal |
| NLS | `nls_monopartita`, `nls_bipartita`, `nls_densidad_basica_10`, `nls_densidad_basica_7`, `nls_carga_neta_nterm20` | Motivos clásicos `K(K/R)X(K/R)` y bipartitos, más la máxima densidad local de K/R |
| TMD alfa-hélice | `tmd_alfa_helices`, `tmd_alfa_hidrofobicidad_max` | Ventanas de 19 residuos con hidrofobicidad media > 1.6 (Kyte-Doolittle) |
| TMD beta | `tmd_beta_momento_hidrofobico` | Momento hidrofóbico de Eisenberg con periodicidad 160° |

Limitaciones honestas: el detector de péptido señal se solapa con anclas de membrana tipo I/II;
el descriptor beta **no** separa las clases porque en proteínas humanas los beta-barriles de
membrana son raros. Para predicciones serias usar SignalP (péptido señal), DeepTMHMM/TMbed
(TMD) y DeepLoc (localización).

### Ajustar el dataset

- `N_POR_CLASE`, `LARGO_MIN`, `LARGO_MAX`: tamaño y filtro de longitud del dataset.
- `SOLO_EVIDENCIA_EXPERIMENTAL`: exigir evidencia a nivel de proteína (`existence:1`).
- `CLASES`: diccionario clase → keyword de UniProt (las 3 del práctico: Núcleo, Citoplasma,
  Membrana).
