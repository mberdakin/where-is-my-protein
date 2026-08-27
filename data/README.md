# data/

## `01_construir_dataset.py`

Descarga proteínas humanas revisadas de UniProt (Swiss-Prot) para las 3 clases del práctico
y calcula directamente las dos representaciones numéricas usadas en Orange.

```bash
pip install requests pandas biopython
python 01_construir_dataset.py
```


### Salidas (se generan en este mismo directorio, no están versionadas)

| Archivo | Contenido |
|---|---|
| `features_composicion_fisicoquimicos.csv` | accession, sequence, length, 20 features de composición de aminoácidos, 10 features fisicoquímicos, localization |
| `features_dipeptidos_fisicoquimicos.csv` | accession, sequence, length, 400 features de dipéptidos, 10 features fisicoquímicos, localization |

Las columnas `accession` y `sequence` no son features — al cargar los CSV en Orange, marcarlas
como **meta** (o *skip*) en el editor de dominio del widget `File`.

### Ajustar el dataset

- `N_POR_CLASE`, `LARGO_MIN`, `LARGO_MAX`: tamaño y filtro de longitud del dataset.
- `CLASES`: diccionario clase → keyword de UniProt. Se puede agregar una cuarta clase
  (por ejemplo `"Mitochondria": "KW-0496"`) sin modificar el resto del script.
