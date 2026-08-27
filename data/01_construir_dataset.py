"""
Construye TODO el material de datos del práctico: descarga las proteínas desde
UniProt y calcula directamente las dos representaciones numéricas que se usan
en Orange (composición + fisicoquímicos, y dipéptidos + fisicoquímicos). El
cálculo de features fisicoquímicos queda resuelto acá (por script), no
depende de que la IA responda bien en el momento de la clase.

Requiere conexión a internet (no funciona dentro de un sandbox sin red).
Uso:
    pip install requests pandas biopython
    python 01_construir_dataset.py

Salidas (todas en el mismo directorio):
    features_composicion_fisicoquimicos.csv - accession, sequence, length,
                                                bag of aminoácidos (20), features
                                                fisicoquímicos (10), localization
    features_dipeptidos_fisicoquimicos.csv   - accession, sequence, length,
                                                dipéptidos (400), features
                                                fisicoquímicos (10), localization

En Orange, al cargar cualquiera de los dos CSV con el widget File, marcar las columnas
'accession' y 'sequence' como meta (o "skip") en el editor de dominio, para que no se
usen como features del modelo pero sigan disponibles para identificar filas puntuales
(por ejemplo, al inspeccionar errores en la Confusion Matrix).
"""

import requests
import pandas as pd
import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

CLASES = {
    "Nucleus":    "KW-0539",
    "Cytoplasm":  "KW-0963",
    "Membrane":   "KW-0472",
}

N_POR_CLASE = 84          # 84 + 83 + 83 = 250
LARGO_MIN = 50
LARGO_MAX = 500            # acota la longitud para que el práctico sea manejable

AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# 1) Descarga desde UniProt
# ---------------------------------------------------------------------------

def descargar_clase(nombre, keyword, n):
    """Descarga proteínas humanas revisadas con esa keyword, excluyendo
    las que además tengan alguna de las otras dos keywords de localización."""
    otras = [kw for c, kw in CLASES.items() if c != nombre]
    query = (
        f"(keyword:{keyword}) AND (reviewed:true) AND (organism_id:9606) "
        f"AND (length:[{LARGO_MIN} TO {LARGO_MAX}]) "
        f"NOT (keyword:{otras[0]}) NOT (keyword:{otras[1]})"
    )
    params = {
        "query": query,
        "fields": "accession,sequence,length",
        "format": "tsv",
        "size": n * 3,   # margen extra, se usa para descartar duplicados/vacíos
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()

    lineas = r.text.strip().split("\n")[1:]  # saltear encabezado
    filas = []
    vistos = set()
    for linea in lineas:
        acc, seq, length = linea.split("\t")
        if acc in vistos or not seq:
            continue
        vistos.add(acc)
        filas.append({"accession": acc, "sequence": seq, "length": int(length),
                       "localization": nombre})
        if len(filas) >= n:
            break
    return filas


def descargar_dataset():
    todas = []
    for nombre, keyword in CLASES.items():
        print(f"Descargando {nombre} ...")
        filas = descargar_clase(nombre, keyword, N_POR_CLASE)
        print(f"  -> {len(filas)} proteínas obtenidas")
        todas.extend(filas)
        time.sleep(1)  # buena práctica: no saturar la API

    df = pd.DataFrame(todas).drop_duplicates(subset="accession")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # mezclar
    return df


# ---------------------------------------------------------------------------
# 2) Representación: bag of aminoácidos (composición, 20 features)
# ---------------------------------------------------------------------------

def composicion_aminoacidica(secuencia):
    """Proporción de cada uno de los 20 aminoácidos en la secuencia."""
    largo = len(secuencia)
    return {aa: secuencia.count(aa) / largo for aa in AMINOACIDOS}


# ---------------------------------------------------------------------------
# 3) Representación: dipéptidos (información local, 400 features)
# ---------------------------------------------------------------------------

def composicion_dipeptidos(secuencia):
    """Frecuencia de cada uno de los 400 dipéptidos posibles en la secuencia."""
    dipeptidos = [a + b for a in AMINOACIDOS for b in AMINOACIDOS]
    conteos = {d: 0 for d in dipeptidos}
    total = len(secuencia) - 1
    for i in range(total):
        par = secuencia[i:i + 2]
        if par in conteos:
            conteos[par] += 1
    return {d: c / total for d, c in conteos.items()}


# ---------------------------------------------------------------------------
# 4) Features fisicoquímicos globales (antes se pedían a la IA en clase;
#    ahora se calculan acá para no depender de eso durante el práctico)
# ---------------------------------------------------------------------------

def calcular_features_fisicoquimicos(secuencia):
    """Devuelve un diccionario con features globales de la secuencia,
    calculados con Biopython (ProtParam)."""
    analisis = ProteinAnalysis(secuencia)
    helice, turn, hoja = analisis.secondary_structure_fraction()
    return {
        "longitud": len(secuencia),
        "peso_molecular": analisis.molecular_weight(),
        "punto_isoelectrico": analisis.isoelectric_point(),
        "hidrofobicidad_promedio": analisis.gravy(),
        "aromaticidad": analisis.aromaticity(),
        "indice_inestabilidad": analisis.instability_index(),
        "carga_neta_pH7": analisis.charge_at_pH(7.0),
        "fraccion_helice": helice,
        "fraccion_turn": turn,
        "fraccion_hoja": hoja,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = descargar_dataset()
    print(f"\nProteínas descargadas: {len(df)}")
    print(df["localization"].value_counts())

    print("\nCalculando bag of aminoácidos...")
    bag_of_aa = df["sequence"].apply(composicion_aminoacidica).apply(pd.Series)

    print("Calculando features fisicoquímicos...")
    fisicoquimicos = df["sequence"].apply(calcular_features_fisicoquimicos).apply(pd.Series)

    print("Calculando dipéptidos (puede tardar unos segundos)...")
    dipeptidos = df["sequence"].apply(composicion_dipeptidos).apply(pd.Series)

    base = df[["accession", "sequence", "length", "localization"]]

    # bag of aminoácidos + fisicoquímicos (accession/sequence incluidos, para
    # marcar como meta en Orange; no como features del modelo)
    export_composicion = pd.concat(
        [base[["accession", "sequence", "length"]], bag_of_aa, fisicoquimicos,
         base[["localization"]]], axis=1
    )
    export_composicion.to_csv("features_composicion_fisicoquimicos.csv", index=False)

    # dipéptidos + fisicoquímicos
    export_dipeptidos = pd.concat(
        [base[["accession", "sequence", "length"]], dipeptidos, fisicoquimicos,
         base[["localization"]]], axis=1
    )
    export_dipeptidos.to_csv("features_dipeptidos_fisicoquimicos.csv", index=False)

    print("\nArchivos generados:")
    print(" - features_composicion_fisicoquimicos.csv ", export_composicion.shape)
    print(" - features_dipeptidos_fisicoquimicos.csv  ", export_dipeptidos.shape)


if __name__ == "__main__":
    main()
