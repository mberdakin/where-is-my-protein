"""
Construye TODO el material de datos del práctico: descarga las proteínas desde
UniProt y calcula directamente las representaciones numéricas que se usan en
Orange. El cálculo de features queda resuelto acá (por script), no depende de
que la IA responda bien en el momento de la clase.

El dataset usa únicamente proteínas humanas revisadas (Swiss-Prot) con
**evidencia experimental a nivel de proteína** (`existence:1`), para que la
localización anotada no sea una inferencia por homología. Cada clase tiene
250 proteínas (750 en total), con longitud entre 50 y 500 aminoácidos.

Requiere conexión a internet (no funciona dentro de un sandbox sin red).
Uso:
    pip install requests pandas biopython
    python 01_construir_dataset.py

Salidas (todas en el mismo directorio):

    features_composicion_fisicoquimicos.csv
        accession, sequence, length, bag of aminoácidos (20),
        fisicoquímicos globales (13), localization

    features_dipeptidos_fisicoquimicos.csv
        accession, sequence, length, dipéptidos (400),
        fisicoquímicos globales (13), localization

    features_composicion_senales_fisicoquimicos.csv
        accession, sequence, length, bag of aminoácidos (20),
        fisicoquímicos globales (13), señales derivadas de la secuencia (11),
        localization
        — la representación "enriquecida": agrega detectores heurísticos de
        péptido señal, señales de localización nuclear (NLS) y dominios
        transmembrana (alfa-hélice y beta-barril).

Las features fisicoquímicas y de señales son heurísticas transparentes,
pensadas para enseñar el mecanismo; NO reemplazan a predictores entrenados
como SignalP, DeepTMHMM, TMbed o DeepLoc. Ver el README para las limitaciones.

En Orange, al cargar cualquiera de los CSV con el widget File, marcar las
columnas 'accession' y 'sequence' como meta (o "skip") en el editor de dominio,
para que no se usen como features del modelo pero sigan disponibles para
identificar filas puntuales (por ejemplo, al inspeccionar errores en la
Confusion Matrix).
"""

import math
import os
import re
import time

import pandas as pd
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_PAGE_SIZE = 500   # máximo que acepta la API REST

# Los CSV se escriben siempre junto a este script, sin importar el cwd.
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASES = {
    "Nucleus":    "KW-0539",
    "Cytoplasm":  "KW-0963",
    "Membrane":   "KW-0472",
}

N_POR_CLASE = 250          # 250 x 3 clases = 750 proteínas
LARGO_MIN = 50
LARGO_MAX = 500            # acota la longitud para que el práctico sea manejable

# Si está en True, solo se usan entradas con evidencia experimental a nivel de
# proteína (UniProt existence:1). Ponerlo en False amplía el pool, pero mete
# localizaciones inferidas por homología.
SOLO_EVIDENCIA_EXPERIMENTAL = True

AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")
AMINOACIDOS_SET = set(AMINOACIDOS)

# Escala de hidrofobicidad de Kyte-Doolittle: hidrofílicos negativos,
# hidrofóbicos positivos. Es la base de las heurísticas de TMD y péptido señal.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# Escala de consenso de Eisenberg para el momento hidrofóbico (Eisenberg et al.,
# 1984). Se usa para estimar propensión a hebra beta transmembrana.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# NLS monopartita clásica: K-(K/R)-X-(K/R) (ej. el NLS de SV40, PKKKRKV).
NLS_MONOPARTITA = re.compile(r"K[KR][A-Z][KR]")
# NLS bipartita: dos clusters básicos separados por ~10 aminoácidos
# (ej. nucleoplasmina, KR[PAATKKAGQA]KKKK).
NLS_BIPARTITA = re.compile(r"[KR]{2}[^KR]{9,12}[KR]{3,5}")


# ---------------------------------------------------------------------------
# 1) Descarga desde UniProt (paginada)
# ---------------------------------------------------------------------------

def _siguiente_pagina(link_header):
    """Extrae la URL con rel="next" del header Link de UniProt, si existe.

    No se puede partir el header por comas: la URL de UniProt contiene comas
    (por ejemplo en `fields=accession,sequence,length`)."""
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link_header)
    return match.group(1) if match else None


def descargar_clase(nombre, keyword, n):
    """Descarga proteínas humanas revisadas con esa keyword, excluyendo las
    que además tengan alguna de las otras dos keywords de localización.
    Pagina la API hasta juntar n proteínas o agotar los resultados."""
    otras = [kw for c, kw in CLASES.items() if c != nombre]
    query = (
        f"(keyword:{keyword}) AND (reviewed:true) AND (organism_id:9606) "
        f"AND (length:[{LARGO_MIN} TO {LARGO_MAX}]) "
        f"NOT (keyword:{otras[0]}) NOT (keyword:{otras[1]})"
    )
    if SOLO_EVIDENCIA_EXPERIMENTAL:
        query += " AND (existence:1)"

    filas = []
    vistos = set()
    params = {
        "query": query,
        "fields": "accession,sequence,length",
        "format": "tsv",
        "size": UNIPROT_PAGE_SIZE,
    }
    url = BASE_URL

    while url and len(filas) < n:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()

        lineas = r.text.strip().split("\n")
        if lineas and lineas[0].startswith("Entry"):
            lineas = lineas[1:]  # saltear encabezado
        for linea in lineas:
            partes = linea.split("\t")
            if len(partes) != 3:
                continue
            acc, seq, length = partes
            if acc in vistos or not seq:
                continue
            # Descartar secuencias con residuos no estándar (U, O, B, Z, X):
            # ProtParam y las heurísticas asumen solo los 20 aminoácidos.
            if not set(seq).issubset(AMINOACIDOS_SET):
                continue
            vistos.add(acc)
            filas.append({
                "accession": acc,
                "sequence": seq,
                "length": int(length),
                "localization": nombre,
            })
            if len(filas) >= n:
                break

        url = _siguiente_pagina(r.headers.get("Link"))
        params = None  # la URL "next" ya trae los parámetros
        if url:
            time.sleep(0.3)  # buena práctica: no saturar la API

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
# 4) Features fisicoquímicos globales (calculados con Biopython / ProtParam)
# ---------------------------------------------------------------------------

def _media_kd(ventana):
    """Hidrofobicidad promedio (Kyte-Doolittle) de una ventana de residuos."""
    return sum(KYTE_DOOLITTLE[aa] for aa in ventana) / len(ventana)


def calcular_features_fisicoquimicos(secuencia):
    """Devuelve features globales de la secuencia calculados con Biopython."""
    analisis = ProteinAnalysis(secuencia)
    helice, turn, hoja = analisis.secondary_structure_fraction()
    extincion_reducida, _ = analisis.molar_extinction_coefficient()

    conteo = {aa: secuencia.count(aa) for aa in AMINOACIDOS}
    n = len(secuencia)
    porcentaje = {aa: 100 * conteo[aa] / n for aa in AMINOACIDOS}

    # Índice alifático (Ikai, 1980): estabilidad térmica relativa de proteínas
    # globulares; alto = rico en A, V, I y L.
    indice_alifatico = (
        porcentaje["A"]
        + 2.9 * porcentaje["V"]
        + 3.9 * (porcentaje["I"] + porcentaje["L"])
    )

    return {
        "longitud": n,
        "peso_molecular": analisis.molecular_weight(),
        "punto_isoelectrico": analisis.isoelectric_point(),
        "hidrofobicidad_promedio": analisis.gravy(),
        "aromaticidad": analisis.aromaticity(),
        "indice_inestabilidad": analisis.instability_index(),
        "carga_neta_pH7": analisis.charge_at_pH(7.0),
        "fraccion_helice": helice,
        "fraccion_turn": turn,
        "fraccion_hoja": hoja,
        "indice_alifatico": indice_alifatico,
        "fraccion_cargados": sum(conteo[aa] for aa in "KRHDE") / n,
        "coef_extincion_molar": float(extincion_reducida),
    }


# ---------------------------------------------------------------------------
# 5) Señales derivadas de la secuencia: péptido señal, NLS y TMD
#
#    Estas features son heurísticas deliberadamente simples y transparentes,
#    para que los estudiantes vean el mecanismo biológico detrás. No son
#    predictores entrenados: sirven para discutir qué información aporta la
#    posición de un motivo, no para reemplazar a SignalP / DeepTMHMM.
# ---------------------------------------------------------------------------

def _features_senal_peptido(secuencia, largo_max_nterm=40):
    """Heurística de péptido señal N-terminal (regiones n, h y c).

    - Región n: primeros residuos, suele tener carga positiva (K/R).
    - Región h: tramo central hidrofóbico (acá: mejor ventana de 9 residuos
      en las primeras ~35 posiciones).
    - Región c: tras la región h, residuos pequeños (A/S/G/T/C) alrededor
      del sitio de corte.

    Devuelve 1 en `senal_peptido` si hay región h hidrofóbica Y región n
    básica; el resto son los puntajes usados para decidir."""
    limite = min(len(secuencia), largo_max_nterm)
    n_region = secuencia[:5]
    n_basicos = sum(n_region.count(aa) for aa in "KR")

    h_max, h_fin = -99.0, None
    for i in range(2, max(3, limite - 9)):
        ventana = secuencia[i:i + 9]
        if len(ventana) < 9:
            break
        media = _media_kd(ventana)
        if media > h_max:
            h_max, h_fin = media, i + 9

    c_region = secuencia[h_fin:h_fin + 6] if h_fin else ""
    c_pequenos = sum(c_region.count(aa) for aa in "ASGTC")

    return {
        "senal_peptido": int(h_max >= 1.8 and n_basicos >= 1),
        "senal_hidrofobicidad_max": round(h_max, 3),
        "senal_n_basicos": n_basicos,
    }


def _densidad_basica_max(secuencia, ventana):
    """Máxima fracción de K/R en cualquier ventana de la secuencia."""
    if len(secuencia) < ventana:
        ventana = len(secuencia)
    if ventana == 0:
        return 0.0
    maximo = 0.0
    for i in range(len(secuencia) - ventana + 1):
        tramo = secuencia[i:i + ventana]
        maximo = max(maximo, sum(tramo.count(aa) for aa in "KR") / ventana)
    return round(maximo, 3)


def _tmd_alfa_helices(secuencia):
    """Cuenta dominios transmembrana alfa-hélice aproximados.

    Una hélice transmembrana típica tiene ~19-21 residuos hidrofóbicos. Se
    marca cada ventana de 19 con hidrofobicidad media > 1.6 (Kyte-Doolittle)
    y se cuentan los tramos contiguos resultantes."""
    ventana = 19
    if len(secuencia) < ventana:
        return 0, 0.0

    marcas = [False] * len(secuencia)
    max_media = -99.0
    for i in range(len(secuencia) - ventana + 1):
        media = _media_kd(secuencia[i:i + ventana])
        max_media = max(max_media, media)
        if media > 1.6:
            for j in range(i, i + ventana):
                marcas[j] = True

    tramos, previo = 0, False
    for marca in marcas:
        if marca and not previo:
            tramos += 1
        previo = marca
    return tramos, round(max_media, 3)


def _tmd_beta_momento_hidrofobico(secuencia, ventana=11):
    """Momento hidrofóbico de Eisenberg con periodicidad beta (160°).

    En una hebra beta transmembrana el patrón hidrofóbico se repite cada 2
    residuos (ángulo ~160°), de modo que un lado de la hebra mira al lípido y
    el otro al poro. Se devuelve el máximo sobre ventanas de 11 residuos.

    Advertencia: en proteínas humanas los beta-barriles de membrana son raros
    y este descriptor simple NO distingue bien las 3 clases del práctico. Para
    una predicción real de TMD beta usar TMbed, DeepTMHMM o BOMP."""
    escala = EISENBERG
    delta = math.radians(160)
    mejor = 0.0
    for i in range(len(secuencia) - ventana + 1):
        hidros = [escala[aa] for aa in secuencia[i:i + ventana]]
        seno = sum(h * math.sin((n + 1) * delta) for n, h in enumerate(hidros))
        coseno = sum(h * math.cos((n + 1) * delta) for n, h in enumerate(hidros))
        mejor = max(mejor, math.sqrt(seno ** 2 + coseno ** 2) / ventana)
    return round(mejor, 3)


def calcular_features_senales(secuencia):
    """Detectores heurísticos de péptido señal, NLS y TMD. Ver docstring del
    módulo para la advertencia sobre su alcance."""
    features = {}
    features.update(_features_senal_peptido(secuencia))

    features["nls_monopartita"] = len(NLS_MONOPARTITA.findall(secuencia))
    features["nls_bipartita"] = len(NLS_BIPARTITA.findall(secuencia))
    features["nls_densidad_basica_10"] = _densidad_basica_max(secuencia, 10)
    features["nls_densidad_basica_7"] = _densidad_basica_max(secuencia, 7)
    n_terminal = secuencia[:20]
    features["nls_carga_neta_nterm20"] = (
        sum(n_terminal.count(aa) for aa in "KR")
        - sum(n_terminal.count(aa) for aa in "DE")
    )

    tmd_alfa, hidrofobicidad_max = _tmd_alfa_helices(secuencia)
    features["tmd_alfa_helices"] = tmd_alfa
    features["tmd_alfa_hidrofobicidad_max"] = hidrofobicidad_max
    # Descriptor exploratorio de propensión a hebra beta transmembrana. No se
    # incluye una versión binaria ("barril sí/no") porque una heurística de
    # secuencia no logra separar beta-barriles del resto en este dataset.
    features["tmd_beta_momento_hidrofobico"] = _tmd_beta_momento_hidrofobico(secuencia)
    return features


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

    print("Calculando señales (péptido señal, NLS, TMD)...")
    senales = df["sequence"].apply(calcular_features_senales).apply(pd.Series)

    print("Calculando dipéptidos (puede tardar unos segundos)...")
    dipeptidos = df["sequence"].apply(composicion_dipeptidos).apply(pd.Series)

    base = df[["accession", "sequence", "length", "localization"]]

    # bag of aminoácidos + fisicoquímicos
    export_composicion = pd.concat(
        [base[["accession", "sequence", "length"]], bag_of_aa, fisicoquimicos,
         base[["localization"]]], axis=1
    )
    export_composicion.to_csv(
        os.path.join(OUT_DIR, "features_composicion_fisicoquimicos.csv"), index=False)

    # dipéptidos + fisicoquímicos
    export_dipeptidos = pd.concat(
        [base[["accession", "sequence", "length"]], dipeptidos, fisicoquimicos,
         base[["localization"]]], axis=1
    )
    export_dipeptidos.to_csv(
        os.path.join(OUT_DIR, "features_dipeptidos_fisicoquimicos.csv"), index=False)

    # bag of aminoácidos + fisicoquímicos + señales posicionales (enriquecida)
    export_senales = pd.concat(
        [base[["accession", "sequence", "length"]], bag_of_aa, fisicoquimicos,
         senales, base[["localization"]]], axis=1
    )
    export_senales.to_csv(
        os.path.join(OUT_DIR, "features_composicion_senales_fisicoquimicos.csv"), index=False)

    print("\nArchivos generados:")
    print(" - features_composicion_fisicoquimicos.csv          ", export_composicion.shape)
    print(" - features_dipeptidos_fisicoquimicos.csv           ", export_dipeptidos.shape)
    print(" - features_composicion_senales_fisicoquimicos.csv  ", export_senales.shape)


if __name__ == "__main__":
    main()
