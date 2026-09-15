"""
Experimentos (sólo 3 clases: Nucleus, Cytoplasm, Membrane) con un conjunto de
TEST separado que el entrenamiento nunca ve.

Diseño:
  - Se descarga todo lo disponible con el filtro del práctico (revisadas,
    humanas, evidencia a nivel de proteína, largo 50-500, keywords mutuamente
    excluyentes entre las 3 clases).
  - Se reservan hasta 1000 proteínas por clase para entrenar (3000 en total) y
    todo el resto forma el test. Como el sobrante está dominado por Membrana,
    se evalúa sobre su subconjunto balanceado más grande posible
    (375 = 125 por clase).

Los TRES experimentos que se usan en clase:
  1. Overfitting: F1 en train (resustitución) vs F1 en el test, en 250/750/
     1500/3000 proteínas de entrenamiento. La brecha es el overfitting.
  2. Curse of dimensionality: con n fija, agregar features hace subir el F1 de
     train mientras el de test se estanca.
  3. Ablación de señales: qué familia hace el trabajo pesado (péptido señal,
     NLS, TMD alfa-hélice, TMD beta).

El Random Forest usa min_samples_leaf=10, max_depth=8, min_samples_split=10 y
max_leaf_nodes=16 para reducir el overfitting (que igual aparece con pocas
muestras o representaciones complejas como los dipéptidos).

El pool se cachea en este directorio (data/*.csv está gitignoreado).

Uso:
    python 03_experimentos.py
"""

import importlib.util
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(OUT_DIR), "img")
SEED = 0
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
N_TRAIN_MAX = 1000        # por clase -> 3000 en total
N_TEST_MAX = 100000       # sin tope: el test es todo lo que sobra
# Tamaños de entrenamiento (por clase): 250, 750, 1500 y 3000 en total.
TAMANOS_POR_CLASE = [83, 250, 500, 1000]

_spec = importlib.util.spec_from_file_location(
    "construir_dataset", os.path.join(OUT_DIR, "01_construir_dataset.py"))
ds01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds01)

AMINOACIDOS = ds01.AMINOACIDOS
FISICOQUIMICOS = [
    "longitud", "peso_molecular", "punto_isoelectrico", "hidrofobicidad_promedio",
    "aromaticidad", "indice_inestabilidad", "carga_neta_pH7", "fraccion_helice",
    "fraccion_turn", "fraccion_hoja", "indice_alifatico", "fraccion_cargados",
    "coef_extincion_molar",
]
# Familias de señales (separadas para la ablación)
SEN_PEPTIDO = ["senal_peptido", "senal_hidrofobicidad_max", "senal_n_basicos"]
SEN_NLS = ["nls_monopartita", "nls_bipartita", "nls_densidad_basica_10",
           "nls_densidad_basica_7", "nls_carga_neta_nterm20"]
SEN_TMD_ALFA = ["tmd_alfa_helices", "tmd_alfa_hidrofobicidad_max"]
SEN_TMD_BETA = ["tmd_beta_momento_hidrofobico"]
SENALES = SEN_PEPTIDO + SEN_NLS + SEN_TMD_ALFA + SEN_TMD_BETA

CLASES3 = {
    "Nucleus": "KW-0539",
    "Cytoplasm": "KW-0963",
    "Membrane": "KW-0472",
}

# Random Forest con límites de crecimiento (profundidad, hojas y mínimo de
# muestras por hoja) para reducir el overfitting.
RF_PARAMS = dict(n_estimators=300, random_state=SEED, n_jobs=1,
                 min_samples_leaf=10, max_depth=8, min_samples_split=10,
                 max_leaf_nodes=16)

MODELOS = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "SVM": make_pipeline(StandardScaler(), SVC(cache_size=1000)),
    "RandomForest": RandomForestClassifier(**RF_PARAMS),
}


def dip_cols(df):
    return [c for c in df.columns if len(c) == 2 and set(c).issubset(set(AMINOACIDOS))]


# ---------------------------------------------------------------------------
# Descarga y pool (con cache)
# ---------------------------------------------------------------------------

def _buscar(query, n):
    filas, vistos, url = [], set(), BASE_URL
    params = {"query": query, "fields": "accession,sequence,length",
              "format": "tsv", "size": 500}
    while url and len(filas) < n:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        lineas = r.text.strip().split("\n")
        if lineas and lineas[0].startswith("Entry"):
            lineas = lineas[1:]
        for linea in lineas:
            partes = linea.split("\t")
            if len(partes) != 3:
                continue
            acc, seq, length = partes
            if acc in vistos or not seq or not set(seq).issubset(ds01.AMINOACIDOS_SET):
                continue
            vistos.add(acc)
            filas.append({"accession": acc, "sequence": seq, "length": int(length)})
            if len(filas) >= n:
                break
        url = ds01._siguiente_pagina(r.headers.get("Link"))
        params = None
        if url:
            time.sleep(0.3)
    return filas


def construir_pool(clases, n_por_clase, largo_max, cache):
    if os.path.exists(cache):
        print(f"  (cache) {os.path.basename(cache)}")
        return pd.read_csv(cache)

    filas = []
    for nombre, kw in clases.items():
        otras = [k for c, k in clases.items() if c != nombre]
        q = (f"(keyword:{kw}) AND (reviewed:true) AND (organism_id:9606) "
             f"AND (existence:1) AND (length:[50 TO {largo_max}]) "
             + " ".join(f"NOT (keyword:{k})" for k in otras))
        obtenidas = _buscar(q, n_por_clase)
        print(f"  {nombre}: {len(obtenidas)}")
        for f in obtenidas:
            f["localization"] = nombre
        filas.extend(obtenidas)
        time.sleep(0.5)

    df = pd.DataFrame(filas).drop_duplicates("accession")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    comp = df["sequence"].apply(ds01.composicion_aminoacidica).apply(pd.Series)
    fisio = df["sequence"].apply(ds01.calcular_features_fisicoquimicos).apply(pd.Series)
    sen = df["sequence"].apply(ds01.calcular_features_senales).apply(pd.Series)
    dip = df["sequence"].apply(ds01.composicion_dipeptidos).apply(pd.Series)

    out = pd.concat(
        [df[["accession", "sequence", "length", "localization"]], comp, fisio, dip, sen],
        axis=1,
    )
    out.to_csv(cache, index=False)
    print(f"  pool guardado: {os.path.basename(cache)} {out.shape}")
    return out


def dividir_train_test(pool, n_train=N_TRAIN_MAX):
    """Reserva hasta n_train por clase para entrenar; el resto es test."""
    tr, te = [], []
    for _, g in pool.groupby("localization"):
        g = g.sample(frac=1, random_state=SEED)
        tr.append(g.head(n_train))
        te.append(g.iloc[n_train:])
    return pd.concat(tr).reset_index(drop=True), pd.concat(te).reset_index(drop=True)


def subconjunto(df, n_por_clase):
    # conserva el orden original de las filas
    return df.groupby("localization", group_keys=False).head(n_por_clase)


# ---------------------------------------------------------------------------
# Evaluación train -> test (los modelos nunca ven el test al entrenar)
# ---------------------------------------------------------------------------

def _metricas(y, pred):
    return {
        "CA": accuracy_score(y, pred),
        "BA": balanced_accuracy_score(y, pred),
        "F1_macro": f1_score(y, pred, average="macro"),
        "reporte": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def evaluar_3clases(train, test, features, modelos=None):
    modelos = modelos or MODELOS
    Xtr, ytr = train[features].to_numpy(float), train["localization"].to_numpy()
    Xte, yte = test[features].to_numpy(float), test["localization"].to_numpy()
    out = {}
    for nombre, modelo in modelos.items():
        modelo.fit(Xtr, ytr)
        out[nombre] = _metricas(yte, modelo.predict(Xte))
    return out


def evaluar_binaria(train, test, features, clases, positiva, modelos=None):
    """clases = lista de las 2 etiquetas; positiva = cuál es la clase 1."""
    modelos = modelos or MODELOS
    tr = train[train["localization"].isin(clases)]
    te = test[test["localization"].isin(clases)]
    ytr = (tr["localization"] == positiva).astype(int).to_numpy()
    yte = (te["localization"] == positiva).astype(int).to_numpy()
    Xtr, Xte = tr[features].to_numpy(float), te[features].to_numpy(float)
    out = {}
    for nombre, modelo in modelos.items():
        modelo.fit(Xtr, ytr)
        out[nombre] = _metricas(yte, modelo.predict(Xte))
    return out


def mejor_ba(res):
    return max(m["BA"] for m in res.values())


# ---------------------------------------------------------------------------
# Experimento 1: overfitting (F1 en train vs F1 en test)
# ---------------------------------------------------------------------------

def exp_overfitting(train, test):
    """Entrena y evalúa en el MISMO train (resustitución) y en el test. La
    brecha F1_train - F1_test es la medida directa de overfitting."""
    print("\n### Experimento 1: overfitting (F1 train vs F1 test)")
    reps = {
        "composicion": AMINOACIDOS + FISICOQUIMICOS,
        "dipeptidos": dip_cols(train) + FISICOQUIMICOS,
        "senales": AMINOACIDOS + FISICOQUIMICOS + SENALES,
    }
    modelos = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
    }
    filas = []
    for n in TAMANOS_POR_CLASE:
        d = subconjunto(train, n)
        ytr, yte = d["localization"].to_numpy(), test["localization"].to_numpy()
        for nombre, feats in reps.items():
            Xtr, Xte = d[feats].to_numpy(float), test[feats].to_numpy(float)
            for mname, clf in modelos.items():
                clf.fit(Xtr, ytr)
                f1_tr = f1_score(ytr, clf.predict(Xtr), average="macro")
                f1_te = f1_score(yte, clf.predict(Xte), average="macro")
                filas.append({"experimento": "overfitting", "n_train": len(d),
                              "n_por_clase": n, "representacion": nombre, "modelo": mname,
                              "F1_train": round(f1_tr, 3), "F1_test": round(f1_te, 3),
                              "F1_gap": round(f1_tr - f1_te, 3)})
                print(f"  n={len(d):4d} {nombre:12s} {mname:12s} "
                      f"F1_train={f1_tr:.3f} F1_test={f1_te:.3f} gap={f1_tr - f1_te:.3f}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Experimento 2: curse of dimensionality (más features, misma n)
# ---------------------------------------------------------------------------

def exp_curse(train, test):
    """Con n fija chica, agrega dipéptidos ordenados por información y muestra
    que F1_train sube hacia 1 mientras F1_test se estanca o baja."""
    print("\n### Experimento 2: curse of dimensionality (F1 train vs test según nº features)")
    ks = [5, 10, 20, 50, 100, 200, 400]
    filas = []
    for n in [83, 250]:  # 250 y 750 proteínas de entrenamiento
        d = subconjunto(train, n)
        Dtr, Dte = d[dip_cols(train)].to_numpy(float), test[dip_cols(train)].to_numpy(float)
        Ftr, Fte = d[FISICOQUIMICOS].to_numpy(float), test[FISICOQUIMICOS].to_numpy(float)
        ytr, yte = d["localization"].to_numpy(), test["localization"].to_numpy()
        for k in ks:
            sel = SelectKBest(f_classif, k=k).fit(Dtr, ytr)
            Xtr = np.hstack([sel.transform(Dtr), Ftr])
            Xte = np.hstack([sel.transform(Dte), Fte])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            clf.fit(Xtr, ytr)
            f1_tr = f1_score(ytr, clf.predict(Xtr), average="macro")
            f1_te = f1_score(yte, clf.predict(Xte), average="macro")
            filas.append({"experimento": "curse", "n_train": len(d), "n_por_clase": n,
                          "n_dipeptidos": k, "n_features": Xtr.shape[1],
                          "F1_train": round(f1_tr, 3), "F1_test": round(f1_te, 3),
                          "F1_gap": round(f1_tr - f1_te, 3)})
            print(f"  n={len(d):4d} dip={k:3d} (features={Xtr.shape[1]:3d}) "
                  f"F1_train={f1_tr:.3f} F1_test={f1_te:.3f} gap={f1_tr - f1_te:.3f}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Experimento 3: ablación de señales
# ---------------------------------------------------------------------------

def exp_ablacion(train, test):
    print("\n### Experimento 3: ablación de señales (train=3000, test reservado)")
    familias = {
        "baseline (C+F)": [],
        "peptido senal": SEN_PEPTIDO,
        "NLS": SEN_NLS,
        "TMD alfa-helice": SEN_TMD_ALFA,
        "TMD beta": SEN_TMD_BETA,
        "todas las senales": SENALES,
    }
    filas = []
    for nombre, fam in familias.items():
        feats = AMINOACIDOS + FISICOQUIMICOS + fam
        r3 = evaluar_3clases(train, test, feats)
        rm = evaluar_binaria(train, test, feats, ["Membrane", "Nucleus", "Cytoplasm"],
                             "Membrane")
        rn = evaluar_binaria(train, test, feats, ["Nucleus", "Cytoplasm"], "Nucleus")
        def mejor(res):
            return max(res.items(), key=lambda kv: kv[1]["BA"])
        _, m3 = mejor(r3)
        _, mm = mejor(rm)
        _, mn = mejor(rn)
        filas.append({"experimento": "ablacion", "familia": nombre, "n_features": len(fam),
                      "BA_3clases": round(m3["BA"], 3),
                      "BA_Membrana_vs_resto": round(mm["BA"], 3),
                      "BA_Nucleo_vs_Citoplasma": round(mn["BA"], 3),
                      "F1_3clases": round(m3["F1_macro"], 3),
                      "CA_3clases": round(m3["CA"], 3)})
        print(f"  {nombre:20s} (+{len(fam):2d} feat)  3clases={m3['BA']:.3f}  "
              f"Membrana_vs_resto={mm['BA']:.3f}  Nucleo_vs_Citoplasma={mn['BA']:.3f}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_overfitting(df):
    reps = [("composicion", "#1b9e77"), ("dipeptidos", "#d95f02"),
            ("senales", "#7570b3")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, modelo in zip(axes, ["LogReg", "RandomForest"]):
        sub = df[df["modelo"] == modelo]
        for rep, color in reps:
            g = (sub[sub["representacion"] == rep]
                 .groupby("n_train")[["F1_train", "F1_test"]].max())
            ax.plot(g.index, g["F1_train"], "o-", label=f"{rep} (train)", color=color)
            ax.plot(g.index, g["F1_test"], "s--", label=f"{rep} (test)", color=color, alpha=0.7)
        ax.set_xlabel("Nº de proteínas de entrenamiento")
        ax.set_title(modelo)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Overfitting: F1 en train (resustitución) vs F1 en el test reservado")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "overfitting.png"), dpi=110)


def figura_curse(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, n in zip(axes, sorted(df["n_train"].unique())):
        sub = df[df["n_train"] == n].sort_values("n_features")
        ax.plot(sub["n_features"], sub["F1_train"], "o-", color="#e41a1c", label="F1 train")
        ax.plot(sub["n_features"], sub["F1_test"], "s--", color="#377eb8", label="F1 test")
        ax.set_xlabel("Nº de features (dipéptidos + fisicoquímicos)")
        ax.set_title(f"Entrenamiento: {n} proteínas")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend()
    fig.suptitle("Curse of dimensionality: más features con pocas muestras")
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "curse_dimensionality.png"), dpi=110)


def figura_ablacion(abl):
    familias = list(abl["familia"])
    x = np.arange(len(familias))
    ancho = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, etiqueta, color) in enumerate([
            ("BA_3clases", "3 clases", "#4daf4a"),
            ("BA_Membrana_vs_resto", "Membrana vs resto", "#e41a1c"),
            ("BA_Nucleo_vs_Citoplasma", "Núcleo vs Citoplasma", "#377eb8")]):
        vals = list(abl[col])
        barras = ax.bar(x + (i - 1) * ancho, vals, ancho, label=etiqueta, color=color)
        ax.bar_label(barras, fmt="%.2f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(" ", "\n") for f in familias], fontsize=8)
    ax.set_ylabel("Balanced accuracy (test)")
    ax.set_ylim(0.50, 1.0)
    ax.set_title("Ablación: ¿qué familia de señales hace el trabajo pesado?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "ablacion_senales.png"), dpi=110)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Construyendo pool completo (largo<=500, todas las que haya)...")
    pool = construir_pool(CLASES3, n_por_clase=N_TEST_MAX + N_TRAIN_MAX,
                          largo_max=500, cache=os.path.join(OUT_DIR, "_exp_pool_full.csv"))
    print("  conteo por clase:", pool["localization"].value_counts().to_dict())

    train, test_completo = dividir_train_test(pool, N_TRAIN_MAX)
    n_test_clase = test_completo["localization"].value_counts().min()
    test = subconjunto(test_completo, n_test_clase).reset_index(drop=True)
    print(f"\nTrain: {len(train)}  {train['localization'].value_counts().to_dict()}")
    print(f"Test balanceado usado: {len(test)}  {test['localization'].value_counts().to_dict()}")

    partes = [exp_overfitting(train, test), exp_curse(train, test), exp_ablacion(train, test)]
    todo = pd.concat(partes, ignore_index=True)
    todo.to_csv(os.path.join(OUT_DIR, "experimentos_resultados.csv"), index=False)
    print("\nGuardado: experimentos_resultados.csv")

    figura_overfitting(partes[0])
    figura_curse(partes[1])
    figura_ablacion(partes[2])


if __name__ == "__main__":
    main()
