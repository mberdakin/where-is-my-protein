"""
ESM-2 embeddings como representación alternativa (no implementada en Orange).

Calcula un embedding por proteína con ESM-2 (modelo de lenguaje de proteínas),
entrena un clasificador lineal encima y lo compara contra composición y
señales, usando EXACTAMENTE el mismo train/test que 03_experimentos.py (test
balanceado de 125 por clase).

Pensado para correr en CPU (y en Colab sin GPU). Modelo:
    facebook/esm2_t6_8M_UR50D     (~8M parámetros, rápido en CPU; rinde ~igual
                                   que modelos más grandes en esta tarea)

Uso:
    python 04_esm2_embeddings.py

Dependencias extra (no están en requirements.txt base):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers

Cachea los embeddings en _esm2_<modelo>.npz (ignorado por git).
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 0
MODELO_DEFECTO = "facebook/esm2_t6_8M_UR50D"
DEVICE = "cpu"            # estudiantes: Colab CPU
BATCH = 16
MAX_LEN = 512

# Reutilizamos pool/split/features de 03 para que la comparación sea idéntica.
_spec = importlib.util.spec_from_file_location(
    "exp03", os.path.join(OUT_DIR, "03_experimentos.py"))
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)

MODELOS = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM": make_pipeline(StandardScaler(), SVC(cache_size=1000)),
    "RandomForest": RandomForestClassifier(500, random_state=SEED, n_jobs=1),
}


def calcular_embeddings(secuencias, modelo):
    """Embedding por proteína: promedio del último hidden state (sin padding)."""
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    red = AutoModel.from_pretrained(modelo).eval().to(DEVICE)

    # Ordenar por largo agrupa secuencias parecidas y reduce el padding.
    orden = sorted(range(len(secuencias)), key=lambda i: len(secuencias[i]))
    salida = np.zeros((len(secuencias), red.config.hidden_size), dtype=np.float32)

    t0 = time.time()
    for inicio in range(0, len(orden), BATCH):
        idx = orden[inicio:inicio + BATCH]
        batch = [secuencias[i] for i in idx]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            hidden = red(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)
        for j, i in enumerate(idx):
            salida[i] = emb[j].cpu().numpy()
        if inicio % (BATCH * 20) == 0:
            hechas = min(inicio + BATCH, len(orden))
            vel = hechas / max(time.time() - t0, 1e-6)
            print(f"    {hechas}/{len(orden)}  ({vel:.1f} seq/s, "
                  f"restante ~{(len(orden)-hechas)/max(vel,1e-6)/60:.1f} min)")
    return salida


def embeddings_de(df, modelo, cache):
    tag = modelo.split("/")[-1]
    cache = cache.replace("<modelo>", tag)
    accesiones = df["accession"].tolist()
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        if list(data["accessions"]) == accesiones:
            print(f"  (cache) {os.path.basename(cache)}")
            return data["X"]
    print(f"  Calculando embeddings con {modelo} ...")
    X = calcular_embeddings(df["sequence"].tolist(), modelo)
    np.savez(cache, X=X, accessions=np.array(accesiones, dtype=object))
    return X


def evaluar(X, train_df, test_df, idx_train, idx_test):
    """Entrena con embeddings de train y evalúa en el test balanceado."""
    out = {}
    Xtr, Xte = X[idx_train], X[idx_test]
    ytr = train_df["localization"].to_numpy()
    yte = test_df["localization"].to_numpy()
    for nombre, modelo in MODELOS.items():
        modelo.fit(Xtr, ytr)
        out[nombre] = exp03._metricas(yte, modelo.predict(Xte))
    return out


def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_DEFECTO
    print(f"Modelo ESM-2: {modelo}")

    cache_pool = os.path.join(OUT_DIR, "_exp_pool_full.csv")
    if os.path.exists(cache_pool):
        pool = pd.read_csv(cache_pool)
    else:
        pool = exp03.construir_pool(exp03.CLASES3, 100000, 500, cache_pool)

    train, test_completo = exp03.dividir_train_test(pool, exp03.N_TRAIN_MAX)
    test = exp03.subconjunto(test_completo, test_completo["localization"].value_counts().min())
    test = test.reset_index(drop=True)
    print(f"Train {len(train)} | Test balanceado {len(test)} "
          f"{test['localization'].value_counts().to_dict()}")

    # índices dentro de un DataFrame que = train + test (mismo orden)
    combinado = pd.concat([train, test], ignore_index=True)
    idx_train = np.arange(len(train))
    idx_test = np.arange(len(train), len(train) + len(test))

    X = embeddings_de(combinado, modelo, os.path.join(OUT_DIR, "_esm2_<modelo>.npz"))

    # --- comparación de representaciones ---
    filas = []

    def registrar(nombre, res, n_features):
        mejor = max(res.items(), key=lambda kv: kv[1]["BA"])
        _, m = mejor
        filas.append({
            "representacion": nombre, "n_features": n_features, "modelo": mejor[0],
            "BA": round(m["BA"], 3), "CA": round(m["CA"], 3),
            "F1_macro": round(m["F1_macro"], 3),
            "F1_Nucleus": round(m["reporte"].get("Nucleus", {}).get("f1-score", np.nan), 3),
            "F1_Cytoplasm": round(m["reporte"].get("Cytoplasm", {}).get("f1-score", np.nan), 3),
            "F1_Membrane": round(m["reporte"].get("Membrane", {}).get("f1-score", np.nan), 3),
        })

    C = exp03.AMINOACIDOS
    F = exp03.FISICOQUIMICOS
    S = exp03.SENALES

    registrar("composicion+fisio", exp03.evaluar_3clases(train, test, C + F), len(C + F))
    registrar("senales", exp03.evaluar_3clases(train, test, C + F + S), len(C + F + S))
    registrar(f"ESM2 ({modelo.split('/')[-1]})",
              evaluar(X, train, test, idx_train, idx_test), X.shape[1])

    tabla = pd.DataFrame(filas)
    print("\n=== Comparación (test balanceado 375) ===")
    print(tabla.to_string(index=False))
    tabla.to_csv(os.path.join(OUT_DIR, "esm2_resultados.csv"), index=False)
    print("\nGuardado: esm2_resultados.csv")


if __name__ == "__main__":
    main()
