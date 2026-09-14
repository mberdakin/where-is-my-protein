"""
Comparación head-to-head de representaciones para el práctico.

Evalúa, con las MISMAS particiones y los mismos modelos, un conjunto de
representaciones sobre dos tamaños de dataset:

  - ~250 proteínas (el tamaño de la versión original del práctico)
  - 750 proteínas (el dataset actual)

Representaciones comparadas:

  1. composicion                              (20)  bag of aminoácidos
  2. composicion + fisicoquimicos             (33)  = método ORIGINAL
  3. dipeptidos                               (400) sólo dipéptidos, sin fisicoquímicos
  4. dipeptidos + fisicoquimicos              (413) = método de dipéptidos original
  5. senales                                  (11)  sólo las señales posicionales
  6. composicion + fisicoquimicos + senales   (44)  = método NUEVO
  7. dipeptidos + fisicoquimicos + senales    (424) dipéptidos + señales

Modelos: Logistic Regression, SVM y Random Forest (los mismos de Orange).
Validación cruzada estratificada de 5 folds, con partición fija.

Uso:
    python 02_comparar_representaciones.py

Requiere que 01_construir_dataset.py ya haya generado los tres CSV.
Es una validación interna (scikit-learn): no reemplaza la práctica en Orange,
sirve para tener números de referencia reproducibles.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N_CHICO = 83  # por clase -> 249 proteínas, comparable con las ~250 originales

AMINOACIDOS = list("ACDEFGHIKLMNPQRSTVWY")
FISICOQUIMICOS = [
    "longitud", "peso_molecular", "punto_isoelectrico", "hidrofobicidad_promedio",
    "aromaticidad", "indice_inestabilidad", "carga_neta_pH7", "fraccion_helice",
    "fraccion_turn", "fraccion_hoja", "indice_alifatico", "fraccion_cargados",
    "coef_extincion_molar",
]
SENALES = [
    "senal_peptido", "senal_hidrofobicidad_max", "senal_n_basicos",
    "nls_monopartita", "nls_bipartita", "nls_densidad_basica_10",
    "nls_densidad_basica_7", "nls_carga_neta_nterm20", "tmd_alfa_helices",
    "tmd_alfa_hidrofobicidad_max", "tmd_beta_momento_hidrofobico",
]

MODELOS = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=SEED),
}


def cargar():
    comp = pd.read_csv(os.path.join(OUT_DIR, "features_composicion_fisicoquimicos.csv"))
    dip = pd.read_csv(os.path.join(OUT_DIR, "features_dipeptidos_fisicoquimicos.csv"))
    sen = pd.read_csv(os.path.join(OUT_DIR, "features_composicion_senales_fisicoquimicos.csv"))
    assert list(comp["accession"]) == list(dip["accession"]) == list(sen["accession"])
    dipeptidos = [c for c in dip.columns if len(c) == 2 and set(c).issubset(AMINOACIDOS)]
    # Tabla combinada dipéptidos + señales (ningún CSV suelto tiene ambas)
    dip_sen = pd.concat([dip, sen[SENALES]], axis=1)
    return comp, dip, sen, dip_sen, dipeptidos


def representaciones(comp, dip, sen, dip_sen, dipeptidos):
    """Devuelve {nombre: (tabla, lista_de_features)} para una escala dada."""
    return {
        "composicion": (comp, AMINOACIDOS),
        "composicion+fisicoquimicos": (comp, AMINOACIDOS + FISICOQUIMICOS),
        "dipeptidos": (dip, dipeptidos),
        "dipeptidos+fisicoquimicos": (dip, dipeptidos + FISICOQUIMICOS),
        "senales": (sen, SENALES),
        "composicion+fisicoquimicos+senales": (sen, AMINOACIDOS + FISICOQUIMICOS + SENALES),
        "dipeptidos+fisicoquimicos+senales": (dip_sen, dipeptidos + FISICOQUIMICOS + SENALES),
    }


def subconjunto_chico(df):
    """Hasta N_CHICO filas por clase, en el orden ya mezclado del dataset."""
    return pd.concat([g.head(N_CHICO) for _, g in df.groupby("localization")])


def evaluar(df, features):
    X = df[features].to_numpy(dtype=float)
    y = df["localization"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fila = {}
    for nombre, modelo in MODELOS.items():
        ca = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
        pred = cross_val_predict(modelo, X, y, cv=cv)
        fila[nombre] = {
            "ca": ca.mean(),
            "ca_std": ca.std(),
            "f1_macro": f1_score(y, pred, average="macro"),
        }
    return fila


def guardar_figura(resultado):
    """Barras con el mejor CA de cada representación, en las dos escalas."""
    orden = [
        "composicion",
        "composicion+fisicoquimicos",
        "dipeptidos",
        "dipeptidos+fisicoquimicos",
        "senales",
        "composicion+fisicoquimicos+senales",
        "dipeptidos+fisicoquimicos+senales",
    ]
    mejor = (resultado.groupby(["representacion", "n", "n_features"])["CA"]
             .max().reset_index())
    escalas = ["~250", "750"]
    x = range(len(orden))
    ancho = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, escala in enumerate(escalas):
        vals = [
            mejor[(mejor["representacion"] == r) & (mejor["n"] == escala)]["CA"].iloc[0]
            for r in orden
        ]
        offset = (i - 0.5) * ancho
        barras = ax.bar([v + offset for v in x], vals, ancho,
                        label=f"n={escala}", color=["#8da0cb", "#fc8d62"][i])
        ax.bar_label(barras, fmt="%.3f", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\n({n} features)" for r, n in
                        mejor.drop_duplicates("representacion").set_index("representacion")
                        .loc[orden, "n_features"].items()], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Mejor accuracy (5-fold CV)")
    ax.set_ylim(0.60, 0.80)
    ax.set_title("Comparación de representaciones: el efecto de agregar señales posicionales")
    ax.legend(title="Tamaño del dataset")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    destino = os.path.join(os.path.dirname(OUT_DIR), "img", "comparacion_representaciones.png")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fig.savefig(destino, dpi=110)
    print("Figura guardada:", destino)


def main():
    comp, dip, sen, dip_sen, dipeptidos = cargar()

    escalas = {
        "~250": (subconjunto_chico(comp), subconjunto_chico(dip),
                 subconjunto_chico(sen), subconjunto_chico(dip_sen)),
        "750": (comp, dip, sen, dip_sen),
    }

    filas = []
    for escala, (c, d, s, ds) in escalas.items():
        reps = representaciones(c, d, s, ds, dipeptidos)
        for nombre, (fuente, features) in reps.items():
            for modelo, m in evaluar(fuente, features).items():
                filas.append({
                    "n": escala,
                    "representacion": nombre,
                    "n_features": len(features),
                    "modelo": modelo,
                    "CA": round(m["ca"], 3),
                    "CA_std": round(m["ca_std"], 3),
                    "F1_macro": round(m["f1_macro"], 3),
                })

    resultado = pd.DataFrame(filas)
    resultado.to_csv(os.path.join(OUT_DIR, "comparacion_representaciones.csv"), index=False)
    guardar_figura(resultado)

    for escala in ["~250", "750"]:
        print(f"\n=== Dataset n={escala} ===")
        bloque = resultado[resultado["n"] == escala]
        pivote = bloque.pivot(index=["representacion", "n_features"],
                              columns="modelo", values="CA")
        pivote = pivote[["LogReg", "SVM", "RandomForest"]]
        pivote["mejor_CA"] = pivote.max(axis=1)
        print(pivote.to_string())

    print("\nGuardado: comparacion_representaciones.csv")


if __name__ == "__main__":
    main()
