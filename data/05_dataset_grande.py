"""
Genera el dataset ampliado que usa el notebook notebooks/overfitting.ipynb.

Toma el pool que arma 03_experimentos.py (proteínas humanas revisadas, con
evidencia experimental, largo 50-500, keywords mutuamente excluyentes) y exporta
solo las columnas de composición + fisicoquímicos + señales, junto con los datos
de identificación y la clase. El resultado (~5.800 proteínas) se guarda en
data_share/ para distribuirlo a los estudiantes.

Uso:
    python 05_dataset_grande.py

Requiere haber corrido 03_experimentos.py al menos una vez (o lo construye acá
la primera vez, descargando de UniProt).
"""

import importlib.util
import os

import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARE_DIR = os.path.join(os.path.dirname(OUT_DIR), "data_share")
DESTINO = os.path.join(SHARE_DIR, "features_grande_composicion_senales.csv")

_spec = importlib.util.spec_from_file_location(
    "exp03", os.path.join(OUT_DIR, "03_experimentos.py"))
exp03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp03)


def main():
    pool = exp03.construir_pool(
        exp03.CLASES3, n_por_clase=100000, largo_max=500,
        cache=os.path.join(OUT_DIR, "_exp_pool_full.csv"))

    columnas = (["accession", "sequence", "length"] + exp03.AMINOACIDOS
                + exp03.FISICOQUIMICOS + exp03.SENALES + ["localization"])
    os.makedirs(SHARE_DIR, exist_ok=True)
    pool[columnas].to_csv(DESTINO, index=False)

    print("Guardado:", DESTINO)
    print("  filas:", pool.shape[0], "| por clase:",
          pool["localization"].value_counts().to_dict())


if __name__ == "__main__":
    main()
