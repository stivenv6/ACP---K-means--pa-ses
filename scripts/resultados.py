"""Reproduce en consola los resultados principales (configuración de referencia).

Uso (desde la raíz del repositorio):
    python scripts/resultados.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline as pl  # noqa: E402
from src.config import nombre  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

r = pl.flujo_completo(periodo=(2015, 2019), n_comp=4, k=4)
prep, acp, grupos, X = r["prep"], r["acp"], r["grupos"], r["X"]

print(f"Países en la muestra final: {len(X)}")
print(f"Países imputados: {len(prep.imputados)} | excluidos por faltantes: {len(prep.paises_excluidos_faltantes)}"
      f" | excluidos por tamaño: {len(prep.paises_excluidos_tamano)}\n")
print("== Varianza explicada ==")
print(acp.varianza.round(2).to_string(index=False), "\n")
print("== Cargas (CP1-CP4) ==")
c = acp.cargas.iloc[:, :4].round(2)
c.index = [nombre(v) for v in c.index]
print(c, "\n")
print("== Codo y silueta (K-Means sobre CP1-CP4) ==")
print(r["tabla_k"].round(3).to_string(index=False), "\n")
med, z = pl.perfiles(prep.originales, prep.estandarizadas, grupos)
med.columns = ["N"] + [nombre(v) for v in med.columns[1:]]
print("== Perfil de los grupos (medianas, unidades originales) ==")
print(med.round(1), "\n")
sil = pl.silueta_por_pais(X, grupos, r["km"])
print(f"Silueta promedio: {sil['silueta'].mean():.3f}")
ct = pd.crosstab(grupos, prep.meta["grupo_ingreso"])
print(f"V de Cramér grupos vs. ingreso: {pl.cramers_v(ct):.2f}\n")
print(f"Colombia -> {grupos['COL']} (silueta {sil.loc['COL', 'silueta']:.2f})")
vec = pl.vecinos(X, "COL", 8)
print("Vecinos más cercanos:", ", ".join(prep.meta.loc[vec.index, "pais"]))
