"""Descarga y arma la base de datos del proyecto.

Fuente: World Development Indicators (WDI) del Banco Mundial, en la copia
pública y versionada que mantiene la Fundación Gapminder en GitHub
(repositorio open-numbers/ddf--open_numbers--world_development_indicators),
que replica directamente los datos del Banco Mundial en archivos CSV.

Se usa esta copia porque no requiere API key, es estable y versionada, y
permite reconstruir exactamente el mismo archivo. Los valores se pueden
contrastar con https://data.worldbank.org usando el código WDI de cada variable.

Uso (desde la raíz del repositorio):
    python scripts/descargar_datos.py

Genera:
    data/wdi_panel_paises.csv   -> panel país-año (2010-2025) con 16 indicadores + población
    data/paises_metadata.csv    -> nombre en español, región ONU y grupo de ingreso del Banco Mundial
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.config import INGRESO_ES, REGION_ES, VARIABLES  # noqa: E402

BASE = ("https://raw.githubusercontent.com/open-numbers/"
        "ddf--open_numbers--world_development_indicators/master")
ANIO_INICIO, ANIO_FIN = 2010, 2025


def _leer_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def _codigo_ddf(codigo_wdi: str) -> str:
    return codigo_wdi.lower().replace(".", "_")


def descargar_indicador(codigo_wdi: str) -> pd.Series:
    ddf = _codigo_ddf(codigo_wdi)
    url = f"{BASE}/datapoints/ddf--datapoints--{ddf}--by--geo--time.csv"
    df = _leer_csv(url)
    df = df[(df["time"] >= ANIO_INICIO) & (df["time"] <= ANIO_FIN)]
    return df.set_index(["geo", "time"])[ddf]


def nombre_es(iso2: str | float, nombre_en: str) -> str:
    try:
        from babel import Locale
        if isinstance(iso2, str):
            return Locale("es").territories.get(iso2.upper(), nombre_en)
    except Exception:
        pass
    return nombre_en


def main() -> None:
    destino = RAIZ / "data"
    destino.mkdir(exist_ok=True)

    print("Descargando metadatos de países…")
    meta = _leer_csv(f"{BASE}/ddf--entities--geo--country.csv")
    # Solo Estados de la ONU (miembros y observadores): excluye territorios dependientes y agregados regionales
    meta = meta[meta["un_state"] == True].copy()  # noqa: E712
    meta["iso3"] = meta["country"].str.upper()
    meta["pais"] = [nombre_es(i2, n) for i2, n in zip(meta["iso3166_1_alpha2"], meta["name"])]
    meta["region"] = meta["un_sdg_region"].map(REGION_ES)
    meta["grupo_ingreso"] = meta["income_groups"].map(INGRESO_ES)
    meta = meta[["iso3", "pais", "region", "grupo_ingreso", "latitude", "longitude"]]
    meta.to_csv(destino / "paises_metadata.csv", index=False)
    print(f"  {len(meta)} países guardados en data/paises_metadata.csv")

    series = {}
    for clave, (codigo, corto, *_rest) in VARIABLES.items():
        print(f"Descargando {codigo:<22} {corto}")
        series[clave] = descargar_indicador(codigo)
    print("Descargando SP.POP.TOTL           Población (solo para filtro de tamaño)")
    series["poblacion"] = descargar_indicador("SP.POP.TOTL")

    panel = pd.concat(series, axis=1).reset_index().rename(columns={"geo": "iso3", "time": "anio"})
    panel["iso3"] = panel["iso3"].str.upper()
    panel = panel[panel["iso3"].isin(meta["iso3"])].sort_values(["iso3", "anio"])
    panel.to_csv(destino / "wdi_panel_paises.csv", index=False)
    print(f"  {len(panel)} filas país-año guardadas en data/wdi_panel_paises.csv")


if __name__ == "__main__":
    main()
