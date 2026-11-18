"""Procesamiento, ACP y K-Means.

Todas las tablas, métricas y gráficos de la aplicación se calculan con estas
funciones a partir de los CSV de la carpeta data/. No hay resultados fijos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (PAIS_FOCO, SEMILLA, TRANSFORMACIONES, UMBRAL_FALTANTES,
                        VARIABLES, nombre)

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# 1. Carga y agregación por periodo
# ---------------------------------------------------------------------------
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(DATA / "wdi_panel_paises.csv")
    meta = pd.read_csv(DATA / "paises_metadata.csv")
    return panel, meta


def promediar_periodo(panel: pd.DataFrame, meta: pd.DataFrame, inicio: int, fin: int,
                      min_anios: int = 3) -> pd.DataFrame:
    """Promedio por país de cada indicador en el periodo [inicio, fin].

    Promediar varios años suaviza choques de un solo año y reduce faltantes.
    Si un país tiene menos de `min_anios` observaciones de un indicador en el
    periodo, ese valor queda como faltante (un solo año no es representativo).
    """
    sub = panel[(panel["anio"] >= inicio) & (panel["anio"] <= fin)]
    cols = list(VARIABLES) + ["poblacion"]
    medias = sub.groupby("iso3")[cols].mean()
    conteos = sub.groupby("iso3")[cols].count()
    medias = medias.where(conteos >= min_anios)
    df = meta.set_index("iso3").join(medias, how="left")
    return df


def duplicados(panel: pd.DataFrame) -> int:
    return int(panel.duplicated(subset=["iso3", "anio"]).sum())


def tabla_faltantes(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    t = pd.DataFrame({
        "Variable": [nombre(v) for v in variables],
        "clave": variables,
        "Código WDI": [VARIABLES[v][0] for v in variables],
        "Países sin dato": [int(df[v].isna().sum()) for v in variables],
    })
    t["% faltante"] = 100 * t["Países sin dato"] / len(df)
    t["Decisión"] = np.where(t["% faltante"] > 100 * UMBRAL_FALTANTES,
                             "Excluir (supera el umbral)", "Conservar")
    return t.sort_values("% faltante", ascending=False)


# ---------------------------------------------------------------------------
# 2. Preparación
# ---------------------------------------------------------------------------
def winsorizar(s: pd.Series, p_inf: float = 0.025, p_sup: float = 0.975) -> pd.Series:
    return s.clip(s.quantile(p_inf), s.quantile(p_sup))


def transformar(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = df[variables].copy()
    for v in variables:
        regla = TRANSFORMACIONES.get(v)
        if regla == "log":
            out[v] = np.log(out[v].clip(lower=1e-6))
        elif regla == "winsor":
            out[v] = winsorizar(out[v])
    return out


@dataclass
class Preparacion:
    originales: pd.DataFrame          # variables en unidades originales (países finales)
    transformadas: pd.DataFrame       # tras log / winsorización (si aplica)
    estandarizadas: pd.DataFrame      # media 0, desviación 1
    variables: list[str]
    excluidas_faltantes: list[str]
    paises_excluidos_faltantes: list[str]
    paises_excluidos_tamano: list[str]
    meta: pd.DataFrame
    aplica_transformacion: bool
    imputados: pd.DataFrame
    base_tamano: pd.DataFrame          # países tras el filtro de tamaño, antes de faltantes
    scaler: StandardScaler = field(repr=False, default=None)


def preparar(df: pd.DataFrame, variables: list[str], excluir_pequenos: bool = True,
             aplicar_transformacion: bool = True, umbral: float = UMBRAL_FALTANTES,
             imputar_un_faltante: bool = True) -> Preparacion:
    base = df.copy()
    pequenos = []
    if excluir_pequenos:
        mask = base["poblacion"] < 1_000_000
        pequenos = base.index[mask].tolist()
        base = base[~mask]
    base_filtrada = base.copy()
    # Variables con demasiados faltantes se excluyen antes de eliminar países
    falt = base[variables].isna().mean()
    excluidas = [v for v in variables if falt[v] > umbral]
    usadas = [v for v in variables if v not in excluidas]
    imputados = []
    if imputar_un_faltante:
        # Países a los que les falta UNA sola variable: se imputa con la mediana
        # de los países de su mismo grupo de ingreso (comparables en desarrollo).
        n_falt = base[usadas].isna().sum(axis=1)
        for iso in base.index[n_falt == 1]:
            v = base.loc[iso, usadas].index[base.loc[iso, usadas].isna()][0]
            grupo = base.loc[iso, "grupo_ingreso"]
            ref = base[base["grupo_ingreso"] == grupo] if isinstance(grupo, str) else base
            valor = ref[v].median()
            base.loc[iso, v] = valor
            imputados.append({"iso3": iso, "País": base.loc[iso, "pais"], "Variable imputada": nombre(v),
                              "Valor imputado": round(float(valor), 2),
                              "Referencia": grupo if isinstance(grupo, str) else "Todos los países"})
    sin_dato = base.index[base[usadas].isna().any(axis=1)].tolist()
    base = base.drop(index=sin_dato)
    originales = base[usadas]
    trans = transformar(base, usadas) if aplicar_transformacion else originales.copy()
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(trans), index=trans.index, columns=usadas)
    meta = base[["pais", "region", "grupo_ingreso"]]
    return Preparacion(originales, trans, z, usadas, excluidas, sin_dato, pequenos,
                       meta, aplicar_transformacion, pd.DataFrame(imputados), base_filtrada, scaler)


# ---------------------------------------------------------------------------
# 3. ACP
# ---------------------------------------------------------------------------
@dataclass
class ResultadoACP:
    pca: PCA
    varianza: pd.DataFrame     # por componente
    cargas: pd.DataFrame       # correlación variable-componente
    puntajes: pd.DataFrame     # coordenadas de cada país
    n_kaiser: int
    n_70: int
    n_80: int


def ejecutar_acp(z: pd.DataFrame) -> ResultadoACP:
    pca = PCA(random_state=SEMILLA).fit(z)
    k = len(pca.explained_variance_ratio_)
    nombres = [f"CP{i+1}" for i in range(k)]
    var = pd.DataFrame({
        "Componente": nombres,
        "Valor propio": pca.explained_variance_,
        "% varianza": 100 * pca.explained_variance_ratio_,
    })
    var["% acumulado"] = var["% varianza"].cumsum()
    # Cargas como correlaciones (vector propio × raíz del valor propio): más fáciles de interpretar
    cargas = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                          index=z.columns, columns=nombres)
    # Convención de signo: CP1 positivo = mayor PIB per cápita (o la variable de mayor carga)
    ref = "pib_pc" if "pib_pc" in cargas.index else cargas.index[0]
    for c in nombres:
        signo = np.sign(cargas.loc[ref, c]) if c == "CP1" else np.sign(cargas[c].iloc[cargas[c].abs().argmax()])
        if signo < 0:
            cargas[c] *= -1
            pca.components_[nombres.index(c)] *= -1
    puntajes = pd.DataFrame(pca.transform(z), index=z.index, columns=nombres)
    n_kaiser = int((pca.explained_variance_ > 1).sum())
    n_70 = int(np.searchsorted(var["% acumulado"].values, 70) + 1)
    n_80 = int(np.searchsorted(var["% acumulado"].values, 80) + 1)
    return ResultadoACP(pca, var, cargas, puntajes, n_kaiser, n_70, n_80)


def describir_componente(cargas: pd.Series, n: int = 3, umbral: float = 0.4) -> tuple[list, list]:
    """Variables con mayor carga positiva y negativa (|carga| >= umbral)."""
    pos = cargas[cargas >= umbral].sort_values(ascending=False).head(n)
    neg = cargas[cargas <= -umbral].sort_values().head(n)
    return ([(nombre(v), round(c, 2)) for v, c in pos.items()],
            [(nombre(v), round(c, 2)) for v, c in neg.items()])


# ---------------------------------------------------------------------------
# 4. K-Means
# ---------------------------------------------------------------------------
def evaluar_k(X: pd.DataFrame, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit(X)
        filas.append({"K": k, "Inercia (WCSS)": km.inertia_,
                      "Silueta": silhouette_score(X, km.labels_)})
    t = pd.DataFrame(filas)
    t["Reducción de inercia (%)"] = -100 * t["Inercia (WCSS)"].pct_change()
    return t


def k_codo(tabla: pd.DataFrame) -> int:
    """Codo por máxima distancia a la recta entre el primer y último punto."""
    x = tabla["K"].values.astype(float)
    y = tabla["Inercia (WCSS)"].values
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    d = np.abs(yn - (1 - xn)) / np.sqrt(2)
    return int(x[np.argmax(d)])


LETRAS = "ABCDEFGHIJ"


def ejecutar_kmeans(X: pd.DataFrame, k: int) -> tuple[pd.Series, KMeans]:
    km = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit(X)
    # Etiquetas con letras asignadas por tamaño del grupo (A = el más grande).
    # Las letras son nombres, no un orden ni una jerarquía.
    tam = pd.Series(km.labels_).value_counts().index.tolist()
    mapa = {lab: f"Grupo {LETRAS[i]}" for i, lab in enumerate(tam)}
    etiquetas = pd.Series([mapa[l] for l in km.labels_], index=X.index, name="grupo")
    return etiquetas, km


def perfiles(originales: pd.DataFrame, z: pd.DataFrame, grupos: pd.Series):
    """Perfil de cada grupo en variables originales (mediana) y estandarizadas (media)."""
    med = originales.groupby(grupos).median()
    med.insert(0, "N.º países", grupos.value_counts())
    zmed = z.groupby(grupos).mean()
    return med.sort_index(), zmed.sort_index()


FEMENINO = {"inflacion", "ied", "cuenta_corriente", "formacion_capital", "esperanza_vida",
            "urbanizacion", "deuda_publica"}


def nombre_descriptivo(zfila: pd.Series, n: int = 2) -> str:
    """Nombre del grupo a partir de sus dos rasgos más distintivos (mayor |z| promedio)."""
    orden = zfila.abs().sort_values(ascending=False).index[:n]
    partes = []
    for v in orden:
        adj = ("alta" if zfila[v] > 0 else "baja") if v in FEMENINO else ("alto" if zfila[v] > 0 else "bajo")
        partes.append(f"{nombre(v)} {adj}")
    texto = ", ".join(partes)
    return texto[0].upper() + texto[1:]


def silueta_por_pais(X: pd.DataFrame, grupos: pd.Series, km: KMeans) -> pd.DataFrame:
    s = silhouette_samples(X, km.labels_)
    dist = km.transform(X)
    orden = np.argsort(dist, axis=1)
    # mapear índice de centroide -> nombre de grupo
    mapa = {}
    for lab, g in zip(km.labels_, grupos):
        mapa[lab] = g
    segundo = [mapa[o[1]] for o in orden]
    return pd.DataFrame({"grupo": grupos, "silueta": s, "grupo_vecino": segundo}, index=X.index)


def vecinos(puntajes: pd.DataFrame, iso: str = PAIS_FOCO, n: int = 8) -> pd.Series:
    d = np.sqrt(((puntajes - puntajes.loc[iso]) ** 2).sum(axis=1))
    return d.drop(iso).sort_values().head(n)


def comparar_particiones(a: pd.Series, b: pd.Series) -> float:
    comunes = a.index.intersection(b.index)
    return adjusted_rand_score(a.loc[comunes], b.loc[comunes])


def cramers_v(tabla: pd.DataFrame) -> float:
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(tabla)[0]
    n = tabla.values.sum()
    r, c = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


# ---------------------------------------------------------------------------
# 5. Flujo completo (útil para el README y pruebas)
# ---------------------------------------------------------------------------
def flujo_completo(periodo=(2015, 2019), variables=None, n_comp=None, k=None,
                   excluir_pequenos=True, aplicar_transformacion=True):
    from src.config import VARIABLES_DEFECTO
    panel, meta = cargar_datos()
    df = promediar_periodo(panel, meta, *periodo)
    prep = preparar(df, variables or VARIABLES_DEFECTO, excluir_pequenos, aplicar_transformacion)
    acp = ejecutar_acp(prep.estandarizadas)
    n_comp = n_comp or acp.n_kaiser
    X = acp.puntajes.iloc[:, :n_comp]
    tabla_k = evaluar_k(X)
    k = k or k_codo(tabla_k)
    grupos, km = ejecutar_kmeans(X, k)
    return dict(df=df, prep=prep, acp=acp, X=X, tabla_k=tabla_k, k=k, grupos=grupos, km=km)
