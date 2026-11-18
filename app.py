"""Aplicación Streamlit: Perfiles macroeconómicos del mundo con ACP y K-Means.

¿Con qué economías comparte Colombia su perfil macroeconómico?

Ejecutar localmente:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import skew

from src import pipeline as pl
from src.config import (ORDEN_INGRESO, PAIS_FOCO, PERIODO_DEFECTO, PERIODOS,
                        TRANSFORMACIONES, UMBRAL_FALTANTES, VARIABLES,
                        VARIABLES_DEFECTO, descripcion, nombre)

st.set_page_config(page_title="Observatorio macro · ¿A quién se parece Colombia?",
                   page_icon=":material/travel_explore:", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Identidad visual: estilo editorial ("observatorio económico")
# ---------------------------------------------------------------------------
TINTA, PAPEL, BORDO, PETROLEO, MOSTAZA = "#1e2a33", "#f7f4ee", "#8c2f39", "#2c6e6a", "#c58b2b"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {{font-family:'IBM Plex Sans', sans-serif;}}
h1, h2, h3, h4 {{font-family:'Fraunces', serif !important; color:{TINTA}; letter-spacing:-0.01em;}}
.block-container {{padding-top:1.2rem; max-width:1240px;}}
/* barra de marca */
.marca {{display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid {TINTA};
        padding-bottom:0.45rem; margin-bottom:0.8rem; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase;}}
.marca b {{color:{BORDO};}}
/* encabezado de cada capítulo */
.capitulo {{display:flex; gap:1.4rem; align-items:flex-end; margin:1.3rem 0 1.2rem 0;}}
.capitulo .num {{font-family:'Fraunces', serif; font-size:4.2rem; line-height:0.85; color:{BORDO}; font-weight:700;}}
.capitulo .tit {{font-family:'Fraunces', serif; font-size:2.1rem; line-height:1.1; color:{TINTA}; font-weight:700;}}
.capitulo .baj {{font-size:1rem; color:#5b6670; margin-top:0.3rem;}}
/* recuadros de lectura */
.nota, .hallazgo, .atencion {{padding:0.85rem 1.1rem; margin:0.8rem 0; border-radius:2px; line-height:1.55;}}
.nota {{background:#ffffff; border:1px solid #e2dccf; border-left:5px solid {PETROLEO};}}
.hallazgo {{background:#fbf3e4; border:1px solid #ecdcbc; border-left:5px solid {MOSTAZA};}}
.atencion {{background:#f8eaea; border:1px solid #ebcfd1; border-left:5px solid {BORDO};}}
.etq {{display:block; font-size:0.7rem; font-weight:600; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:0.3rem;}}
.nota .etq {{color:{PETROLEO};}} .hallazgo .etq {{color:#9a6a17;}} .atencion .etq {{color:{BORDO};}}
/* indicadores */
[data-testid="stMetric"] {{background:#ffffff; border:1px solid #e2dccf; border-top:4px solid {BORDO};
        padding:0.7rem 0.9rem; border-radius:2px;}}
[data-testid="stMetricValue"] {{font-family:'Fraunces', serif; color:{TINTA};}}
/* panel lateral oscuro */
[data-testid="stSidebar"] {{background:{TINTA};}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{color:#f3efe6 !important;}}
[data-testid="stSidebar"] [data-testid="stMetric"] {{background:#263540; border:1px solid #34444f; border-top:4px solid {MOSTAZA};}}
[data-testid="stSidebar"] [data-testid="stMetricValue"], [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{color:#f3efe6 !important;}}
[data-testid="stSidebar"] hr {{border-color:#3b4a55;}}
/* navegación superior */
[data-testid="stButtonGroup"] button {{font-size:0.85rem;}}
.pie {{margin-top:2.5rem; padding-top:0.6rem; border-top:1px solid #d8d1c3; font-size:0.78rem; color:#6b747c;}}
</style>
""", unsafe_allow_html=True)

COLORES = ["#8c2f39", "#2c6e6a", "#c58b2b", "#3d4f7a", "#7a8b5a",
           "#b86b4b", "#5c4b6e", "#4a4a4a", "#9aa5b1", "#d4a5a5"]
COLOR_GRUPO = {f"Grupo {l}": c for l, c in zip(pl.LETRAS, COLORES)}
COLOR_REGION = {
    "América Latina y el Caribe": "#8c2f39", "Europa y Norteamérica": "#3d4f7a",
    "África subsahariana": "#c58b2b", "Norte de África y Asia occidental": "#7a8b5a",
    "Asia central y meridional": "#b86b4b", "Asia oriental y sudoriental": "#2c6e6a",
    "Oceanía": "#5c4b6e",
}
DIVERGENTE = [[0, PETROLEO], [0.5, "#f4f1ea"], [1, BORDO]]
SECUENCIAL = [[0, "#f4f1ea"], [0.5, "#d9a7a1"], [1, BORDO]]

import plotly.io as pio
pio.templates["observatorio"] = go.layout.Template(pio.templates["simple_white"])
pio.templates["observatorio"].layout.update(
    font=dict(family="IBM Plex Sans, sans-serif", color=TINTA, size=13),
    title=dict(font=dict(family="Fraunces, serif", size=18, color=TINTA)),
    colorway=COLORES, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="#e6e0d4", linecolor=TINTA),
    yaxis=dict(showgrid=True, gridcolor="#e6e0d4", linecolor=TINTA),
    shapedefaults=dict(line=dict(width=1.6), opacity=0.9, fillcolor="rgba(0,0,0,0)"),
)
PLANTILLA = "observatorio"


def nota(texto: str, clase: str = "nota"):
    etiqueta = {"nota": "Lectura", "hallazgo": "Hallazgo", "atencion": "Atención"}.get(clase, "")
    st.markdown(f'<div class="{clase}"><span class="etq">{etiqueta}</span>{texto}</div>', unsafe_allow_html=True)


def capitulo(i: int, titulo: str, bajada: str = ""):
    st.markdown(f'<div class="capitulo"><div class="num">{i:02d}</div><div><div class="tit">{titulo}</div>'
                f'<div class="baj">{bajada}</div></div></div>', unsafe_allow_html=True)


def estilo(fig, titulo, x=None, y=None, alto=450):
    fig.update_layout(title=dict(text=titulo, x=0, xanchor="left"), template=PLANTILLA,
                      height=alto, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(title=None, orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)))
    if x is not None:
        fig.update_xaxes(title_text=x)
    if y is not None:
        fig.update_yaxes(title_text=y)
    return fig


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Cálculos (todos se ejecutan en vivo, con caché para agilizar la navegación)
# ---------------------------------------------------------------------------
@st.cache_data
def datos():
    return pl.cargar_datos()


@st.cache_data(show_spinner="Preparando datos y ejecutando ACP…")
def analizar(periodo, variables, excluir, transf, imputar):
    panel, meta = datos()
    df = pl.promediar_periodo(panel, meta, *PERIODOS[periodo])
    prep = pl.preparar(df, list(variables), excluir, transf, imputar_un_faltante=imputar)
    acp = pl.ejecutar_acp(prep.estandarizadas)
    return df, prep, acp


@st.cache_data(show_spinner="Evaluando valores de K…")
def tabla_k(X):
    return pl.evaluar_k(X)


@st.cache_data(show_spinner="Ejecutando K-Means…")
def segmentar(X, k):
    grupos, km = pl.ejecutar_kmeans(X, k)
    sil = pl.silueta_por_pais(X, grupos, km)
    centroides = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    centroides["grupo"] = [grupos[km.labels_ == i].iloc[0] for i in range(k)]
    return grupos, sil, centroides


# ---------------------------------------------------------------------------
# Barra lateral: navegación y parámetros
# ---------------------------------------------------------------------------
PAGINAS = ["Contexto", "Exploración", "Preparación", "ACP", "K-Means", "Colombia", "Robustez", "Conclusiones"]
st.markdown('<div class="marca"><span><b>Observatorio macro</b> &nbsp;·&nbsp; ¿A quién se parece Colombia?</span>'
            '<span>ACP + K-Means &nbsp;·&nbsp; WDI, Banco Mundial</span></div>', unsafe_allow_html=True)
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGINAS[0]
pagina = st.segmented_control("Navegación", PAGINAS, key="nav", label_visibility="collapsed")
if pagina is None:
    pagina = PAGINAS[0]

st.sidebar.markdown("## Panel de control")
st.sidebar.caption("Cada cambio vuelve a ejecutar la preparación, el ACP y K-Means.")
periodo = st.sidebar.selectbox("Periodo (promedio de años)", list(PERIODOS),
                               index=list(PERIODOS).index(PERIODO_DEFECTO))
variables = st.sidebar.multiselect("Variables del modelo", list(VARIABLES), default=VARIABLES_DEFECTO,
                                   format_func=nombre)
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos 4 variables cuantitativas.")
    st.stop()
excluir = st.sidebar.toggle("Excluir países con menos de 1 millón de habitantes", value=True)
transf = st.sidebar.toggle("Aplicar transformaciones (log / winsorización)", value=True)
imputar = st.sidebar.toggle("Imputar países a los que les falta una sola variable", value=True)

df, prep, acp = analizar(periodo, tuple(variables), excluir, transf, imputar)
n_var = len(prep.variables)
if n_var < 4:
    st.error("Tras excluir variables con muchos faltantes quedan menos de 4. Agrega otras variables.")
    st.stop()

n_comp = st.sidebar.slider("Componentes principales a conservar", 2, n_var,
                           value=min(max(acp.n_kaiser, 2), n_var), key=f"ncomp_{n_var}_{periodo}",
                           help="Por defecto: criterio de Kaiser (valor propio > 1).")
k = st.sidebar.slider("Número de grupos (K)", 2, 10, value=4,
                      help="Compara el método del codo y la silueta en la sección 5.")

X = acp.puntajes.iloc[:, :n_comp]
tk = tabla_k(X)
grupos, sil, centroides = segmentar(X, k)
med_grupo, z_grupo = pl.perfiles(prep.originales, prep.estandarizadas, grupos)
NOMBRE_GRUPO = {g: pl.nombre_descriptivo(z_grupo.loc[g]) for g in z_grupo.index}
ETIQ = {g: f"{g} · {NOMBRE_GRUPO[g]}" for g in NOMBRE_GRUPO}
COLOR_ETIQ = {ETIQ[g]: COLOR_GRUPO[g] for g in ETIQ}
var_ret = acp.varianza["% acumulado"].iloc[n_comp - 1]
meta = prep.meta
tiene_col = PAIS_FOCO in X.index

ES_DEFECTO = (periodo == PERIODO_DEFECTO and set(variables) == set(VARIABLES_DEFECTO)
              and excluir and transf and imputar and n_comp == 4 and k == 4)

st.sidebar.markdown("---")
st.sidebar.markdown("### Resultado actual")
st.sidebar.metric("Países analizados", len(X))
st.sidebar.metric("Información conservada por el ACP", f"{fmt(var_ret)} %")
st.sidebar.metric("Silueta promedio", fmt(sil["silueta"].mean(), 3))
if not ES_DEFECTO:
    st.sidebar.info("Estás viendo una configuración distinta a la de referencia. Las tablas y "
                    "gráficos se recalculan; los textos interpretativos fijos describen la configuración "
                    "de referencia (2015–2019, 12 variables, 4 componentes, K = 4).")


def aviso_config():
    if not ES_DEFECTO:
        nota("La lectura escrita de esta sección corresponde a la configuración de referencia. "
             "Las cifras entre líneas y todos los gráficos sí reflejan tu configuración actual.", "atencion")


def nombres_paises(isos):
    return ", ".join(meta.loc[[i for i in isos if i in meta.index], "pais"].tolist())


# ===========================================================================
# 1. CONTEXTO
# ===========================================================================
if pagina == PAGINAS[0]:
    capitulo(1, "¿A quién se parece Colombia?", "Perfiles macroeconómicos de los países del mundo con Análisis de Componentes Principales y K-Means")
    st.markdown("""
**Pregunta principal.** ¿Con qué economías comparte Colombia su perfil macroeconómico y de desarrollo,
y ese parecido coincide con su región o con su nivel de ingreso?

**Preguntas que orientan el ejercicio**
1. ¿Qué dimensiones resumen las diferencias entre las economías del mundo? ¿Basta con el nivel de ingreso o
   aparecen otras (dinamismo, estabilidad, apertura)?
2. ¿Existen grupos de países con perfiles similares? ¿Qué los caracteriza?
3. ¿Los grupos replican la clasificación oficial por ingreso del Banco Mundial o la contradicen?
4. ¿En qué grupo queda Colombia, con quién comparte grupo y qué la diferencia de sus pares?
""")
    nota("<b>Unidad de observación:</b> cada fila es un <b>país</b> (Estado de la ONU). Cada variable es el "
         "<b>promedio del periodo</b> elegido (2015–2019 o 2020–2024), lo que suaviza choques de un solo año.")

    st.markdown("### ¿Por qué ACP y K-Means para este caso?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Análisis de Componentes Principales (ACP)**
- Los indicadores de desarrollo están muy correlacionados entre sí (PIB per cápita, esperanza de vida,
  uso de internet y urbanización tienen correlaciones cercanas a 0,8–0,9). Usarlos todos por separado
  repite la misma información.
- El ACP los resume en pocas **dimensiones no correlacionadas** e interpretables, y permite **ver** a
  más de 130 países en un plano, algo imposible con 12 variables a la vez.
- Además, permite saber si la "economía" de un país se resume en una sola cifra (el ingreso) o si
  hay otras dimensiones relevantes.
""")
    with c2:
        st.markdown("""
**K-Means**
- La clasificación oficial del Banco Mundial agrupa a los países **solo por ingreso** (INB per cápita).
  K-Means agrupa por el **perfil completo**: crecimiento, inflación, desempleo, apertura,
  profundidad financiera y desarrollo humano.
- Eso permite responder si países con ingreso parecido tienen estructuras distintas, y con quién se
  parece realmente Colombia.
- Se aplica sobre los componentes del ACP: así cada dimensión pesa por la información que aporta y no
  por cuántas variables redundantes la miden.
""")

    st.markdown("### Fuente y cobertura")
    st.markdown(f"""
- **Fuente:** *World Development Indicators* (WDI) del Banco Mundial, en la copia pública y versionada que
  mantiene la Fundación Gapminder en GitHub (repositorio `open-numbers/ddf--open_numbers--world_development_indicators`),
  que replica los datos del Banco Mundial. El script `scripts/descargar_datos.py` reconstruye la base.
- **Cobertura:** 195 Estados de la ONU (193 miembros más Palestina y la Santa Sede, observadores), panel anual 2010–2025 (excluye territorios dependientes y agregados regionales).
- **Clasificaciones auxiliares (no entran al modelo):** región ONU y grupo de ingreso del Banco Mundial,
  usados solo para interpretar los grupos.
- **Población** (`SP.POP.TOTL`): solo se usa para el filtro de tamaño, no como variable del modelo.
""")
    tabla = pd.DataFrame([{
        "Variable": v[1], "Descripción": v[2], "Código WDI": v[0], "Dimensión": v[3],
        "En el modelo por defecto": "Sí" if k_ in VARIABLES_DEFECTO else "No (disponible)",
        "Transformación": {"log": "Logaritmo", "winsor": "Winsorización 2,5–97,5 %"}.get(TRANSFORMACIONES.get(k_), "—"),
    } for k_, v in VARIABLES.items()])
    st.dataframe(tabla, hide_index=True)

    st.markdown("### Limitaciones de la fuente")
    st.markdown("""
- **Faltantes no aleatorios:** los países en conflicto o con estadísticas débiles (Venezuela, Siria, Yemen,
  Corea del Norte, Somalia…) tienen más huecos y terminan fuera del análisis. La muestra final está sesgada
  hacia países con mejor capacidad estadística.
- **Comparabilidad:** aunque el Banco Mundial armoniza las series, las metodologías nacionales (por ejemplo,
  de desempleo o del IPC) difieren entre países.
- **Promedios:** resumir cinco años en un promedio esconde la dinámica interna del periodo; 2020–2024
  mezcla la caída por la pandemia y el rebote posterior.
- **Revisiones:** el Banco Mundial revisa las series con el tiempo; el script fija la fuente para que el
  resultado sea reproducible.
- **Variables fiscales:** la deuda pública tiene demasiados faltantes y no puede usarse, y el recaudo tributario
  tiene huecos importantes, aunque ambas serían relevantes (ver sección 2).
""")

# ===========================================================================
# 2. EXPLORACIÓN
# ===========================================================================
elif pagina == PAGINAS[1]:
    capitulo(2, "Exploración de los datos", "Estructura, calidad, distribuciones y correlaciones de la base")
    panel, meta_full = datos()
    c = st.columns(5)
    c[0].metric("Filas país-año", f"{len(panel):,}".replace(",", "."))
    c[1].metric("Países", panel["iso3"].nunique())
    c[2].metric("Años", f"{panel['anio'].min()}–{panel['anio'].max()}")
    c[3].metric("Variables cuantitativas", len(VARIABLES))
    c[4].metric("Registros duplicados (país-año)", pl.duplicados(panel))

    st.markdown("### Estructura de la base")
    c1, c2 = st.columns([1, 2])
    with c1:
        tipos = pd.DataFrame({"Columna": panel.columns, "Tipo": panel.dtypes.astype(str).values,
                              "Rol": ["Identificador (no entra al modelo)", "Tiempo (se usa para promediar)"]
                              + ["Indicador" for _ in VARIABLES] + ["Solo para filtro de tamaño"]})
        st.dataframe(tipos, hide_index=True, height=420)
    with c2:
        st.caption("Primeras filas del panel país-año (datos crudos tal como se descargan)")
        st.dataframe(panel.head(12), hide_index=True)
    nota("El panel tiene una fila por país y año. <b>iso3</b> es un identificador y no entra al modelo. "
         f"Para el análisis se colapsa a <b>una fila por país</b> promediando el periodo <b>{periodo}</b>; "
         "un valor solo se calcula si el país tiene al menos 3 años con dato.")

    st.markdown(f"### Estadísticas descriptivas · promedio {periodo}")
    base = prep.base_tamano
    desc = base[list(VARIABLES)].describe().T
    desc.insert(0, "Variable", [nombre(v) for v in desc.index])
    desc["Asimetría"] = [skew(base[v].dropna()) for v in desc.index]
    st.dataframe(desc.round(2), hide_index=True)
    nota("Varias variables son muy asimétricas: el <b>PIB per cápita</b> va de unos cientos a más de 80.000 dólares; "
         "la <b>inflación</b> tiene casos de hiperinflación; el <b>comercio</b> supera el 300 % del PIB en economías "
         "de tránsito como Singapur. Estos extremos dominarían las distancias de K-Means, por eso se tratan en la sección 3.")

    st.markdown("### Datos faltantes")
    falt = pl.tabla_faltantes(base, list(VARIABLES))
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.bar(falt.sort_values("% faltante"), x="% faltante", y="Variable", orientation="h",
                     color="Decisión", color_discrete_map={"Conservar": "#2c6e6a", "Excluir (supera el umbral)": "#8c2f39"})
        fig.add_vline(x=100 * UMBRAL_FALTANTES, line_dash="dash", line_color="#8c2f39",
                      annotation_text=f"Umbral {int(100*UMBRAL_FALTANTES)} %")
        st.plotly_chart(estilo(fig, "Porcentaje de países sin dato por variable", "% de países sin dato", None, 480))
    with c2:
        reg = base.groupby("region")[list(VARIABLES)].apply(lambda d: d.isna().mean() * 100)
        reg.columns = [nombre(v) for v in reg.columns]
        fig = px.imshow(reg.T, color_continuous_scale=SECUENCIAL, aspect="auto", text_auto=".0f")
        fig.update_coloraxes(colorbar_title="% sin dato")
        st.plotly_chart(estilo(fig, "Faltantes por región (%)", "Región", "Variable", 480))
    nota("La <b>deuda pública</b> supera ampliamente el umbral de faltantes (a casi tres de cada cuatro países les falta) "
         "y no puede usarse. El <b>recaudo tributario</b> queda cerca del umbral, por lo que no se incluye por defecto: "
         "obligaría a imputar o excluir a muchos países. Los faltantes se concentran en África subsahariana y Asia occidental: "
         "no son aleatorios, lo que se reconoce como limitación.")

    st.markdown("### Distribución de cada variable")
    v_sel = st.selectbox("Variable", list(VARIABLES), format_func=nombre, key="dist")
    c1, c2 = st.columns(2)
    d = base[[v_sel, "pais", "region"]].dropna()
    with c1:
        fig = px.histogram(d, x=v_sel, nbins=40, color_discrete_sequence=["#2c6e6a"])
        st.plotly_chart(estilo(fig, f"Histograma · {nombre(v_sel)}", descripcion(v_sel), "Número de países", 380))
    with c2:
        fig = px.box(d, x=v_sel, y="region", points="outliers", hover_name="pais", color="region",
                     color_discrete_map=COLOR_REGION)
        fig.update_layout(showlegend=False)
        st.plotly_chart(estilo(fig, f"Por región · {nombre(v_sel)}", descripcion(v_sel), None, 380))
    if PAIS_FOCO in base.index and pd.notna(base.loc[PAIS_FOCO, v_sel]):
        pct = (base[v_sel] < base.loc[PAIS_FOCO, v_sel]).mean() * 100
        st.caption(f"Colombia: {fmt(base.loc[PAIS_FOCO, v_sel], 2)} · por encima del {fmt(pct, 0)} % de los países.")

    st.markdown("### Correlaciones entre las variables del modelo")
    corr = prep.transformadas.corr()
    corr.index = corr.columns = [nombre(v) for v in corr.columns]
    fig = px.imshow(corr, color_continuous_scale=DIVERGENTE, zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_coloraxes(colorbar_title="r de Pearson")
    st.plotly_chart(estilo(fig, f"Matriz de correlaciones ({len(prep.transformadas)} países, variables ya transformadas)", None, None, 600))
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False)
    top = "; ".join(f"{a} – {b} ({fmt(r, 2)})" for (a, b), r in tri.head(5).items())
    nota(f"<b>Pares más correlacionados:</b> {top}. Este bloque de variables de desarrollo muy correlacionadas es "
         "la razón principal para aplicar ACP: contienen en buena parte la misma información. Otras variables "
         "(crecimiento, IED, formación de capital) se correlacionan poco con ese bloque y aportarán dimensiones propias.")

# ===========================================================================
# 3. PREPARACIÓN
# ===========================================================================
elif pagina == PAGINAS[2]:
    capitulo(3, "Preparación", "Cada decisión de limpieza, con su justificación y su efecto sobre la muestra")
    total = len(df)
    n_peq = len(prep.paises_excluidos_tamano)
    n_falt = len(prep.paises_excluidos_faltantes)
    pasos = pd.DataFrame({
        "Paso": ["Estados de la ONU", "Filtro de tamaño (< 1 millón de hab.)",
                 "Exclusión por faltantes", "Muestra final"],
        "Países": [total, total - n_peq, total - n_peq - n_falt, len(X)],
    })
    fig = px.funnel(pasos, x="Países", y="Paso", color_discrete_sequence=["#2c6e6a"])
    st.plotly_chart(estilo(fig, "De los Estados de la ONU a la muestra final", "Número de países", None, 330))

    st.markdown("### Decisiones y su justificación")
    st.markdown(f"""
1. **Promedio del periodo ({periodo}).** Reduce el ruido de un año atípico y aprovecha años con dato cuando
   falta otro. Se exige un mínimo de 3 años con dato por indicador.
2. **Duplicados.** Se verificó que no haya filas país-año repetidas ({pl.duplicados(datos()[0])} encontradas).
3. **Identificadores fuera del modelo.** El código ISO, el nombre, la región y el grupo de ingreso no entran
   a ACP ni a K-Means; solo se usan para interpretar.
4. **Filtro de tamaño** ({"activado" if excluir else "desactivado"}). Las micro-economías (islas del Caribe y del
   Pacífico, Luxemburgo, Malta…) tienen perfiles extremos de comercio, IED o turismo que distorsionan las distancias
   y no son comparables con Colombia. {"Se excluyeron " + str(n_peq) + " países." if excluir else ""}
5. **Variables con muchos faltantes.** Se excluye toda variable con más de {int(UMBRAL_FALTANTES*100)} % de países sin dato.
   Excluidas en esta configuración: **{", ".join(nombre(v) for v in prep.excluidas_faltantes) or "ninguna"}**.
6. **Imputación mínima** ({"activada" if imputar else "desactivada"}). Si a un país le falta **una sola** variable,
   se completa con la mediana de los países de su mismo grupo de ingreso. Así no se pierden países relevantes
   (por ejemplo Argentina, cuyo IPC oficial no está en el WDI varios años, o Canadá). Con dos o más faltantes,
   el país se excluye.
7. **Transformaciones** ({"activadas" if transf else "desactivadas"}). Logaritmo para variables positivas muy
   sesgadas (PIB per cápita, comercio, crédito) y winsorización 2,5–97,5 % para variables con colas extremas en
   ambos lados (inflación, IED, cuenta corriente, crecimiento).
8. **Estandarización.** ACP y K-Means dependen de escalas y distancias: sin estandarizar, el PIB per cápita (en
   miles de dólares) dominaría a la inflación (en puntos porcentuales). Cada variable queda con media 0 y desviación 1.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Países excluidos por faltantes ({n_falt})**")
        st.write(nombres_paises(prep.paises_excluidos_faltantes) or "Ninguno")
        if excluir:
            st.markdown(f"**Países excluidos por tamaño ({n_peq})**")
            st.write(", ".join(df.loc[prep.paises_excluidos_tamano, "pais"].tolist()))
    with c2:
        st.markdown(f"**Países imputados ({len(prep.imputados)})**")
        if len(prep.imputados):
            st.dataframe(prep.imputados.drop(columns="iso3"), hide_index=True)
        else:
            st.write("Ninguno")

    st.markdown("### Efecto de las transformaciones")
    asim = pd.DataFrame({
        "Variable": [nombre(v) for v in prep.variables],
        "Regla": [{"log": "Logaritmo", "winsor": "Winsorización"}.get(TRANSFORMACIONES.get(v), "—") for v in prep.variables],
        "Asimetría original": [skew(prep.originales[v]) for v in prep.variables],
        "Asimetría transformada": [skew(prep.transformadas[v]) for v in prep.variables],
    })
    st.dataframe(asim.round(2), hide_index=True)
    vt = st.selectbox("Comparar distribución", [v for v in prep.variables if v in TRANSFORMACIONES] or prep.variables,
                      format_func=nombre)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(prep.originales, x=vt, nbins=35, color_discrete_sequence=["#7a8b5a"])
        st.plotly_chart(estilo(fig, f"{nombre(vt)} · original", descripcion(vt), "Número de países", 340))
    with c2:
        fig = px.histogram(prep.transformadas, x=vt, nbins=35, color_discrete_sequence=["#2c6e6a"])
        st.plotly_chart(estilo(fig, f"{nombre(vt)} · {'transformada' if transf else 'sin transformar'}",
                               "Valor transformado", "Número de países", 340))

    st.markdown("### Verificación de la estandarización")
    ver = pd.DataFrame({"Variable": [nombre(v) for v in prep.variables],
                        "Media": prep.estandarizadas.mean().values,
                        "Desviación estándar": prep.estandarizadas.std(ddof=0).values})
    st.dataframe(ver.round(3), hide_index=True)

    final = meta.join(prep.originales)
    st.download_button("Descargar base final (CSV, variables originales)", final.to_csv().encode("utf-8"),
                       "base_final_paises.csv", "text/csv")

# ===========================================================================
# 4. ACP
# ===========================================================================
elif pagina == PAGINAS[3]:
    capitulo(4, "Análisis de Componentes Principales", "Qué dimensiones resumen las diferencias entre economías")
    c = st.columns(4)
    c[0].metric("Componentes conservados", n_comp)
    c[1].metric("Varianza conservada", f"{fmt(var_ret)} %")
    c[2].metric("Criterio de Kaiser (λ > 1)", acp.n_kaiser)
    c[3].metric("Componentes para ≥ 70 % / ≥ 80 %", f"{acp.n_70} / {acp.n_80}")

    st.markdown("### Varianza explicada")
    c1, c2 = st.columns(2)
    v = acp.varianza
    with c1:
        fig = go.Figure()
        fig.add_bar(x=v["Componente"], y=v["% varianza"], name="% de varianza", marker_color="#2c6e6a",
                    text=v["% varianza"].round(1), textposition="outside")
        fig.add_scatter(x=v["Componente"], y=v["% acumulado"], name="% acumulado", mode="lines+markers",
                        line=dict(color="#c58b2b"), yaxis="y")
        fig.add_hline(y=70, line_dash="dot", line_color="gray", annotation_text="70 %")
        fig.add_vline(x=n_comp - 0.5, line_dash="dash", line_color="#8c2f39", annotation_text="Corte elegido")
        st.plotly_chart(estilo(fig, "Varianza explicada y acumulada por componente", "Componente", "% de la varianza total"))
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=v["Componente"], y=v["Valor propio"], mode="lines+markers", marker_color="#2c6e6a",
                        name="Valor propio")
        fig.add_hline(y=1, line_dash="dash", line_color="#8c2f39", annotation_text="Kaiser: λ = 1")
        st.plotly_chart(estilo(fig, "Gráfico de sedimentación (valores propios)", "Componente", "Valor propio (λ)"))
    st.dataframe(v.round(2), hide_index=True)

    nota(f"<b>¿Cuántos componentes conservar?</b> El criterio de Kaiser sugiere <b>{acp.n_kaiser}</b> componentes "
         f"(valor propio mayor que 1, es decir, que resumen más que una variable original). Con {acp.n_kaiser} se conserva el "
         f"<b>{fmt(v['% acumulado'].iloc[acp.n_kaiser-1])} %</b> de la varianza, por encima del 70 % que suele considerarse suficiente; "
         f"el gráfico de sedimentación muestra un quiebre después del primer componente y una pendiente suave luego del "
         f"componente {acp.n_kaiser}. El CP1 solo resume el {fmt(v['% varianza'].iloc[0])} %: <b>reducir a un único eje "
         f"de 'ingreso' dejaría por fuera más de la mitad de la información</b>. Actualmente estás usando {n_comp} componentes "
         f"({fmt(var_ret)} %).")

    st.markdown("### Cargas: ¿qué significa cada componente?")
    cargas = acp.cargas.iloc[:, :n_comp].copy()
    cargas.index = [nombre(x) for x in cargas.index]
    fig = px.imshow(cargas, color_continuous_scale=DIVERGENTE, zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_coloraxes(colorbar_title="Carga (correlación)")
    st.plotly_chart(estilo(fig, "Cargas de las variables (correlación entre la variable y el componente)",
                           "Componente", None, 520))
    st.caption("Las cargas se expresan como correlaciones (vector propio × √valor propio). Valores cercanos a ±1 "
               "indican que la variable define al componente. El signo de cada componente es arbitrario; se fijó para "
               "que el CP1 aumente con el PIB per cápita.")

    ETIQUETAS_CP = {
        "CP1": "Nivel de desarrollo",
        "CP2": "Dinamismo inversor frente a inestabilidad",
        "CP3": "Apertura con desempleo alto",
        "CP4": "Inversión financiada con déficit externo",
    }
    for cp in cargas.columns:
        pos, neg = pl.describir_componente(acp.cargas[cp])
        txt_pos = ", ".join(f"{a} ({fmt(b, 2)})" for a, b in pos) or "ninguna con carga ≥ 0,40"
        txt_neg = ", ".join(f"{a} ({fmt(b, 2)})" for a, b in neg) or "ninguna con carga ≤ −0,40"
        etiqueta = f" — <i>{ETIQUETAS_CP[cp]}</i>" if ES_DEFECTO and cp in ETIQUETAS_CP else ""
        nota(f"<b>{cp}</b> ({fmt(v.loc[v.Componente == cp, '% varianza'].iloc[0])} %){etiqueta}<br>"
             f"Valores altos: {txt_pos}.<br>Valores bajos: {txt_neg}.")
    if ES_DEFECTO:
        st.markdown("""
**Lectura de los componentes (configuración de referencia)**
- **CP1 · Nivel de desarrollo (≈ 42 %).** PIB per cápita, uso de internet, esperanza de vida, crédito privado y
  urbanización cargan por encima de 0,8. Es el eje "rico–pobre", con inflación baja y superávit externo del lado alto.
  Es la dimensión principal, pero no la única.
- **CP2 · Dinamismo inversor frente a inestabilidad (≈ 15 %).** Crecimiento, IED y formación de capital en un extremo;
  desempleo e inflación en el otro. Separa a los países que crecen invirtiendo y atrayendo IED (Vietnam, Camboya, Panamá, Hungría,
  Irlanda) de los estancados con desequilibrios (Argentina, Sudáfrica, Angola, Zimbabue), sin importar su ingreso.
- **CP3 · Apertura con desempleo (≈ 10 %).** Desempleo, comercio e IED altos a la vez. Destacan economías abiertas del
  sur de África con desempleo muy alto (Esuatini, Namibia, Lesoto) y, por su apertura, Irlanda, Singapur y Chipre.
- **CP4 · Inversión con déficit externo (≈ 8 %).** Formación de capital alta frente a cuenta corriente negativa: países
  que invierten más de lo que ahorran y lo financian con el exterior.
""")

    st.markdown("### Los países en el espacio de componentes")
    c1, c2, c3 = st.columns(3)
    ejes = list(acp.puntajes.columns[:max(n_comp, 2)])
    ex = c1.selectbox("Eje horizontal", ejes, index=0)
    ey = c2.selectbox("Eje vertical", ejes, index=1)
    color_por = c3.selectbox("Colorear por", ["Región", "Grupo de ingreso (Banco Mundial)"])
    vectores = st.toggle("Mostrar vectores de las variables (biplot)", value=True)
    pts = acp.puntajes[[ex, ey]].join(meta)
    col = "region" if color_por == "Región" else "grupo_ingreso"
    pts[col] = pts[col].fillna("Sin clasificar")
    fig = px.scatter(pts, x=ex, y=ey, color=col, hover_name="pais",
                     color_discrete_map=COLOR_REGION if col == "region" else None,
                     category_orders={"grupo_ingreso": ORDEN_INGRESO})
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.5, color="white")))
    if tiene_col:
        fig.add_scatter(x=[pts.loc[PAIS_FOCO, ex]], y=[pts.loc[PAIS_FOCO, ey]], mode="markers+text",
                        marker=dict(symbol="star-diamond", size=19, color="#1e2a33", line=dict(width=2, color="#f7f4ee")),
                        text=["Colombia"], textposition="top center", name="Colombia")
    if vectores:
        esc = 0.85 * min(pts[ex].abs().max(), pts[ey].abs().max())
        for var_ in prep.variables:
            cx, cy = acp.cargas.loc[var_, ex] * esc, acp.cargas.loc[var_, ey] * esc
            fig.add_annotation(x=cx, y=cy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor="#555", arrowwidth=1.2, text="")
            fig.add_annotation(x=cx * 1.1, y=cy * 1.1, text=nombre(var_), showarrow=False,
                               font=dict(size=10, color="#333"))
    pex = v.loc[v.Componente == ex, "% varianza"].iloc[0]
    pey = v.loc[v.Componente == ey, "% varianza"].iloc[0]
    st.plotly_chart(estilo(fig, f"Países en el plano {ex}–{ey}: esta vista resume el {fmt(pex + pey)} % de la información",
                           f"{ex} ({fmt(pex)} %)", f"{ey} ({fmt(pey)} %)", 650))
    nota(f"Este plano muestra solo el <b>{fmt(pex + pey)} %</b> de la variación total; el {fmt(100 - pex - pey)} % restante "
         f"está en los otros componentes. Dos países cercanos aquí pueden ser distintos en los componentes no graficados, "
         f"por eso K-Means usa los {n_comp} componentes conservados ({fmt(var_ret)} %) y no solo esta vista.")

    with st.expander("Ver puntajes de todos los países"):
        st.dataframe(acp.puntajes.iloc[:, :n_comp].join(meta[["pais", "region"]]).round(3))

# ===========================================================================
# 5. K-MEANS
# ===========================================================================
elif pagina == PAGINAS[4]:
    capitulo(5, "Segmentación con K-Means", "Grupos de países con perfiles similares, descritos con las variables originales")
    nota(f"K-Means se aplica sobre los <b>{n_comp} componentes conservados</b> ({fmt(var_ret)} % de la varianza). "
         "Los componentes ya están en escala comparable, no están correlacionados y concentran la señal; los "
         "componentes descartados contienen sobre todo ruido. Se usan 50 inicializaciones aleatorias (n_init = 50) "
         "y semilla fija para que el resultado sea reproducible.")

    st.markdown("### ¿Cuántos grupos? Codo y silueta")
    codo = pl.k_codo(tk)
    k_sil = int(tk.loc[tk["Silueta"].idxmax(), "K"])
    tk3 = tk[tk["K"] >= 3]
    k_sil3 = int(tk3.loc[tk3["Silueta"].idxmax(), "K"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(tk, x="K", y="Inercia (WCSS)", markers=True, color_discrete_sequence=["#2c6e6a"])
        fig.add_vline(x=codo, line_dash="dash", line_color="#c58b2b", annotation_text=f"Codo ≈ {codo}")
        fig.add_vline(x=k, line_dash="dot", line_color="#8c2f39", annotation_text=f"K elegido = {k}",
                      annotation_position="bottom right")
        st.plotly_chart(estilo(fig, "Método del codo", "Número de grupos (K)", "Inercia intra-grupo (WCSS)", 400))
    with c2:
        fig = px.line(tk, x="K", y="Silueta", markers=True, color_discrete_sequence=["#3d4f7a"])
        fig.add_vline(x=k, line_dash="dot", line_color="#8c2f39", annotation_text=f"K elegido = {k}")
        st.plotly_chart(estilo(fig, "Coeficiente de silueta promedio", "Número de grupos (K)", "Silueta (−1 a 1)", 400))
    st.dataframe(tk.round(3), hide_index=True)
    c = st.columns(3)
    c[0].metric("Codo (máxima curvatura)", codo)
    c[1].metric("Mayor silueta", f"K = {k_sil} ({fmt(tk['Silueta'].max(), 3)})")
    c[2].metric("Mayor silueta con K ≥ 3", f"K = {k_sil3} ({fmt(tk3['Silueta'].max(), 3)})")
    if ES_DEFECTO:
        nota("<b>Elección de K = 4.</b> La silueta más alta se obtiene con K = 2 (≈ 0,37), pero esa partición solo separa "
             "países de ingreso alto del resto: repite lo que ya dice el CP1. Entre K = 3 y K = 6 la silueta es "
             "prácticamente plana (≈ 0,27–0,28), así que la silueta no discrimina entre esas opciones. El codo se ubica "
             "entre 4 y 6, donde la reducción de inercia empieza a estabilizarse. Se elige <b>K = 4</b>, el menor K que, "
             "además del gradiente de ingreso, separa un grupo definido por otra dimensión (países que invierten mucho "
             "con déficit externo), y que da grupos de tamaño razonable. Con K = 5 o 6 aparecen grupos más pequeños que "
             "fragmentan sin mejorar la silueta.")
    else:
        nota(f"En tu configuración el codo sugiere K ≈ {codo} y la mayor silueta se da con K = {k_sil}. Si la silueta es "
             "casi igual para varios K, elige el que tenga grupos interpretables con las variables originales.")

    st.markdown("### Los grupos en el espacio de componentes")
    pts = X.join(meta).join(grupos)
    pts["Grupo"] = pts["grupo"].map(ETIQ)
    fig = px.scatter(pts, x="CP1", y="CP2", color="Grupo", hover_name="pais", color_discrete_map=COLOR_ETIQ,
                     hover_data={"region": True, "grupo_ingreso": True, "Grupo": False})
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.add_scatter(x=centroides["CP1"], y=centroides["CP2"], mode="markers", name="Centroides",
                    marker=dict(symbol="x", size=14, color="black"))
    if tiene_col:
        fig.add_scatter(x=[pts.loc[PAIS_FOCO, "CP1"]], y=[pts.loc[PAIS_FOCO, "CP2"]], mode="markers+text",
                        marker=dict(symbol="star-diamond", size=19, color="#1e2a33", line=dict(width=2, color="#f7f4ee")),
                        text=["Colombia"], textposition="top center", name="Colombia")
    p12 = acp.varianza["% varianza"].iloc[:2].sum()
    st.plotly_chart(estilo(fig, f"Grupos de K-Means proyectados en CP1–CP2 ({fmt(p12)} % de la información)",
                           f"CP1 ({fmt(acp.varianza['% varianza'].iloc[0])} %)",
                           f"CP2 ({fmt(acp.varianza['% varianza'].iloc[1])} %)", 600))
    st.caption(f"Los grupos se calcularon con {n_comp} componentes; en este plano de 2 dimensiones algunos se "
               "superponen porque se separan en CP3 o CP4. Las letras son nombres, no indican orden ni jerarquía "
               "(se asignan por tamaño: A es el grupo más grande).")

    mapa = meta.join(grupos).reset_index()
    mapa["Grupo"] = mapa["grupo"].map(ETIQ)
    fig = px.choropleth(mapa, locations="iso3", color="Grupo", hover_name="pais", color_discrete_map=COLOR_ETIQ,
                        projection="natural earth")
    fig.update_geos(showcountries=True, countrycolor="white", showframe=False, landcolor="#e9e9e9")
    st.plotly_chart(estilo(fig, "Mapa de los grupos (gris: países fuera de la muestra)", None, None, 520))

    st.markdown("### Perfil de cada grupo en las variables originales")
    tabla = med_grupo.copy()
    tabla.columns = ["N.º países"] + [nombre(c_) for c_ in tabla.columns[1:]]
    tabla.insert(0, "Perfil (rasgos más distintivos)", [NOMBRE_GRUPO[g] for g in tabla.index])
    mundo = prep.originales.median()
    fila_mundo = pd.DataFrame([["Mediana de todos los países", len(X)] + mundo.tolist()],
                              columns=tabla.columns, index=["Total"])
    st.dataframe(pd.concat([tabla, fila_mundo]).round(1))
    st.caption("Medianas por grupo en unidades originales (US$ para PIB per cápita, % para el resto, años para "
               "esperanza de vida). Se usa la mediana porque es robusta a los valores extremos.")
    zz = z_grupo.copy()
    zz.columns = [nombre(c_) for c_ in zz.columns]
    zz.index = [ETIQ[g] for g in zz.index]
    fig = px.imshow(zz, color_continuous_scale=DIVERGENTE, zmin=-2, zmax=2, text_auto=".2f", aspect="auto")
    fig.update_coloraxes(colorbar_title="Desv. estándar<br>frente a la media")
    st.plotly_chart(estilo(fig, "Perfil estandarizado: cuánto se aleja cada grupo de la media mundial", None, None, 420))

    if ES_DEFECTO:
        st.markdown("""
**Descripción de los grupos (configuración de referencia)**
- **Grupo A · Economías de ingreso alto (44 países).** PIB per cápita mediano ≈ US$ 31.700, esperanza de vida de 81 años,
  85 % de uso de internet, crédito privado ≈ 85 % del PIB, inflación ≈ 1 % y superávit en cuenta corriente. Incluye a Europa
  occidental, Norteamérica, Asia desarrollada, el Golfo y, en América Latina, a **Chile y Costa Rica**.
- **Grupo B · Ingreso medio con desempleo alto y bajo crecimiento (37 países).** PIB per cápita ≈ US$ 6.150, desempleo
  mediano de 11,4 % (el más alto), crecimiento de solo 2,2 % y formación de capital baja. Es el grupo de **Colombia**, junto
  con Brasil, México, Perú, Argentina, Uruguay, Ecuador, Turquía, Sudáfrica, Rusia y los Balcanes. Su rasgo común es el
  estancamiento con un mercado laboral débil.
- **Grupo C · Ingreso bajo, en expansión (33 países).** PIB per cápita ≈ US$ 1.000, esperanza de vida de 63 años, 17 % de
  uso de internet y crédito privado de 14 % del PIB, pero el mayor crecimiento (4,9 %). África subsahariana y Asia meridional.
- **Grupo D · Economías que invierten con déficit externo (24 países).** Ingreso medio-bajo (≈ US$ 3.200) pero con la
  formación de capital más alta (31 % del PIB), crecimiento de 4,4 % y el mayor déficit en cuenta corriente (−5,6 %). China,
  Vietnam, Indonesia, Filipinas, Panamá, República Dominicana, Bolivia, Marruecos. **Este grupo no lo define el ingreso**: mezcla
  países de ingreso bajo, medio-bajo, medio-alto e incluso uno de ingreso alto.
""")

    with st.expander("Ver los países de cada grupo"):
        for g in sorted(grupos.unique()):
            isos = grupos[grupos == g].index
            st.markdown(f"**{ETIQ[g]}** ({len(isos)} países): {nombres_paises(isos)}")

    st.markdown("### ¿Qué tan claros son los grupos? Silueta por país")
    s = sil.join(meta).sort_values(["grupo", "silueta"], ascending=[True, False]).reset_index()
    s["Grupo"] = s["grupo"].map(ETIQ)
    s["orden"] = range(len(s))
    fig = px.bar(s, x="orden", y="silueta", color="Grupo", hover_name="pais", color_discrete_map=COLOR_ETIQ,
                 hover_data={"orden": False, "grupo_vecino": True})
    fig.add_hline(y=sil["silueta"].mean(), line_dash="dash", line_color="black",
                  annotation_text=f"Promedio = {fmt(sil['silueta'].mean(), 3)}")
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(estilo(fig, "Coeficiente de silueta de cada país, ordenado dentro de su grupo",
                           "Países (pasa el cursor para ver el nombre)", "Silueta", 420))
    c1, c2 = st.columns(2)
    with c1:
        sg = sil.groupby("grupo")["silueta"].agg(["mean", "min", "count"]).rename(
            columns={"mean": "Silueta media", "min": "Mínima", "count": "Países"})
        sg.index = [ETIQ[g] for g in sg.index]
        st.dataframe(sg.round(3))
    with c2:
        frontera = sil[sil["silueta"] < 0.05].join(meta).sort_values("silueta")
        st.markdown(f"**Países frontera (silueta < 0,05): {len(frontera)}**")
        frontera = frontera.assign(**{"Grupo asignado": frontera["grupo"].map(ETIQ),
                                      "Grupo más cercano": frontera["grupo_vecino"].map(ETIQ)})
        st.dataframe(frontera[["pais", "silueta", "Grupo asignado", "Grupo más cercano"]].round(3), hide_index=True)
    nota(f"Una silueta cercana a 1 indica que el país está bien dentro de su grupo; cerca de 0, que está en la frontera "
         f"entre dos; negativa, que estaría mejor en otro. La silueta promedio de <b>{fmt(sil['silueta'].mean(), 3)}</b> "
         "indica una <b>estructura moderada</b>: hay grupos reconocibles, pero los países forman más un continuo que "
         "islas separadas. Esto es esperable en datos macroeconómicos, donde el desarrollo es gradual.")

    st.markdown("### ¿Los grupos replican la clasificación por ingreso del Banco Mundial?")
    ct = pd.crosstab(grupos.map(ETIQ), meta["grupo_ingreso"]).reindex(columns=[c_ for c_ in ORDEN_INGRESO if c_ in meta["grupo_ingreso"].unique()])
    fig = px.imshow(ct, text_auto=True, color_continuous_scale=SECUENCIAL, aspect="auto")
    fig.update_coloraxes(colorbar_title="Países")
    st.plotly_chart(estilo(fig, "Tabla cruzada: grupos de K-Means vs. grupo de ingreso oficial",
                           "Grupo de ingreso (Banco Mundial)", None, 380))
    v_c = pl.cramers_v(ct)
    nota(f"<b>V de Cramér = {fmt(v_c, 2)}</b> (0 = sin relación, 1 = correspondencia perfecta). La asociación es "
         f"{'fuerte' if v_c > 0.5 else 'moderada' if v_c > 0.3 else 'débil'}: el ingreso explica buena parte de la "
         "agrupación, pero no toda. Los grupos que mezclan niveles de ingreso son justamente el aporte de K-Means: "
         "países con ingreso parecido pero estructuras distintas (crecimiento, inversión, desempleo).", "hallazgo")

# ===========================================================================
# 6. COLOMBIA
# ===========================================================================
elif pagina == PAGINAS[5]:
    capitulo(6, "¿Dónde está Colombia?", "Su grupo, sus vecinos más cercanos y lo que la diferencia")
    if not tiene_col:
        st.error("Colombia quedó fuera de la muestra con esta configuración (revisa variables y filtros).")
        st.stop()
    g_col = grupos[PAIS_FOCO]
    s_col = sil.loc[PAIS_FOCO]
    c = st.columns(4)
    c[0].metric("Grupo de Colombia", g_col)
    c[1].metric("Países en su grupo", int((grupos == g_col).sum()))
    c[2].metric("Silueta de Colombia", fmt(s_col["silueta"], 3))
    c[3].metric("Grupo más cercano después del suyo", s_col["grupo_vecino"])

    st.markdown(f"**Perfil del grupo de Colombia:** {NOMBRE_GRUPO[g_col]}.")
    mismos = grupos[grupos == g_col].index.drop(PAIS_FOCO)
    latam = meta.index[meta["region"] == "América Latina y el Caribe"]
    st.markdown(f"**Comparte grupo con ({len(mismos)}):** {nombres_paises(mismos)}")

    st.markdown("### Los vecinos más cercanos de Colombia")
    vec = pl.vecinos(X, PAIS_FOCO, 10).rename("Distancia").to_frame().join(meta).join(grupos)
    vec["Mismo grupo"] = np.where(vec["grupo"] == g_col, "Sí", "No")
    st.dataframe(vec[["pais", "region", "grupo_ingreso", "grupo", "Mismo grupo", "Distancia"]].round(2)
                 .rename(columns={"pais": "País", "region": "Región", "grupo_ingreso": "Grupo de ingreso",
                                  "grupo": "Grupo K-Means"}), hide_index=True)
    st.caption(f"Distancia euclidiana en el espacio de los {n_comp} componentes conservados.")
    n_lat = vec["region"].eq("América Latina y el Caribe").sum()
    nota(f"De los 10 países más parecidos a Colombia, <b>{n_lat}</b> son latinoamericanos. "
         f"El más cercano es <b>{vec['pais'].iloc[0]}</b>. Colombia "
         f"{'está bien ubicada en su grupo' if s_col['silueta'] > 0.25 else 'está cerca de la frontera con otro grupo'} "
         f"(silueta {fmt(s_col['silueta'], 2)}); el grupo alternativo más cercano es el "
         f"<b>{ETIQ[s_col['grupo_vecino']]}</b>.")

    st.markdown("### Colombia frente a su grupo, América Latina y el mundo")
    comp = pd.DataFrame({
        "Colombia": prep.originales.loc[PAIS_FOCO],
        f"Mediana {g_col}": prep.originales.loc[grupos[grupos == g_col].index].median(),
        "Mediana América Latina": prep.originales.loc[prep.originales.index.intersection(latam)].median(),
        "Mediana mundo (muestra)": prep.originales.median(),
    })
    comp.index = [nombre(i) for i in comp.index]
    st.dataframe(comp.round(1))
    zc = pd.DataFrame({
        "Colombia": prep.estandarizadas.loc[PAIS_FOCO],
        f"Promedio {g_col}": prep.estandarizadas.loc[grupos[grupos == g_col].index].mean(),
        "Promedio América Latina": prep.estandarizadas.loc[prep.estandarizadas.index.intersection(latam)].mean(),
    })
    zc.index = [nombre(i) for i in zc.index]
    zl = zc.reset_index().melt(id_vars="index", var_name="Serie", value_name="z")
    fig = px.bar(zl, x="z", y="index", color="Serie", barmode="group", orientation="h",
                 color_discrete_sequence=["#1e2a33", COLOR_GRUPO[g_col], "#c58b2b"])
    fig.add_vline(x=0, line_color="black")
    st.plotly_chart(estilo(fig, "Perfil estandarizado (0 = media de la muestra)",
                           "Desviaciones estándar respecto a la media mundial", None, 560))
    dif = (zc["Colombia"] - zc[f"Promedio {g_col}"]).sort_values(key=abs, ascending=False)
    rasgos = ", ".join(f"{i} ({'más alto' if d > 0 else 'más bajo'})" for i, d in dif.head(3).items())
    nota(f"<b>Lo que diferencia a Colombia de su propio grupo:</b> {rasgos}.", "hallazgo")
    if ES_DEFECTO:
        st.markdown("""
**Lectura (configuración de referencia, 2015–2019)**
- Colombia pertenece al grupo de **ingreso medio con desempleo alto y bajo crecimiento**, junto con México, Brasil, Perú,
  Argentina, Uruguay, Ecuador y Paraguay, pero también con **Turquía, Kazajistán, Bielorrusia y Sudáfrica**. Su vecino más
  cercano no es latinoamericano: es **Turquía**, seguido de México.
- **El parecido no lo explica la región:** Chile y Costa Rica quedan en el grupo de ingreso alto, y Bolivia, Panamá,
  República Dominicana, Honduras y Nicaragua en el de economías que invierten con déficit externo. América Latina se reparte
  en tres grupos distintos.
- **Su rasgo más distintivo es la baja apertura comercial:** el comercio exterior equivale a ≈ 37 % del PIB, muy por debajo de
  la mediana de su grupo (≈ 74 %) y de la mundial. Colombia es una de las economías más cerradas de la muestra, algo que su
  grupo no comparte.
- También tiene **urbanización y esperanza de vida algo superiores** a sus pares. Su desempleo (≈ 9 %) es alto frente al
  mundo, aunque menor que la mediana de su grupo (≈ 11 %), y mantiene un déficit en cuenta corriente (≈ −4,6 % del PIB),
  propio de una economía que depende del financiamiento externo.
- Con silueta ≈ 0,31, Colombia está razonablemente bien ubicada; el grupo alternativo más cercano es el de economías que
  invierten con déficit externo, no el de ingreso alto.
""")

    st.markdown("### América Latina en los grupos")
    lat = meta.loc[meta.index.intersection(latam)].join(grupos)
    tab = lat.groupby("grupo")["pais"].apply(lambda s: ", ".join(sorted(s))).rename("Países").to_frame()
    tab.insert(0, "N.º", lat.groupby("grupo").size())
    tab.index = [ETIQ[g] for g in tab.index]
    st.dataframe(tab)
    pts = X.join(meta).join(grupos)
    pts = pts[pts.index.isin(latam)]
    pts["Grupo"] = pts["grupo"].map(ETIQ)
    fig = px.scatter(pts, x="CP1", y="CP2", color="Grupo", text="pais", color_discrete_map=COLOR_ETIQ)
    fig.update_traces(textposition="top center", marker=dict(size=11))
    st.plotly_chart(estilo(fig, "Países latinoamericanos en el plano CP1–CP2, coloreados por grupo",
                           "CP1 · nivel de desarrollo", "CP2", 560))

# ===========================================================================
# 7. ROBUSTEZ
# ===========================================================================
elif pagina == PAGINAS[6]:
    capitulo(7, "Robustez", "Qué pasa si cambian el periodo, las transformaciones o K")
    st.markdown("Se repite el análisis cambiando una decisión a la vez para ver si las conclusiones se sostienen.")

    st.markdown("### a) Cambiar el periodo: antes vs. después de la pandemia")
    otro = [p for p in PERIODOS if p != periodo][0]
    _, prep2, acp2 = analizar(otro, tuple(variables), excluir, transf, imputar)
    X2 = acp2.puntajes.iloc[:, :min(n_comp, len(prep2.variables))]
    g2, sil2, _ = segmentar(X2, k)
    _, z2 = pl.perfiles(prep2.originales, prep2.estandarizadas, g2)
    et2 = {g: f"{g} · {pl.nombre_descriptivo(z2.loc[g])}" for g in z2.index}
    ari = pl.comparar_particiones(grupos, g2)
    c = st.columns(4)
    c[0].metric("Países en ambos periodos", len(grupos.index.intersection(g2.index)))
    c[1].metric("Índice de Rand ajustado (ARI)", fmt(ari, 2), help="1 = misma partición; 0 = coincidencia al azar")
    c[2].metric(f"Varianza CP1 · {periodo[:9]}", f"{fmt(acp.varianza['% varianza'].iloc[0])} %")
    c[3].metric(f"Varianza CP1 · {otro[:9]}", f"{fmt(acp2.varianza['% varianza'].iloc[0])} %")
    comunes = grupos.index.intersection(g2.index)
    tr = pd.crosstab(grupos.loc[comunes].map(ETIQ), g2.loc[comunes].map(et2))
    fig = px.imshow(tr, text_auto=True, color_continuous_scale=SECUENCIAL, aspect="auto")
    fig.update_coloraxes(colorbar_title="Países")
    st.plotly_chart(estilo(fig, f"Transiciones de grupo: {periodo[:9]} (filas) → {otro[:9]} (columnas)", None, None, 420))
    st.caption("Las letras se asignan por tamaño en cada corrida, así que una misma letra no significa lo mismo en los dos "
               "periodos: compara por el perfil descriptivo.")
    if PAIS_FOCO in g2.index and tiene_col:
        socios1 = set(grupos[grupos == grupos[PAIS_FOCO]].index)
        socios2 = set(g2[g2 == g2[PAIS_FOCO]].index)
        estables = (socios1 & socios2) - {PAIS_FOCO}
        nota(f"Colombia está en <b>{ETIQ[grupos[PAIS_FOCO]]}</b> en {periodo[:9]} y en <b>{et2[g2[PAIS_FOCO]]}</b> en "
             f"{otro[:9]}. Países que la acompañan en ambos periodos ({len(estables)}): {nombres_paises(sorted(estables))}.")
    if ES_DEFECTO:
        nota("<b>Lectura.</b> El ARI cercano a 0,36 indica que la partición cambia de forma importante entre periodos. Pero el "
             "cambio tiene una explicación económica: en 2020–2024 aparece un grupo pequeño de <b>crisis inflacionaria</b> "
             "(Argentina, Líbano, Sudán, Zimbabue, Haití) que, por sus valores extremos, \"gasta\" uno de los cuatro grupos, y el "
             "de ingreso alto absorbe a buena parte del ingreso medio, Colombia incluida. Si en 2020–2024 se usa K = 5 (prueba "
             "moviendo el control), vuelve a aparecer un grupo de ingreso medio separado del de ingreso alto, donde queda Colombia "
             "con Brasil, Perú, México y Paraguay. El <b>núcleo latinoamericano de Colombia se mantiene</b>; lo que cambia es "
             "cuántos grupos hacen falta para verlo.", "hallazgo")

    st.markdown("### b) Sin transformaciones (log / winsorización)")
    _, prep3, acp3 = analizar(periodo, tuple(variables), excluir, not transf, imputar)
    X3 = acp3.puntajes.iloc[:, :min(n_comp, len(prep3.variables))]
    g3, sil3, _ = segmentar(X3, k)
    c = st.columns(3)
    c[0].metric("ARI con la versión actual", fmt(pl.comparar_particiones(grupos, g3), 2))
    c[1].metric("Tamaños de grupo (actual)", " / ".join(str(x) for x in grupos.value_counts().values))
    c[2].metric(f"Tamaños {'sin' if transf else 'con'} transformación", " / ".join(str(x) for x in g3.value_counts().values))
    peq = g3.value_counts()
    peq = peq[peq <= 5]
    if len(peq):
        st.markdown(f"**Grupos diminutos {'sin' if transf else 'con'} transformación:** " +
                    "; ".join(f"{g} ({nombres_paises(g3[g3 == g].index)})" for g in peq.index))
    nota("Sin transformar, unos pocos países extremos (en 2015–2019, economías de enorme apertura e IED: Singapur, "
         "Irlanda, Chipre y Yibuti) forman un grupo propio y el resto queda agrupado de forma más gruesa. Las "
         "transformaciones evitan que el modelo dedique un grupo entero a explicar atípicos, a costa de suavizar "
         "diferencias reales en los extremos.")

    st.markdown("### c) Sensibilidad a K: ¿quién acompaña siempre a Colombia?")
    if tiene_col:
        filas, conjuntos = [], []
        for kk in range(3, 7):
            gk, _, _ = segmentar(X, kk)
            socios = set(gk[gk == gk[PAIS_FOCO]].index) - {PAIS_FOCO}
            conjuntos.append(socios)
            filas.append({"K": kk, "Tamaño del grupo de Colombia": len(socios) + 1,
                          "Silueta promedio": float(tk.loc[tk.K == kk, "Silueta"].iloc[0])})
        nucleo = set.intersection(*conjuntos)
        st.dataframe(pd.DataFrame(filas).round(3), hide_index=True)
        nota(f"<b>Núcleo estable:</b> {len(nucleo)} países comparten grupo con Colombia para todo K entre 3 y 6: "
             f"{nombres_paises(sorted(nucleo))}. Son los socios de perfil más confiables, porque no dependen del K elegido.",
             "hallazgo")

# ===========================================================================
# 8. CONCLUSIONES
# ===========================================================================
elif pagina == PAGINAS[7]:
    capitulo(8, "Discusión y conclusiones", "Qué aprendimos de los datos y con qué cautelas leerlo")
    aviso_config()
    cp1_pos, _ = pl.describir_componente(acp.cargas["CP1"])
    cp2_pos, cp2_neg = pl.describir_componente(acp.cargas["CP2"])
    v_c = pl.cramers_v(pd.crosstab(grupos, meta["grupo_ingreso"]))
    frontera = (sil["silueta"] < 0.05).sum()

    st.markdown("### 1. ¿Qué variables definen los componentes principales?")
    st.markdown(f"""
El **CP1** ({fmt(acp.varianza['% varianza'].iloc[0])} % de la varianza) lo definen
{", ".join(a for a, _ in cp1_pos)}: es un eje de **nivel de desarrollo**. El **CP2**
({fmt(acp.varianza['% varianza'].iloc[1])} %) opone {", ".join(a for a, _ in cp2_pos) or "—"} a
{", ".join(a for a, _ in cp2_neg) or "—"}: separa a las economías **dinámicas que invierten** de las **estancadas con
desequilibrios**, con independencia de su ingreso. Los CP3 y CP4 recogen la apertura con desempleo y la inversión financiada
con déficit externo. La conclusión es que **el ingreso es la dimensión dominante, pero no la única**: más de la mitad de la
variación entre países tiene que ver con dinamismo, estabilidad y sector externo.
""")
    st.markdown("### 2. ¿Qué proporción de la información se conservó?")
    st.markdown(f"""
Con **{n_comp} componentes** se conserva el **{fmt(var_ret)} %** de la varianza de las {n_var} variables estandarizadas
(criterio de Kaiser, por encima del 70 %). El plano CP1–CP2 que se usa para graficar resume solo el
{fmt(acp.varianza['% varianza'].iloc[:2].sum())} %, por eso los grupos se calcularon con todos los componentes conservados y
algunos se superponen en el gráfico.
""")
    st.markdown("### 3. ¿Qué diferencias reales hay entre los grupos?")
    filas = []
    for g in med_grupo.index:
        m = med_grupo.loc[g]
        filas.append(f"- **{ETIQ[g]}** ({int(m['N.º países'])} países): PIB per cápita mediano US$ {fmt(m.get('pib_pc', np.nan), 0)}"
                     + (f", crecimiento {fmt(m['crec_pib'])} %" if 'crec_pib' in m else "")
                     + (f", desempleo {fmt(m['desempleo'])} %" if 'desempleo' in m else "")
                     + (f", inflación {fmt(m['inflacion'])} %" if 'inflacion' in m else "")
                     + (f", formación de capital {fmt(m['formacion_capital'])} % del PIB" if 'formacion_capital' in m else "") + ".")
    st.markdown("\n".join(filas))
    st.markdown(f"""
Las diferencias más grandes se dan en desarrollo humano y profundidad financiera (ingreso alto frente a ingreso bajo), pero
**dos de los grupos no se ordenan por ingreso**: uno se define por el desempleo alto y el bajo crecimiento, y otro por la
inversión alta con déficit externo. La relación con la clasificación oficial es **fuerte pero imperfecta** (V de Cramér =
{fmt(v_c, 2)}): K-Means aporta información que el ingreso por sí solo no da.
""")
    st.markdown("### 4. ¿Son grupos claramente distinguibles?")
    st.markdown(f"""
**Parcialmente.** La silueta promedio es **{fmt(sil['silueta'].mean(), 3)}**: hay estructura, pero los países forman un
**continuo** más que grupos aislados. Los grupos extremos (ingreso alto e ingreso bajo) son los más compactos; los
intermedios son los más difusos, y hay **{frontera} países frontera** (silueta < 0,05) que podrían estar en dos grupos. Que la
silueta sea máxima con K = 2 y casi plana entre 3 y 6 confirma que no hay un número "natural" de grupos: la elección de K
es una decisión analítica, justificada por la interpretabilidad. Los resultados también cambian con el periodo (sección 7),
sobre todo por los episodios de alta inflación de 2020–2024.
""")
    st.markdown("### 5. Respuesta a la pregunta: ¿a quién se parece Colombia?")
    if tiene_col:
        vec = pl.vecinos(X, PAIS_FOCO, 5)
        st.markdown(f"""
Colombia queda en el **{ETIQ[grupos[PAIS_FOCO]]}**. Sus cinco vecinos más cercanos son
**{nombres_paises(vec.index)}**.

- Colombia **se parece más a las grandes economías de ingreso medio con mercado laboral débil** (México, Brasil, Perú,
  Turquía, Sudáfrica) **que a una región**: los vecinos de perfil cruzan continentes, y América Latina se reparte en tres grupos.
- **Chile y Costa Rica ya no son pares de Colombia** en este análisis: comparten el perfil de las economías de ingreso alto.
- El rasgo que **más distingue** a Colombia, incluso dentro de su grupo, es su **baja apertura comercial**, junto con un
  desempleo alto y un déficit externo persistente.
- **Implicación:** para comparar políticas (empleo, comercio, financiamiento externo), los referentes naturales de Colombia
  son México, Perú, Brasil y Turquía, más que el promedio de América Latina o la OCDE. Y la brecha con el grupo de ingreso
  alto no es solo de ingreso: es de profundidad financiera, inflación y equilibrio externo.
""")
    st.markdown("### Limitaciones")
    st.markdown("""
1. **Sesgo de selección por faltantes:** los países con peores estadísticas (en conflicto o muy pobres) quedan fuera, así que
   los grupos de menor ingreso están subrepresentados.
2. **Grupos sensibles a las decisiones:** el periodo, las transformaciones y K cambian una parte de las asignaciones (ARI
   entre periodos ≈ 0,36). Los resultados deben leerse como **perfiles aproximados**, no como una clasificación definitiva;
   los países frontera, en particular, no deberían rotularse con un solo grupo.
3. **Ausencia de variables fiscales e institucionales:** deuda pública, recaudo, calidad institucional o desigualdad no
   pudieron incluirse y podrían cambiar los grupos.
4. **Promedios y causalidad:** el análisis es descriptivo; muestra parecidos, no por qué ocurren, y los promedios de cinco
   años ocultan trayectorias dentro del periodo.
5. **Imputación:** a los países con un solo dato faltante se les asignó la mediana de su grupo de ingreso, lo que puede
   acercarlos artificialmente al centro de ese grupo.
""")

st.markdown('<div class="pie">Giuliano Stiven Vera Campo · Machine Learning aplicado a Economía y Finanzas · '
            'Pontificia Universidad Javeriana Cali · Datos: World Development Indicators, Banco Mundial.</div>',
            unsafe_allow_html=True)
