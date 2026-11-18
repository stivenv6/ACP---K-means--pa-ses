# ¿A quién se parece Colombia? Perfiles macroeconómicos con ACP y K-Means

**Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali**
Autor: Giuliano Stiven Vera Campo

Aplicación desplegada: `https://<tu-app>.streamlit.app` *(reemplazar por el enlace de Streamlit Community Cloud)*

Cuaderno paso a paso: [`analisis_paso_a_paso.ipynb`](analisis_paso_a_paso.ipynb) ·
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stivenv6/ACP---K-means--pa-ses/blob/main/analisis_paso_a_paso.ipynb)

---

## 1. Contexto y pregunta

La clasificación oficial del Banco Mundial ordena a los países **solo por su ingreso** (INB per cápita). Pero dos
economías con ingreso parecido pueden ser muy distintas: una puede crecer invirtiendo, otra estar estancada con
desempleo alto, otra vivir una crisis inflacionaria. Este proyecto busca responder:

> **¿Con qué economías comparte Colombia su perfil macroeconómico y de desarrollo, y ese parecido coincide con su
> región o con su nivel de ingreso?**

Preguntas que orientan el ejercicio:

1. ¿Qué dimensiones resumen las diferencias entre las economías del mundo? ¿Basta con el ingreso?
2. ¿Existen grupos de países con perfiles similares y qué los caracteriza?
3. ¿Los grupos replican la clasificación por ingreso del Banco Mundial?
4. ¿En qué grupo queda Colombia, con quién lo comparte y qué la diferencia de sus pares?

**Unidad de observación:** cada fila es un país (Estado de la ONU), con el promedio de cada indicador en el
periodo de análisis (2015–2019 por defecto; 2020–2024 como prueba de robustez).

### ¿Por qué ACP y K-Means?

- **ACP:** los indicadores de desarrollo están muy correlacionados (PIB per cápita, internet, esperanza de vida y
  urbanización tienen correlaciones de 0,8–0,9). El ACP los resume en pocas dimensiones no correlacionadas e
  interpretables, permite visualizar a más de 130 países en un plano y deja ver si la economía de un país se resume
  en una sola cifra (el ingreso) o si hay otras dimensiones relevantes.
- **K-Means:** agrupa a los países por su **perfil completo** y no solo por el ingreso. Permite ver si hay países de
  ingreso parecido con estructuras distintas e identificar los pares reales de Colombia. Se aplica sobre los
  componentes del ACP para que cada dimensión pese por la información que aporta y no por cuántas variables
  redundantes la miden.

## 2. Datos

| | |
|---|---|
| **Fuente** | *World Development Indicators* (WDI), Banco Mundial, en la copia pública y versionada que mantiene la Fundación Gapminder en GitHub: [`open-numbers/ddf--open_numbers--world_development_indicators`](https://github.com/open-numbers/ddf--open_numbers--world_development_indicators), que replica los datos del Banco Mundial. Licencia CC BY 4.0. |
| **Cobertura** | 195 Estados de la ONU (193 miembros más Palestina y la Santa Sede, observadores), panel anual 2010–2025 (3.104 filas país-año). Excluye territorios dependientes y agregados regionales. |
| **Archivos** | `data/wdi_panel_paises.csv` (panel país-año) y `data/paises_metadata.csv` (nombre en español, región ONU, grupo de ingreso). |
| **Cómo regenerarlos** | `python scripts/descargar_datos.py` |

Cada valor puede verificarse en `https://data.worldbank.org/indicator/<CÓDIGO>`.

| Variable | Código WDI | ¿En el modelo? | Transformación |
|---|---|---|---|
| PIB per cápita (US$ constantes 2015) | NY.GDP.PCAP.KD | Sí | Logaritmo |
| Crecimiento del PIB real (% anual) | NY.GDP.MKTP.KD.ZG | Sí | Winsorización |
| Inflación, IPC (% anual) | FP.CPI.TOTL.ZG | Sí | Winsorización |
| Desempleo (% fuerza laboral, OIT) | SL.UEM.TOTL.ZS | Sí | — |
| Comercio exterior (% PIB) | NE.TRD.GNFS.ZS | Sí | Logaritmo |
| IED, entradas netas (% PIB) | BX.KLT.DINV.WD.GD.ZS | Sí | Winsorización |
| Cuenta corriente (% PIB) | BN.CAB.XOKA.GD.ZS | Sí | Winsorización |
| Formación bruta de capital (% PIB) | NE.GDI.TOTL.ZS | Sí | — |
| Crédito al sector privado (% PIB) | FS.AST.PRVT.GD.ZS | Sí | Logaritmo |
| Esperanza de vida (años) | SP.DYN.LE00.IN | Sí | — |
| Población urbana (%) | SP.URB.TOTL.IN.ZS | Sí | — |
| Uso de internet (% población) | IT.NET.USER.ZS | Sí | — |
| Gasto en salud (% PIB) | SH.XPD.CHEX.GD.ZS | Opcional | — |
| Gasto en educación (% PIB) | SE.XPD.TOTL.GD.ZS | Opcional | — |
| Deuda pública (% PIB) | GC.DOD.TOTL.GD.ZS | No: 73 % faltante | — |
| Recaudo tributario (% PIB) | GC.TAX.TOTL.GD.ZS | Opcional (23 % faltante, cerca del umbral) | — |
| Población | SP.POP.TOTL | Solo para filtro de tamaño | — |

**Limitaciones de la fuente:** faltantes no aleatorios (los países en conflicto o con estadísticas débiles quedan
fuera), diferencias metodológicas nacionales (desempleo, IPC), revisiones periódicas de las series y ausencia de
variables fiscales con cobertura suficiente.

## 3. Metodología

Todo el procesamiento está en `src/pipeline.py`; la aplicación lo ejecuta en vivo (no hay resultados precalculados) y el cuaderno `analisis_paso_a_paso.ipynb` lo recorre paso a paso.

1. **Agregación:** promedio por país del periodo elegido; un valor solo se calcula con al menos 3 años con dato.
2. **Calidad de datos:** se verificó que no haya duplicados país-año; identificadores (ISO, nombre, región, ingreso) fuera del modelo.
3. **Filtro de tamaño:** se excluyen los países con menos de 1 millón de habitantes (37), cuyos perfiles extremos de
   comercio, IED o turismo distorsionan las distancias.
4. **Faltantes:** se excluye toda variable con más del 25 % de países sin dato (la deuda pública, con 73 %). A los países
   con **un solo** faltante se les imputa la mediana de su grupo de ingreso (12 países, entre ellos Argentina y
   Canadá); los países con dos o más faltantes se excluyen (20, entre ellos Venezuela, Siria, Yemen y Corea del Norte).
   **Muestra final: 138 países.**
5. **Transformaciones:** logaritmo para variables positivas muy sesgadas y winsorización 2,5–97,5 % para colas extremas.
6. **Estandarización** (media 0, desviación 1), necesaria porque ACP y K-Means dependen de escalas y distancias.
7. **ACP** (`sklearn.decomposition.PCA`): varianza explicada, gráfico de sedimentación y cargas expresadas como
   correlaciones. El número de componentes se elige con el criterio de Kaiser (valor propio > 1) y se verifica que
   supere el 70 % de varianza acumulada.
8. **K-Means** (`sklearn.cluster.KMeans`, `n_init=50`, semilla 42) sobre los componentes conservados. Se compara K = 2…10
   con el método del codo y el coeficiente de silueta. Los grupos se describen con medianas de las variables
   originales y se contrastan con la clasificación de ingreso del Banco Mundial (V de Cramér).
9. **Robustez:** se repite el análisis con el periodo 2020–2024, sin transformaciones y con K entre 3 y 6 (índice de
   Rand ajustado y "núcleo estable" de países que siempre acompañan a Colombia).

## 4. Resultados principales (configuración de referencia: 2015–2019, 12 variables, 4 componentes, K = 4)

Se reproducen con `python scripts/resultados.py` o ejecutando el cuaderno.

**ACP.** Cuatro componentes tienen valor propio mayor que 1 y conservan el **74,5 %** de la varianza.

| Componente | % varianza | Interpretación (cargas principales) |
|---|---|---|
| CP1 | 41,6 % | **Nivel de desarrollo:** PIB per cápita (0,96), internet (0,95), esperanza de vida (0,90), crédito privado (0,84), urbanización (0,83) |
| CP2 | 14,6 % | **Dinamismo inversor frente a inestabilidad:** crecimiento (0,67), IED (0,61), formación de capital (0,57) frente a desempleo (−0,42) e inflación (−0,38) |
| CP3 | 10,0 % | **Apertura con desempleo:** desempleo (0,66), comercio (0,56), IED (0,54) |
| CP4 | 8,4 % | **Inversión con déficit externo:** formación de capital (0,65) frente a cuenta corriente (−0,59) |

El ingreso es la dimensión dominante, pero solo explica el 42 %: más de la mitad de la variación entre países tiene
que ver con dinamismo, estabilidad y sector externo.

**K-Means.** La silueta es máxima con K = 2 (0,37), pero esa partición solo separa a los países ricos del resto. Entre
K = 3 y 6 la silueta es casi plana (0,27–0,28) y el codo se ubica entre 4 y 6. Se elige **K = 4**, el menor K que
separa un grupo definido por una dimensión distinta del ingreso.

| Grupo | Países | PIB pc mediano (US$) | Rasgos | Ejemplos |
|---|---|---|---|---|
| A · Ingreso alto | 44 | 31.700 | Esperanza de vida 81 años, crédito 85 % PIB, inflación 1 %, superávit externo | EE. UU., Alemania, Japón, **Chile, Costa Rica** |
| B · Ingreso medio, desempleo alto y bajo crecimiento | 37 | 6.150 | Desempleo 11,4 %, crecimiento 2,2 %, inversión baja | **Colombia**, México, Brasil, Perú, Argentina, Turquía, Sudáfrica |
| C · Ingreso bajo en expansión | 33 | 1.000 | Crecimiento 4,9 %, internet 17 %, crédito 14 % PIB | India, Kenia, Etiopía, Bangladés |
| D · Inversión alta con déficit externo | 24 | 3.200 | Formación de capital 31 % PIB, déficit externo −5,6 % | China, Vietnam, Indonesia, Panamá, Bolivia |

- **Silueta promedio: 0,267.** La estructura es moderada: los países forman un continuo más que islas separadas. Los
  grupos extremos (A y C) son compactos; los intermedios (B y D) son difusos.
- **V de Cramér con la clasificación por ingreso: 0,68.** La relación es fuerte pero imperfecta: el grupo D mezcla
  todos los niveles de ingreso.

**Colombia** queda en el grupo B (silueta 0,31). Sus vecinos más cercanos son **Turquía, México, Bielorrusia,
Kazajistán, Perú y Ecuador**. Se parece más a las grandes economías de ingreso medio con mercado laboral débil que a
"América Latina" (que se reparte en tres grupos). Su rasgo más distintivo, incluso frente a su grupo, es la **baja
apertura comercial** (comercio ≈ 37 % del PIB frente a ≈ 74 % de la mediana de su grupo).

**Robustez.** Entre periodos el índice de Rand ajustado es 0,36: en 2020–2024 aparece un grupo de crisis
inflacionaria (Argentina, Líbano, Sudán, Zimbabue, Haití) que absorbe uno de los cuatro grupos. Aun así, 26 países
acompañan a Colombia para todo K entre 3 y 6 (entre ellos México, Brasil, Perú, Ecuador, Paraguay, Uruguay y Turquía).

**Limitaciones:** sesgo de selección por faltantes, sensibilidad de los grupos a las decisiones (periodo, K,
transformaciones), ausencia de variables fiscales e institucionales, carácter descriptivo (no causal) e imputación
de un dato en 12 países.

## 5. Estructura del repositorio

```
├── app.py                     # Aplicación Streamlit (8 secciones)
├── analisis_paso_a_paso.ipynb # Cuaderno con todo el análisis ejecutado (abre en GitHub o Colab)
├── src/
│   ├── config.py              # Variables, códigos WDI, transformaciones y parámetros
│   └── pipeline.py            # Preparación, ACP, K-Means, silueta, perfiles, robustez
├── scripts/
│   ├── descargar_datos.py     # Descarga y arma la base desde la fuente
│   └── resultados.py          # Reproduce en consola los resultados del README
├── data/
│   ├── wdi_panel_paises.csv   # Panel país-año 2010–2025
│   └── paises_metadata.csv    # Nombre, región y grupo de ingreso
├── .streamlit/config.toml     # Tema visual
└── requirements.txt
```

## 6. Cómo ejecutar el proyecto

```bash
git clone https://github.com/stivenv6/ACP---K-means--pa-ses.git
cd ACP---K-means--pa-ses
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/descargar_datos.py   # opcional: regenera data/ desde la fuente
python scripts/resultados.py        # opcional: resultados en consola
streamlit run app.py
# opcional: pip install jupyter && jupyter notebook analisis_paso_a_paso.ipynb
```

**Cuaderno:** `analisis_paso_a_paso.ipynb` reproduce todo el análisis con gráficos estáticos y ya incluye las salidas,
así que puede leerse directamente en GitHub. En Colab, la primera celda clona el repositorio e instala las dependencias
(si cambias el nombre del repositorio, actualiza `REPO_URL` en esa celda y el enlace del botón *Open in Colab*).

**Despliegue:** en [share.streamlit.io](https://share.streamlit.io) → *Create app* → elegir el repositorio, rama `main`
y archivo principal `app.py`.

## 7. Secciones de la aplicación

1. **Contexto y pregunta:** problema, justificación de los métodos, fuente, variables y limitaciones.
2. **Exploración:** estructura, tipos, descriptivas, faltantes por variable y región, distribuciones y correlaciones.
3. **Preparación:** cada decisión de limpieza con su efecto, países excluidos e imputados, transformaciones y estandarización.
4. **ACP:** varianza, sedimentación, cargas, interpretación de componentes y biplot con Colombia resaltada.
5. **K-Means:** codo, silueta, grupos en el plano y en el mapa, perfiles en variables originales, silueta por país,
   países frontera y comparación con la clasificación por ingreso.
6. **¿Dónde está Colombia?:** grupo, vecinos más cercanos, comparación con su grupo y con América Latina.
7. **Robustez:** cambio de periodo, sin transformaciones y sensibilidad a K.
8. **Discusión y conclusiones.**

Desde la barra lateral se pueden cambiar el periodo, las variables, los filtros, las transformaciones, el número de
componentes y K; todo se recalcula.

## 8. Uso de herramientas de IA

Se usaron herramientas de inteligencia artificial como apoyo en el desarrollo del código y la redacción. Las
decisiones metodológicas, el código y los resultados fueron revisados y pueden sustentarse.
