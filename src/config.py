"""Configuración central del proyecto: variables, periodos y etiquetas.

Cada variable del análisis proviene de los World Development Indicators (WDI)
del Banco Mundial. Se guarda el código oficial del WDI para que cualquier dato
pueda verificarse en https://data.worldbank.org/indicator/<CODIGO>.
"""

# clave interna -> (código WDI, nombre corto, descripción, dimensión temática)
VARIABLES = {
    "pib_pc": ("NY.GDP.PCAP.KD", "PIB per cápita", "PIB per cápita (US$ constantes de 2015)", "Ingreso"),
    "crec_pib": ("NY.GDP.MKTP.KD.ZG", "Crecimiento del PIB", "Crecimiento del PIB real (% anual)", "Dinamismo"),
    "inflacion": ("FP.CPI.TOTL.ZG", "Inflación", "Inflación, precios al consumidor (% anual)", "Estabilidad"),
    "desempleo": ("SL.UEM.TOTL.ZS", "Desempleo", "Desempleo (% de la fuerza laboral, estimación OIT)", "Mercado laboral"),
    "comercio": ("NE.TRD.GNFS.ZS", "Comercio exterior", "Exportaciones + importaciones (% del PIB)", "Apertura"),
    "ied": ("BX.KLT.DINV.WD.GD.ZS", "IED", "Inversión extranjera directa, entradas netas (% del PIB)", "Apertura"),
    "cuenta_corriente": ("BN.CAB.XOKA.GD.ZS", "Cuenta corriente", "Saldo de la cuenta corriente (% del PIB)", "Sector externo"),
    "formacion_capital": ("NE.GDI.TOTL.ZS", "Formación de capital", "Formación bruta de capital (% del PIB)", "Inversión"),
    "credito_privado": ("FS.AST.PRVT.GD.ZS", "Crédito privado", "Crédito interno al sector privado (% del PIB)", "Profundidad financiera"),
    "esperanza_vida": ("SP.DYN.LE00.IN", "Esperanza de vida", "Esperanza de vida al nacer (años)", "Desarrollo humano"),
    "urbanizacion": ("SP.URB.TOTL.IN.ZS", "Urbanización", "Población urbana (% del total)", "Desarrollo humano"),
    "internet": ("IT.NET.USER.ZS", "Uso de internet", "Personas que usan internet (% de la población)", "Desarrollo humano"),
    "gasto_salud": ("SH.XPD.CHEX.GD.ZS", "Gasto en salud", "Gasto corriente en salud (% del PIB)", "Gasto social"),
    "gasto_educacion": ("SE.XPD.TOTL.GD.ZS", "Gasto en educación", "Gasto público en educación (% del PIB)", "Gasto social"),
    "deuda_publica": ("GC.DOD.TOTL.GD.ZS", "Deuda pública", "Deuda del gobierno central (% del PIB)", "Finanzas públicas"),
    "recaudo": ("GC.TAX.TOTL.GD.ZS", "Recaudo tributario", "Ingresos tributarios (% del PIB)", "Finanzas públicas"),
}

# Variables propuestas por defecto para el modelo (el usuario puede cambiarlas en la app)
VARIABLES_DEFECTO = [
    "pib_pc", "crec_pib", "inflacion", "desempleo", "comercio", "ied",
    "cuenta_corriente", "formacion_capital", "credito_privado",
    "esperanza_vida", "urbanizacion", "internet",
]

# Transformaciones sugeridas para variables muy asimétricas
#   "log": logaritmo natural (variables estrictamente positivas y sesgadas a la derecha)
#   "winsor": recorte en percentiles 2,5 y 97,5 (variables con colas extremas en ambos lados)
TRANSFORMACIONES = {
    "pib_pc": "log",
    "inflacion": "winsor",
    "comercio": "log",
    "ied": "winsor",
    "cuenta_corriente": "winsor",
    "credito_privado": "log",
    "crec_pib": "winsor",
}

PERIODOS = {
    "2015–2019 (antes de la pandemia)": (2015, 2019),
    "2020–2024 (pandemia y recuperación)": (2020, 2024),
}
PERIODO_DEFECTO = "2015–2019 (antes de la pandemia)"

UMBRAL_FALTANTES = 0.25   # una variable con más de 25 % de países sin dato se excluye
SEMILLA = 42

INGRESO_ES = {
    "low_income": "Ingreso bajo",
    "lower_middle_income": "Ingreso medio-bajo",
    "upper_middle_income": "Ingreso medio-alto",
    "high_income": "Ingreso alto",
}
ORDEN_INGRESO = ["Ingreso bajo", "Ingreso medio-bajo", "Ingreso medio-alto", "Ingreso alto"]

REGION_ES = {
    "un_latin_america_and_the_caribbean": "América Latina y el Caribe",
    "un_europe_and_northern_america": "Europa y Norteamérica",
    "un_sub_saharan_africa": "África subsahariana",
    "un_northern_africa_and_western_asia": "Norte de África y Asia occidental",
    "un_central_and_southern_asia": "Asia central y meridional",
    "un_eastern_and_south_eastern_asia": "Asia oriental y sudoriental",
    "un_australia_and_new_zealand": "Oceanía",
    "un_oceania_exc_australia_and_new_zealand": "Oceanía",
}

PAIS_FOCO = "COL"


def nombre(var: str) -> str:
    return VARIABLES[var][1]


def descripcion(var: str) -> str:
    return VARIABLES[var][2]
