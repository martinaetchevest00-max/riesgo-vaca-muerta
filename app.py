"""
Mapa de riesgo por lluvias y ríos — Vaca Muerta
================================================
App web en Streamlit. Para correrla (desde la carpeta donde está este archivo):

    streamlit run app.py

Datos:
- Pozos: pozos_vaca_muerta.csv (lo genera preparar_datos.py desde NoConvencional.xlsx)
- Lluvia: Open-Meteo (pronóstico hasta 15 días adelante, o histórico)
- Caudal de ríos: Open-Meteo Flood API (modelo GloFAS)
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Riesgo hídrico · Vaca Muerta", page_icon="🌧️", layout="wide")

CARPETA = Path(__file__).parent
ARCHIVO_POZOS = CARPETA / "pozos_vaca_muerta.csv"
TAMANIO_CELDA = 0.25  # grados (~25 km): se pide la lluvia por celda, no por pozo

URL_PRONOSTICO = "https://api.open-meteo.com/v1/forecast"
URL_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
URL_CAUDAL = "https://flood-api.open-meteo.com/v1/flood"

# Niveles de riesgo según lluvia acumulada (mm) en la ventana elegida.
# La zona es árida (~150-200 mm/año), así que 25 mm en pocos días ya corta caminos de tierra.
NIVELES = [
    ("Bajo", 0, 10, [46, 160, 67]),
    ("Moderado", 10, 25, [240, 190, 40]),
    ("Alto", 25, 50, [240, 120, 30]),
    ("Muy alto", 50, 1e9, [200, 30, 30]),
]

# Trazados APROXIMADOS de los ríos principales (lat, lon)
RIOS = {
    "Río Neuquén": [(-37.38, -70.27), (-37.95, -69.85), (-38.20, -69.35), (-38.35, -68.95),
                    (-38.40, -68.75), (-38.60, -68.40), (-38.83, -68.13), (-38.97, -68.03)],
    "Río Limay": [(-40.05, -70.07), (-39.55, -69.35), (-39.27, -68.75), (-39.01, -68.43),
                  (-38.96, -68.23), (-38.97, -68.03)],
    "Río Negro": [(-38.97, -68.03), (-39.04, -67.58), (-39.10, -67.00), (-39.30, -66.30)],
    "Río Colorado": [(-36.85, -69.85), (-37.05, -69.70), (-37.39, -68.93), (-37.60, -68.30),
                     (-37.85, -67.78), (-38.20, -67.20), (-38.60, -66.00)],
}

# Puntos de control de caudal sobre los ríos
PUNTOS_RIO = pd.DataFrame([
    ("Neuquén en Chos Malal", -37.38, -70.27),
    ("Neuquén en Centenario", -38.83, -68.13),
    ("Limay en Plottier", -38.96, -68.23),
    ("Negro en Gral. Roca", -39.04, -67.58),
    ("Colorado en Rincón de los Sauces", -37.39, -68.93),
    ("Colorado en Catriel", -37.88, -67.79),
], columns=["nombre", "lat", "lon"])


# ----------------------------------------------------------------------------
# Datos de pozos
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_pozos():
    if ARCHIVO_POZOS.exists():
        return pd.read_csv(ARCHIVO_POZOS), False
    # Si todavía no generaste el CSV, la app arranca con pozos de ejemplo
    rng = np.random.default_rng(42)
    zonas = [("Loma Campana", -38.25, -68.75), ("Fortín de Piedra", -38.45, -68.95),
             ("La Amarga Chica", -38.20, -68.95), ("Bajo del Choique", -37.95, -69.05),
             ("Aguada Pichana", -38.40, -68.45), ("Rincón de los Sauces", -37.45, -69.05)]
    filas = []
    for i in range(400):
        nombre, la, lo = zonas[i % len(zonas)]
        filas.append({
            "idpozo": 100000 + i, "sigla": f"EJ-{i:04d}", "empresa": f"Empresa {chr(65 + i % 5)}",
            "area": nombre, "provincia": "Neuquén", "tipopozo": "Petrolífero",
            "estado": "Extracción Efectiva",
            "lat": la + rng.normal(0, 0.06), "lon": lo + rng.normal(0, 0.06),
            "prod_pet_12m": rng.gamma(2, 3000), "prod_gas_12m": rng.gamma(2, 2000), "prod_agua_12m": 0,
        })
    return pd.DataFrame(filas), True


# ----------------------------------------------------------------------------
# Consultas a Open-Meteo
# ----------------------------------------------------------------------------
def _pedir(url, lats, lons, params):
    """Pide datos para muchas coordenadas, en tandas de 50."""
    resultados = []
    for i in range(0, len(lats), 50):
        p = dict(params,
                 latitude=",".join(f"{x:.4f}" for x in lats[i:i + 50]),
                 longitude=",".join(f"{x:.4f}" for x in lons[i:i + 50]))
        r = requests.get(url, params=p, timeout=30)
        r.raise_for_status()
        js = r.json()
        resultados.extend(js if isinstance(js, list) else [js])
    return resultados


@st.cache_data(ttl=3600, show_spinner="Consultando lluvia en Open-Meteo...")
def lluvia_por_celda(celdas, inicio, fin):
    """Devuelve una tabla: celda x día con la lluvia (mm)."""
    hoy = date.today()
    url = URL_PRONOSTICO if inicio >= hoy - timedelta(days=80) else URL_HISTORICO
    lats = [c[0] for c in celdas]
    lons = [c[1] for c in celdas]
    params = {"daily": "precipitation_sum", "timezone": "America/Argentina/Buenos_Aires",
              "start_date": inicio.isoformat(), "end_date": fin.isoformat()}
    respuestas = _pedir(url, lats, lons, params)
    filas = []
    for celda, js in zip(celdas, respuestas):
        for dia, mm in zip(js["daily"]["time"], js["daily"]["precipitation_sum"]):
            filas.append({"celda_lat": celda[0], "celda_lon": celda[1],
                          "dia": dia, "mm": mm if mm is not None else 0.0})
    return pd.DataFrame(filas), ("pronóstico" if url == URL_PRONOSTICO else "histórico")


@st.cache_data(ttl=3600, show_spinner="Consultando caudal de ríos...")
def estado_rios(fecha):
    """Compara el caudal del día con el del último año en cada punto de control."""
    inicio = fecha - timedelta(days=365)
    params = {"daily": "river_discharge", "start_date": inicio.isoformat(), "end_date": fecha.isoformat()}
    respuestas = _pedir(URL_CAUDAL, PUNTOS_RIO["lat"].tolist(), PUNTOS_RIO["lon"].tolist(), params)
    salida = PUNTOS_RIO.copy()
    caudales, percentiles = [], []
    for js in respuestas:
        serie = pd.Series(js["daily"]["river_discharge"], dtype="float").dropna()
        if serie.empty:
            caudales.append(np.nan)
            percentiles.append(np.nan)
            continue
        valor = serie.iloc[-1]
        caudales.append(valor)
        percentiles.append((serie <= valor).mean() * 100)
    salida["caudal_m3s"] = caudales
    salida["percentil"] = percentiles
    salida["estado"] = pd.cut(salida["percentil"], [-1, 75, 90, 101],
                              labels=["Normal", "Alto", "Crecida"]).astype(object).fillna("Sin dato")
    return salida


def nivel_de_riesgo(mm):
    for nombre, desde, hasta, color in NIVELES:
        if desde <= mm < hasta:
            return nombre, color
    return NIVELES[-1][0], NIVELES[-1][3]


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
pozos, es_ejemplo = cargar_pozos()
hoy = date.today()

st.sidebar.title("🌧️ Consulta")
fecha = st.sidebar.date_input("Fecha a consultar", value=hoy, min_value=date(1990, 1, 1),
                              max_value=hoy + timedelta(days=15), format="DD/MM/YYYY")
ventana = st.sidebar.slider("Días de lluvia acumulada (hasta la fecha)", 1, 7, 3,
                            help="Los caminos de tierra se ven afectados por la lluvia de varios días seguidos.")
empresas = sorted(pozos["empresa"].dropna().unique())
filtro_emp = st.sidebar.multiselect("Empresa", empresas, placeholder="Todas")
ver_rios = st.sidebar.checkbox("Mostrar ríos y caudales", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Niveles de riesgo (lluvia acumulada)**")
for nombre, desde, hasta, color in NIVELES:
    rango = f"≥ {desde:.0f} mm" if hasta > 1e8 else f"{desde:.0f}–{hasta:.0f} mm"
    st.sidebar.markdown(
        f"<span style='color:rgb({color[0]},{color[1]},{color[2]})'>●</span> {nombre}: {rango}",
        unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("**El Niño:** en años Niño la Patagonia norte suele tener más lluvias de primavera "
                "y verano. Consultá el estado actual en el informe ENSO del SMN.")

if filtro_emp:
    pozos = pozos[pozos["empresa"].isin(filtro_emp)]
if pozos.empty:
    st.warning("No hay pozos para ese filtro.")
    st.stop()

# ----------------------------------------------------------------------------
# Cálculo del riesgo
# ----------------------------------------------------------------------------
pozos = pozos.copy()
pozos["celda_lat"] = (pozos["lat"] / TAMANIO_CELDA).round() * TAMANIO_CELDA
pozos["celda_lon"] = (pozos["lon"] / TAMANIO_CELDA).round() * TAMANIO_CELDA
celdas = tuple(sorted(set(zip(pozos["celda_lat"].round(4), pozos["celda_lon"].round(4)))))
pozos["celda_lat"] = pozos["celda_lat"].round(4)
pozos["celda_lon"] = pozos["celda_lon"].round(4)

inicio = fecha - timedelta(days=ventana - 1)
try:
    lluvia, fuente = lluvia_por_celda(celdas, inicio, fecha)
except Exception as e:
    st.error(f"No se pudo consultar la lluvia en Open-Meteo ({e}). Revisá tu conexión a internet.")
    st.stop()

acumulado = lluvia.groupby(["celda_lat", "celda_lon"], as_index=False)["mm"].sum()
pozos = pozos.merge(acumulado, on=["celda_lat", "celda_lon"], how="left").fillna({"mm": 0})
pozos["riesgo"] = pozos["mm"].apply(lambda v: nivel_de_riesgo(v)[0])
pozos["color"] = pozos["mm"].apply(lambda v: nivel_de_riesgo(v)[1])
pozos["nombre"] = pozos["sigla"].astype(str) + " · " + pozos["empresa"].astype(str)
pozos["detalle"] = (pozos["area"].astype(str) + "<br>Lluvia " + pozos["mm"].round(1).astype(str)
                    + " mm → riesgo " + pozos["riesgo"])

# ----------------------------------------------------------------------------
# Encabezado y métricas
# ----------------------------------------------------------------------------
st.title("Riesgo hídrico en caminos y ríos · Vaca Muerta")
st.caption(f"Lluvia acumulada del {inicio:%d/%m/%Y} al {fecha:%d/%m/%Y} (dato {fuente}). "
           "Fuente: Open-Meteo. Pozos: Secretaría de Energía (datos.energia.gob.ar).")
if es_ejemplo:
    st.warning("Estás viendo **pozos de ejemplo**. Ejecutá `preparar_datos.py` en tu notebook para "
               "generar `pozos_vaca_muerta.csv` y ponelo en la misma carpeta que esta app.")

conteo = pozos["riesgo"].value_counts()
columnas = st.columns(len(NIVELES))
for col, (nombre, *_r) in zip(columnas, NIVELES):
    col.metric(f"Pozos con riesgo {nombre.lower()}", f"{conteo.get(nombre, 0):,}")

# ----------------------------------------------------------------------------
# Mapa
# ----------------------------------------------------------------------------
capas = [pdk.Layer("ScatterplotLayer", data=pozos, get_position=["lon", "lat"], get_fill_color="color",
                   get_radius=600, radius_min_pixels=2, radius_max_pixels=8, opacity=0.8, pickable=True)]

rios_estado = None
if ver_rios:
    trazados = pd.DataFrame([{"nombre": n, "detalle": "Trazado aproximado",
                              "path": [[lo, la] for la, lo in pts]} for n, pts in RIOS.items()])
    capas.insert(0, pdk.Layer("PathLayer", data=trazados, get_path="path", get_color=[40, 110, 200],
                              width_min_pixels=3, pickable=True))
    try:
        rios_estado = estado_rios(fecha)
        colores = {"Normal": [40, 110, 200], "Alto": [240, 120, 30], "Crecida": [200, 30, 30]}
        rios_estado["color"] = rios_estado["estado"].map(colores).apply(
            lambda c: c if isinstance(c, list) else [150, 150, 150])
        rios_estado["detalle"] = ("Caudal " + rios_estado["caudal_m3s"].round(0).astype(str)
                                  + " m³/s<br>Estado: " + rios_estado["estado"])
        capas.append(pdk.Layer("ScatterplotLayer", data=rios_estado, get_position=["lon", "lat"],
                               get_fill_color="color", get_line_color=[255, 255, 255], stroked=True,
                               line_width_min_pixels=2, get_radius=3000, radius_min_pixels=7,
                               pickable=True))
    except Exception:
        st.info("No se pudo consultar el caudal de los ríos; se muestran solo los trazados.")

vista = pdk.ViewState(latitude=float(pozos["lat"].mean()), longitude=float(pozos["lon"].mean()),
                      zoom=7.5)
st.pydeck_chart(pdk.Deck(layers=capas, initial_view_state=vista, map_provider="carto", map_style="light",
                         tooltip={"html": "<b>{nombre}</b><br>{detalle}"}), use_container_width=True)

# ----------------------------------------------------------------------------
# Tablas y gráficos
# ----------------------------------------------------------------------------
izq, der = st.columns([3, 2])

with izq:
    st.subheader("Áreas con más lluvia")
    por_area = (pozos.groupby("area")
                .agg(pozos=("idpozo", "count"), lluvia_mm=("mm", "max"), empresas=("empresa", "nunique"))
                .sort_values("lluvia_mm", ascending=False))
    por_area["riesgo"] = por_area["lluvia_mm"].apply(lambda v: nivel_de_riesgo(v)[0])
    st.bar_chart(por_area.head(15)["lluvia_mm"], y_label="mm acumulados", x_label="Área")
    st.dataframe(por_area, use_container_width=True)

with der:
    st.subheader("Lluvia día por día")
    diaria = lluvia.groupby("dia")["mm"].max().rename("máximo en la zona (mm)")
    st.bar_chart(diaria)
    if rios_estado is not None:
        st.subheader("Estado de los ríos")
        st.caption("Compara el caudal del día con el último año. Neuquén y Limay están regulados por represas.")
        st.dataframe(rios_estado[["nombre", "caudal_m3s", "percentil", "estado"]].round(1),
                     hide_index=True, use_container_width=True)

st.download_button("⬇️ Descargar pozos con su riesgo (CSV)",
                   pozos.drop(columns=["color", "detalle", "nombre"]).to_csv(index=False).encode("utf-8"),
                   file_name=f"riesgo_vaca_muerta_{fecha:%Y%m%d}.csv", mime="text/csv")
