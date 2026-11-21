# -*- coding: utf-8 -*-
import os
import io
import base64
import unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy.special import gamma
from scipy.ndimage import zoom, distance_transform_edt
from shapely.geometry import Point
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import folium
import branca.colormap as cm_branca
from streamlit_folium import st_folium
import streamlit as st

# Compatibilité NumPy >= 2.0 (trapezoid) et < 2.0 (trapz)
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz")

# ==============================================================================
# CONFIGURATION DE LA PAGE & STYLE INSTITUTIONNEL
# ==============================================================================
st.set_page_config(
    page_title="Atlas Éolien National — Bénin (2000–2060)",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; }
        .atlas-banner {
            background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #1565c0 100%);
            padding: 16px 22px;
            border-radius: 10px;
            margin-bottom: 14px;
            box-shadow: 0 3px 10px rgba(26, 35, 126, 0.18);
        }
        .atlas-banner h2 {
            color: #ffffff;
            margin: 0;
            text-align: center;
            font-size: 21px;
            letter-spacing: 0.4px;
        }
        .atlas-banner p {
            color: #dbeafe;
            margin: 5px 0 0 0;
            text-align: center;
            font-size: 13.5px;
        }
        .site-bar {
            background-color: #f8fafc;
            border-left: 5px solid #1565c0;
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 14px;
            color: #1e293b;
        }
        .classe-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 12.5px;
            color: #ffffff;
            margin-left: 8px;
        }
    </style>
    <div class="atlas-banner">
        <h2>SIMULATEUR &amp; ATLAS ÉOLIEN NATIONAL DU BÉNIN (2000–2060)</h2>
        <p>Outil Interactif d'Aide à la Décision — Fusion In Situ (ASECNA) &amp; ERA5 par Modèle Hybride XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CHEMINS RELATIFS & RÉFÉRENTIELS SCIENTIFIQUES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GEOJSON_BENIN = os.path.join(DATA_DIR, "geoBoundaries-BEN-ADM1.geojson")
CSV_DEPT = os.path.join(DATA_DIR, "Bilan_Multi_Altitudes_12_Departements.csv")

RASTERS_VITESSE = {
    "10 m":  os.path.join(DATA_DIR, "Raster_Vitesse_10m_XGBoost.tif"),
    "20 m":  os.path.join(DATA_DIR, "Raster_Vitesse_20m_XGBoost.tif"),
    "53 m":  os.path.join(DATA_DIR, "Raster_Vitesse_53m_XGBoost.tif"),
    "86 m":  os.path.join(DATA_DIR, "Raster_Vitesse_86m_XGBoost.tif"),
    "120 m": os.path.join(DATA_DIR, "Raster_Vitesse_120m_XGBoost.tif"),
    "150 m": os.path.join(DATA_DIR, "Raster_Vitesse_150m_XGBoost.tif"),
}

ALTITUDES_NUM = {"10 m": 10, "20 m": 20, "53 m": 53, "86 m": 86, "120 m": 120, "150 m": 150}

STATIONS_ASECNA = {
    "Cotonou (Aéroport)":       {"lat": 6.3500,  "lon": 2.3833, "dept": "Littoral"},
    "Bohicon":                  {"lat": 7.1667,  "lon": 2.0667, "dept": "Zou"},
    "Savè":                     {"lat": 8.0333,  "lon": 2.4667, "dept": "Collines"},
    "Parakou (Aéroport)":       {"lat": 9.3500,  "lon": 2.6000, "dept": "Borgou"},
    "Natitingou (Péporiyakou)": {"lat": 10.3783, "lon": 1.3589, "dept": "Atacora"},
    "Kandi (Aéroport)":         {"lat": 11.1333, "lon": 2.9333, "dept": "Alibori"},
}

TURBINES = {
    "Vestas V112 (3,0 MW - Grand Parc)":   {"P_nom": 3000.0, "v_in": 3.0, "v_rat": 12.0, "v_out": 25.0},
    "Nordex N100 (2,5 MW - Littoral)":     {"P_nom": 2500.0, "v_in": 3.0, "v_rat": 12.5, "v_out": 20.0},
    "Gamesa G97 (2,0 MW - Classe III)":    {"P_nom": 2000.0, "v_in": 3.0, "v_rat": 11.0, "v_out": 25.0},
    "Turbine Moyenne (500 kW - Atacora)":  {"P_nom": 500.0,  "v_in": 3.0, "v_rat": 11.5, "v_out": 25.0},
    "Petit Éolien Hybride (50 kW - Nord)": {"P_nom": 50.0,   "v_in": 2.5, "v_rat": 10.5, "v_out": 25.0},
}

SCENARIOS = {
    "Référence Historique (2000–2024)": 1.000,
    "Horizon 2030 — Pacte National Énergie (+1,3%)": 1.013,
    "Horizon 2060 — Vision Alafia SSP2-4.5 (+2,8%)": 1.028,
    "Horizon 2060 — Vision Alafia SSP5-8.5 (+4,5%)": 1.045,
}

# Positions extérieures avec lignes de rappel pour les 6 petits départements du Sud (évite toute surcharge)
CALLOUTS_SUD = {
    "couffo":     {"xytext": (1.02, 7.18), "ha": "right"},
    "kouffo":     {"xytext": (1.02, 7.18), "ha": "right"},
    "mono":       {"xytext": (1.02, 6.56), "ha": "right"},
    "atlantique": {"xytext": (1.52, 6.02), "ha": "center"},
    "littoral":   {"xytext": (2.56, 6.02), "ha": "center"},
    "oueme":      {"xytext": (3.42, 6.56), "ha": "left"},
    "plateau":    {"xytext": (3.45, 7.22), "ha": "left"},
}


# ==============================================================================
# FONCTIONS UTILITAIRES & TRAITEMENTS SPATIAUX (MIS EN CACHE)
# ==============================================================================
def normaliser_nom(texte):
    txt = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode("ascii")
    return txt.strip().lower()


def qualifier_gisement(dpe):
    """Qualification énergétique du gisement éolien à partir de la DPE (W/m²)."""
    if dpe < 100:
        return "Classe 1 — Faible (Petit éolien / Hybride)", "#64748b"
    elif dpe < 200:
        return "Classe 2 — Marginal à Modéré (Éolien local)", "#d97706"
    elif dpe < 300:
        return "Classe 3 — Modéré à Bon (Faisabilité ciblée)", "#ea580c"
    elif dpe < 400:
        return "Classe 4 — Bon potentiel (Parc viable)", "#16a34a"
    else:
        return "Classe 5+ — Très bon potentiel (Grand parc)", "#0d9488"


@st.cache_data
def charger_geojson(path):
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    col_nom = "shapeName" if "shapeName" in gdf.columns else gdf.columns[0]
    gdf = gdf[[col_nom, "geometry"]].rename(columns={col_nom: "Departement"})
    geom_union = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    gdf_frontiere = gpd.GeoDataFrame(geometry=[geom_union], crs="EPSG:4326")
    return gdf, gdf_frontiere


@st.cache_data
def charger_raster_brut(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        transform = src.transform
    return data, extent, transform


@st.cache_data
def preparer_raster_lisse_clippe(tif_path, geojson_path):
    """Lissage bilinéaire haute qualité et découpage strict sur les frontières exactes du Bénin."""
    data, extent, _ = charger_raster_brut(tif_path)
    gdf_benin, _ = charger_geojson(geojson_path)

    # Remplissage temporaire des NaN par plus proche voisin pour éviter l'érosion des bords au lissage
    mask_nan = np.isnan(data)
    if np.any(mask_nan) and not np.all(mask_nan):
        _, indices = distance_transform_edt(mask_nan, return_indices=True)
        data_filled = data[tuple(indices)]
    else:
        data_filled = data.copy()

    # Facteur d'agrandissement adaptatif (cible ~520 px de haut pour rapidité et netteté)
    facteur = float(np.clip(520.0 / max(data.shape[0], 1), 1.0, 8.0))
    data_smooth = zoom(data_filled, zoom=facteur, order=1)

    new_h, new_w = data_smooth.shape
    new_transform = from_bounds(extent[0], extent[2], extent[1], extent[3], new_w, new_h)

    # Découpage topologique strict sur les frontières du Bénin
    masque_benin = geometry_mask(
        gdf_benin.geometry,
        out_shape=(new_h, new_w),
        transform=new_transform,
        invert=True,
    )
    data_clip = np.where(masque_benin, data_smooth, np.nan).astype(np.float32)
    return data_clip, extent


def creer_image_overlay_uri(data_clip, vmin, vmax, cmap_nom="YlOrRd"):
    """Convertit le raster découpé en image PNG transparente encodée en base64 pour Folium."""
    cmap = mpl.colormaps[cmap_nom]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(np.nan_to_num(data_clip, nan=vmin)))
    rgba[..., 3] = np.where(np.isnan(data_clip), 0.0, 1.0)
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extraire_vitesse_pixel(data, transform, extent, lon, lat, dans_benin):
    """Extrait la vitesse du raster au point (lon, lat), avec tolérance en bordure côtière."""
    if not ((extent[0] <= lon <= extent[1]) and (extent[2] <= lat <= extent[3])):
        return np.nan, False

    row, col = rasterio.transform.rowcol(transform, lon, lat)
    nrows, ncols = data.shape
    if 0 <= row < nrows and 0 <= col < ncols and not np.isnan(data[row, col]):
        return float(data[row, col]), False

    if dans_benin:
        for rayon in range(1, 5):
            r0, r1 = max(0, row - rayon), min(nrows, row + rayon + 1)
            c0, c1 = max(0, col - rayon), min(ncols, col + rayon + 1)
            voisinage = data[r0:r1, c0:c1]
            if not np.all(np.isnan(voisinage)):
                return float(np.nanmean(voisinage)), True

    return np.nan, False


def calculer_weibull_et_aep(v_moy, dept_nom, spec):
    """Calcule Weibull (k, c), DPE (W/m²), AEP (MWh/an), Facteur de charge (%) et CO2 évité (t/an)."""
    est_cotier = normaliser_nom(dept_nom) in {"littoral", "atlantique", "oueme", "mono"}
    k = 2.85 if est_cotier else 2.35
    c = v_moy / gamma(1.0 + 1.0 / k)
    dpe = 0.5 * 1.18 * (c ** 3) * gamma(1.0 + 3.0 / k)

    v_grid = np.linspace(0.5, 25.0, 500)
    pdf = (k / c) * ((v_grid / c) ** (k - 1)) * np.exp(-((v_grid / c) ** k))

    p_courbe = np.zeros_like(v_grid)
    mask_up = (v_grid >= spec["v_in"]) & (v_grid < spec["v_rat"])
    mask_rat = (v_grid >= spec["v_rat"]) & (v_grid <= spec["v_out"])
    p_courbe[mask_up] = spec["P_nom"] * (
        (v_grid[mask_up] ** 3 - spec["v_in"] ** 3) / (spec["v_rat"] ** 3 - spec["v_in"] ** 3)
    )
    p_courbe[mask_rat] = spec["P_nom"]

    p_moy_kw = float(_trapezoid(p_courbe * pdf, v_grid))
    fc_pct = (p_moy_kw / spec["P_nom"]) * 100.0
    aep_mwh = (p_moy_kw * 8760.0) / 1000.0
    co2_t = aep_mwh * 0.58
    return k, c, dpe, v_grid, pdf, p_courbe, fc_pct, aep_mwh, co2_t


# ==============================================================================
# INITIALISATION DES DONNÉES ET DE L'ÉTAT DE SESSION (SYNCHRONISATION CLIC)
# ==============================================================================
if not os.path.exists(GEOJSON_BENIN):
    st.error(f"Fichier GeoJSON introuvable : {GEOJSON_BENIN}")
    st.stop()

gdf_benin, gdf_frontiere = charger_geojson(GEOJSON_BENIN)
noms_dept = sorted(gdf_benin["Departement"].tolist())

MODES = [
    "🖱️ Clic direct sur la carte",
    "🗺️ Par Département",
    "📍 Par Coordonnées GPS",
    "📡 Par Station ASECNA",
]

if "mode_sel" not in st.session_state:
    st.session_state["mode_sel"] = "🖱️ Clic direct sur la carte"
if "lat_sel" not in st.session_state:
    st.session_state["lat_sel"] = 6.3700
if "lon_sel" not in st.session_state:
    st.session_state["lon_sel"] = 2.4200
if "last_map_click" not in st.session_state:
    st.session_state["last_map_click"] = None
if "pending_click" in st.session_state and st.session_state["pending_click"] is not None:
    clat, clon = st.session_state["pending_click"]
    st.session_state["lat_sel"] = round(float(clat), 4)
    st.session_state["lon_sel"] = round(float(clon), 4)
    st.session_state["mode_sel"] = "🖱️ Clic direct sur la carte"
    st.session_state["pending_click"] = None

# ==============================================================================
# BARRE LATÉRALE : SÉLECTION SPATIALE & PARAMÈTRES TECHNIQUES
# ==============================================================================
st.sidebar.header("📍 1. Localisation du Site")
mode_sel = st.sidebar.radio("Mode de sélection :", MODES, key="mode_sel")

sub_gdf = gdf_benin.iloc[0:0]
dans_benin = True

if mode_sel == "🗺️ Par Département":
    idx_lit = noms_dept.index("Littoral") if "Littoral" in noms_dept else 0
    dept = st.sidebar.selectbox("Département du Bénin :", noms_dept, index=idx_lit)
    sub_gdf = gdf_benin[gdf_benin["Departement"] == dept]
    pt_rep = sub_gdf.geometry.representative_point().iloc[0]
    lon_sel, lat_sel = float(pt_rep.x), float(pt_rep.y)
    nom_site = f"Département : {dept}"

elif mode_sel == "📡 Par Station ASECNA":
    station_nom = st.sidebar.selectbox("Station synoptique ASECNA :", list(STATIONS_ASECNA.keys()))
    lat_sel = STATIONS_ASECNA[station_nom]["lat"]
    lon_sel = STATIONS_ASECNA[station_nom]["lon"]
    pt_geom = Point(lon_sel, lat_sel)
    match = gdf_benin[gdf_benin.geometry.contains(pt_geom)]
    if not match.empty:
        sub_gdf = match.iloc[[0]]
        dept = sub_gdf["Departement"].iloc[0]
    else:
        dept = STATIONS_ASECNA[station_nom]["dept"]
        sub_gdf = gdf_benin[gdf_benin["Departement"].apply(normaliser_nom) == normaliser_nom(dept)]
    dans_benin = True
    nom_site = f"Station ASECNA : {station_nom}"

else:
    if mode_sel == "🖱️ Clic direct sur la carte":
        st.sidebar.info("💡 Cliquez directement sur n'importe quel point de la carte du Bénin pour l'analyser.")
        lat_sel = float(st.session_state["lat_sel"])
        lon_sel = float(st.session_state["lon_sel"])
        st.sidebar.caption(f"Point actif : **{lat_sel:.4f}°N, {lon_sel:.4f}°E**")
    else:
        lat_sel = st.sidebar.number_input(
            "Latitude (°N) :", min_value=6.10, max_value=12.45,
            value=float(np.clip(st.session_state["lat_sel"], 6.10, 12.45)),
            step=0.05, format="%.4f",
        )
        lon_sel = st.sidebar.number_input(
            "Longitude (°E) :", min_value=0.70, max_value=3.90,
            value=float(np.clip(st.session_state["lon_sel"], 0.70, 3.90)),
            step=0.05, format="%.4f",
        )
        st.session_state["lat_sel"] = lat_sel
        st.session_state["lon_sel"] = lon_sel

    pt_geom = Point(lon_sel, lat_sel)
    match = gdf_benin[gdf_benin.geometry.contains(pt_geom)]
    if not match.empty:
        sub_gdf = match.iloc[[0]]
        dept = sub_gdf["Departement"].iloc[0]
        dans_benin = True
    else:
        dept = "Hors frontières du Bénin"
        dans_benin = False
    nom_site = f"Point ({lat_sel:.3f}°N, {lon_sel:.3f}°E)"

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Paramètres Techniques")

altitudes = list(RASTERS_VITESSE.keys())
alt = st.sidebar.selectbox("Altitude du moyeu :", altitudes, index=altitudes.index("120 m"))
turb_nom = st.sidebar.selectbox("Aérogénérateur de référence :", list(TURBINES.keys()))
scen = st.sidebar.selectbox("Horizon & Scénario Climatique :", list(SCENARIOS.keys()))

st.sidebar.markdown("---")
st.sidebar.header("🗺️ 3. Habillage de la Carte")
afficher_noms_dept = st.sidebar.checkbox("Afficher les noms des départements", value=True)
afficher_stations = st.sidebar.checkbox("Afficher les 6 stations ASECNA", value=True)
opacite_raster = st.sidebar.slider("Opacité de la couche vent :", min_value=0.35, max_value=0.95, value=0.78, step=0.05)

facteur_scen = SCENARIOS[scen]
tif_path = RASTERS_VITESSE[alt]
if not os.path.exists(tif_path):
    st.error(f"Raster introuvable : {os.path.basename(tif_path)}")
    st.stop()

# ==============================================================================
# EXTRACTION DU RASTER & CALCULS PHYSIQUES
# ==============================================================================
data_brut, extent, transform_brut = charger_raster_brut(tif_path)
data_brut_scen = data_brut * facteur_scen
data_clip_base, _ = preparer_raster_lisse_clippe(tif_path, GEOJSON_BENIN)
data_clip_scen = data_clip_base * facteur_scen

vmin_map = float(np.nanpercentile(data_clip_scen, 2))
vmax_map = float(np.nanpercentile(data_clip_scen, 98))
if vmax_map <= vmin_map:
    vmin_map, vmax_map = float(np.nanmin(data_clip_scen)), float(np.nanmax(data_clip_scen))

v_moy, est_voisin = extraire_vitesse_pixel(data_brut_scen, transform_brut, extent, lon_sel, lat_sel, dans_benin)
point_valide = not np.isnan(v_moy)

spec = TURBINES[turb_nom]
if point_valide:
    k, c, dpe, v_grid, pdf, p_courbe, fc_pct, aep_mwh, co2_t = calculer_weibull_et_aep(v_moy, dept, spec)
    classe_txt, classe_col = qualifier_gisement(dpe)

    bilan = {
        "Mode_Selection": mode_sel,
        "Site": nom_site,
        "Departement": dept,
        "Latitude_N": round(lat_sel, 4),
        "Longitude_E": round(lon_sel, 4),
        "Altitude": alt,
        "Scenario": scen,
        "Turbine": turb_nom,
        "V_moy_ms": round(v_moy, 2),
        "Weibull_k": round(k, 2),
        "Weibull_c_ms": round(c, 2),
        "DPE_Wm2": round(dpe, 1),
        "Qualification": classe_txt,
        "AEP_MWh_an": round(aep_mwh, 1),
        "Facteur_Charge_pct": round(fc_pct, 1),
        "CO2_Evite_t_an": round(co2_t, 1),
    }
    csv_bytes = pd.DataFrame([bilan]).to_csv(index=False, sep=";").encode("utf-8-sig")
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="💾 Télécharger le bilan (CSV)",
        data=csv_bytes,
        file_name=f"Bilan_Eolien_{normaliser_nom(dept).replace(' ', '_')}_{alt.replace(' ', '')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ==============================================================================
# BANDEAU D'INDICATEURS CLÉS (KPI)
# ==============================================================================
if point_valide:
    st.markdown(
        f"""
        <div class="site-bar">
            <strong>Site analysé :</strong> {nom_site} &nbsp;|&nbsp;
            <strong>Département :</strong> {dept} &nbsp;|&nbsp;
            <strong>GPS :</strong> <code>{lat_sel:.4f}°N, {lon_sel:.4f}°E</code>
            <span class="classe-badge" style="background-color:{classe_col};">{classe_txt}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not dans_benin:
        st.caption("ℹ️ Point situé en bordure frontalière immédiate (valeur extraite du pixel valide adjacent).")
    elif est_voisin:
        st.caption("ℹ️ Point littoral/frontalier : valeur calculée sur le pixel valide immédiatement voisin.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Vitesse moyenne", f"{v_moy:.2f} m/s")
    k2.metric("Densité de puissance", f"{dpe:.1f} W/m²")
    k3.metric("Productible (AEP)", f"{aep_mwh:,.1f} MWh/an".replace(",", " "))
    k4.metric("Facteur de charge", f"{fc_pct:.1f} %")
    k5.metric("CO₂ évité", f"{co2_t:,.1f} t/an".replace(",", " "))
    k6.metric("Weibull (k / c)", f"k={k:.2f} | c={c:.2f}")
else:
    st.warning(
        f"⚠️ Le point cliqué (`{lat_sel:.4f}°N, {lon_sel:.4f}°E`) est **hors du territoire du Bénin**. "
        "Cliquez à l'intérieur des frontières du Bénin sur la carte ci-dessous pour afficher les indicateurs."
    )

# ==============================================================================
# ESPACE PRINCIPAL : CARTE INTERACTIVE (GAUCHE) + ANALYSES & ATLAS HD (DROITE)
# ==============================================================================
col_gauche, col_droite = st.columns([1.05, 1.25], gap="medium")

with col_gauche:
    st.subheader(f"🗺️ Carte Interactive du Bénin ({alt})")
    st.caption("Cliquez sur n'importe quel endroit du Bénin pour mettre à jour instantanément tous les calculs.")

    # Construction de la carte Folium épurée (CartoDB Positron)
    m = folium.Map(
        location=[9.35, 2.30],
        zoom_start=7,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )

    # 1. Couche Raster XGBoost lissée et découpée au millimètre sur le Bénin
    img_uri = creer_image_overlay_uri(data_clip_scen, vmin_map, vmax_map, cmap_nom="YlOrRd")
    folium.raster_layers.ImageOverlay(
        name=f"Vitesse du vent ({alt})",
        image=img_uri,
        bounds=[[extent[2], extent[0]], [extent[3], extent[1]]],
        opacity=opacite_raster,
        interactive=False,
        zindex=2,
    ).add_to(m)

    # 2. Limites des 12 départements avec surbrillance et info-bulle au survol
    dept_actif_norm = normaliser_nom(dept)

    def style_departement(feature):
        nom_d = feature["properties"].get("Departement", "")
        if normaliser_nom(nom_d) == dept_actif_norm:
            return {"fillColor": "transparent", "color": "#0d47a1", "weight": 2.6, "fillOpacity": 0.0}
        return {"fillColor": "transparent", "color": "#334155", "weight": 1.0, "fillOpacity": 0.0}

    folium.GeoJson(
        gdf_benin.__geo_interface__,
        name="Départements du Bénin",
        style_function=style_departement,
        highlight_function=lambda _: {"color": "#0284c7", "weight": 2.4, "fillOpacity": 0.08},
        tooltip=folium.GeoJsonTooltip(
            fields=["Departement"],
            aliases=["Département :"],
            style="background-color:white; color:#0f172a; font-weight:600; font-size:12px; border:1px solid #cbd5e1;",
        ),
    ).add_to(m)

    # 3. Frontière nationale extérieure renforcée
    folium.GeoJson(
        gdf_frontiere.__geo_interface__,
        name="Frontière Nationale",
        style_function=lambda _: {"fillColor": "transparent", "color": "#0f172a", "weight": 2.2, "fillOpacity": 0.0},
        interactive=False,
    ).add_to(m)

    # 4. Noms des départements lisibles et non surchargés (les 6 grands en direct, les 6 petits du Sud compacts)
    if afficher_noms_dept:
        decalages_folium = {
            "littoral":   (6.31, 2.42),
            "atlantique": (6.68, 2.18),
            "oueme":      (6.58, 2.55),
            "mono":       (6.52, 1.82),
            "couffo":     (7.02, 1.80),
            "plateau":    (7.22, 2.66),
        }
        for _, row_d in gdf_benin.iterrows():
            nom_d = row_d["Departement"]
            cle_d = normaliser_nom(nom_d)
            if cle_d in decalages_folium:
                lbl_lat, lbl_lon = decalages_folium[cle_d]
                taille_txt = "9.5px"
            else:
                pt_d = row_d.geometry.representative_point()
                lbl_lat, lbl_lon = pt_d.y, pt_d.x
                taille_txt = "11px"

            folium.Marker(
                location=[lbl_lat, lbl_lon],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:{taille_txt}; font-weight:700; color:#111827; '
                        f'text-align:center; white-space:nowrap; pointer-events:none; '
                        f'transform:translate(-50%, -50%); '
                        f'text-shadow: -1px -1px 2px #fff, 1px -1px 2px #fff, '
                        f'-1px 1px 2px #fff, 1px 1px 2px #fff, 0 0 4px #fff;">'
                        f'{nom_d}</div>'
                    )
                ),
                interactive=False,
            ).add_to(m)

    # 5. Marqueurs discrets des 6 stations ASECNA
    if afficher_stations:
        for st_nom, st_info in STATIONS_ASECNA.items():
            folium.CircleMarker(
                location=[st_info["lat"], st_info["lon"]],
                radius=4.5,
                color="#0f172a",
                weight=1.5,
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.95,
                tooltip=f"Station ASECNA : {st_nom}",
            ).add_to(m)

    # 6. Marqueur principal du point analysé
    popup_html = (
        f"<b>{nom_site}</b><br>Département : {dept}<br>"
        f"Vitesse ({alt}) : <b>{v_moy:.2f} m/s</b><br>DPE : <b>{dpe:.1f} W/m²</b><br>AEP : <b>{aep_mwh:,.1f} MWh/an</b>"
        if point_valide
        else f"<b>Point hors Bénin</b><br>{lat_sel:.4f}°N, {lon_sel:.4f}°E"
    )
    folium.Marker(
        location=[lat_sel, lon_sel],
        popup=folium.Popup(popup_html, max_width=240),
        tooltip=f"Point actif ({lat_sel:.3f}°N, {lon_sel:.3f}°E)",
        icon=folium.Icon(color="blue" if point_valide else "red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    # 7. Barre de légende colorimétrique intégrée
    cmap_mpl = mpl.colormaps["YlOrRd"]
    couleurs_hex = [mcolors.to_hex(cmap_mpl(x)) for x in np.linspace(0, 1, 8)]
    legende_cm = cm_branca.LinearColormap(
        colors=couleurs_hex,
        vmin=round(vmin_map, 2),
        vmax=round(vmax_map, 2),
        caption=f"Vitesse moyenne à {alt} (m/s)",
    )
    legende_cm.add_to(m)

    # Affichage interactif et écoute des clics souris
    retour_carte = st_folium(
        m,
        height=640,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="carte_interactive_benin",
    )

    if retour_carte and retour_carte.get("last_clicked"):
        click_lat = round(float(retour_carte["last_clicked"]["lat"]), 4)
        click_lon = round(float(retour_carte["last_clicked"]["lng"]), 4)
        if st.session_state["last_map_click"] != (click_lat, click_lon):
            st.session_state["last_map_click"] = (click_lat, click_lon)
            st.session_state["pending_click"] = (click_lat, click_lon)
            st.rerun()

# ==============================================================================
# PANNEAU DROIT : ONGLETS SCIENTIFIQUES & CARTE ATLAS HD EXPORTABLE
# ==============================================================================
with col_droite:
    tab_weib, tab_profil, tab_atlas = st.tabs([
        "📊 Weibull & Turbine",
        "📈 Profil Vertical (10–150 m)",
        "🖼️ Carte Atlas HD (Export)",
    ])

    # --- ONGLET 1 : DISTRIBUTION DE WEIBULL & COURBE DE PUISSANCE ---
    with tab_weib:
        if point_valide:
            fig_w, ax1 = plt.subplots(figsize=(6.8, 4.3), dpi=130)
            ax1.plot(v_grid, pdf, color="#1565c0", lw=2.3, label=f"Loi de Weibull (k={k:.2f}, c={c:.2f} m/s)")
            ax1.fill_between(v_grid, pdf, color="#bfdbfe", alpha=0.45)
            ax1.axvline(v_moy, color="#dc2626", linestyle="--", lw=1.6, label=f"Vitesse moy. = {v_moy:.2f} m/s")
            ax1.axvline(spec["v_in"], color="#16a34a", linestyle=":", lw=1.4, label=f"Démarrage = {spec['v_in']} m/s")
            ax1.set_xlabel("Vitesse du vent (m/s)", fontsize=9.5, fontweight="bold")
            ax1.set_ylabel("Densité de probabilité f(v)", fontsize=9.5, color="#1565c0", fontweight="bold")
            ax1.tick_params(axis="y", labelcolor="#1565c0")
            ax1.grid(True, linestyle=":", alpha=0.45)

            # Second axe : Courbe de puissance de l'aérogénérateur
            ax2 = ax1.twinx()
            ax2.plot(v_grid, p_courbe, color="#ea580c", lw=2.0, linestyle="-.", label=f"Courbe {spec['P_nom']:.0f} kW")
            ax2.set_ylabel("Puissance électrique (kW)", fontsize=9.5, color="#ea580c", fontweight="bold")
            ax2.tick_params(axis="y", labelcolor="#ea580c")
            ax2.set_ylim(0, spec["P_nom"] * 1.15)

            lignes1, labels1 = ax1.get_legend_handles_labels()
            lignes2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lignes1 + lignes2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.92)
            ax1.set_title(f"Distribution Statistique & Couplage Turbine à {alt} — {dept}", fontsize=10, fontweight="bold")
            fig_w.tight_layout()
            st.pyplot(fig_w)
            plt.close(fig_w)

            st.markdown("#### Fiche Technique du Site Sélectionné")
            df_recap = pd.DataFrame(
                {
                    "Indicateur": [
                        "Site / Localisation", "Département", "Coordonnées GPS",
                        "Altitude du moyeu", "Scénario climatique", "Aérogénérateur",
                        "Paramètre de forme (k)", "Paramètre d'échelle (c)",
                        "Qualification du gisement",
                    ],
                    "Valeur": [
                        nom_site, dept, f"{lat_sel:.4f}°N, {lon_sel:.4f}°E",
                        alt, scen, turb_nom, f"{k:.2f}", f"{c:.2f} m/s", classe_txt,
                    ],
                }
            )
            st.dataframe(df_recap, hide_index=True, use_container_width=True)
        else:
            st.info("Sélectionnez un point à l'intérieur du Bénin pour afficher la loi de Weibull et la fiche technique.")

    # --- ONGLET 2 : PROFIL VERTICAL MULTI-ALTITUDES AU POINT SÉLECTIONNÉ ---
    with tab_profil:
        if point_valide:
            lignes_prof = []
            for a_label, a_path in RASTERS_VITESSE.items():
                if os.path.exists(a_path):
                    d_raw, ext_a, tr_a = charger_raster_brut(a_path)
                    v_a, _ = extraire_vitesse_pixel(d_raw * facteur_scen, tr_a, ext_a, lon_sel, lat_sel, dans_benin)
                    if not np.isnan(v_a):
                        k_a, c_a, dpe_a, _, _, _, fc_a, aep_a, _ = calculer_weibull_et_aep(v_a, dept, spec)
                        lignes_prof.append({
                            "Altitude_m": ALTITUDES_NUM[a_label],
                            "Altitude": a_label,
                            "Vitesse (m/s)": round(v_a, 2),
                            "DPE (W/m²)": round(dpe_a, 1),
                            "AEP (MWh/an)": round(aep_a, 1),
                            "Fact. Charge (%)": round(fc_a, 1),
                        })

            if lignes_prof:
                df_prof = pd.DataFrame(lignes_prof).sort_values("Altitude_m")
                fig_p, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(7.0, 4.1), dpi=130, sharey=True)

                ax_v.plot(df_prof["Vitesse (m/s)"], df_prof["Altitude_m"], marker="o", lw=2.3, color="#1565c0")
                ax_v.axhline(ALTITUDES_NUM[alt], color="#dc2626", linestyle="--", lw=1.2, alpha=0.8)
                ax_v.set_title("Profil de Vitesse", fontsize=10, fontweight="bold")
                ax_v.set_xlabel("Vitesse moyenne (m/s)", fontsize=9, fontweight="bold")
                ax_v.set_ylabel("Altitude au-dessus du sol (m)", fontsize=9, fontweight="bold")
                ax_v.grid(True, linestyle=":", alpha=0.5)

                ax_d.plot(df_prof["DPE (W/m²)"], df_prof["Altitude_m"], marker="s", lw=2.3, color="#ea580c")
                ax_d.axhline(ALTITUDES_NUM[alt], color="#dc2626", linestyle="--", lw=1.2, alpha=0.8, label=f"Moyeu actif ({alt})")
                ax_d.set_title("Densité de Puissance (DPE)", fontsize=10, fontweight="bold")
                ax_d.set_xlabel("DPE (W/m²)", fontsize=9, fontweight="bold")
                ax_d.grid(True, linestyle=":", alpha=0.5)
                ax_d.legend(loc="lower right", fontsize=8)

                fig_p.suptitle(f"Évolution Verticale du Gisement — {dept} ({lat_sel:.2f}°N, {lon_sel:.2f}°E)", fontsize=10.5, fontweight="bold")
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)

                st.dataframe(
                    df_prof[["Altitude", "Vitesse (m/s)", "DPE (W/m²)", "AEP (MWh/an)", "Fact. Charge (%)"]],
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.info("Sélectionnez un point au Bénin pour tracer le profil vertical de 10 m à 150 m.")

    # --- ONGLET 3 : CARTE ATLAS HD (STYLE PUBLICATION / SOUTENANCE 300 DPI) ---
    with tab_atlas:
        fig_hd, ax_hd = plt.subplots(figsize=(6.4, 7.6), dpi=150)
        fig_hd.patch.set_facecolor("white")
        ax_hd.set_facecolor("#ffffff")

        # Raster lissé et découpé strictement sur le Bénin
        im_hd = ax_hd.imshow(
            data_clip_scen, cmap="YlOrRd", extent=extent, origin="upper",
            vmin=vmin_map, vmax=vmax_map, interpolation="bilinear", zorder=2,
        )

        # Limites départementales fines + frontière nationale épaisse + département actif
        gdf_benin.plot(ax=ax_hd, facecolor="none", edgecolor="#334155", linewidth=0.75, zorder=3)
        gdf_frontiere.plot(ax=ax_hd, facecolor="none", edgecolor="#0f172a", linewidth=1.7, zorder=4)
        if not sub_gdf.empty:
            sub_gdf.plot(ax=ax_hd, facecolor="none", edgecolor="#0d47a1", linewidth=2.3, zorder=5)

        # Halo blanc autour des textes pour une lisibilité parfaite
        halo = [pe.withStroke(linewidth=2.6, foreground="white")]

        # Étiquettes des 12 départements sans aucun chevauchement (lignes de rappel au Sud)
        if afficher_noms_dept:
            for _, row_d in gdf_benin.iterrows():
                nom_d = row_d["Departement"]
                cle_d = normaliser_nom(nom_d)
                pt_d = row_d.geometry.representative_point()

                if cle_d in CALLOUTS_SUD:
                    cfg = CALLOUTS_SUD[cle_d]
                    ax_hd.annotate(
                        nom_d,
                        xy=(pt_d.x, pt_d.y),
                        xytext=cfg["xytext"],
                        ha=cfg["ha"],
                        va="center",
                        fontsize=7.6,
                        fontweight="bold",
                        color="#1e293b",
                        path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color="#475569", lw=0.75, shrinkA=0, shrinkB=2),
                        zorder=8,
                    )
                else:
                    ax_hd.text(
                        pt_d.x, pt_d.y, nom_d.upper(),
                        ha="center", va="center",
                        fontsize=8.2, fontweight="bold", color="#0f172a",
                        path_effects=halo, zorder=8,
                    )

        # Repères discrets des pays voisins et de l'Océan Atlantique
        ax_hd.text(0.88, 9.20, "TOGO", color="#94a3b8", fontsize=9, fontweight="bold", rotation=86, ha="center")
        ax_hd.text(1.30, 11.75, "BURKINA FASO", color="#94a3b8", fontsize=8, fontweight="bold", ha="center")
        ax_hd.text(3.38, 12.18, "NIGER", color="#94a3b8", fontsize=9, fontweight="bold", ha="center")
        ax_hd.text(3.82, 9.25, "NIGÉRIA", color="#94a3b8", fontsize=9, fontweight="bold", rotation=-88, ha="center")
        ax_hd.text(2.30, 5.74, "OCÉAN ATLANTIQUE", color="#3b82f6", fontsize=8.5, style="italic", fontweight="bold", ha="center")

        # Stations ASECNA
        if afficher_stations:
            st_lons = [v["lon"] for v in STATIONS_ASECNA.values()]
            st_lats = [v["lat"] for v in STATIONS_ASECNA.values()]
            ax_hd.scatter(
                st_lons, st_lats,
                color="white", edgecolor="#0f172a", s=34, linewidth=1.0, zorder=9, label="Stations ASECNA",
            )

        # Marqueur du point analysé
        ax_hd.scatter(
            [lon_sel], [lat_sel],
            color="#00e5ff", edgecolor="#0f172a", marker="*", s=210, linewidth=1.1, zorder=11, label="Site sélectionné",
        )

        # Flèche du Nord (en haut à gauche)
        ax_hd.annotate(
            "N", xy=(0.86, 12.32), xytext=(0.86, 11.75),
            arrowprops=dict(facecolor="#0f172a", width=3.5, headwidth=9, headlength=9),
            ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a", zorder=10,
        )

        # Barre d'échelle graphique (~100 km à ~9°N ≈ 0.91°)
        ax_hd.plot([3.00, 3.91], [5.86, 5.86], color="#0f172a", lw=2.5, zorder=10)
        ax_hd.plot([3.00, 3.00], [5.82, 5.90], color="#0f172a", lw=1.4, zorder=10)
        ax_hd.plot([3.91, 3.91], [5.82, 5.90], color="#0f172a", lw=1.4, zorder=10)
        ax_hd.text(3.455, 5.95, "100 km", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0f172a")

        ax_hd.set_xlim(0.55, 4.15)
        ax_hd.set_ylim(5.60, 12.55)
        ax_hd.set_aspect("equal", adjustable="box")
        ax_hd.set_xlabel("Longitude (°E)", fontsize=8.5, fontweight="bold")
        ax_hd.set_ylabel("Latitude (°N)", fontsize=8.5, fontweight="bold")
        ax_hd.grid(True, linestyle="--", alpha=0.25, color="#64748b", zorder=1)
        ax_hd.legend(loc="upper right", fontsize=7.8, framealpha=0.92)
        ax_hd.set_title(
            f"Potentiel Éolien du Bénin à {alt} — Modèle XGBoost\n{scen}",
            fontsize=10.5, fontweight="bold", pad=10,
        )

        cbar = fig_hd.colorbar(im_hd, ax=ax_hd, fraction=0.042, pad=0.03, shrink=0.78)
        cbar.set_label("Vitesse moyenne du vent (m/s)", fontsize=8.5, fontweight="bold")
        fig_hd.tight_layout()

        # Préparation du téléchargement PNG Haute Définition (300 DPI)
        buf_hd = io.BytesIO()
        fig_hd.savefig(buf_hd, format="PNG", dpi=300, bbox_inches="tight", facecolor="white")
        st.pyplot(fig_hd)
        plt.close(fig_hd)

        st.download_button(
            label="🖼️ Télécharger cette carte en Haute Définition (PNG 300 DPI)",
            data=buf_hd.getvalue(),
            file_name=f"Atlas_Eolien_Benin_{alt.replace(' ', '')}_{normaliser_nom(dept).replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )

# ==============================================================================
# TABLEAU RÉCAPITULATIF MULTI-ALTITUDES DES 12 DÉPARTEMENTS
# ==============================================================================
if os.path.exists(CSV_DEPT):
    with st.expander("📊 Consulter le bilan complet des 12 départements (multi-altitudes)"):
        st.dataframe(pd.read_csv(CSV_DEPT), use_container_width=True)
