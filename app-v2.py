# -*- coding: utf-8 -*-
"""
Atlas Éolien National & Simulateur Interactif du Bénin (2000–2060)
Modèle Hybride XGBoost (Météo Bénin + ERA5) — Version Web Pro v2.0
Auteur : CHABI GUIYA Orou Séko Idelphonse

Nouveautés v2.0 :
- En-tête entièrement visible sur tous les écrans (correction du décalage supérieur Streamlit)
- Hauteur de moyeu libre et continue de 10 m à 150 m (interpolation physique de Hellman entre rasters)
- Ajout & positionnement interactif d'éoliennes / sites sur la carte (bilan cumulé de parc + export CSV)
- Option d'ajout d'une turbine personnalisée & nombre d'éoliennes (dimensionnement de parc + ménages alimentés)
"""

import io
import os
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy.special import gamma
from scipy.ndimage import zoom
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from PIL import Image
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import streamlit as st

# Compatibilité stricte NumPy >= 2.0 (trapezoid) et < 2.0 (trapz)
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else getattr(np, "trapz")

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE & CSS (EN-TÊTE 100 % VISIBLE)
# ==============================================================================
st.set_page_config(
    page_title="Atlas Éolien & Simulateur Bénin (2000–2060)",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Laisse toujours l'espace suffisant sous la barre supérieure Streamlit Cloud
           pour que l'en-tête soit 100 % visible sur PC, tablette et smartphone */
        .stAppViewBlockContainer,
        .stMainBlockContainer,
        div[data-testid="stAppViewBlockContainer"],
        .block-container {
            padding-top: 4.0rem !important;
            padding-bottom: 2.0rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
        /* Barre supérieure Streamlit légère et non intrusive */
        header[data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.92) !important;
            backdrop-filter: blur(6px);
            height: 2.8rem !important;
        }
        /* Bandeau titre principal entièrement visible et responsive */
        .hero-banner {
            background: linear-gradient(135deg, #0f2942 0%, #1b4965 55%, #0e7c7b 100%);
            color: #ffffff;
            padding: 1.15rem 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #f59e0b;
            box-shadow: 0 4px 14px rgba(15, 41, 66, 0.18);
            margin-top: 0.25rem;
            margin-bottom: 1.05rem;
        }
        .hero-kicker {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #fbbf24;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            font-size: clamp(1.15rem, 2.0vw, 1.65rem);
            font-weight: 800;
            line-height: 1.25;
            margin: 0;
            color: #ffffff;
        }
        .hero-sub {
            font-size: 0.90rem;
            color: #dbeafe;
            margin-top: 0.35rem;
            margin-bottom: 0;
        }
        /* Cartes KPI propres */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-top: 3px solid #0e7c7b;
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
        }
        .classe-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.88rem;
            margin-top: 0.2rem;
            margin-bottom: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CHEMINS DES DONNÉES & CONSTANTES SCIENTIFIQUES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GEOJSON_BENIN = os.path.join(DATA_DIR, "geoBoundaries-BEN-ADM1.geojson")
CSV_DEPT = os.path.join(DATA_DIR, "Bilan_Multi_Altitudes_12_Departements.csv")

ALTITUDES_REF = [10, 20, 53, 86, 120, 150]
RASTERS_VITESSE = {
    z: os.path.join(DATA_DIR, f"Raster_Vitesse_{z}m_XGBoost.tif")
    for z in ALTITUDES_REF
}

STATIONS_ASECNA = {
    "Cotonou Aéroport (Littoral)":        {"lat": 6.3500,  "lon": 2.3833, "dept": "Littoral"},
    "Bohicon (Zou)":                      {"lat": 7.1667,  "lon": 2.0667, "dept": "Zou"},
    "Savè (Collines)":                    {"lat": 8.0333,  "lon": 2.4667, "dept": "Collines"},
    "Parakou Aéroport (Borgou)":          {"lat": 9.3500,  "lon": 2.6000, "dept": "Borgou"},
    "Natitingou Péporiyakou (Atacora)":   {"lat": 10.3783, "lon": 1.3589, "dept": "Atacora"},
    "Kandi Aéroport (Alibori)":           {"lat": 11.1333, "lon": 2.9333, "dept": "Alibori"},
}

TURBINES_STD = {
    "Vestas V112 (3,0 MW — Grand Parc)":   {"P_nom": 3000.0, "v_in": 3.0, "v_rat": 12.0, "v_out": 25.0, "diam": 112.0},
    "Nordex N100 (2,5 MW — Littoral)":     {"P_nom": 2500.0, "v_in": 3.0, "v_rat": 12.5, "v_out": 20.0, "diam": 100.0},
    "Gamesa G97 (2,0 MW — Classe III)":    {"P_nom": 2000.0, "v_in": 3.0, "v_rat": 11.0, "v_out": 25.0, "diam": 97.0},
    "Turbine Moyenne (500 kW — Transition)": {"P_nom": 500.0, "v_in": 3.0, "v_rat": 11.5, "v_out": 25.0, "diam": 44.0},
    "Petit Éolien Hybride (50 kW — Rural)": {"P_nom": 50.0,  "v_in": 2.5, "v_rat": 10.5, "v_out": 25.0, "diam": 15.0},
}

SCENARIOS = {
    "Référence Historique (2000–2024)":               1.000,
    "Horizon 2030 — Pacte National Énergie (+1,3 %)": 1.013,
    "Horizon 2060 — Vision Alafia SSP2-4.5 (+2,8 %)": 1.028,
    "Horizon 2060 — Vision Alafia SSP5-8.5 (+4,5 %)": 1.045,
}

# Départements côtiers (régime maritime plus régulier)
DEPTS_COTIERS = {"Littoral", "Atlantique", "Ouémé", "Oueme", "Mono"}

# Lignes de rappel extérieures pour les 6 petits départements du Sud (carte HD 300 DPI)
CALLOUTS_SUD = {
    "Mono":       (1.18, 6.45),
    "Couffo":     (1.12, 7.02),
    "Kouffo":     (1.12, 7.02),
    "Atlantique": (1.28, 6.08),
    "Littoral":   (2.42, 5.95),
    "Ouémé":      (3.22, 6.50),
    "Oueme":      (3.22, 6.50),
    "Plateau":    (3.28, 7.25),
}


# ==============================================================================
# 3. CHARGEMENT MIS EN CACHE & INTERPOLATION VERTICALE CONTINUE (10 m → 150 m)
# ==============================================================================
@st.cache_data(show_spinner=False)
def charger_geometrie():
    gdf = gpd.read_file(GEOJSON_BENIN)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    col_nom = "shapeName" if "shapeName" in gdf.columns else gdf.columns[0]
    gdf = gdf.rename(columns={col_nom: "Departement"})
    union_geom = gdf.geometry.unary_union if hasattr(gdf.geometry, "unary_union") else gdf.geometry.union_all()
    return gdf, union_geom


@st.cache_data(show_spinner=False)
def charger_bilan_csv():
    if os.path.exists(CSV_DEPT):
        return pd.read_csv(CSV_DEPT)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def charger_raster_ref(z_ref: int):
    path = RASTERS_VITESSE[z_ref]
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        transform = src.transform
    return arr, bounds, transform


@st.cache_data(show_spinner=False)
def obtenir_raster_altitude(z_m: int, facteur_scen: float):
    """
    Retourne le champ de vitesse spatialisé pour N'IMPORTE QUELLE hauteur z_m ∈ [10, 150].
    - Si z_m correspond à une altitude de référence (10, 20, 53, 86, 120, 150 m),
      utilise directement le raster GeoTIFF XGBoost natif.
    - Sinon, applique une interpolation verticale physique de Hellman pixel par pixel
      entre les deux rasters encadrants [z_low, z_high] :
      alpha_loc = ln(V_high / V_low) / ln(z_high / z_low), V(z) = V_low * (z / z_low)^alpha_loc.
    """
    if z_m in RASTERS_VITESSE:
        arr, bounds, transform = charger_raster_ref(z_m)
        return arr * facteur_scen, bounds, transform, f"Raster natif {z_m} m"

    # Encadrement [z_low, z_high]
    z_low = max(z for z in ALTITUDES_REF if z < z_m)
    z_high = min(z for z in ALTITUDES_REF if z > z_m)

    arr_low, bounds, transform = charger_raster_ref(z_low)
    arr_high, _, _ = charger_raster_ref(z_high)

    # Interpolation logarithmique de Hellman exacte et stable
    valid = (~np.isnan(arr_low)) & (~np.isnan(arr_high)) & (arr_low > 0.05) & (arr_high > 0.05)
    out = np.full_like(arr_low, np.nan, dtype=np.float32)

    ratio_z = np.log(z_m / z_low) / np.log(z_high / z_low)
    # V(z) = V_low * (V_high / V_low)^(ln(z/z_low)/ln(z_high/z_low))
    out[valid] = arr_low[valid] * np.power(arr_high[valid] / arr_low[valid], ratio_z)
    out = out * facteur_scen
    return out, bounds, transform, f"Interpolation Hellman ({z_low} m ↔ {z_high} m)"


def extraire_vitesse_point(data, bounds, lon: float, lat: float):
    """Extrait par interpolation bilinéaire la valeur du raster au point (lon, lat)."""
    lon_min, lat_min, lon_max, lat_max = bounds
    nrows, ncols = data.shape
    col_f = (lon - lon_min) / (lon_max - lon_min) * (ncols - 1)
    row_f = (lat_max - lat) / (lat_max - lat_min) * (nrows - 1)

    c0 = int(np.clip(np.floor(col_f), 0, ncols - 1))
    r0 = int(np.clip(np.floor(row_f), 0, nrows - 1))
    c1 = min(c0 + 1, ncols - 1)
    r1 = min(r0 + 1, nrows - 1)
    dc = np.clip(col_f - c0, 0.0, 1.0)
    dr = np.clip(row_f - r0, 0.0, 1.0)

    vals = np.array([data[r0, c0], data[r0, c1], data[r1, c0], data[r1, c1]], dtype=float)
    weights = np.array([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc], dtype=float)
    mask = ~np.isnan(vals)
    if np.any(mask):
        return float(np.sum(vals[mask] * weights[mask]) / np.sum(weights[mask]))
    # Repli sur le pixel valide le plus proche
    yy, xx = np.where(~np.isnan(data))
    if len(yy) == 0:
        return float("nan")
    dist2 = (yy - r0) ** 2 + (xx - c0) ** 2
    idx = int(np.argmin(dist2))
    return float(data[yy[idx], xx[idx]])


def detecter_departement(gdf, lon: float, lat: float):
    pt = Point(lon, lat)
    for _, row in gdf.iterrows():
        if row.geometry.contains(pt) or row.geometry.touches(pt):
            return row["Departement"], True
    # Plus proche département si en bordure
    dists = gdf.geometry.distance(pt)
    idx_min = dists.idxmin()
    return gdf.loc[idx_min, "Departement"], False


def calculer_physique_eolienne(v_moy: float, dept_nom: str, spec: dict, nb_turbines: int = 1, pertes_pct: float = 10.0):
    """Calcule Weibull (k, c), DPE, puissance moyenne, facteur de charge, AEP net, CO2 évité et ménages alimentés."""
    k = 2.85 if dept_nom in DEPTS_COTIERS else 2.35
    c = v_moy / gamma(1.0 + 1.0 / k)
    rho_air = 1.18  # Masse volumique moyenne tropicale (kg/m³)
    dpe = 0.5 * rho_air * (c ** 3) * gamma(1.0 + 3.0 / k)

    v_grid = np.linspace(0.5, 25.0, 500)
    pdf = (k / c) * ((v_grid / c) ** (k - 1)) * np.exp(-((v_grid / c) ** k))

    p_courbe = np.zeros_like(v_grid)
    mask_up = (v_grid >= spec["v_in"]) & (v_grid < spec["v_rat"])
    mask_rat = (v_grid >= spec["v_rat"]) & (v_grid <= spec["v_out"])
    denom = max(spec["v_rat"] ** 3 - spec["v_in"] ** 3, 1e-6)
    p_courbe[mask_up] = spec["P_nom"] * ((v_grid[mask_up] ** 3 - spec["v_in"] ** 3) / denom)
    p_courbe[mask_rat] = spec["P_nom"]

    p_moy_brut_kw = float(_trapezoid(p_courbe * pdf, v_grid))
    rendement_net = 1.0 - (pertes_pct / 100.0)
    p_moy_net_kw = p_moy_brut_kw * rendement_net

    fc_pct = (p_moy_net_kw / spec["P_nom"]) * 100.0 if spec["P_nom"] > 0 else 0.0
    aep_1_mwh = (p_moy_net_kw * 8760.0) / 1000.0
    aep_parc_mwh = aep_1_mwh * nb_turbines
    puissance_parc_mw = (spec["P_nom"] * nb_turbines) / 1000.0
    co2_parc_t = aep_parc_mwh * 0.58  # Facteur d'émission thermique de référence (tCO2/MWh)
    menages = int(round((aep_parc_mwh * 1000.0) / 1500.0))  # ~1 500 kWh/ménage/an au Bénin

    return {
        "k": k, "c": c, "dpe": dpe, "v_grid": v_grid, "pdf": pdf, "p_courbe": p_courbe,
        "fc_pct": fc_pct, "aep_1_mwh": aep_1_mwh, "aep_parc_mwh": aep_parc_mwh,
        "puissance_parc_mw": puissance_parc_mw, "co2_parc_t": co2_parc_t, "menages": menages,
    }


def qualifier_gisement(dpe: float):
    if dpe >= 200:
        return "Classe 2–3 NREL • Très favorable (Grands parcs industriels)", "#15803d", "#dcfce7"
    if dpe >= 120:
        return "Classe 1–2 NREL • Favorable (Parcs côtiers IEC III)", "#0e7c7b", "#e6f4f4"
    if dpe >= 60:
        return "Zone de transition • Moyen éolien / Micro-réseaux", "#b45309", "#fef3c7"
    return "Zone continentale • Recommandé : Hybride Solaire PV + Petit Éolien", "#475569", "#f1f5f9"


@st.cache_data(show_spinner=False)
def generer_overlay_png_base64(z_m: int, facteur_scen: float):
    """Lisse le raster et le masque strictement sur les frontières du Bénin pour Folium."""
    gdf, _ = charger_geometrie()
    data, bounds, _ = obtenir_raster_altitude(z_m, facteur_scen)[:3]

    # Remplissage temporaire des NaN avant zoom pour éviter les halos de bord
    nan_mask = np.isnan(data)
    mean_fill = float(np.nanmean(data)) if not np.all(nan_mask) else 0.0
    filled = np.where(nan_mask, mean_fill, data)

    facteur_zoom = 6
    smooth = zoom(filled, facteur_zoom, order=1)
    h_s, w_s = smooth.shape
    lon_min, lat_min, lon_max, lat_max = bounds
    transform_s = from_bounds(lon_min, lat_min, lon_max, lat_max, w_s, h_s)

    # Masquage topologique strict aux frontières nationales du Bénin
    inside = ~geometry_mask(gdf.geometry, out_shape=(h_s, w_s), transform=transform_s, invert=False)
    smooth = np.where(inside, smooth, np.nan)

    vmin = float(np.nanpercentile(smooth, 2))
    vmax = float(np.nanpercentile(smooth, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = plt.get_cmap("YlOrRd")
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.nan_to_num(smooth, nan=vmin)))
    rgba[..., 3] = np.where(np.isnan(smooth), 0.0, 0.78)

    img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}", bounds, vmin, vmax


# ==============================================================================
# 4. INITIALISATION DES DONNÉES & ÉTAT DE SESSION
# ==============================================================================
try:
    gdf_benin, benin_union = charger_geometrie()
    df_bilan_dept = charger_bilan_csv()
except Exception as exc:
    st.error(f"Erreur lors du chargement des données dans `{DATA_DIR}` : {exc}")
    st.stop()

noms_depts = sorted(gdf_benin["Departement"].unique().tolist())

if "lat_sel" not in st.session_state:
    st.session_state.lat_sel = 6.3650
if "lon_sel" not in st.session_state:
    st.session_state.lon_sel = 2.4183
if "z_m" not in st.session_state:
    st.session_state.z_m = 120
if "sites_positionnes" not in st.session_state:
    st.session_state.sites_positionnes = []
if "last_map_click" not in st.session_state:
    st.session_state.last_map_click = None

# ==============================================================================
# 5. EN-TÊTE PRINCIPAL (100 % VISIBLE)
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-kicker">République du Bénin • Fusion Multi-Sources (Météo Bénin + ERA5) &amp; Apprentissage Automatique XGBoost</div>
        <h1 class="hero-title">🌬️ SIMULATEUR DU POTENTIEL ÉOLIEN DU BÉNIN (2000–2060)</h1>
        <p class="hero-sub">
            Atlas interactif continu de <b>10 m à 150 m</b> • Positionnement de sites &amp; dimensionnement de parcs • Projections PONADER 2030 &amp; Vision Bénin 2060 Alafia
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 6. BARRE LATÉRALE : PARAMÈTRES COMPLETS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Paramètres de Simulation")

    # A. Mode de sélection spatiale
    mode_sel = st.radio(
        "1. Mode de sélection du site :",
        [
            "🖱️ Clic direct sur la carte",
            "🏛️ Par Département (12)",
            "📍 Coordonnées GPS libres",
            "📡 Station ASECNA (6)",
        ],
        index=0,
    )

    if mode_sel == "🏛️ Par Département (12)":
        dept_choisi = st.selectbox(
            "Département :",
            noms_depts,
            index=noms_depts.index("Littoral") if "Littoral" in noms_depts else 0,
        )
        sub_g = gdf_benin[gdf_benin["Departement"] == dept_choisi]
        pt_rep = sub_g.geometry.representative_point().iloc[0]
        st.session_state.lat_sel = float(pt_rep.y)
        st.session_state.lon_sel = float(pt_rep.x)

    elif mode_sel == "📍 Coordonnées GPS libres":
        c_lat, c_lon = st.columns(2)
        with c_lat:
            st.session_state.lat_sel = st.number_input(
                "Latitude (°N)", min_value=5.80, max_value=12.60,
                value=float(st.session_state.lat_sel), step=0.01, format="%.4f"
            )
        with c_lon:
            st.session_state.lon_sel = st.number_input(
                "Longitude (°E)", min_value=0.60, max_value=4.00,
                value=float(st.session_state.lon_sel), step=0.01, format="%.4f"
            )

    elif mode_sel == "📡 Station ASECNA (6)":
        st_nom = st.selectbox("Station synoptique :", list(STATIONS_ASECNA.keys()))
        st.session_state.lat_sel = STATIONS_ASECNA[st_nom]["lat"]
        st.session_state.lon_sel = STATIONS_ASECNA[st_nom]["lon"]
    else:
        st.caption(
            f"Point actif : **{st.session_state.lat_sel:.4f}° N, {st.session_state.lon_sel:.4f}° E** "
            "(cliquez n'importe où sur la carte du Bénin)."
        )

    st.markdown("---")

    # B. HAUTEUR DE MOYEU LIBRE DE 10 m À 150 m (Curseur + Saisie exacte + Raccourcis)
    st.markdown("**2. Hauteur de moyeu libre (10 m à 150 m) :**")

    if "slider_z" not in st.session_state:
        st.session_state.slider_z = int(st.session_state.z_m)
    if "input_z" not in st.session_state:
        st.session_state.input_z = int(st.session_state.z_m)

    def _sync_from_slider():
        val = int(st.session_state.slider_z)
        st.session_state.z_m = val
        st.session_state.input_z = val

    def _sync_from_input():
        val = int(st.session_state.input_z)
        st.session_state.z_m = val
        st.session_state.slider_z = val

    def _set_preset_z(z_val: int):
        st.session_state.z_m = int(z_val)
        st.session_state.slider_z = int(z_val)
        st.session_state.input_z = int(z_val)

    st.slider(
        "Ajuster la hauteur (m)",
        min_value=10, max_value=150, step=1,
        key="slider_z", on_change=_sync_from_slider,
        help="Choisissez n'importe quelle hauteur entre 10 m et 150 m. Les hauteurs intermédiaires sont calculées par interpolation physique de Hellman entre les rasters XGBoost encadrants.",
    )
    st.number_input(
        "Hauteur exacte (m)",
        min_value=10, max_value=150, step=1,
        key="input_z", on_change=_sync_from_input,
    )
    # Boutons raccourcis hauteurs standards
    st.caption("Raccourcis rapides (m) :")
    rb_cols = st.columns(6)
    for col_b, z_preset in zip(rb_cols, [10, 50, 80, 100, 120, 150]):
        col_b.button(
            f"{z_preset}",
            key=f"btn_z_{z_preset}",
            on_click=_set_preset_z,
            args=(z_preset,),
            use_container_width=True,
        )

    z_actuel = int(st.session_state.z_m)

    st.markdown("---")

    # C. Aérogénérateur (5 modèles constructeurs + 1 turbine personnalisée) & Taille du parc
    st.markdown("**3. Aérogénérateur & Dimensionnement :**")
    options_turb = list(TURBINES_STD.keys()) + ["⚙️ Turbine personnalisée (ajouter mes paramètres)"]
    turb_nom = st.selectbox("Modèle d'aérogénérateur :", options_turb, index=0)

    if turb_nom.startswith("⚙️"):
        p_nom_custom = st.number_input("Puissance nominale P_nom (kW)", min_value=5.0, max_value=15000.0, value=1500.0, step=50.0)
        c1, c2 = st.columns(2)
        with c1:
            v_in_c = st.number_input("Démarrage v_in (m/s)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
            v_out_c = st.number_input("Coupure v_out (m/s)", min_value=15.0, max_value=35.0, value=25.0, step=1.0)
        with c2:
            v_rat_c = st.number_input("Nominale v_rat (m/s)", min_value=7.0, max_value=18.0, value=11.5, step=0.5)
            diam_c = st.number_input("Diamètre rotor (m)", min_value=5.0, max_value=220.0, value=82.0, step=2.0)
        spec_active = {"P_nom": p_nom_custom, "v_in": v_in_c, "v_rat": v_rat_c, "v_out": v_out_c, "diam": diam_c}
        turb_label = f"Personnalisée ({p_nom_custom:.0f} kW)"
    else:
        spec_active = TURBINES_STD[turb_nom]
        turb_label = turb_nom

    c_nb, c_loss = st.columns(2)
    with c_nb:
        nb_turbines = st.number_input("Nb d'éoliennes", min_value=1, max_value=100, value=1, step=1)
    with c_loss:
        pertes_pct = st.number_input("Pertes (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)

    st.markdown("---")

    # D. Scénario climatique & Fond de carte
    scen_nom = st.selectbox("4. Horizon climatique (CMIP6) :", list(SCENARIOS.keys()), index=0)
    facteur_scen = SCENARIOS[scen_nom]

    fond_carte = st.selectbox(
        "5. Fond cartographique :",
        ["CartoDB Clair (Épuré)", "Esri Topo (Relief)", "OpenStreetMap", "Satellite Esri"],
        index=0,
    )
    afficher_stations = st.checkbox("Afficher les 6 stations ASECNA", value=True)
    afficher_sillage = st.checkbox("Afficher rayon anti-sillage (5D) sur les sites positionnés", value=True)

# ==============================================================================
# 7. CALCULS AU POINT ACTIF
# ==============================================================================
data_z, bounds_z, _, methode_z = obtenir_raster_altitude(z_actuel, facteur_scen)
lat_cur = float(st.session_state.lat_sel)
lon_cur = float(st.session_state.lon_sel)
dept_cur, dans_benin = detecter_departement(gdf_benin, lon_cur, lat_cur)
v_moy_cur = extraire_vitesse_point(data_z, bounds_z, lon_cur, lat_cur)

res_cur = calculer_physique_eolienne(v_moy_cur, dept_cur, spec_active, nb_turbines=nb_turbines, pertes_pct=pertes_pct)
classe_txt, classe_fg, classe_bg = qualifier_gisement(res_cur["dpe"])

if not dans_benin:
    st.warning(
        f"⚠️ Le point ({lat_cur:.4f}° N, {lon_cur:.4f}° E) est légèrement hors des frontières du Bénin "
        f"(extraction sur le pixel frontalier le plus proche — département : **{dept_cur}**)."
    )

# ==============================================================================
# 8. INDICATEURS CLÉS (KPI) + BARRE DE POSITIONNEMENT DE SITES / ÉOLIENNES
# ==============================================================================
st.markdown(
    f'<span class="classe-badge" style="color:{classe_fg}; background:{classe_bg}; border:1px solid {classe_fg};">'
    f'📍 {dept_cur} ({lat_cur:.4f}° N, {lon_cur:.4f}° E) &nbsp;|&nbsp; Hauteur : {z_actuel} m ({methode_z}) &nbsp;|&nbsp; {classe_txt}'
    f'</span>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Vitesse moyenne", f"{v_moy_cur:.2f} m/s", f"H = {z_actuel} m")
k2.metric("Densité de puissance", f"{res_cur['dpe']:.1f} W/m²", f"k={res_cur['k']:.2f} | c={res_cur['c']:.2f}")
k3.metric("Productible Net (AEP)", f"{res_cur['aep_parc_mwh']:,.1f} MWh/an", f"{nb_turbines} × {spec_active['P_nom']:.0f} kW")
k4.metric("Facteur de charge", f"{res_cur['fc_pct']:.1f} %", f"Pertes -{pertes_pct:.0f}%")
k5.metric("CO₂ évité", f"{res_cur['co2_parc_t']:,.0f} t/an", "0,58 tCO₂/MWh")
k6.metric("Ménages alimentés", f"{res_cur['menages']:,}".replace(",", " "), f"Parc {res_cur['puissance_parc_mw']:.2f} MW")

# Barre d'ajout / positionnement d'un site ou d'une éolienne sur la carte
with st.expander("➕ Positionner & ajouter ce site / cette éolienne dans votre projet de parc (comparaison multi-sites)", expanded=True):
    col_n, col_b1, col_b2 = st.columns([3.2, 1.6, 1.2])
    with col_n:
        nom_defaut = f"Site {len(st.session_state.sites_positionnes) + 1} — {dept_cur} ({z_actuel} m)"
        nom_site_ajout = st.text_input("Nom du repère / de l'éolienne à positionner :", value=nom_defaut, label_visibility="collapsed", placeholder="Ex : Parc Sèmè-Podji — Éolienne 1")
    with col_b1:
        if st.button("📌 Positionner sur la carte", type="primary", use_container_width=True):
            st.session_state.sites_positionnes.append({
                "Nom": nom_site_ajout.strip() or nom_defaut,
                "Departement": dept_cur,
                "Lat": round(lat_cur, 4),
                "Lon": round(lon_cur, 4),
                "Hauteur_m": z_actuel,
                "Turbine": turb_label,
                "Nb_Turbines": int(nb_turbines),
                "Puissance_MW": round(res_cur["puissance_parc_mw"], 3),
                "Diametre_m": float(spec_active["diam"]),
                "Scenario": scen_nom,
                "V_moy_ms": round(v_moy_cur, 2),
                "DPE_Wm2": round(res_cur["dpe"], 1),
                "Facteur_Charge_pct": round(res_cur["fc_pct"], 1),
                "AEP_MWh_an": round(res_cur["aep_parc_mwh"], 1),
                "CO2_t_an": round(res_cur["co2_parc_t"], 1),
                "Menages": int(res_cur["menages"]),
            })
            st.toast(f"✅ Positionné : {nom_site_ajout}", icon="📍")
            st.rerun()
    with col_b2:
        if st.button("🗑️ Vider les repères", use_container_width=True, disabled=(len(st.session_state.sites_positionnes) == 0)):
            st.session_state.sites_positionnes = []
            st.rerun()

# ==============================================================================
# 9. ONGLETS PRINCIPAUX
# ==============================================================================
tab_carte, tab_analyse, tab_parc, tab_bilan = st.tabs([
    "🗺️ Carte Interactive & Positionnement",
    "📈 Weibull, Profil Vertical (10–150 m) & Export HD",
    f"🏗️ Projets & Sites Positionnés ({len(st.session_state.sites_positionnes)})",
    "📊 Classement des 12 Départements",
])

# ------------------------------------------------------------------------------
# ONGLET 1 : CARTE INTERACTIVE FOLIUM
# ------------------------------------------------------------------------------
with tab_carte:
    col_map, col_info = st.columns([2.25, 1.0])

    with col_map:
        overlay_uri, ov_bounds, vmin, vmax = generer_overlay_png_base64(z_actuel, facteur_scen)

        tiles_cfg = {
            "CartoDB Clair (Épuré)": ("CartoDB positron", None),
            "Esri Topo (Relief)": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                "Esri World Topo",
            ),
            "OpenStreetMap": ("OpenStreetMap", None),
            "Satellite Esri": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Esri World Imagery",
            ),
        }
        t_url, t_attr = tiles_cfg[fond_carte]
        m = folium.Map(
            location=[9.35, 2.30],
            zoom_start=7,
            tiles=t_url if t_attr is None else None,
            control_scale=True,
        )
        if t_attr is not None:
            folium.TileLayer(tiles=t_url, attr=t_attr, name=fond_carte).add_to(m)

        # Overlay raster lissé et découpé strictement sur le Bénin
        folium.raster_layers.ImageOverlay(
            image=overlay_uri,
            bounds=[[ov_bounds[1], ov_bounds[0]], [ov_bounds[3], ov_bounds[2]]],
            opacity=0.82,
            name=f"Vitesse XGBoost ({z_actuel} m)",
            interactive=False,
        ).add_to(m)

        # Limites des 12 départements avec survol
        folium.GeoJson(
            gdf_benin,
            name="Limites des 12 départements",
            style_function=lambda feat: {
                "fillColor": "transparent",
                "color": "#0f2942" if feat["properties"]["Departement"] == dept_cur else "#334155",
                "weight": 2.6 if feat["properties"]["Departement"] == dept_cur else 1.1,
            },
            tooltip=folium.GeoJsonTooltip(fields=["Departement"], aliases=["Département : "]),
        ).add_to(m)

        # 6 stations ASECNA de référence
        if afficher_stations:
            fg_st = folium.FeatureGroup(name="6 Stations ASECNA (Réf.)")
            for st_n, info_st in STATIONS_ASECNA.items():
                folium.CircleMarker(
                    location=[info_st["lat"], info_st["lon"]],
                    radius=5.5,
                    color="#0f2942",
                    weight=2,
                    fill=True,
                    fill_color="#ffffff",
                    fill_opacity=0.95,
                    tooltip=f"Station ASECNA : {st_n}",
                ).add_to(fg_st)
            fg_st.add_to(m)

        # Sites / éoliennes positionnés par l'utilisateur
        if st.session_state.sites_positionnes:
            fg_pos = folium.FeatureGroup(name="Sites / Éoliennes positionnés")
            for idx_p, site in enumerate(st.session_state.sites_positionnes, 1):
                popup_html = (
                    f"<div style='font-family:sans-serif;font-size:12px;min-width:190px;'>"
                    f"<b>#{idx_p} • {site['Nom']}</b><br>"
                    f"Département : {site['Departement']}<br>"
                    f"Hauteur : <b>{site['Hauteur_m']} m</b><br>"
                    f"Vitesse : <b>{site['V_moy_ms']} m/s</b> ({site['DPE_Wm2']} W/m²)<br>"
                    f"Parc : {site['Nb_Turbines']} × {site['Turbine']}<br>"
                    f"AEP : <b>{site['AEP_MWh_an']:,.1f} MWh/an</b><br>"
                    f"CO₂ évité : <b>{site['CO2_t_an']:,.1f} t/an</b></div>"
                )
                folium.Marker(
                    location=[site["Lat"], site["Lon"]],
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"#{idx_p} {site['Nom']} ({site['V_moy_ms']} m/s à {site['Hauteur_m']} m)",
                    icon=folium.Icon(color="green", icon="flash", prefix="glyphicon"),
                ).add_to(fg_pos)
                if afficher_sillage:
                    rayon_m = max(150.0, 5.0 * site["Diametre_m"])
                    folium.Circle(
                        location=[site["Lat"], site["Lon"]],
                        radius=rayon_m,
                        color="#15803d",
                        weight=1.5,
                        fill=True,
                        fill_color="#22c55e",
                        fill_opacity=0.14,
                        tooltip=f"Zone anti-sillage 5D ({rayon_m:.0f} m)",
                    ).add_to(fg_pos)
            fg_pos.add_to(m)

        # Marqueur du curseur actif
        folium.Marker(
            location=[lat_cur, lon_cur],
            tooltip=f"Point actif : {dept_cur} | {v_moy_cur:.2f} m/s à {z_actuel} m",
            icon=folium.Icon(color="red", icon="screenshot", prefix="glyphicon"),
        ).add_to(m)

        # Barre de légende couleur
        colormap = LinearColormap(
            colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
            vmin=round(vmin, 2),
            vmax=round(vmax, 2),
            caption=f"Vitesse moyenne du vent à {z_actuel} m (m/s) — {scen_nom[:22]}",
        )
        colormap.add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)

        map_state = st_folium(
            m,
            width="100%",
            height=580,
            returned_objects=["last_clicked"],
            key="benin_wind_map",
        )

        # Mise à jour uniquement sur un NOUVEAU clic souris (n'écrase jamais les modes Département / GPS / Station)
        if map_state and map_state.get("last_clicked"):
            c_lat = round(float(map_state["last_clicked"]["lat"]), 5)
            c_lon = round(float(map_state["last_clicked"]["lng"]), 5)
            new_click = (c_lat, c_lon)
            if new_click != st.session_state.last_map_click:
                st.session_state.last_map_click = new_click
                st.session_state.lat_sel = c_lat
                st.session_state.lon_sel = c_lon
                st.rerun()

    with col_info:
        st.markdown("#### 📋 Fiche Technique du Point Actif")
        fiche_df = pd.DataFrame([
            ("Département", dept_cur),
            ("Coordonnées GPS", f"{lat_cur:.4f}° N, {lon_cur:.4f}° E"),
            ("Hauteur de moyeu", f"{z_actuel} m"),
            ("Méthode altitude", methode_z),
            ("Scénario climatique", scen_nom),
            ("Vitesse moyenne", f"{v_moy_cur:.2f} m/s"),
            ("Paramètres Weibull", f"k = {res_cur['k']:.2f} | c = {res_cur['c']:.2f} m/s"),
            ("Densité de puissance", f"{res_cur['dpe']:.1f} W/m²"),
            ("Turbine sélectionnée", turb_label),
            ("Parc installé", f"{nb_turbines} unité(s) • {res_cur['puissance_parc_mw']:.2f} MW"),
            ("Productible 1 turbine", f"{res_cur['aep_1_mwh']:,.1f} MWh/an"),
            ("Productible Total Parc", f"{res_cur['aep_parc_mwh']:,.1f} MWh/an"),
            ("Facteur de charge net", f"{res_cur['fc_pct']:.1f} %"),
            ("CO₂ évité", f"{res_cur['co2_parc_t']:,.1f} tCO₂/an"),
            ("Ménages alimentés", f"~ {res_cur['menages']:,} foyers".replace(",", " ")),
        ], columns=["Paramètre", "Valeur"])
        st.dataframe(fiche_df, use_container_width=True, hide_index=True, height=455)

        csv_bytes = pd.DataFrame([{
            "Departement": dept_cur, "Latitude": round(lat_cur, 4), "Longitude": round(lon_cur, 4),
            "Hauteur_m": z_actuel, "Scenario": scen_nom, "Turbine": turb_label,
            "Nb_Turbines": nb_turbines, "Puissance_Parc_MW": round(res_cur["puissance_parc_mw"], 3),
            "V_moy_ms": round(v_moy_cur, 2), "Weibull_k": round(res_cur["k"], 2),
            "Weibull_c_ms": round(res_cur["c"], 2), "DPE_Wm2": round(res_cur["dpe"], 1),
            "Facteur_Charge_pct": round(res_cur["fc_pct"], 1),
            "AEP_Net_MWh_an": round(res_cur["aep_parc_mwh"], 1),
            "CO2_Evite_t_an": round(res_cur["co2_parc_t"], 1),
            "Menages_Alimentes": res_cur["menages"],
        }]).to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            "💾 Télécharger la Fiche Technique (CSV)",
            data=csv_bytes,
            file_name=f"Fiche_Eolienne_{dept_cur}_{z_actuel}m.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ------------------------------------------------------------------------------
# ONGLET 2 : WEIBULL, PROFIL VERTICAL CONTINU (10–150 m) & CARTE HD 300 DPI
# ------------------------------------------------------------------------------
with tab_analyse:
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(f"#### Distribution de Weibull & Courbe de Puissance ({z_actuel} m)")
        fig_w, ax1 = plt.subplots(figsize=(6.4, 4.3), dpi=130)
        ax1.plot(res_cur["v_grid"], res_cur["pdf"], color="#1b4965", lw=2.3, label=f"Weibull (k={res_cur['k']:.2f}, c={res_cur['c']:.2f} m/s)")
        ax1.fill_between(res_cur["v_grid"], res_cur["pdf"], color="#bfdbfe", alpha=0.45)
        ax1.axvline(v_moy_cur, color="#dc2626", ls="--", lw=1.6, label=f"V_moy = {v_moy_cur:.2f} m/s")
        ax1.set_xlabel("Vitesse du vent (m/s)", fontweight="bold")
        ax1.set_ylabel("Densité de probabilité f(v)", color="#1b4965", fontweight="bold")
        ax1.grid(True, ls=":", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.plot(res_cur["v_grid"], res_cur["p_courbe"], color="#d97706", lw=2.0, ls="-.", label=f"Puissance {turb_label} (kW)")
        ax2.set_ylabel("Puissance turbine (kW)", color="#d97706", fontweight="bold")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.2)
        ax1.set_title(f"{dept_cur} — Couplage Vent / Turbine à {z_actuel} m", fontweight="bold")
        fig_w.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with c_right:
        st.markdown("#### Profil Vertical Continu (10 m → 150 m) au point actif")
        z_curve = np.arange(10, 151, 5)
        v_curve = []
        dpe_curve = []
        for zz in z_curve:
            d_tmp, b_tmp, _, _ = obtenir_raster_altitude(int(zz), facteur_scen)
            vz = extraire_vitesse_point(d_tmp, b_tmp, lon_cur, lat_cur)
            v_curve.append(vz)
            cz = vz / gamma(1.0 + 1.0 / res_cur["k"])
            dpe_curve.append(0.5 * 1.18 * (cz ** 3) * gamma(1.0 + 3.0 / res_cur["k"]))

        fig_p, axp1 = plt.subplots(figsize=(6.4, 4.3), dpi=130)
        axp1.plot(v_curve, z_curve, color="#0e7c7b", lw=2.4, label="Vitesse v(z) [m/s]")
        axp1.scatter([v_moy_cur], [z_actuel], color="#dc2626", s=70, zorder=5, label=f"Hauteur choisie ({z_actuel} m : {v_moy_cur:.2f} m/s)")
        axp1.axhline(z_actuel, color="#dc2626", ls=":", lw=1.2)
        axp1.set_xlabel("Vitesse moyenne du vent (m/s)", color="#0e7c7b", fontweight="bold")
        axp1.set_ylabel("Hauteur au-dessus du sol (m)", fontweight="bold")
        axp1.set_ylim(5, 155)
        axp1.grid(True, ls=":", alpha=0.5)

        axp2 = axp1.twiny()
        axp2.plot(dpe_curve, z_curve, color="#d97706", lw=1.8, ls="--", label="DPE(z) [W/m²]")
        axp2.set_xlabel("Densité de Puissance Éolienne (W/m²)", color="#d97706", fontweight="bold")

        hp1, lp1 = axp1.get_legend_handles_labels()
        hp2, lp2 = axp2.get_legend_handles_labels()
        axp1.legend(hp1 + hp2, lp1 + lp2, loc="lower right", fontsize=8.2)
        axp1.set_title(f"Profil Vertical — {dept_cur} ({lat_cur:.3f}°N, {lon_cur:.3f}°E)", fontweight="bold", pad=12)
        fig_p.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

    # Export Carte HD 300 DPI avec callouts propres & sites positionnés
    st.markdown("---")
    st.markdown(f"#### 🖨️ Export Carte Haute Définition (300 DPI) à {z_actuel} m")
    if st.button("🎨 Générer la carte 300 DPI prête pour le mémoire / rapport"):
        with st.spinner("Génération de la carte haute résolution (300 DPI)..."):
            fig_hd, ax_hd = plt.subplots(figsize=(7.5, 10.2), dpi=300)
            fig_hd.patch.set_facecolor("white")
            ax_hd.set_facecolor("#e0f2fe")

            lon_min, lat_min, lon_max, lat_max = bounds_z
            im_hd = ax_hd.imshow(
                data_z, cmap="YlOrRd",
                extent=[lon_min, lon_max, lat_min, lat_max],
                origin="upper", interpolation="bilinear",
            )
            gdf_benin.plot(ax=ax_hd, facecolor="none", edgecolor="#1e293b", linewidth=0.9)

            # Étiquettes propres des départements avec lignes de rappel au Sud
            for _, row_d in gdf_benin.iterrows():
                dname = row_d["Departement"]
                rp = row_d.geometry.representative_point()
                if dname in CALLOUTS_SUD:
                    tx, ty = CALLOUTS_SUD[dname]
                    ax_hd.annotate(
                        dname.upper(), xy=(rp.x, rp.y), xytext=(tx, ty),
                        fontsize=7.2, fontweight="bold", color="#0f2942", ha="center",
                        arrowprops=dict(arrowstyle="-", color="#334155", lw=0.7),
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                    )
                else:
                    ax_hd.text(
                        rp.x, rp.y, dname.upper(), fontsize=7.6, fontweight="bold",
                        color="#0f2942", ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
                    )

            # Point actif + sites positionnés
            ax_hd.scatter([lon_cur], [lat_cur], c="#dc2626", edgecolors="white", s=95, zorder=6, label=f"Point actif ({v_moy_cur:.2f} m/s)")
            for idx_s, s_item in enumerate(st.session_state.sites_positionnes, 1):
                ax_hd.scatter([s_item["Lon"]], [s_item["Lat"]], c="#15803d", edgecolors="white", marker="^", s=95, zorder=7)
                ax_hd.text(
                    s_item["Lon"] + 0.06, s_item["Lat"] + 0.04, f"#{idx_s}",
                    fontsize=7.5, fontweight="bold", color="#15803d",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                )

            cbar = fig_hd.colorbar(im_hd, ax=ax_hd, shrink=0.58, pad=0.03)
            cbar.set_label(f"Vitesse moyenne à {z_actuel} m (m/s)", fontweight="bold")
            ax_hd.set_xlim(0.55, 4.05)
            ax_hd.set_ylim(5.75, 12.55)
            ax_hd.set_xlabel("Longitude (°E)", fontweight="bold")
            ax_hd.set_ylabel("Latitude (°N)", fontweight="bold")
            ax_hd.set_title(
                f"Atlas Éolien du Bénin — Vitesse Moyenne à {z_actuel} m\nModèle XGBoost • {scen_nom}",
                fontsize=12, fontweight="bold", pad=10,
            )
            ax_hd.grid(True, ls="--", lw=0.4, alpha=0.5)
            fig_hd.tight_layout()

            buf_hd = io.BytesIO()
            fig_hd.savefig(buf_hd, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig_hd)
            st.image(buf_hd.getvalue(), caption=f"Aperçu Carte HD 300 DPI ({z_actuel} m)", width=460)
            st.download_button(
                "⬇️ Télécharger la Carte PNG (300 DPI)",
                data=buf_hd.getvalue(),
                file_name=f"Carte_Eolienne_Benin_{z_actuel}m_300DPI.png",
                mime="image/png",
            )

# ------------------------------------------------------------------------------
# ONGLET 3 : PROJETS & SITES POSITIONNÉS (BILAN CUMULÉ DE PARC)
# ------------------------------------------------------------------------------
with tab_parc:
    st.markdown("#### 🏗️ Tableau Comparatif & Bilan Cumulé des Éoliennes / Sites Positionnés")
    if not st.session_state.sites_positionnes:
        st.info(
            "Aucun site n'est encore positionné. Cliquez sur la carte (ou choisissez un département / GPS), "
            "réglez la hauteur entre **10 m et 150 m** et cliquez sur **« 📌 Positionner sur la carte »** ci-dessus."
        )
    else:
        df_pos = pd.DataFrame(st.session_state.sites_positionnes)
        # Totaux cumulés du projet
        tot_p = df_pos["Puissance_MW"].sum()
        tot_aep = df_pos["AEP_MWh_an"].sum()
        tot_co2 = df_pos["CO2_t_an"].sum()
        tot_men = df_pos["Menages"].sum()
        tot_turb = df_pos["Nb_Turbines"].sum()

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Puissance cumulée du projet", f"{tot_p:.2f} MW", f"{tot_turb} éolienne(s)")
        p2.metric("Productible cumulé (AEP)", f"{tot_aep:,.1f} MWh/an")
        p3.metric("CO₂ évité cumulé", f"{tot_co2:,.1f} t/an")
        p4.metric("Foyers béninois alimentés", f"{tot_men:,}".replace(",", " "))

        st.dataframe(df_pos, use_container_width=True, hide_index=True)

        # Suppression sélective d'un site
        c_del1, c_del2 = st.columns([2.5, 1.5])
        with c_del1:
            noms_pos = [f"#{i+1} — {s['Nom']}" for i, s in enumerate(st.session_state.sites_positionnes)]
            sel_del = st.selectbox("Retirer un repère spécifique :", noms_pos)
        with c_del2:
            st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
            if st.button("❌ Supprimer ce repère", use_container_width=True):
                idx_rm = noms_pos.index(sel_del)
                st.session_state.sites_positionnes.pop(idx_rm)
                st.rerun()

        csv_parc = df_pos.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "💾 Exporter tout le tableau des sites positionnés (CSV)",
            data=csv_parc,
            file_name="Projet_Parc_Eolien_Benin_Sites_Positionnes.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------------------
# ONGLET 4 : CLASSEMENT DES 12 DÉPARTEMENTS
# ------------------------------------------------------------------------------
with tab_bilan:
    st.markdown(f"#### 📊 Bilan Officiel des 12 Départements du Bénin (Hauteur active : {z_actuel} m)")
    rows_live = []
    for _, r_d in gdf_benin.iterrows():
        dn = r_d["Departement"]
        pt_d = r_d.geometry.representative_point()
        vd = extraire_vitesse_point(data_z, bounds_z, float(pt_d.x), float(pt_d.y))
        calc_d = calculer_physique_eolienne(vd, dn, spec_active, nb_turbines=nb_turbines, pertes_pct=pertes_pct)
        rows_live.append({
            "Département": dn,
            f"Vitesse à {z_actuel} m (m/s)": round(vd, 2),
            f"DPE à {z_actuel} m (W/m²)": round(calc_d["dpe"], 1),
            "Weibull k": round(calc_d["k"], 2),
            "Facteur de charge (%)": round(calc_d["fc_pct"], 1),
            f"AEP ({nb_turbines}× {turb_label[:14]}) [MWh/an]": round(calc_d["aep_parc_mwh"], 1),
            "CO₂ évité (t/an)": round(calc_d["co2_parc_t"], 1),
        })
    df_live = pd.DataFrame(rows_live).sort_values(by=f"DPE à {z_actuel} m (W/m²)", ascending=False).reset_index(drop=True)
    df_live.index = df_live.index + 1
    df_live.index.name = "Rang"
    st.dataframe(df_live, use_container_width=True)

    if not df_bilan_dept.empty:
        with st.expander("📁 Voir le tableau CSV complet multi-altitudes de référence (10 m à 120 m)"):
            st.dataframe(df_bilan_dept, use_container_width=True, hide_index=True)
