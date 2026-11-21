# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.special import gamma
import matplotlib.pyplot as plt
import streamlit as st

# Compatibilité NumPy >= 2.0 et < 2.0
_trapezoid = getattr(np, "trapezoid", np.trapz)

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Atlas Éolien National — Bénin (2000–2060)",
    page_icon="🌬️",
    layout="wide",
)

# Bandeau institutionnel
st.markdown(
    """
    <div style="background-color:#1a237e; padding:16px 20px; border-radius:8px; margin-bottom:18px;">
        <h2 style="color:white; margin:0; text-align:center; font-size:22px;">
            SIMULATEUR DU POTENTIEL ÉOLIEN DU BÉNIN (2000–2060)
        </h2>
        <p style="color:#c5cae9; margin:4px 0 0 0; text-align:center; font-size:14px;">
            Atlas Éolien National &amp; Outil d'Aide à la Décision — Modèle Hybride XGBoost
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CHEMINS RELATIFS & PARAMÈTRES
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

TURBINES = {
    "Vestas V112 (3,0 MW - Grand Parc)":   {"P_nom": 3000.0, "v_in": 3.0, "v_rat": 12.0, "v_out": 25.0},
    "Nordex N100 (2,5 MW - Littoral)":     {"P_nom": 2500.0, "v_in": 3.0, "v_rat": 12.5, "v_out": 20.0},
    "Gamesa G97 (2,0 MW - Classe III)":    {"P_nom": 2000.0, "v_in": 3.0, "v_rat": 11.0, "v_out": 25.0},
    "Turbine Moyenne (500 kW - Atacora)":  {"P_nom": 500.0,  "v_in": 3.0, "v_rat": 11.5, "v_out": 25.0},
    "Petit Éolien Hybride (50 kW - Nord)": {"P_nom": 50.0,   "v_in": 2.5, "v_rat": 10.5, "v_out": 25.0},
}

SCENARIOS = {
    "Référence Historique (2000–2024)": 1.0,
    "Horizon 2030 — Pacte National Énergie (+1,3%)": 1.013,
    "Horizon 2060 — Vision Alafia SSP2-4.5 (+2,8%)": 1.028,
    "Horizon 2060 — Vision Alafia SSP5-8.5 (+4,5%)": 1.045,
}


# ==============================================================================
# CHARGEMENT MIS EN CACHE (ACCÉLÈRE L'APPLICATION WEB)
# ==============================================================================
@st.cache_data
def charger_geojson(path):
    return gpd.read_file(path)


@st.cache_data
def charger_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        transform = src.transform
    return data, extent, transform


if not os.path.exists(GEOJSON_BENIN):
    st.error(f"Fichier GeoJSON introuvable : {GEOJSON_BENIN}")
    st.stop()

gdf_benin = charger_geojson(GEOJSON_BENIN)
noms_dept = (
    sorted(gdf_benin["shapeName"].tolist())
    if "shapeName" in gdf_benin.columns
    else [
        "Alibori", "Atacora", "Atlantique", "Borgou", "Collines", "Couffo",
        "Donga", "Littoral", "Mono", "Ouémé", "Plateau", "Zou",
    ]
)

# ==============================================================================
# BARRE LATÉRALE : PARAMÈTRES DE SIMULATION
# ==============================================================================
st.sidebar.header("⚙️ Paramètres de Simulation")

idx_littoral = noms_dept.index("Littoral") if "Littoral" in noms_dept else 0
dept = st.sidebar.selectbox("1. Département du Bénin :", noms_dept, index=idx_littoral)

altitudes = list(RASTERS_VITESSE.keys())
alt = st.sidebar.selectbox("2. Altitude du moyeu :", altitudes, index=altitudes.index("120 m"))

turb_nom = st.sidebar.selectbox("3. Aérogénérateur de référence :", list(TURBINES.keys()))
scen = st.sidebar.selectbox("4. Horizon & Scénario Climatique :", list(SCENARIOS.keys()))

facteur_scen = SCENARIOS[scen]
tif_path = RASTERS_VITESSE[alt]

if not os.path.exists(tif_path):
    st.error(f"Le raster {os.path.basename(tif_path)} est introuvable dans le dossier data/.")
    st.stop()

# ==============================================================================
# CALCULS SPATIAUX & PHYSIQUES (WEIBULL + PRODUCTIBLE AEP)
# ==============================================================================
data_brut, extent, transform = charger_raster(tif_path)
data = data_brut * facteur_scen

if "shapeName" in gdf_benin.columns:
    sub_gdf = gdf_benin[gdf_benin["shapeName"] == dept]
    if sub_gdf.empty:
        sub_gdf = gdf_benin.iloc[[0]]
else:
    sub_gdf = gdf_benin.iloc[[0]]

point_rep = sub_gdf.geometry.representative_point().iloc[0]
row, col = rasterio.transform.rowcol(transform, point_rep.x, point_rep.y)
row = int(np.clip(row, 0, data.shape[0] - 1))
col = int(np.clip(col, 0, data.shape[1] - 1))

v_moy = float(data[row, col])
if np.isnan(v_moy):
    v_moy = float(np.nanmean(data))

# Paramètres de Weibull & Densité de Puissance Éolienne (DPE)
k = 2.85 if dept in ["Littoral", "Atlantique", "Ouémé", "Mono"] else 2.35
c = v_moy / gamma(1.0 + 1.0 / k)
dpe = 0.5 * 1.18 * (c ** 3) * gamma(1.0 + 3.0 / k)

# Courbe de puissance de l'aérogénérateur
spec = TURBINES[turb_nom]
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

bilan = {
    "Departement": dept,
    "Altitude": alt,
    "Scenario": scen,
    "Turbine": turb_nom,
    "V_moy_ms": round(v_moy, 2),
    "Weibull_k": round(k, 2),
    "Weibull_c_ms": round(c, 2),
    "DPE_Wm2": round(dpe, 1),
    "AEP_MWh_an": round(aep_mwh, 1),
    "Facteur_Charge_pct": round(fc_pct, 1),
    "CO2_Evite_t_an": round(co2_t, 1),
}

# Bouton d'export CSV dans la barre latérale
csv_bytes = pd.DataFrame([bilan]).to_csv(index=False, sep=";").encode("utf-8-sig")
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="💾 Télécharger le rapport (CSV)",
    data=csv_bytes,
    file_name=f"Bilan_Eolien_{dept}_{alt.replace(' ', '')}.csv",
    mime="text/csv",
    use_container_width=True,
)

# ==============================================================================
# INDICATEURS CLÉS (KPI)
# ==============================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vitesse moyenne", f"{v_moy:.2f} m/s")
c2.metric("Densité de puissance", f"{dpe:.1f} W/m²")
c3.metric("Productible (AEP)", f"{aep_mwh:,.1f} MWh/an".replace(",", " "))
c4.metric("Facteur de charge", f"{fc_pct:.1f} %")
c5.metric("CO₂ évité", f"{co2_t:,.1f} t/an".replace(",", " "))

# ==============================================================================
# GRAPHIQUES : CARTE RASTER XGBOOST & COURBE DE WEIBULL
# ==============================================================================
col_map, col_weib = st.columns([1.05, 1.15])

with col_map:
    fig_map, ax_map = plt.subplots(figsize=(5.5, 6.2), dpi=120)
    im = ax_map.imshow(data, cmap="YlOrRd", extent=extent, origin="upper")
    gdf_benin.plot(ax=ax_map, facecolor="none", edgecolor="#263238", linewidth=0.8)
    sub_gdf.plot(ax=ax_map, facecolor="none", edgecolor="#0d47a1", linewidth=2.4)
    ax_map.set_xlim(0.65, 4.0)
    ax_map.set_ylim(6.1, 12.5)
    ax_map.set_title(f"Raster XGBoost ({alt}) — {dept}", fontsize=10.5, fontweight="bold")
    ax_map.set_xlabel("Longitude (°E)", fontsize=8.5)
    ax_map.set_ylabel("Latitude (°N)", fontsize=8.5)
    cbar = fig_map.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Vitesse du vent (m/s)", fontsize=8.5)
    fig_map.tight_layout()
    st.pyplot(fig_map)
    plt.close(fig_map)

with col_weib:
    fig_w, ax_w = plt.subplots(figsize=(6.0, 5.2), dpi=120)
    ax_w.plot(v_grid, pdf, color="#1565c0", lw=2.3, label=f"Weibull (k={k:.2f}, c={c:.2f} m/s)")
    ax_w.fill_between(v_grid, pdf, color="#bbdefb", alpha=0.45)
    ax_w.axvline(v_moy, color="#c62828", linestyle="--", lw=1.6, label=f"Vmoy = {v_moy:.2f} m/s")
    ax_w.axvline(spec["v_in"], color="#2e7d32", linestyle=":", lw=1.4, label=f"Démarrage = {spec['v_in']} m/s")
    ax_w.set_title(f"Distribution Statistique à {alt} ({dept})", fontsize=10.5, fontweight="bold")
    ax_w.set_xlabel("Vitesse du vent (m/s)", fontsize=9)
    ax_w.set_ylabel("Densité de probabilité f(v)", fontsize=9)
    ax_w.grid(True, linestyle=":", alpha=0.5)
    ax_w.legend(fontsize=8.5)
    fig_w.tight_layout()
    st.pyplot(fig_w)
    plt.close(fig_w)

    st.markdown("#### Synthèse technique détaillée")
    df_recap = pd.DataFrame(
        {
            "Paramètre": [
                "Département", "Altitude du moyeu", "Scénario climatique",
                "Aérogénérateur", "Puissance nominale", "Paramètre de forme (k)",
                "Paramètre d'échelle (c)",
            ],
            "Valeur": [
                dept, alt, scen, turb_nom,
                f"{spec['P_nom']:.0f} kW", f"{k:.2f}", f"{c:.2f} m/s",
            ],
        }
    )
    st.dataframe(df_recap, hide_index=True, use_container_width=True)

# Tableau multi-altitudes optionnel s'il est présent dans data/
if os.path.exists(CSV_DEPT):
    with st.expander("📊 Consulter le bilan complet des 12 départements (multi-altitudes)"):
        st.dataframe(pd.read_csv(CSV_DEPT), use_container_width=True)
