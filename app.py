# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:00:17 2026

@author: HP
"""

import os
import shutil

DOSSIER_WEB = r"C:\ATLAS_EOLIEN_WEB"
DOSSIER_DATA = os.path.join(DOSSIER_WEB, "data")
os.makedirs(DOSSIER_DATA, exist_ok=True)

fichiers_sources = [
    r"D:\Desktop\MEMOIRE\Stations\geoBoundaries-BEN-ADM1.geojson",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Bilan_Multi_Altitudes_12_Departements.csv",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_10m_XGBoost.tif",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_20m_XGBoost.tif",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_53m_XGBoost.tif",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_86m_XGBoost.tif",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_120m_XGBoost.tif",
    r"C:\RESULTATS_MEMOIRE_EOLIEN\ATLAS_MULTI_ALTITUDES\Raster_Vitesse_150m_XGBoost.tif",
]

for f in fichiers_sources:
    if os.path.exists(f):
        shutil.copy2(f, DOSSIER_DATA)
        print("Copié :", os.path.basename(f))
    else:
        print("Introuvable :", f)

# Création automatique du fichier requirements.txt pour Streamlit Cloud
requirements = """streamlit
numpy
pandas
scipy
geopandas
rasterio
matplotlib
"""
with open(os.path.join(DOSSIER_WEB, "requirements.txt"), "w", encoding="utf-8") as req:
    req.write(requirements.strip() + "\n")

print("\nDossier Web prêt :", DOSSIER_WEB)