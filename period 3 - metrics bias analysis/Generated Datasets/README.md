# 📝 Period 3 — Metrics & Bias Analysis

In Period 3, we moved from exploration to a **structured quantification of ERA5 temperature bias**.  
After confirming that Reggio–Messina lacked reliable ground truth, we shifted to **Central Italy**, where ECAD station coverage is strong.

We selected **12 stations** representing rural/urban and coastal/interior environments to understand how land cover and geography influence the ERA5 cold bias.

We generated a **daily dataset (2020–2023)** with **18 features per station**, combining:
- ECAD Tmax  
- ERA5 Tmax, wind, precipitation  
- NDVI (Sentinel-2)  
- Urban fraction (impervious %)  
- Distance to sea  

These variables describe both **atmospheric drivers** and **surface characteristics**, forming the basis for all metrics.

📺 **Period 3 Video:**  
https://www.youtube.com/watch?v=S9myeSHVJIE

---

## 📁 Scripts in This Folder

**`00_diagnosis_all_stations.ipynb`**  
Evaluates ECAD station quality across Italy to identify valid stations.

**`00_select_reference_stations_central_italy.ipynb`**  
Applies filtering criteria to select the **12 final stations** for the study.

**`01_station_ERA5_build_daily_table.ipynb`**  
Integrates ECAD + ERA5 + EO datasets and produces the **full daily table with 18 features**.

**`02_metrics_season_UHI.ipynb`**  
Computes seasonal ERA5 bias by NDVI class, UHI category, and environment type.

**`03_metrics_distance_wind_rain_story.ipynb`**  
Quantifies how distance to sea, wind regime, and rainfall influence ERA5 error.

---

## 📄 Generated Dataset

Inside the **generated_datasets/** folder:

**`stations_daily_with_features.csv`**  
**`stations_daily_with_features.parquet`**

These files contain the **complete 2020–2023 daily dataset** for the 12 selected stations, including:
- station & ERA5 Tmax  
- wind speed + categories  
- precipitation + rainy-day frequency  
- NDVI  
- urban fraction  
- distance to sea  
- seasons  

This dataset supports both **metrics (Period 3)** and **modelling (Period 4)**.

---
