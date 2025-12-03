# 📝 Period 1 – Warmup & Exploration

During Period 1, our goal was to explore the core datasets that would define the rest of the hackathon: **ERA5 reanalysis**, **Sentinel NDVI products**, and **weather stations**. We focused on understanding data quality, spatial coverage, and early physical insights about **Reggio Calabria** and the surrounding region.

---

## 🌬️ ERA5 Analysis
We examined **10m wind**, **2m temperature**, and **total precipitation** time series for Reggio Calabria.

**Key findings:**
- Wind intensity is *stable across years*, suggesting limited long-term wind-driven variability.
- Summer shows *persistent dryness*, amplifying potential UHI and cold-bias effects.
- ERA5 spatial anomalies revealed *localized warm and cool cores*, indicating spatial heterogeneity relevant for bias studies.

---

## 🛰️ Sentinel NDVI (Sentinel-2 & Sentinel-3)

We compared **Sentinel-2 (S2)** and **Sentinel-3 (S3)** NDVI products.

**Challenges:**
- Different CRS between datasets.  
- Reprojecting S3 caused memory issues.  
- Initial scaling in S3 (0–255) required correction.  

**Outcome:**
- We processed NDVI directly in each dataset’s **native CRS**.  
- S2 provides **high spatial detail**, ideal for local vegetation analysis.  
- S3 provides **consistent temporal coverage**, useful for long-term behaviour.

---

## 🌡️ Weather Stations

We analyzed ECAD station availability and found an important limitation:

- No ECAD station exists in **Reggio Calabria**, our original AOI.
- The nearest stations (Messina, San Pier Niceto, Torregrotta) are across the Strait of Messina.
- These stations differ in **climate, topography, and urban form**, making it difficult to ensure representativeness.

---

## 📌 Conclusion

Period 1 highlighted the **data strengths and weaknesses** of Reggio Calabria.  
While the area presents a compelling UHI context, the **lack of local ground stations** raises questions about data reliability and future analysis feasibility.

---
