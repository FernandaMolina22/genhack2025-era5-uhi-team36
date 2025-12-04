# ERA5 Bias in Tmax — From Diagnosis to Action
_codellera andina — December 2025_

> **Purpose.** Turn a clear **ERA5 cold-bias diagnosis** into a **transparent, physics-based correction**, using only interpretable drivers (vegetation, urbanization, distance to sea, wind, rain).

---

## Table of Contents
1. [Executive Summary](#executive-summary)  
2. [Objectives](#objectives)  
3. [Data & Preprocessing](#data--preprocessing)  
4. [Diagnostic Analysis](#diagnostic-analysis)  
5. [Modeling Framework](#modeling-framework)  
6. [Results & Interpretation](#results--interpretation)  
7. [Operational Correction & Deployment](#operational-correction--deployment)  
8. [Sensitivity, Limitations, Next Steps](#sensitivity-limitations-next-steps)  
9. [Repository Structure](#repository-structure)  
10. [Code Snippets](#selected-code-snippets)
---

## Executive Summary
- ERA5 exhibits a **systematic cold bias** in Tmax (≈ **–1.5 to –1.8 °C**) with **larger dispersion in summer**.  
- Bias is **context-dependent**: amplified **near the coast** and in **urbanized/vegetated** settings; **wind** and **rain** tend to **mitigate** it.  
- A **linear model**, fit on **station×season** aggregates (48 rows; 12 stations × 4 seasons), explains the bias using **physical drivers** and yields a **portable correction**.  
- **Generalization:** **LOO RMSE ≈ 1.578 °C**.  
- **Standardized coefficients:** distance to sea **+0.965**, wind speed **+0.369**, vegetation (1–NDVI) **–0.299**.  
- **Correction formula:**  

$$
\boxed{T_{\text{adjusted}} = T_{\text{ERA5}} - \widehat{\text{bias}}}$$

- **So‑what:** Retain ERA5’s spatial patterns while delivering **decision-ready temperatures**.

---

## Objectives
- Quantify **ERA5 Tmax bias** relative to ground stations.  
- Understand how physical factors (vegetation, urbanization, distance to sea, wind, rainfall) influence this bias.  
- Build a **simple regression model** to explain the bias.  
- Communicate insights through visuals, metrics, and a clear narrative.

## Data & Preprocessing

### Station Selection
We selected **12 stations** across Central Italy, evenly distributed across rural/urban and coastal/interior environments, to study how land cover and geography influence the ERA5 cold bias.

### Sources
- **Stations:** 12 Central Italy stations, **2020–2023**.  
- **Reanalysis:** ERA5 Tmax collocated to stations.  
- **Surface/Meteo drivers:** NDVI, env_class/urban fraction, distance to sea (km), wind speed (WS), wind regime, precipitation.

### Core definitions
Let $y$ be station Tmax and $\tilde y$ ERA5 Tmax. Define **daily error**:  

$$
\boxed{\text{error}_t = \tilde y_t - y_t.}$$

Aggregate to **station × season** (for season $s$ at station $i$):  

$$
\boxed{\overline{\text{bias}}_{i,s} = \frac{1}{n_{i,s}}\sum_{t\in (i,s)} \text{error}_t.}$$

**Metrics** used throughout (for a set $\mathcal{E}$ of errors):

$$
\boxed{\mu = \frac{1}{n}\sum_{e\in\mathcal E} e,\quad
\sigma = \sqrt{\frac{1}{n-1}\sum_{e\in\mathcal E}(e-\mu)^2},\quad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{e\in\mathcal E} e^2}.}$$

### Feature construction (daily → seasonal)
- **Distance to sea:** use $\log(1 + d_{\text{km}})$ to stabilize coastal gradients.  
- **NDVI:** seasonal mean.  
- **Urbanization:** env_class or urban fraction.  
- **Wind:** seasonal mean WS; optional regimes (cardinal).  
- **Rain:** seasonal frequency $\mathbb{1}(\text{precip} > 0)$.

---

## Diagnostic Analysis

### Binning schemes
**Distance categories:**

$$
\boxed{\text{dist\\_cat} = \begin{cases}
\text{< 10 km}, & d < 10 \\
\text{10–50 km}, & 10 \le d \le 50 \\
\text{> 50 km}, & d > 50
\end{cases}}$$

**Wind speed categories:** with seasonal percentiles $P_{25}, P_{75}$ of WS,

$$
\boxed{\text{weak}:\; \text{WS} < P_{25},\quad
\text{medium}: P_{25} \le \text{WS} \le P_{75},\quad
\text{strong}: \text{WS} > P_{75}.}$$

**Rain categories:**

$$
\boxed{\text{dry}:\; \text{precip} = 0,\quad
\text{wet}:\; \text{precip} > 0,\quad
\text{heavy\\_rain}:\; \text{precip} \ge P_{90}(\text{precip}\mid \text{precip}>0).}$$

### Summary findings (from station/daily diagnostics)
- **Coast:** **<10 km → ~–2.2 °C** underestimation; **>50 km → ~+0.3 °C** (slightly positive).  
- **Wind:** bias ≈ **–1.7 °C** across categories; **northerlies ~–1.3 °C**, **southerly/westerly ~–2.0 °C**.  
- **Rain:** wet/heavy-rain days **reduce** cold bias.  
- **NDVI:** **low NDVI → –2 to –3 °C errors**; greener = smaller bias; Reggio Calabria: NDVI declines align with **hotter zones/UHI**.

---

## Modeling Framework

### Linear model
Let the station-season bias be $b_{i,s}$. With features $x_{i,s}\in\mathbb R^p$:

$$
\boxed{b_{i,s} = \beta_0 + \sum_{j=1}^p \beta_j x_{i,s,j} + \varepsilon_{i,s},\quad \varepsilon_{i,s}\sim\text{i.i.d.}}$$

**Standardization** for interpretability:  

$$
\boxed{x'_j = \frac{x_j - \mu_j}{\sigma_j},\quad \beta_j^{(std)} = \beta_j \cdot \sigma_j.}$$

**Group cross‑validation by station** (LOO): each fold holds out all seasons for one station to test generalization to unseen locations.

**Bias correction:** for any ERA5 Tmax:  

$$
\boxed{\;T^{\text{adjusted}} = T^{\text{ERA5}} - \widehat b(x)\;}$$

---

## Results & Interpretation

### Generalization
- **LOO RMSE**: **1.578 °C** (robust across stations).

### Standardized effects (sign & magnitude)
- **Distance to sea (+0.965):** moving **inland** reduces the cold bias; strongest underestimation **near the coast**.  
- **Wind speed (+0.369):** **ventilation** homogenizes temperature and reduces localized errors.  
- **Vegetation, 1–NDVI (–0.299):** **greener areas → stronger cold bias**, consistent with ERA5 **over‑cooling via ET** where sub-grid heterogeneity is large.

### Physical read‑out
The cold bias intensifies where **sub‑grid surface energy balance** dominates (coastal boundary layers, vegetated surfaces with strong ET). **Wind** and **inland fetch** mitigate this by mixing and reducing decoupling.

---

## Operational Correction & Deployment

### Formula

$$
\boxed{T^{\text{adjusted}} = T^{\text{ERA5}} - \widehat b\!\left(x_{\text{NDVI}},\;x_{\text{urban}},\;\log(1+d),\;\text{WS},\;\text{rain}\right).}$$

### Deployment checklist
1. Build seasonal features: **NDVI**, **env_class/urban fraction**, **log(1+distance_to_sea_km)**, **mean WS**, **rain frequency**.  
2. Predict $\widehat b$ using the fitted linear model.  
3. Compute **ERA5‑corrected Tmax** via $T^{adj}=T^{ERA5}-\widehat b$.  
4. **QA** against 1–2 independent stations; verify seasonal distribution and coastal gradient.  
5. Publish **bias maps** with metadata and driver layers for transparency.

---

## Sensitivity, Limitations, Next Steps
- **Nonlinearity near coast**: log transform is adequate; future work could add a **coastal regime term** within 10–30 km.  
- **Wind sector**: include **sector-wise effects** if directional data are stable.  
- **Rain intensity**: heavy‑rain signal suggests exploring **P90/P95** thresholds seasonally.  
- **Transferability**: validate coefficients in each new basin; **refit intercepts** with minimal local data.  
- **Add terrain/elevation** for more complex inland topography if extending beyond coastal basins.

---

## Repository Structure
The repository follows the four official hackathon periods:
```text
period1 - warmup exploration/     → Data exploration (ERA5, NDVI, stations)
period2 - visualization uhi/      → UHI visualization & communication
period3 - metrics bias analysis/  → Bias metrics + quantitative insight
period4 - modelling correction/   → Explanatory model (Linear/Ridge)
```
Each period folder contains:
- The **Jupyter notebooks**
- The **slides** for that week
- The **video**
- A local **README.md** summarizing the period

## Code Snippets

### Load & derive error
```python
import pandas as pd, numpy as np

df = pd.read_csv("/mnt/data/stations_daily_with_features.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Prefer existing 'error'; else ERA5 - station
if "error" not in df.columns:
    era5 = [c for c in df.columns if "era5" in c.lower() and any(k in c.lower() for k in ["t2m","temp","temperature"])][0]
    obs  = [c for c in df.columns if any(k in c.lower() for k in ["obs","station","meas"]) and any(k in c.lower() for k in ["t2m","temp","temperature"])][0]
    df["error"] = df[era5] - df[obs]
```

### Categories (distance, wind, rain)
```python
# Distance bins
dist = df["distance_to_sea_km"].astype(float)
df["dist_cat"] = pd.cut(dist, bins=[-np.inf,10,50,np.inf], labels=["< 10 km","10–50 km","> 50 km"], ordered=True)

# Wind categories (2022–2023 only)
c2 = df[df["date"] >= "2022-01-01"].copy()
p25, p75 = np.nanpercentile(c2["WS"], [25,75])
c2["wind_cat"] = pd.Categorical(
    np.where(c2["WS"] < p25, "weak", np.where(c2["WS"] > p75, "strong", "medium")),
    categories=["weak","medium","strong"], ordered=True
)

# Rain categories (2020–2023)
c3 = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2023-12-31")].copy()
c3["rain_cat"] = np.where(c3["precip"] > 0, "wet", "dry")
pos = c3.loc[c3["precip"] > 0, "precip"]
thr = np.nanpercentile(pos, 90) if len(pos) else np.nan
if np.isfinite(thr) and (pos >= thr).sum() >= 20:
    c3.loc[c3["precip"] >= thr, "rain_cat"] = "heavy_rain"
```

### Metrics & summaries
```python
def rmse(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan

def summarize(df, by, col="error"):
    g = df.groupby(by)[col]
    out = g.agg(error_mean="mean", error_std="std", n="count").reset_index()
    out["error_rmse"] = g.apply(rmse).values
    return out

summary_distance = summarize(df.dropna(subset=["dist_cat","error"]), ["dist_cat"])
summary_wind     = summarize(c2.dropna(subset=["wind_cat","error"]), ["wind_cat"])
summary_rain     = summarize(c3.dropna(subset=["rain_cat","error"]), ["rain_cat"])
```

### Linear model & correction (group CV by station)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import numpy as np

feat_num = ["ndvi","urban_fraction","log_dist_km","wind_speed","rain_freq"]
feat_cat = ["env_class","wind_regime"]
df["log_dist_km"] = np.log1p(df["distance_to_sea_km"])

X = df[feat_num + feat_cat].copy()
y = df["error"].astype(float).values
groups = df["station_id"].astype(str).values

ct = ColumnTransformer([
    ("num", StandardScaler(), feat_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feat_cat)
])

pipe = Pipeline([("prep", ct), ("lin", LinearRegression())])

gkf = GroupKFold(n_splits=5)
pred = np.full_like(y, np.nan, dtype=float)

for tr, te in gkf.split(X, y, groups):
    m = Pipeline([("prep", ct), ("lin", LinearRegression())])
    m.fit(X.iloc[tr], y[tr])
    pred[te] = m.predict(X.iloc[te])

rmse_cv = float(np.sqrt(np.nanmean((y - pred)**2)))
print("LOO-like RMSE:", rmse_cv)

# Fit final & correct ERA5
pipe.fit(X, y)
df["predicted_bias"] = pipe.predict(X)
df["era5_corrected"] = df["era5_t2m"] - df["predicted_bias"]
```
