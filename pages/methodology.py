import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="📖", layout="wide")

st.title("📖 Methodology & Documentation")
st.subheader("Maize Yield Prediction Model — Uasin Gishu County, Kenya")
st.divider()

# ── OVERVIEW ──────────────────────────────────────────────────
st.subheader("🎯 Project Overview")
st.markdown("""
This project develops a **Machine Learning model** to predict maize harvest yields 
in Uasin Gishu County, Kenya. The model uses historical climate, satellite, soil, 
and seed variety data to forecast yield in tonnes per hectare (t/ha).

**Why this matters:**
- Uasin Gishu is Kenya's largest maize-producing county, contributing over 30% of national output
- Early yield predictions help farmers, county government, and food security planners make better decisions
- Climate variability is increasing — data-driven tools help communities adapt

**Target users:**
- County Agriculture Officers
- Farmers and cooperatives
- Researchers and students
- Food security planners
""")

st.divider()



# ── DATA SOURCES ──────────────────────────────────────────────
st.subheader("📂 Data Sources")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🌦 Weather & Climate")
    st.markdown("""
    **Source:** NASA POWER (power.larc.nasa.gov)  
    **Coverage:** 2005–2024 (20 years)  
    **Resolution:** Monthly  
    **Location:** Lat 0.5204, Lon 35.2699 (Eldoret)  
    **Variables collected:**
    - Annual and seasonal rainfall (mm)
    - Average, min, max temperature (°C)
    - Solar radiation (MJ/m²/day)
    - Relative humidity (%)
    - Root zone soil wetness (0–1)
    
    **Why NASA POWER?** Free, satellite-derived, globally 
    consistent, and specifically designed for agricultural 
    applications. Zero missing values in our dataset.
    """)

    st.markdown("#### 🪨 Soil Properties")
    st.markdown("""
    **Source:** ISRIC SoilGrids (rest.isric.org)  
    **Coverage:** Static (soil doesn't change year to year)  
    **Depth:** 0–30cm topsoil layer  
    **Sampling:** 4 locations across Uasin Gishu averaged  
    **Variables collected:**
    - Soil pH (water): 6.05
    - Nitrogen (g/kg): 3.245
    - Clay content (%): 40.0
    - Organic carbon (g/kg): 36.85
    
    **Key finding:** Uasin Gishu has slightly acidic, 
    nitrogen-rich Nitisol soils — ideal for maize production.
    """)

with col2:
    st.markdown("#### 🛰 Satellite Vegetation (NDVI)")
    st.markdown("""
    **Source:** NASA MODIS via ORNL DAAC (modis.ornl.gov)  
    **Product:** MOD13Q1 — Vegetation Indices 16-Day 250m  
    **Coverage:** 2005–2024 (20 years)  
    **Variables collected:**
    - NDVI Long Rains (March–July average)
    - NDVI Short Rains (October–December average)
    
    **What is NDVI?** Normalized Difference Vegetation Index 
    measures crop health from satellite imagery. Values range 
    from 0 (bare soil) to 1 (dense vegetation). Uasin Gishu 
    values range 0.23–0.29 — typical for mixed highland farmland.
    
    **Why two seasons?** Kenya has two maize growing seasons. 
    Long Rains (March–July) is the main season, Short Rains 
    (October–December) is secondary.
    """)

    st.markdown("#### 🌽 Maize Yield Records")
    st.markdown("""
    **Source:** Uasin Gishu County Agriculture Office  
    **Coverage:** 2012–2023 (confirmed), 2005–2011 (estimated)  
    **Variables collected:**
    - Area planted (Hectares)
    - Total production (Metric Tonnes)
    - Yield (Tonnes per Hectare)
    
    **Note:** 2005–2011 figures are research-based estimates 
    from published Kenya agricultural surveys. These are clearly 
    flagged as "Estimated" in the dataset.
    """)

    st.markdown("#### 🌱 Seed Variety Data")
    st.markdown("""
    **Source:** KEPHIS + CIMMYT adoption studies  
    **Coverage:** 2005–2024  
    **Variables collected:**
    - Dominant variety per year
    - Yield potential range (t/ha)
    - Maturity period (months)
    
    **Key varieties:**
    - H614D (2005–2012): older hybrid, 6–8 t/ha potential
    - DH04 (2013–2016): improved hybrid, 7–9 t/ha potential
    - DK8031 (2017–2022): modern hybrid, 8–10 t/ha potential
    - H6213 (2023–2024): modern hybrid, 7–9 t/ha potential
    """)

st.divider()

# ── DATA PIPELINE ─────────────────────────────────────────────
st.subheader("⚙️ Data Collection Pipeline")
st.markdown("""
All data was collected programmatically using Python scripts organized into Jupyter notebooks:

| Notebook | Purpose | Output |
|----------|---------|--------|
| 01_weather_download.ipynb | Download NASA POWER climate data | weather_clean.csv |
| 02_soil_data.ipynb | Query ISRIC SoilGrids API | soil_clean.csv |
| 03_ndvi_satellite.ipynb | Extract MODIS NDVI via ORNL API | ndvi_clean.csv |
| 04_yield_and_seeds.ipynb | Compile yield and seed records | yield_data.csv, seeds_clean.csv |
| 05_master_merge.ipynb | Merge all sources into one dataset | master_dataset.csv |
| 06_modelling.ipynb | Train and evaluate ML models | maize_yield_model.pkl |
""")

st.divider()

# ── MASTER DATASET ────────────────────────────────────────────
st.subheader("📋 Master Dataset Structure")
st.markdown("""
The final dataset has **19 complete rows** (2005–2023) and **24 columns**:

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| Year | int | All | 2005–2023 |
| **Yield_MT_Ha** | float | County office | **TARGET variable** |
| Rainfall_mm | float | NASA POWER | Annual total |
| Rainfall_LongRains_mm | float | NASA POWER | March–May sum |
| Rainfall_ShortRains_mm | float | NASA POWER | October–December sum |
| Temp_Avg_C | float | NASA POWER | Annual mean |
| Temp_Max_C | float | NASA POWER | Heat stress indicator |
| Temp_Min_C | float | NASA POWER | Cold stress indicator |
| Solar_Radiation | float | NASA POWER | MJ/m²/day |
| Humidity_Pct | float | NASA POWER | Relative humidity |
| Soil_Wetness | float | NASA POWER | Root zone 0–1 |
| NDVI_LongRains | float | MODIS | Vegetation health Mar–Jul |
| NDVI_ShortRains | float | MODIS | Vegetation health Oct–Dec |
| Soil_pH | float | ISRIC | Static — 6.05 |
| Soil_Nitrogen | float | ISRIC | Static — 3.245 g/kg |
| Clay_Pct | float | ISRIC | Static — 40% |
| Organic_Carbon | float | ISRIC | Static — 36.85 g/kg |
| Dominant_Variety | string | KEPHIS | Seed variety name |
| Variety_Code | int | KEPHIS | Encoded: 1–4 |
| Yield_Potential_Mid | float | KEPHIS | Theoretical ceiling t/ha |
| Maturity_Months | float | KEPHIS | Growing period |
| Area_Ha | int | County office | Hectares planted |
| Production_MT | float | County office | Total production |
| Source | string | — | "County Report" or "Estimated" |
""")

st.divider()

# ── MODELLING ─────────────────────────────────────────────────
st.subheader("🤖 Modelling Approach")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    #### Model Selection
    Four models were trained and compared:
    
    | Model | MAE | RMSE | R² |
    |-------|-----|------|----|
    | **Linear Regression** | **0.300** | **0.364** | **0.763** |
    | Ridge Regression | 0.265 | 0.377 | 0.745 |
    | Random Forest | 0.288 | 0.375 | 0.748 |
    | Gradient Boosting | 0.295 | 0.402 | 0.711 |
    
    **Linear Regression was selected** as the final model because:
    - Highest R² score (0.763)
    - Simple and interpretable — important for stakeholder trust
    - With only 19 data points, simpler models generalise better
    - Coefficients are directly explainable to non-technical users
    """)

with col_m2:
    st.markdown("""
    #### Validation Strategy
    **Leave-One-Out Cross Validation (LOO-CV)** was used because:
    - Dataset has only 19 rows — standard train/test split wastes too much data
    - LOO-CV trains on 18 years and tests on 1, repeated for every year
    - Gives the most honest estimate of real-world performance on small datasets
    
    #### Key Findings
    - **Seed variety** is the strongest predictor (41% importance)
    - **Yield potential** of the variety is second (40% importance)
    - **Rainfall** and **NDVI** add meaningful but smaller signals
    - **Temperature** negatively correlates with yield in this region
    - The jump from H614D to modern hybrids explains most yield gains since 2013
    """)

st.divider()

# ── LIMITATIONS ───────────────────────────────────────────────
st.subheader("⚠️ Limitations & Future Work")

col_l1, col_l2 = st.columns(2)

with col_l1:
    st.markdown("""
    #### Current Limitations
    - **Small dataset (19 rows)** — more years would improve accuracy
    - **County-level only** — sub-county or ward-level data would be more precise
    - **2005–2011 yield estimates** — not confirmed official records
    - **Static soil data** — soil properties assumed constant across years
    - **Single dominant variety** — actual fields have mixed variety adoption
    - **No pest/disease data** — outbreaks like MLN (Maize Lethal Necrosis) affect yield
    - **2020 outlier** — extreme rainfall year was poorly predicted
    """)

with col_l2:
    st.markdown("""
    #### Recommended Future Improvements
    - Collect sub-county yield data to increase dataset size
    - Add pest and disease outbreak records as binary features
    - Incorporate fertilizer application rates per year (actual, not recommended)
    - Add irrigation coverage data
    - Retrain annually as new county reports are published
    - Explore LSTM (time series deep learning) when more data is available
    - Validate 2005–2011 estimates with KNBS agricultural surveys
    """)

st.divider()

# ── HOW TO RUN ────────────────────────────────────────────────
st.subheader("🚀 How to Run This Project")
st.markdown("""
#### Requirements
- Python 3.10 or higher
- VS Code (recommended)

#### Installation
1. Clone the repository:
   `git clone https://github.com/patience-2510789/maize_yield_prediction.git`

2. Install dependencies:
   `py -m pip install -r requirements.txt`

3. Run the app:
   `py -m streamlit run app.py`

#### Reproducing the Data Collection
Run the notebooks in order inside the `notebooks/` folder:
1. `01_weather_download.ipynb`
2. `02_soil_data.ipynb`
3. `03_ndvi_satellite.ipynb`
4. `04_yield_and_seeds.ipynb`
5. `05_master_merge.ipynb`
6. `06_modelling.ipynb`
""")

st.divider()

# ── REFERENCES ────────────────────────────────────────────────
st.subheader("📚 References & Data Citations")
st.markdown("""
1. **NASA POWER Project** — Prediction of Worldwide Energy Resources. 
   NASA Langley Research Center. https://power.larc.nasa.gov

2. **ISRIC World Soil Information** — SoilGrids: Global gridded soil information. 
   https://soilgrids.org

3. **NASA MODIS** — MOD13Q1 Vegetation Indices 16-Day 250m. 
   ORNL DAAC. https://modis.ornl.gov

4. **KEPHIS** — Kenya Plant Health Inspectorate Service. 
   Variety catalogue. https://www.kephis.org

5. **CIMMYT** — International Maize and Wheat Improvement Center. 
   Maize variety adoption studies, Kenya highlands.

6. **County Government of Uasin Gishu** — Annual Agricultural Production Reports 2012–2023.

7. **KNBS** — Kenya National Bureau of Statistics. 
   Economic Survey, Agriculture Chapter. https://www.knbs.or.ke
""")

st.caption("Methodology Documentation — Maize Yield Prediction Project | KCA University | 2026")
