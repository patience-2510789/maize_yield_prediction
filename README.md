# 🌽 Maize Yield Prediction — Uasin Gishu County, Kenya

A machine learning web application that predicts maize harvest yields in Uasin Gishu County using historical climate, satellite, soil, and seed variety data.

🔗 **Live App:** https://maizeyieldprediction-kujseettxbbr5yg2yzqixa.streamlit.app/

---

## 📌 Project Summary

| Item | Detail |
|------|--------|
| **Location** | Uasin Gishu County, Kenya |
| **Target** | Maize Yield (Tonnes/Hectare) |
| **Data Range** | 2005–2024 (20 years) |
| **Model** | Linear Regression (R² = 0.76, MAE = ±0.30 t/ha) |
| **Institution** | KCA University |

---

## 🗂 Project Structure
```
maize-prediction/
├── app.py                          # Main Streamlit app (Predictor)
├── requirements.txt                # Python dependencies
├── pages/
│   ├── 01_dashboard.py             # Charts & insights dashboard
│   └── 02_methodology.py           # Full methodology & documentation
├── notebooks/
│   ├── 01_weather_download.ipynb   # NASA POWER weather data
│   ├── 02_soil_data.ipynb          # ISRIC soil data
│   ├── 03_ndvi_satellite.ipynb     # MODIS NDVI satellite data
│   ├── 04_yield_and_seeds.ipynb    # Yield & seed variety data
│   ├── 05_master_merge.ipynb       # Merge all datasets
│   └── 06_modelling.ipynb          # Model training & evaluation
├── data/
│   ├── raw/                        # Raw downloaded files
│   └── processed/                  # Cleaned, merged datasets
└── outputs/                        # Charts and saved model
```

---

## 📂 Data Sources

| Data Type | Source | Years |
|-----------|--------|-------|
| Weather & Climate | NASA POWER | 2005–2024 |
| Satellite NDVI | NASA MODIS via ORNL | 2005–2024 |
| Soil Properties | ISRIC SoilGrids | Static |
| Maize Yield | Uasin Gishu County Agriculture Office | 2012–2023 |
| Seed Varieties | KEPHIS + CIMMYT | 2005–2024 |

---

## 🤖 Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Linear Regression** ✅ | **0.300** | **0.364** | **0.763** |
| Ridge Regression | 0.265 | 0.377 | 0.745 |
| Random Forest | 0.288 | 0.375 | 0.748 |
| Gradient Boosting | 0.295 | 0.402 | 0.711 |

Validated using **Leave-One-Out Cross Validation** — best strategy for small datasets.

---

## 🔑 Key Findings

- **Seed variety** is the strongest predictor of yield (41% importance)
- Adoption of modern hybrids (DH04, DK8031) after 2013 explains the yield jump from ~2.2 to ~4.0 t/ha
- **Long Rains rainfall** (March–May) positively correlates with yield
- **Higher temperatures** are associated with lower yields in this highland region
- **2024 predicted yield:** 3.415 t/ha (range: 3.115–3.715 t/ha)

---

## 🚀 Run Locally
```bash
# Clone the repository
git clone https://github.com/patience-2510789/maize_yield_prediction.git
cd maize_yield_prediction

# Install dependencies
py -m pip install -r requirements.txt

# Run the app
py -m streamlit run app.py
```

---

## 📓 Reproduce the Data Collection

Run notebooks in order from the `notebooks/` folder:

1. `01_weather_download.ipynb` — Downloads 20 years of NASA climate data
2. `02_soil_data.ipynb` — Queries ISRIC SoilGrids API for soil properties
3. `03_ndvi_satellite.ipynb` — Extracts MODIS NDVI via ORNL API
4. `04_yield_and_seeds.ipynb` — Compiles yield and seed variety records
5. `05_master_merge.ipynb` — Merges all sources into master dataset
6. `06_modelling.ipynb` — Trains, evaluates and compares ML models

---

## ⚠️ Limitations

- Small dataset (19 confirmed rows) — more years would improve accuracy
- 2005–2011 yield figures are research-based estimates, not official records
- County-level predictions only — sub-county granularity needs more data
- No pest/disease outbreak data included
- 2020 was an outlier year (abnormal rainfall) with the highest prediction error

---

## 📚 References

1. NASA POWER — https://power.larc.nasa.gov
2. ISRIC SoilGrids — https://soilgrids.org
3. NASA MODIS ORNL — https://modis.ornl.gov
4. KEPHIS — https://www.kephis.org
5. County Government of Uasin Gishu — Annual Agricultural Reports
6. KNBS — https://www.knbs.or.ke

---

## 📄 License

This project was developed for academic research purposes at KCA University, 2026.