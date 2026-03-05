import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Maize Yield Predictor - Uasin Gishu",
    page_icon="🌽",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("data/processed/master_dataset.csv")
    df_model = df.dropna(subset=["Yield_MT_Ha"]).copy()
    features = [
        "Rainfall_mm", "Rainfall_LongRains_mm", "Rainfall_ShortRains_mm",
        "Temp_Avg_C", "Temp_Max_C", "Temp_Min_C",
        "Solar_Radiation", "Humidity_Pct", "Soil_Wetness",
        "NDVI_LongRains", "NDVI_ShortRains",
        "Variety_Code", "Yield_Potential_Mid", "Maturity_Months"
    ]
    X = df_model[features]
    y = df_model["Yield_MT_Ha"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    pipeline.fit(X, y)
    return pipeline, df_model, y.mean()

model, df_hist, avg_yield = load_model_and_data()

st.title("🌽 Maize Yield Predictor")
st.subheader("Uasin Gishu County, Kenya")
st.markdown("Predict maize harvest yield using weather, satellite and seed data.")
st.divider()

st.sidebar.title("Input Parameters")
st.sidebar.subheader("🌱 Seed Variety")
variety_options = {
    "H614D  (older, 6-8 t/ha)":   (1, 7.0, 6.5),
    "DH04   (medium, 7-9 t/ha)":  (2, 8.0, 7.0),
    "DK8031 (modern, 8-10 t/ha)": (3, 9.0, 6.5),
    "H6213  (modern, 7-9 t/ha)":  (4, 8.0, 7.0),
}
selected_variety = st.sidebar.selectbox("Dominant Variety", list(variety_options.keys()))
variety_code, yield_potential, maturity_months = variety_options[selected_variety]

st.sidebar.subheader("🌧 Rainfall")
rainfall_mm = st.sidebar.slider("Annual Rainfall (mm)", 2.0, 12.0, 6.0, 0.1)
rainfall_lr = st.sidebar.slider("Long Rains Mar-May (mm)", 10.0, 45.0, 22.0, 0.5)
rainfall_sr = st.sidebar.slider("Short Rains Oct-Dec (mm)", 5.0, 30.0, 15.0, 0.5)

st.sidebar.subheader("🌡 Temperature")
temp_avg = st.sidebar.slider("Average Temp (C)", 17.0, 21.0, 19.0, 0.1)
temp_max = st.sidebar.slider("Max Temp (C)", 25.0, 35.0, 31.0, 0.1)
temp_min = st.sidebar.slider("Min Temp (C)", 7.0, 13.0, 10.0, 0.1)

st.sidebar.subheader("Other Weather")
solar        = st.sidebar.slider("Solar Radiation", 19.0, 24.0, 21.5, 0.1)
humidity     = st.sidebar.slider("Humidity (%)", 65.0, 85.0, 75.0, 0.5)
soil_wetness = st.sidebar.slider("Soil Wetness (0-1)", 0.4, 1.0, 0.7, 0.01)

st.sidebar.subheader("🛰 Satellite NDVI")
ndvi_lr = st.sidebar.slider("NDVI Long Rains", 0.20, 0.35, 0.27, 0.001)
ndvi_sr = st.sidebar.slider("NDVI Short Rains", 0.15, 0.30, 0.25, 0.001)

input_df = pd.DataFrame([{
    "Rainfall_mm":            rainfall_mm,
    "Rainfall_LongRains_mm":  rainfall_lr,
    "Rainfall_ShortRains_mm": rainfall_sr,
    "Temp_Avg_C":             temp_avg,
    "Temp_Max_C":             temp_max,
    "Temp_Min_C":             temp_min,
    "Solar_Radiation":        solar,
    "Humidity_Pct":           humidity,
    "Soil_Wetness":           soil_wetness,
    "NDVI_LongRains":         ndvi_lr,
    "NDVI_ShortRains":        ndvi_sr,
    "Variety_Code":           variety_code,
    "Yield_Potential_Mid":    yield_potential,
    "Maturity_Months":        maturity_months
}])

predicted = round(float(model.predict(input_df)[0]), 3)
low  = round(predicted - 0.3, 3)
high = round(predicted + 0.3, 3)
diff = round(predicted - avg_yield, 3)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Yield", str(predicted) + " t/ha")
with col2:
    st.metric("Likely Range", str(low) + " - " + str(high) + " t/ha")
with col3:
    st.metric("vs Historical Avg", str(diff) + " t/ha")
with col4:
    st.metric("Historical Average", str(round(avg_yield, 3)) + " t/ha")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Historical Yield vs Your Prediction")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_hist["Year"], df_hist["Yield_MT_Ha"],
            "o-", color="green", linewidth=2, label="Historical")
    ax.axhline(predicted, color="orange", linewidth=2,
               linestyle="--", label="Your prediction: " + str(predicted) + " t/ha")
    ax.fill_between(
        [df_hist["Year"].min(), df_hist["Year"].max()],
        low, high, alpha=0.15, color="orange", label="Prediction range"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (t/ha)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Seed Variety Comparison")
    variety_info = {
        "H614D":  (1, 7.0, 6.5),
        "DH04":   (2, 8.0, 7.0),
        "DK8031": (3, 9.0, 6.5),
        "H6213":  (4, 8.0, 7.0),
    }
    variety_preds = {}
    for vname, (vcode, vpot, vmat) in variety_info.items():
        test = input_df.copy()
        test["Variety_Code"]       = vcode
        test["Yield_Potential_Mid"] = vpot
        test["Maturity_Months"]    = vmat
        variety_preds[vname] = round(float(model.predict(test)[0]), 3)

    selected_name = selected_variety.split()[0]
    bar_colors = ["orange" if v == selected_name else "steelblue"
                  for v in variety_preds.keys()]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(variety_preds.keys(), variety_preds.values(),
                   color=bar_colors, alpha=0.85)
    ax2.bar_label(bars, fmt="%.3f", padding=3)
    ax2.set_ylabel("Predicted Yield (t/ha)")
    ax2.set_title("Same weather conditions, different seed varieties")
    ax2.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig2)
    plt.close()

st.divider()
st.subheader("Historical Data")
st.dataframe(
    df_hist[[
        "Year", "Yield_MT_Ha", "Rainfall_mm",
        "Temp_Avg_C", "NDVI_LongRains", "Dominant_Variety", "Source"
    ]].rename(columns={
        "Yield_MT_Ha":      "Yield (t/ha)",
        "Rainfall_mm":      "Rainfall (mm)",
        "Temp_Avg_C":       "Avg Temp (C)",
        "NDVI_LongRains":   "NDVI Long Rains",
        "Dominant_Variety": "Seed Variety"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption("Model: Linear Regression | R2 = 0.76 | MAE = 0.30 t/ha | Data: NASA POWER, MODIS, ISRIC, County Agriculture Office")
