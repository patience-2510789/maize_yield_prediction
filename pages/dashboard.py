import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/master_dataset.csv")

df = load_data()
df_model = df.dropna(subset=["Yield_MT_Ha"]).copy()

st.title("📊 Project Dashboard")
st.subheader("Maize Yield Analysis — Uasin Gishu County, Kenya (2005–2023)")
st.markdown("A visual summary of historical trends, climate patterns, and model insights.")
st.divider()

# ── KPI METRICS ───────────────────────────────────────────────
st.subheader("Key Statistics")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Years of Data", "19")
with col2:
    st.metric("Avg Yield", str(round(df_model["Yield_MT_Ha"].mean(), 2)) + " t/ha")
with col3:
    st.metric("Best Year", str(df_model.loc[df_model["Yield_MT_Ha"].idxmax(), "Year"]))
with col4:
    st.metric("Worst Year", str(df_model.loc[df_model["Yield_MT_Ha"].idxmin(), "Year"]))
with col5:
    st.metric("Model R²", "0.76")

st.divider()

# ── CHART 1: YIELD OVER TIME ──────────────────────────────────
st.subheader("🌽 Maize Yield Trend 2005–2023")

fig1, ax1 = plt.subplots(figsize=(14, 4))
variety_colors = {
    "H614D": "steelblue", "DH04": "green",
    "DK8031": "orange", "H6213": "purple"
}
for _, row in df_model.iterrows():
    color = variety_colors.get(row["Dominant_Variety"], "gray")
    ax1.bar(row["Year"], row["Yield_MT_Ha"], color=color, alpha=0.85)

ax1.plot(df_model["Year"], df_model["Yield_MT_Ha"],
         "o-", color="black", linewidth=1.5, markersize=4, alpha=0.6)
ax1.axhline(df_model["Yield_MT_Ha"].mean(), color="red",
            linestyle="--", linewidth=1.5,
            label="Average: " + str(round(df_model["Yield_MT_Ha"].mean(), 2)) + " t/ha")

from matplotlib.patches import Patch
legend_elements = [Patch(color=c, label=v) for v, c in variety_colors.items()]
legend_elements.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Average"))
ax1.legend(handles=legend_elements, loc="upper left")
ax1.set_xlabel("Year")
ax1.set_ylabel("Yield (t/ha)")
ax1.grid(True, alpha=0.2, axis="y")
st.pyplot(fig1)
plt.close()

st.caption("Bars coloured by dominant seed variety. Clear yield jump when DH04 and DK8031 were adopted after 2012.")
st.divider()

# ── CHART 2: WEATHER TRENDS ───────────────────────────────────
st.subheader("🌦 Climate Trends 2005–2024")

col_w1, col_w2 = st.columns(2)

with col_w1:
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(df["Year"], df["Rainfall_mm"], "o-", color="steelblue", linewidth=2)
    ax2.fill_between(df["Year"], df["Rainfall_mm"],
                     alpha=0.15, color="steelblue")
    ax2.set_title("Annual Rainfall (mm)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

with col_w2:
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.plot(df["Year"], df["Rainfall_LongRains_mm"],
             "o-", color="green", linewidth=2, label="Long Rains (Mar-May)")
    ax3.plot(df["Year"], df["Rainfall_ShortRains_mm"],
             "o-", color="orange", linewidth=2, label="Short Rains (Oct-Dec)")
    ax3.set_title("Seasonal Rainfall Comparison")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Rainfall (mm)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()

col_w3, col_w4 = st.columns(2)

with col_w3:
    fig4, ax4 = plt.subplots(figsize=(7, 3.5))
    ax4.plot(df["Year"], df["Temp_Avg_C"], "o-", color="red", linewidth=2)
    ax4.fill_between(df["Year"], df["Temp_Min_C"], df["Temp_Max_C"],
                     alpha=0.15, color="red", label="Min-Max range")
    ax4.set_title("Temperature Range (C)")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Temperature (C)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    plt.close()

with col_w4:
    fig5, ax5 = plt.subplots(figsize=(7, 3.5))
    ax5.plot(df["Year"], df["NDVI_LongRains"],
             "o-", color="darkgreen", linewidth=2, label="Long Rains NDVI")
    ax5.plot(df["Year"], df["NDVI_ShortRains"],
             "o-", color="limegreen", linewidth=2, label="Short Rains NDVI")
    ax5.set_title("Vegetation Index (NDVI) by Season")
    ax5.set_xlabel("Year")
    ax5.set_ylabel("NDVI (0-1)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    plt.close()

st.divider()

# ── CHART 3: CORRELATIONS ─────────────────────────────────────
st.subheader("🔗 What Correlates With Yield?")

features_to_check = [
    "Rainfall_mm", "Rainfall_LongRains_mm", "Rainfall_ShortRains_mm",
    "Temp_Avg_C", "NDVI_LongRains", "NDVI_ShortRains",
    "Soil_Wetness", "Humidity_Pct", "Variety_Code"
]

correlations = df_model[features_to_check + ["Yield_MT_Ha"]].corr()["Yield_MT_Ha"].drop("Yield_MT_Ha").sort_values()

fig6, ax6 = plt.subplots(figsize=(10, 4))
colors_corr = ["red" if c < 0 else "green" for c in correlations.values]
bars = ax6.barh(correlations.index, correlations.values, color=colors_corr, alpha=0.8)
ax6.axvline(x=0, color="black", linewidth=0.8)
ax6.set_title("Correlation with Maize Yield (green = positive, red = negative)")
ax6.set_xlabel("Correlation Coefficient")
ax6.grid(True, alpha=0.3, axis="x")
st.pyplot(fig6)
plt.close()

st.caption("Variety_Code has the strongest positive correlation — confirming seed variety is the biggest yield driver.")
st.divider()

# ── CHART 4: YIELD BY SEED VARIETY ───────────────────────────
st.subheader("🌱 Average Yield by Seed Variety")

variety_avg = df_model.groupby("Dominant_Variety")["Yield_MT_Ha"].agg(["mean","min","max","count"]).reset_index()
variety_avg.columns = ["Variety", "Avg Yield", "Min Yield", "Max Yield", "Years Used"]
variety_avg = variety_avg.sort_values("Avg Yield", ascending=False)

col_v1, col_v2 = st.columns(2)

with col_v1:
    fig7, ax7 = plt.subplots(figsize=(7, 4))
    colors_v = [variety_colors.get(v, "gray") for v in variety_avg["Variety"]]
    bars7 = ax7.bar(variety_avg["Variety"], variety_avg["Avg Yield"],
                    color=colors_v, alpha=0.85)
    ax7.bar_label(bars7, fmt="%.2f", padding=3)
    ax7.set_ylabel("Average Yield (t/ha)")
    ax7.set_title("Average Yield by Seed Variety")
    ax7.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig7)
    plt.close()

with col_v2:
    st.dataframe(
        variety_avg,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ── MODEL PERFORMANCE ─────────────────────────────────────────
st.subheader("🤖 Model Performance Summary")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

features = [
    "Rainfall_mm", "Rainfall_LongRains_mm", "Rainfall_ShortRains_mm",
    "Temp_Avg_C", "Temp_Max_C", "Temp_Min_C",
    "Solar_Radiation", "Humidity_Pct", "Soil_Wetness",
    "NDVI_LongRains", "NDVI_ShortRains",
    "Variety_Code", "Yield_Potential_Mid", "Maturity_Months"
]

X = df_model[features]
y = df_model["Yield_MT_Ha"]

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
y_pred = cross_val_predict(pipeline, X, y, cv=LeaveOneOut())

mae = round(mean_absolute_error(y, y_pred), 4)
r2  = round(r2_score(y, y_pred), 4)

col_m1, col_m2 = st.columns(2)

with col_m1:
    fig8, ax8 = plt.subplots(figsize=(7, 4))
    ax8.plot(df_model["Year"], y.values, "o-", color="green",
             linewidth=2, label="Actual")
    ax8.plot(df_model["Year"], y_pred, "s--", color="orange",
             linewidth=2, label="Predicted")
    ax8.fill_between(df_model["Year"], y.values, y_pred,
                     alpha=0.2, color="red", label="Error")
    ax8.set_xlabel("Year")
    ax8.set_ylabel("Yield (t/ha)")
    ax8.set_title("Actual vs Predicted Yield")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    st.pyplot(fig8)
    plt.close()

with col_m2:
    st.markdown("### Model Metrics")
    st.metric("R² Score", str(r2), help="How much yield variation the model explains (1.0 = perfect)")
    st.metric("Mean Absolute Error", str(mae) + " t/ha", help="Average prediction error")
    st.metric("Training Years", "19")
    st.metric("Algorithm", "Linear Regression")
    st.markdown("""
    **Interpretation:**
    - R² of 0.76 means the model explains **76% of yield variation**
    - Average predictions are off by only **0.30 t/ha**
    - Worst prediction was 2020 due to abnormally high rainfall
    - Model performs best for years with normal weather patterns
    """)

st.caption("Dashboard — Maize Yield Prediction Project | Uasin Gishu County | 2026")