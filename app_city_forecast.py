import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

DATA_PATH = r"C:\Users\praka_32k187u\Downloads\archive (1)\city_day.csv"

try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("❌ Dataset not found! Please check your path.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"])

st.set_page_config(page_title="🌍 AI-Based Air Quality Predictor", page_icon="🌆", layout="centered")
st.title("🌆 Air Quality Predictor Dashboard")
st.subheader("AI-based City Air Quality Forecast System")

cities = sorted(data["City"].dropna().unique())
selected_city = st.selectbox("🏙️ Select your city:", cities)

if selected_city:
    city_data = data[data["City"] == selected_city].copy()
    city_data = city_data.sort_values("Date")

    city_data["AQI"] = pd.to_numeric(city_data["AQI"], errors="coerce")
    city_data["AQI"] = city_data["AQI"].interpolate()
    city_data = city_data.dropna(subset=["AQI"])

    city_data = city_data[(city_data["AQI"] > 0) & (city_data["AQI"] < 500)]

    city_data["Days"] = (city_data["Date"] - city_data["Date"].min()).dt.days

    if city_data.empty:
        st.warning("⚠️ Not enough valid AQI data for this city.")
        st.stop()

    X = city_data[["Days"]]
    y = city_data["AQI"]

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)

    st.write("### 🔮 AQI Forecast Trend (Predicted Only)")

    forecast_days = st.slider("Select how many future days to predict", 3, 30, 7)

    today = pd.Timestamp.now().normalize()
    future_dates = [today + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_X = np.arange(city_data["Days"].max() + 1,
                         city_data["Days"].max() + forecast_days + 1).reshape(-1, 1)
    predicted_aqi = model.predict(future_X)
    predicted_aqi = np.clip(predicted_aqi, 0, 500)


    def get_aqi_category(value):
        if value <= 50:
            return "Good", "#9CFF9C"
        elif value <= 100:
            return "Satisfactory", "#FFFF9C"
        elif value <= 200:
            return "Moderate", "#FFD79C"
        elif value <= 300:
            return "Poor", "#FF9C9C"
        elif value <= 400:
            return "Very Poor", "#B19CFF"
        else:
            return "Severe", "#FF4C4C"

    first_category, bg_color = get_aqi_category(predicted_aqi[0])

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


    st.write("### 📅 Detailed AQI Forecast")

    for date, value in zip(future_dates, predicted_aqi):
        category, color = get_aqi_category(value)
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                border-radius:10px;
                padding:10px 15px;
                margin-bottom:8px;
                font-size:16px;
                color:#000;
            ">
            🌤️ <b>{date.strftime('%d %B %Y')}</b> → 
            Predicted AQI for <b>{selected_city}</b>: <b>{value:.1f}</b> 
            ({category})
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_dates, predicted_aqi, color="orange", marker="o", linewidth=2)
    ax.fill_between(future_dates, predicted_aqi, color="orange", alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"Predicted AQI Trend for Next {forecast_days} Days in {selected_city}")
    ax.grid(True)
    st.pyplot(fig)

    # ============================================
    # ============================================
    pollutant_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]
    available_cols = [col for col in pollutant_cols if col in city_data.columns]
    pollutant_means = city_data[available_cols].mean().dropna()

    st.write("### 🧪 Average Pollutant Composition")

    if pollutant_means.empty or pollutant_means.sum() == 0:
        st.warning("⚠️ No pollutant data available for this city.")
    else:
        fig2, ax2 = plt.subplots()
        ax2.pie(
            pollutant_means,
            labels=pollutant_means.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)

    # ============================================
    # ============================================
    model_dir = "city_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{selected_city}.pkl")
    joblib.dump(model, model_path)
    st.caption(f"💾 Model for {selected_city} saved successfully!")

# ============================================
# ============================================
st.markdown("---")
st.caption("Developed by CODEVENT | AI-Based Air Quality Predictor 🌍")