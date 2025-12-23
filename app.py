import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------
# 🎯 APP CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌤️",
    layout="wide"
)

st.title("🌍 Air Quality Predictor Dashboard")
st.markdown("Predict tomorrow’s AQI and view today’s live data dynamically for your city.")

# ----------------------------
# ⚙️ LOAD ML MODEL (Ensure model.pkl is in same folder)
# ----------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("❌ Model file not found! Please ensure 'model.pkl' exists in this folder.")
    st.stop()

# ----------------------------
# 🌐 FETCH LIVE CITY LIST (Dynamic)
# ----------------------------
st.header("📍 Select Location")

API_TOKEN = "92d92e0a62986754f283368a25884572b56be8e7"
url = f"https://api.waqi.info/map/bounds/?token={API_TOKEN}&latlng=6.0,68.0,36.0,98.0"  # Roughly India bounds

cities = {}

try:
    res = requests.get(url, timeout=10).json()
    if res["status"] == "ok":
        for s in res["data"]:
            if s.get("station") and s.get("lat") and s.get("lon"):
                cities[s["station"]["name"]] = (s["lat"], s["lon"])

        if not cities:
            st.warning("⚠️ No live city data found — switching to manual entry.")
            city_name = st.text_input("Enter your city name:")
            lat, lon = None, None
        else:
            city_name = st.selectbox("Select your city (live data):", list(cities.keys()))
            lat, lon = cities[city_name]
    else:
        st.warning("⚠️ Unable to fetch city list from API.")
        city_name = st.text_input("Enter your city name manually:")
        lat, lon = None, None
except Exception as e:
    st.error("❌ Error fetching city data.")
    city_name = st.text_input("Enter your city name manually:")
    lat, lon = None, None

# ----------------------------
# 🌤️ FETCH LIVE AQI DATA
# ----------------------------
if city_name:
    st.subheader(f"📊 Live AQI Data for {city_name}")

    if lat and lon:
        live_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={API_TOKEN}"
    else:
        live_url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"

    try:
        live_data = requests.get(live_url, timeout=10).json()
        if live_data["status"] == "ok":
            aqi = live_data["data"]["aqi"]
            dominent = live_data["data"].get("dominentpol", "N/A").upper()
            st.metric(label="Current AQI", value=aqi)
            st.write(f"**Dominant Pollutant:** {dominent}")
        else:
            st.warning("⚠️ Live AQI data not available.")
            aqi = None
    except Exception:
        st.error("❌ Failed to fetch live AQI data.")
        aqi = None
else:
    st.info("Please select or enter your city to continue.")
    st.stop()

# ----------------------------
# 📈 USER INPUT FOR PREDICTION
# ----------------------------
st.header("🤖 Predict Tomorrow's AQI")

col1, col2, col3 = st.columns(3)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=60.0)
with col2:
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=100.0)
with col3:
    temp = st.number_input("Temperature (°C)", min_value=-10.0, value=30.0)

col4, col5, col6 = st.columns(3)
with col4:
    humidity = st.number_input("Humidity (%)", min_value=0.0, value=45.0)
with col5:
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=2.0)
with col6:
    pressure = st.number_input("Pressure (hPa)", min_value=800.0, value=1000.0)

# ----------------------------
# 🧮 PREDICTION
# ----------------------------
if st.button("🚀 Predict Tomorrow's AQI"):
    try:
        input_data = np.array([[pm25, pm10, temp, humidity, wind_speed, pressure]])
        pred_aqi = model.predict(input_data)[0]
        st.success(f"🌤️ Predicted AQI for Tomorrow in {city_name}: **{pred_aqi:.2f}**")

        # Visualization
        today = datetime.now().strftime("%b %d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%b %d")
        plt.figure(figsize=(5, 3))
        plt.bar([today, tomorrow], [aqi if aqi else 0, pred_aqi], color=["skyblue", "orange"])
        plt.title("Today's vs Tomorrow's AQI")
        plt.ylabel("AQI Value")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----------------------------
# 📉 FOOTER
# ----------------------------
st.markdown("---")
st.caption("Developed by Raj R | Powered by WAQI API + Machine Learning")
