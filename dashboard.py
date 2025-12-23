import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------
# Load trained model
# ----------------------------------------
with open("aqi_model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------------------
# Load dataset (for city lookup)
# ----------------------------------------
DATA_PATH = r"C:\Users\praka_32k187u\Downloads\archive (1)\city_day.csv"
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["AQI"], inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(page_title="AI-Based Air Quality Predictor", page_icon="🌍", layout="centered")
st.title("🌍 AI-Based Air Quality Predictor")
st.markdown("### Predict Tomorrow's AQI for Your City (Offline Dataset Model)")

# City selection
cities = sorted(df["City"].dropna().unique())
city = st.selectbox("🏙️ Select your city:", cities)

# ----------------------------------------
# Predict AQI based on historical last day
# ----------------------------------------
if st.button("🔍 Predict Tomorrow's AQI"):
    city_data = df[df["City"] == city].sort_values("Date")

    if city_data.empty:
        st.error("❌ No data available for this city.")
    else:
        latest = city_data.iloc[-1]  # last available record

        # Extract pollutant features (same used in training)
        features = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        input_values = [latest[f] if f in city_data.columns else 0 for f in features]

        input_data = np.array([input_values])
        predicted_aqi = model.predict(input_data)[0]

        # AQI Category
        def get_aqi_category(aqi):
            if aqi <= 50:
                return "Good", "🟢", "#90EE90"
            elif aqi <= 100:
                return "Satisfactory", "🟡", "#FFFF99"
            elif aqi <= 200:
                return "Moderate", "🟠", "#FFD580"
            elif aqi <= 300:
                return "Poor", "🔴", "#FF9999"
            elif aqi <= 400:
                return "Very Poor", "🟣", "#C44DFF"
            else:
                return "Severe", "⚫", "#808080"

        category, emoji, color = get_aqi_category(predicted_aqi)

        # Background color
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {color};
                    color: black;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display results
        st.markdown(f"## 🌆 City: **{city}**")
        st.markdown(f"### 📅 Based on: **{latest['Date'].date()}**")
        st.markdown(f"### 🌤 Predicted AQI for Tomorrow: **{predicted_aqi:.2f}**")
        st.markdown(f"### {emoji} Air Quality Category: **{category}**")

        # Pollutant Visualization
        st.markdown("### 📊 Latest Pollutant Levels")
        pollutants = ["PM2.5", "PM10", "NO₂", "SO₂", "CO", "O₃"]
        values = input_values

        fig, ax = plt.subplots()
        ax.bar(pollutants, values, color="skyblue", edgecolor="black")
        ax.set_ylabel("Concentration (µg/m³ or mg/m³)")
        ax.set_title(f"{city} - Pollutant Levels")
        st.pyplot(fig)

        # AQI trend visualization
        st.markdown("### 📈 AQI Trend (Last 7 Days)")
        trend = city_data.tail(7)
        plt.figure()
        plt.plot(trend["Date"], trend["AQI"], marker='o')
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.title(f"AQI Trend for {city}")
        st.pyplot(plt)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption("Developed by CODEVENT | Offline AI-Based Air Quality Predictor 🌍")
