# ============================================
# ============================================

import pandas as pd
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA

# ===============================
# ===============================
file_path = r"C:\Users\praka_32k187u\Downloads\archive (1)\city_day.csv"

if not os.path.exists(file_path):
    print("❌ Dataset not found! Please check your path.")
    exit()

data = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# ===============================
# ===============================
print("\n🧹 Cleaning dataset...")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with missing essential columns
data = data.dropna(subset=['AQI', 'City', 'Date'])

data = data.sort_values(['City', 'Date'])


pollutant_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]
for col in pollutant_cols:
    if col in data.columns:
        data[col] = data.groupby('City')[col].transform(lambda x: x.interpolate(limit_direction='both'))

cleaned_data_path = os.path.join(os.getcwd(), "cleaned_city_day.csv")
data.to_csv(cleaned_data_path, index=False)

print(f"✅ Cleaned dataset saved at: {cleaned_data_path}")
print("\n🔹 Preview of Cleaned Dataset:")
print(data.head(5))

# ===============================
models_dir = "city_models"
os.makedirs(models_dir, exist_ok=True)

cities = data['City'].unique()
print(f"\n🏙️ Training models for {len(cities)} cities...")

for city in cities:
    city_data = data[data['City'] == city]
    city_data = city_data.set_index('Date')
    city_data = city_data.asfreq('D')  # ensure daily frequency
    city_data['AQI'] = city_data['AQI'].interpolate()  # fill missing AQI

    try:
        model = ARIMA(city_data['AQI'], order=(2, 1, 2))
        model_fit = model.fit()
        model_path = os.path.join(models_dir, f"model_{city}.pkl")
        joblib.dump(model_fit, model_path)
        print(f"✅ Model trained and saved for {city}")
    except Exception as e:
        print(f"⚠️ Skipped {city} due to error: {e}")

print("\n✅ All city models saved in 'city_models' folder.")
print(f"📁 Cleaned dataset available at: {cleaned_data_path}")

# ===============================
# ===============================
try:
    print("\n📂 Opening cleaned dataset in Excel...")
    os.startfile(cleaned_data_path)  # Opens CSV directly in Excel
except Exception as e:
    print(f"⚠️ Could not open Excel automatically: {e}")
