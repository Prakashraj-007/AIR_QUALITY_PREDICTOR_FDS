import pandas as pd

# Load dataset from your Downloads path
df = pd.read_csv(r"C:\Users\praka_32k187u\Downloads\archive (1)\city_day.csv")

# Keep only useful columns
cols_needed = ["City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
df = df[cols_needed]

# Drop missing values
df = df.dropna()

# Save cleaned dataset into your project folder
df.to_csv("AirQuality.csv", index=False)
print("✅ Cleaned dataset saved as AirQuality.csv with", len(df), "rows.")
