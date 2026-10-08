import pandas as pd
import joblib

# Load model and data
models = joblib.load("all_models.pkl")
rf_model = models["hotspot_model"]

# Load dataset
df = pd.read_csv("crime_data.csv", parse_dates=["Month"])
df.dropna(subset=["Longitude", "Latitude"], inplace=True)

# Fill missing postcode with LSOA code
df['Area'] = df['Postcode'].fillna(df['LSOA code'])

# Group by Area (Postcode or LSOA), get average coordinates + crime count
grouped = df.groupby('Area').agg({
    'Longitude': 'mean',
    'Latitude': 'mean',
    'Area': 'count'
}).rename(columns={'Area': 'CrimeCount'}).reset_index()

# Predict hotspot
X = grouped[['Longitude', 'Latitude', 'CrimeCount']]
grouped['HotspotPrediction'] = rf_model.predict(X)
grouped['HotspotPrediction'] = grouped['HotspotPrediction'].map({1: 'Hotspot', 0: 'Not Hotspot'})

# ✅ Save output as a proper CSV file
grouped[['Area', 'CrimeCount', 'HotspotPrediction']].sort_values(by='CrimeCount', ascending=False).to_csv(
    "hotspot_prediction_summary.csv", index=False)

print("✅ Output saved to 'hotspot_prediction_summary.csv'")



import pandas as pd
import numpy as np
import joblib

# Load data and model
df = pd.read_csv("crime_data.csv", parse_dates=["Month"])
models = joblib.load("all_models.pkl")
xgb_model = models["trend_model"]

# Clean and preprocess
df.dropna(subset=["Longitude", "Latitude", "Crime type", "Month"], inplace=True)
df['Year'] = df['Month'].dt.year
df['MonthNum'] = df['Month'].dt.month
df['CrimeTypeIndex'] = df['Crime type'].factorize()[0]

# Combine postcode/LSOA for area identification
df["Area"] = df["Postcode"].fillna(df["LSOA code"])

# Get latest month from data
latest_month = df["Month"].max()
year, month = latest_month.year, latest_month.month
quarter = (month - 1) // 3 + 1

records = []

# Process each area
for area in df["Area"].dropna().unique():
    area_data = df[df["Area"] == area]
    if area_data.empty:
        continue

    most_common = area_data["Crime type"].mode()[0]
    crime_index = area_data["CrimeTypeIndex"].mode()[0]

    # Previous month calculation
    last_month = month - 1 if month > 1 else 12
    last_year = year if month > 1 else year - 1

    prev_count = len(df[
        (df["Area"] == area) &
        (df["CrimeTypeIndex"] == crime_index) &
        (df["Year"] == last_year) &
        (df["MonthNum"] == last_month)
    ])

    trend_input = np.array([[year, month, crime_index, prev_count, quarter, quarter]])
    predicted = int(xgb_model.predict(trend_input)[0])
    delta = predicted - prev_count

    if delta > 5:
        trend = "Increase"
    elif delta < -5:
        trend = "Decrease"
    else:
        trend = "No Change"

    records.append({
        "Area": area,
        "PrevCount": prev_count,
        "PredictedCount": predicted,
        "TrendPrediction": trend
    })

# Save to CSV
trend_summary = pd.DataFrame(records)
trend_summary.to_csv("trend_prediction_summary.csv", index=False)

print("✅ trend_prediction_summary.csv created!")
