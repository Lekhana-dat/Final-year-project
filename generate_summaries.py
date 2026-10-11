import pandas as pd
import numpy as np
import joblib

models = joblib.load("all_models.pkl")
rf_model = models["hotspot_model"]
xgb_model = models["trend_model"]

df = pd.read_csv("crime_data.csv", parse_dates=["Month"])
df.dropna(subset=["Longitude", "Latitude"], inplace=True)
df["Area"] = df["Postcode"].fillna(df["LSOA code"])

# ============ 1. Hotspot predictions by area (location only) ============
grouped = df.groupby("Area").agg(
    Longitude=("Longitude", "mean"),
    Latitude=("Latitude", "mean"),
    CrimeCount=("Area", "count")
).reset_index()

X = grouped[["Longitude", "Latitude"]]
grouped["HotspotPrediction"] = rf_model.predict(X)
grouped["HotspotPrediction"] = grouped["HotspotPrediction"].map({1: "Hotspot", 0: "Not Hotspot"})
grouped[["Area", "CrimeCount", "HotspotPrediction"]].sort_values(
    by="CrimeCount", ascending=False).to_csv("hotspot_prediction_summary.csv", index=False)
print("Saved hotspot_prediction_summary.csv")

# ============ 2. Trend predictions by area ============
df = df.dropna(subset=["Crime type", "Month"]).copy()
df["Year"] = df["Month"].dt.year
df["MonthNum"] = df["Month"].dt.month
df["CrimeTypeIndex"] = df["Crime type"].factorize()[0]

latest = df["Month"].max()
year, month = latest.year, latest.month
quarter = (month - 1) // 3 + 1
last_month = month - 1 if month > 1 else 12
last_year = year if month > 1 else year - 1

# Most common crime type per area, and its count in the previous month
common = df.groupby("Area")["CrimeTypeIndex"].agg(lambda s: s.mode().iat[0]).rename("CrimeTypeIndex").reset_index()
prev = (df[(df["Year"] == last_year) & (df["MonthNum"] == last_month)]
        .groupby(["Area", "CrimeTypeIndex"]).size().rename("PrevCount").reset_index())
trend = common.merge(prev, on=["Area", "CrimeTypeIndex"], how="left").fillna({"PrevCount": 0})
trend["PrevCount"] = trend["PrevCount"].astype(int)

features = pd.DataFrame({
    "Year": year, "MonthNum": month, "CrimeTypeIndex": trend["CrimeTypeIndex"],
    "PrevCrimeCount": trend["PrevCount"], "MonthQuarter": quarter, "YearQuarter": quarter})
# the model outputs 1 (count expected to rise) or 0 (not expected to rise)
trend["TrendPrediction"] = np.where(xgb_model.predict(features) == 1, "Increase", "No Increase")
trend[["Area", "PrevCount", "TrendPrediction"]].to_csv("trend_prediction_summary.csv", index=False)
print("Saved trend_prediction_summary.csv")
