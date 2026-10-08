from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from geopy.distance import geodesic
import os
import time
import folium
from flask import redirect

app = Flask(__name__)

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}


# Load models
models = joblib.load('all_models.pkl')
log_reg_model = models['type_model']
scaler = models['scaler']

# Load crime data
crime_df = pd.read_csv("crime_data.csv", parse_dates=["Month"])
crime_df.dropna(subset=["Longitude", "Latitude", "Crime type", "Month"], inplace=True)
crime_df["Year"] = crime_df["Month"].dt.year
crime_df["MonthNum"] = crime_df["Month"].dt.month
crime_df["CrimeTypeIndex"] = crime_df["Crime type"].factorize()[0]

# Area coordinates
area_coords = crime_df.groupby("Postcode").agg({'Latitude': 'mean', 'Longitude': 'mean'}).dropna()
area_coords.index = area_coords.index.str.upper()

# Load predictions
try:
    hotspot_df = pd.read_csv("hotspot_prediction_summary.csv", usecols=["Area", "CrimeCount", "HotspotPrediction"])
    hotspot_df["Area"] = hotspot_df["Area"].str.strip().str.upper()
except:
    hotspot_df = pd.DataFrame(columns=["Area", "CrimeCount", "HotspotPrediction"])

try:
    trend_df = pd.read_csv("trend_prediction_summary.csv")
    trend_df["Area"] = trend_df["Area"].str.strip().str.upper()
except:
    trend_df = pd.DataFrame(columns=["Area", "PrevCount", "PredictedCount", "TrendPrediction"])

postcode_cache = {}

# ====================== UTILITY FUNCTIONS ======================

def get_hotspot_status(area_code, lat, lon):
    area_code = area_code.strip().upper()
    match = hotspot_df[hotspot_df["Area"] == area_code]
    if not match.empty:
        return match.iloc[0]["HotspotPrediction"]

    valid_coords = area_coords.loc[area_coords.index.isin(hotspot_df["Area"].values)].copy()
    if not valid_coords.empty:
        valid_coords["distance"] = valid_coords.apply(
            lambda row: geodesic((lat, lon), (row["Latitude"], row["Longitude"])).km, axis=1)
        nearest_area = valid_coords.sort_values("distance").index[0]
        return hotspot_df[hotspot_df["Area"] == nearest_area]["HotspotPrediction"].values[0]
    return "Unknown"

def get_trend_status(area_code, lat, lon):
    area_code = area_code.strip().upper()
    match = trend_df[trend_df["Area"] == area_code]

    if not match.empty:
        pred = match.iloc[0]
    else:
        # Fallback to nearest area
        valid_coords = area_coords.loc[area_coords.index.isin(trend_df["Area"].values)].copy()
        if valid_coords.empty:
            return "Unknown"
        valid_coords["distance"] = valid_coords.apply(
            lambda row: geodesic((lat, lon), (row["Latitude"], row["Longitude"])).km, axis=1)
        nearest_area = valid_coords.sort_values("distance").index[0]
        nearby_match = trend_df[trend_df["Area"] == nearest_area]
        if nearby_match.empty:
            return "Unknown"
        pred = nearby_match.iloc[0]

    if pred["PredictedCount"] > pred["PrevCount"]:
        return "Predicted Increase"
    elif pred["PredictedCount"] < pred["PrevCount"]:
        return "Predicted Decrease"
    else:
        return "No Change"



def get_location_from_postcode(area):
    if not area or len(area.strip()) < 3:
        return None
    area = area.strip()

    if area in postcode_cache:
        return postcode_cache[area]

    try:
        # First, try postcode API (only if looks like postcode)
        if ' ' in area or any(char.isdigit() for char in area):
            res = requests.get(f"https://api.postcodes.io/postcodes/{area}", timeout=4)
            data = res.json()
            if data['status'] == 200:
                result = {
                    "latitude": data['result']['latitude'],
                    "longitude": data['result']['longitude'],
                    "address": f"{data['result']['admin_ward']}, {data['result']['admin_district']}",
                    "area_code": data['result']['postcode'].upper()
                }
                postcode_cache[area] = result
                return result
    except:
        pass

    try:
        # Fallback: Try general location lookup with Nominatim
        res = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={area}, London, UK&format=json&limit=1",
            headers={'User-Agent': 'CrimeAwareApp'}, timeout=5)
        data = res.json()
        if data:
            display = data[0]['display_name']
            result = {
                "latitude": float(data[0]['lat']),
                "longitude": float(data[0]['lon']),
                "address": display,
                "area_code": area.upper()
            }
            postcode_cache[area] = result
            return result
    except:
        pass

    return None


def get_crime_count_nearby(lat, lon, radius_km=5.0):
    lat_margin = radius_km / 111
    lon_margin = radius_km / (111 * np.cos(np.radians(lat)))
    filtered = crime_df[
        (crime_df["Latitude"].between(lat - lat_margin, lat + lat_margin)) &
        (crime_df["Longitude"].between(lon - lon_margin, lon + lon_margin))
    ]
    crimes_nearby = [row for _, row in filtered.iterrows()
                     if geodesic((lat, lon), (row["Latitude"], row["Longitude"])).km <= radius_km]
    return len(crimes_nearby), crimes_nearby

# ======================== ROUTES ========================
@app.route('/')
def home():
    return render_template("index.html", crimes=[], center_lat=51.5074, center_lon=-0.1278, area=None)
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    area = data.get("area", "").strip()

    if not area:
        return jsonify({"success": False, "error": "No area provided."})

    try:
        location = get_location_from_postcode(area)
        if not location:
            return jsonify({"success": False, "error": "Area not found."})

        lat = location["latitude"]
        lon = location["longitude"]

        if area.lower() == "london":
            # London bounding box (approximate)
            min_lat, max_lat = 51.28, 51.70
            min_lon, max_lon = -0.53, 0.23

            # Filter crimes within London bounds
            nearby = crime_df[
                (crime_df["Latitude"].between(min_lat, max_lat)) &
                (crime_df["Longitude"].between(min_lon, max_lon))
            ].copy()
        else:
            # Nearby search logic for other areas
            radius_km = 1.0
            lat_margin = radius_km / 111
            lon_margin = radius_km / (111 * np.cos(np.radians(lat)))

            nearby = crime_df[
                (crime_df["Latitude"].between(lat - lat_margin, lat + lat_margin)) &
                (crime_df["Longitude"].between(lon - lon_margin, lon + lon_margin))
            ].copy().head(300)

        crimes = []
        for _, row in nearby.iterrows():
            crimes.append({
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
                "Crime type": row["Crime type"],
                "Month": str(row["Month"].date()),
                "Hotspot": "Unknown",
                "Trend": "Unknown",
                "MostCommon": row["Crime type"],
                "TotalCrimes": 1,
                "LSOA": row.get("LSOA name", "Area not specified") if "LSOA name" in row else "Area not specified"
            })

        return jsonify({
            "success": True,
            "lat": lat,
            "lon": lon,
            "crimes": crimes,
            "area": location["address"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

import time

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    postcode = request.args.get('postcode') or request.form.get('postcode')

    if not postcode:
        return render_template("predict.html", error=None)

    location = get_location_from_postcode(postcode)
    if not location:
        return render_template("predict.html", error="Location not found.")

    lat = location['latitude']
    lon = location['longitude']
    address = location['address']
    area_code = location['area_code']

    crime_count, nearby = get_crime_count_nearby(lat, lon, radius_km=2.0)

    crime_types = [r["Crime type"] for r in nearby]
    most_common = max(set(crime_types), key=crime_types.count) if crime_types else "N/A"

    hotspot = get_hotspot_status(area_code, lat, lon)
    trend = get_trend_status(area_code, lat, lon)

    result = {
        "address": address,
        "crime_count": crime_count,
        "crime_type": most_common,
        "hotspot": hotspot,
        "trend": trend,
        "area_code": area_code,
        "postcode": postcode
    }

    return render_template("predict.html", result=result, postcode=postcode)

@app.route('/trends', methods=['GET', 'POST'])
def trends():
    # Get user input from form or URL query
    area_input = request.args.get("area", "") if request.method == "GET" else request.form.get("area", "")
    crime_type_filter = request.form.get("crime_type", "All") if request.method == "POST" else "All"

    filtered_df = crime_df.copy()
    address_display = area_input  # default fallback

    # If area provided, locate and filter by proximity
    if area_input:
        location = get_location_from_postcode(area_input)
        if location:
            lat = location["latitude"]
            lon = location["longitude"]
            address_display = location["address"]
            lat_margin = 2 / 111
            lon_margin = 2 / (111 * np.cos(np.radians(lat)))

            # Filter to crimes within 2km of center
            filtered_df = filtered_df[
                (filtered_df["Latitude"].between(lat - lat_margin, lat + lat_margin)) &
                (filtered_df["Longitude"].between(lon - lon_margin, lon + lon_margin))
            ]

    # Filter by crime type if selected
    if crime_type_filter != "All":
        filtered_df = filtered_df[filtered_df["Crime type"] == crime_type_filter]

    # Generate trend data grouped by month
    if not filtered_df.empty:
        filtered_df["MonthStr"] = filtered_df["Month"].dt.strftime("%Y-%m")
        trend_group = filtered_df.groupby("MonthStr").size().sort_index()
        trend_data = {
            "labels": list(trend_group.index),
            "data": [int(val) for val in trend_group.values]
        }
    else:
        trend_data = None

    return render_template("trends.html",
                           trend_data=trend_data,
                           area=address_display,
                           crime_types=sorted(crime_df["Crime type"].unique()),
                           selected_crime=crime_type_filter)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
