# London Crime Map and Hotspot Prediction (Final Year Project)

Flask web app that maps London street-level crime and predicts crime hotspots and trends by area or postcode.

## What it does
- Interactive Folium map of crime records with postcode and area lookup (postcodes.io and OpenStreetMap geocoding).
- Hotspot prediction by area: a Random Forest classifier flags whether an area falls in the top 20% of London locations by crime volume, using location (longitude, latitude) as input.
- Monthly crime trend charts by area and crime type, plus a next-month "increase / no increase" indicator from an XGBoost classifier.

## Data
Metropolitan and City of London Police street-level crime records, January 2024 to January 2025 (about 1.2 million records). Columns used: Month, Longitude, Latitude, Crime type, LSOA code, Postcode.

## Model results (held-out test set, from `model.py`)
Hotspot model (Random Forest, SMOTE inside cross-validation, tuned for F1):
| Metric | Result |
|---|---|
| Hotspot recall | 0.68 |
| Hotspot F1 | 0.53 |
| Accuracy | 0.76 (an always-"not hotspot" baseline scores 0.80, so recall and F1 are the main measures) |

Crime count is deliberately excluded from the inputs because it defines the hotspot label; including it produces an artificial 100% accuracy.

The trend model is trained on 13 months of monthly counts per crime type (34 test rows), so its output (about 50% accuracy) is indicative only.

## Run order
1. Place `crime_data.csv` in this folder (not included: file size).
2. `pip install -r requirements.txt`
3. `python model.py` trains the models, prints the evaluation and saves `all_models.pkl`
4. `python generate_summaries.py` creates `hotspot_prediction_summary.csv` and `trend_prediction_summary.csv`
5. `python app.py`, then open http://127.0.0.1:5000

## Still to add before publishing
- `templates/` folder (index.html, predict.html, trends.html, about.html, contact.html) and `static/`
- Screenshots of the app
