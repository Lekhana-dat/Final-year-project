# London Crime Map and Hotspot Prediction (Final Year Project)
Flask web app that maps London crime and predicts hotspots and trends by area or postcode.

**Models** (`model.py`): XGBoost (will crime in an area increase next month?), Random Forest (hotspot or not), Logistic Regression; SMOTE for class imbalance.
**Front end** (`app.py`): Flask + Folium map, postcode lookup through postcodes.io and OpenStreetMap geocoding.

## Run order
1. Put `crime_data.csv` in this folder (monthly street-level crime records with columns `Month`, `Longitude`, `Latitude`, `Crime type`).
2. `pip install -r requirements.txt`
3. `python model.py` retrains the models and saves `all_models.pkl`
4. `python generate_summaries.py` creates `hotspot_prediction_summary.csv` and `trend_prediction_summary.csv`
5. `python app.py`, then open http://127.0.0.1:5000

## Missing from this upload (add before publishing)
- `crime_data.csv` and the two summary CSVs
- a `templates/` folder containing `index.html`, `predict.html`, `trends.html`, `about.html`, `contact.html`
- screenshots of the app, and the model accuracy figures from `model.py`'s output
