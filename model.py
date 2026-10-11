import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
warnings.filterwarnings("ignore")

# ---------------------- Load Data ----------------------
df = pd.read_csv("crime_data.csv", dtype={
    'Longitude': 'float32',
    'Latitude': 'float32',
    'Crime type': 'category'
}, parse_dates=['Month'])

df.columns = df.columns.str.strip()
df.dropna(subset=["Longitude", "Latitude", "Crime type", "Month"], inplace=True)
df['Year'] = df['Month'].dt.year.astype('int16')
df['MonthNum'] = df['Month'].dt.month.astype('int8')

# ---------------------- Crime Hotspot Prediction ----------------------
# A location is a hotspot if its crime count is in the top 20% of locations.
# The model predicts this from LOCATION ONLY (longitude, latitude). Crime count
# is deliberately not an input, because it is what defines the label.
hotspot_df = df.groupby(['Longitude', 'Latitude']).size().reset_index(name='CrimeCount')
thresh = np.percentile(hotspot_df['CrimeCount'], 80)
hotspot_df['Hotspot'] = (hotspot_df['CrimeCount'] >= thresh).astype('int8')

X_hotspot = hotspot_df[['Longitude', 'Latitude']]
y_hotspot = hotspot_df['Hotspot']
X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(
    X_hotspot, y_hotspot, test_size=0.2, random_state=42, stratify=y_hotspot)

# SMOTE sits inside the pipeline so it is applied only to the training folds
hotspot_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])
param_grid_hot = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [10, 12, 15, 18],
    'rf__min_samples_split': [2, 4, 6],
    'rf__min_samples_leaf': [1, 2, 3],
}
hotspot_search = RandomizedSearchCV(
    hotspot_pipe,
    param_distributions=param_grid_hot,
    n_iter=10,
    scoring='f1',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    random_state=42,
    n_jobs=1
)
hotspot_search.fit(X_train_hot, y_train_hot)
best_rf_hot = hotspot_search.best_estimator_

# ---------------------- Crime Trend Prediction ----------------------
# Predicts whether a crime type's monthly count will rise compared with the previous month
trend_df = df.groupby(['Year', 'MonthNum', 'Crime type']).size().reset_index(name='CrimeCount')
trend_df['CrimeTypeIndex'] = trend_df['Crime type'].factorize()[0].astype('int16')
trend_df['PrevCrimeCount'] = trend_df.groupby('Crime type')['CrimeCount'].shift(1)
trend_df.dropna(inplace=True)
trend_df['CrimeIncrease'] = (trend_df['CrimeCount'] > trend_df['PrevCrimeCount']).astype('int8')
trend_df['MonthQuarter'] = ((trend_df['MonthNum'] - 1) // 3 + 1).astype('int8')
trend_df['YearQuarter'] = trend_df['MonthQuarter']

X_trend = trend_df[['Year', 'MonthNum', 'CrimeTypeIndex', 'PrevCrimeCount', 'MonthQuarter', 'YearQuarter']]
y_trend = trend_df['CrimeIncrease']
X_train_trend, X_test_trend, y_train_trend, y_test_trend = train_test_split(
    X_trend, y_trend, test_size=0.2, random_state=42)
X_train_trend_res, y_train_trend_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(
    X_train_trend, y_train_trend)

xgb_trend = xgb.XGBClassifier(
    random_state=42, objective='binary:logistic', eval_metric='logloss',
    n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0)
xgb_trend.fit(X_train_trend_res, y_train_trend_res)

# ---------------------- Evaluation ----------------------
hot_pred = best_rf_hot.predict(X_test_hot)
print("\n===== Crime Hotspot Model (Random Forest, location only) =====")
print("Always-'not hotspot' baseline accuracy:", round(1 - y_test_hot.mean(), 3))
print("Accuracy:", round(accuracy_score(y_test_hot, hot_pred), 3))
print("Hotspot recall:", round(recall_score(y_test_hot, hot_pred), 3))
print("Hotspot F1:", round(f1_score(y_test_hot, hot_pred), 3))
print(classification_report(y_test_hot, hot_pred, digits=3))

trend_pred = xgb_trend.predict(X_test_trend)
print("===== Crime Trend Model (XGBoost) =====")
print("Test rows:", len(y_test_trend), "| Accuracy:", round(accuracy_score(y_test_trend, trend_pred), 3))

# ---------------------- Save Models ----------------------
joblib.dump({'trend_model': xgb_trend, 'hotspot_model': best_rf_hot}, 'all_models.pkl', compress=3)
print("\nModels saved to 'all_models.pkl'")
