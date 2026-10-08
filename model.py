import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import warnings
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
df['CrimeTypeIndex'] = df['Crime type'].factorize()[0].astype('int16')

# ---------------------- Crime Trend Prediction ----------------------
trend_df = df.groupby(['Year', 'MonthNum', 'Crime type']).size().reset_index(name='CrimeCount')
trend_df['CrimeTypeIndex'] = trend_df['Crime type'].factorize()[0].astype('int16')
trend_df['PrevCrimeCount'] = trend_df.groupby('Crime type')['CrimeCount'].shift(1)
trend_df.dropna(inplace=True)

trend_df['CrimeIncrease'] = (trend_df['CrimeCount'] > trend_df['PrevCrimeCount']).astype('int8')
trend_df['MonthQuarter'] = ((trend_df['MonthNum'] - 1) // 3 + 1).astype('int8')
trend_df['YearQuarter'] = trend_df['MonthQuarter']

X_trend = trend_df[['Year', 'MonthNum', 'CrimeTypeIndex', 'PrevCrimeCount', 'MonthQuarter', 'YearQuarter']]
y_trend = trend_df['CrimeIncrease']
X_train_trend, X_test_trend, y_train_trend, y_test_trend = train_test_split(X_trend, y_trend, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_trend_resampled, y_train_trend_resampled = smote.fit_resample(X_train_trend, y_train_trend)

xgb_trend = xgb.XGBClassifier(
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0
)
xgb_trend.fit(X_train_trend_resampled, y_train_trend_resampled)

# ---------------------- Crime Hotspot Prediction ----------------------
hotspot_df = df.groupby(['Longitude', 'Latitude']).size().reset_index(name='CrimeCount')
thresh = np.percentile(hotspot_df['CrimeCount'], 80)
hotspot_df['Hotspot'] = (hotspot_df['CrimeCount'] >= thresh).astype('int8')

X_hotspot = hotspot_df[['Longitude', 'Latitude', 'CrimeCount']]
y_hotspot = hotspot_df['Hotspot']
X_train_hot, X_test_hot, y_train_hot, y_test_hot = train_test_split(X_hotspot, y_hotspot, test_size=0.2, random_state=42, stratify=y_hotspot)

smote_hot = SMOTE(random_state=42)
X_train_hot_res, y_train_hot_res = smote_hot.fit_resample(X_train_hot, y_train_hot)

rf_hot = RandomForestClassifier(random_state=42)

param_grid_hot = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 15, 18],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'class_weight': ['balanced']
}

hotspot_search = RandomizedSearchCV(
    rf_hot,
    param_distributions=param_grid_hot,
    n_iter=10,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3),
    verbose=1,
    random_state=42,
    n_jobs=-1
)

hotspot_search.fit(X_train_hot_res, y_train_hot_res)
best_rf_hot = hotspot_search.best_estimator_

# ---------------------- Crime Type Prediction ----------------------
X_type = df[['Year', 'MonthNum', 'CrimeTypeIndex', 'Longitude', 'Latitude']]
y_type = df['CrimeTypeIndex']
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_type_scaled = scaler.fit_transform(X_train_type)
X_test_type_scaled = scaler.transform(X_test_type)

log_reg = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=42)
log_reg.fit(X_train_type_scaled, y_train_type)

# ---------------------- Evaluation ----------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

evaluate_model("Crime Trend Model", y_test_trend, xgb_trend.predict(X_test_trend))
evaluate_model("Crime Hotspot Model", y_test_hot, best_rf_hot.predict(X_test_hot))
evaluate_model("Crime Type Model", y_test_type, log_reg.predict(X_test_type_scaled))

# ---------------------- Accuracy Comparison ----------------------
accuracies = [
    accuracy_score(y_test_trend, xgb_trend.predict(X_test_trend)),
    accuracy_score(y_test_hot, best_rf_hot.predict(X_test_hot)),
    accuracy_score(y_test_type, log_reg.predict(X_test_type_scaled))
]
labels = ['Trend Model', 'Hotspot Model', 'Type Model']

plt.figure(figsize=(8, 5))
plt.barh(labels, accuracies, color='mediumseagreen')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0.5, 1.0)
plt.grid(True, axis='x', linestyle='--')
plt.tight_layout()
plt.show()

# ---------------------- Save All Models ----------------------
joblib.dump({
    'trend_model': xgb_trend,
    'hotspot_model': best_rf_hot,
    'type_model': log_reg,
    'scaler': scaler
}, 'all_models.pkl', compress=3)

print(" All models trained and saved successfully as 'all_models.pkl'")
