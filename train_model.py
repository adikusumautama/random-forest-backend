import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import json
# from xgboost import XGBRegressor
from datetime import datetime

# --- Load Data ---
df = pd.read_csv('data/damiu.csv')  # Ganti sesuai lokasi file kamu

# --- Parsing & Urutkan Tanggal ---
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df = df.sort_values(by='Tanggal')

# --- Fitur Waktu Dasar ---
df['hari_ke'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days
df['hari_dalam_minggu'] = df['Tanggal'].dt.dayofweek
df['bulan'] = df['Tanggal'].dt.month
df['minggu_ke'] = df['Tanggal'].dt.isocalendar().week
df['tahun'] = df['Tanggal'].dt.year
df['is_weekend'] = df['hari_dalam_minggu'].isin([5, 6]).astype(int)

# --- Fitur Awal/Akhir Bulan ---
df['is_awal_bulan'] = (df['Tanggal'].dt.day <= 3).astype(int)
df['is_akhir_bulan'] = (df['Tanggal'].dt.day >= 28).astype(int)

# --- Lag Fitur & Rolling ---
df['penjualan_kemarin'] = df['Galon Terjual'].shift(1)
df['penjualan_2hari_lalu'] = df['Galon Terjual'].shift(2)

df['rata2_3hari'] = df['Galon Terjual'].rolling(window=3).mean()
df['rata2_7hari'] = df['Galon Terjual'].rolling(window=7).mean()
df['rata2_14hari'] = df['Galon Terjual'].rolling(window=14).mean()

df['std_7hari'] = df['Galon Terjual'].rolling(window=7).std()

df['delta_penjualan'] = df['Galon Terjual'].diff()

# --- Bersihkan NaN akibat rolling & shift ---
df.dropna(inplace=True)

# --- Fitur & Target ---
feature_column_names = [
    'hari_ke',
    'hari_dalam_minggu',
    'bulan',
    'minggu_ke',
    'tahun',
    'is_weekend',
    'is_awal_bulan',
    'is_akhir_bulan',
    'penjualan_kemarin',
    'penjualan_2hari_lalu',
    'rata2_3hari',
    'rata2_7hari',
    'rata2_14hari',
    'std_7hari',
    'delta_penjualan'
]

target_column_name = 'Galon Terjual'

X = df[feature_column_names]
y = df[target_column_name]

# --- Split Data Train & Test (berdasarkan urutan waktu) ---
train_size_ratio = 0.8
train_index_end = int(len(df) * train_size_ratio)

X_train, X_test = X.iloc[:train_index_end], X.iloc[train_index_end:]
y_train, y_test = y.iloc[:train_index_end], y.iloc[train_index_end:]

# --- Latih Model Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
rf_model.fit(X_train, y_train)

# --- Evaluasi Random Forest ---
y_pred_test = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nEvaluasi Model pada Data Uji:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- Simpan Model Random Forest ---
joblib.dump(rf_model, 'random_forest_gallon_model.joblib')

# # --- Latih Model XGBoost ---
# xgb_model = XGBRegressor(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )
# xgb_model.fit(X_train, y_train)

# --- Evaluasi XGBoost ---
# y_pred_xgb = xgb_model.predict(X_test)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)
# rmse_xgb = np.sqrt(mse_xgb)
# r2_xgb = r2_score(y_test, y_pred_xgb)

# print(f"\nEvaluasi XGBoost pada Data Uji:")
# print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.2f}")
# print(f"R-squared (R2): {r2_xgb:.2f}")

# --- Simpan Model XGBoost ---
# joblib.dump(xgb_model, 'xgboost_gallon_model.joblib')



model_metadata = {
    'features': feature_column_names,
    'training_start_date': df['Tanggal'].min().isoformat(),
    'training_end_date': df['Tanggal'].max().isoformat(),
    'model_type': 'RandomForestRegressor',
    'rmse': rmse,
    'r2': r2
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)

print("\nModel dan metadata berhasil disimpan.")
