import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import json
from datetime import datetime

# Ganti dengan path file data Anda
# Contoh: df = pd.read_csv('D:/Skripsi/Data/damiu_sales_data.csv')
df = pd.read_csv('data/damiu.csv') # Menggunakan path relatif jika file ada di direktori yang sama

# Pastikan kolom tanggal di-parse dengan benar
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce') # Gunakan 'Tanggal' sesuai CSV
df.dropna(subset=['Tanggal'], inplace=True) # Hapus baris dengan tanggal yang tidak valid
df = df.sort_values(by='Tanggal')

print(df.head())
print(df.info())

# --- Feature Engineering ---
df['hari_ke'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days
df['hari_dalam_minggu'] = df['Tanggal'].dt.dayofweek # Senin=0, Minggu=6
df['bulan'] = df['Tanggal'].dt.month
# Tambahkan fitur lain yang relevan jika ada, misal:
# df['minggu_ke_dalam_tahun'] = df['Tanggal'].dt.isocalendar().week
# df['tahun'] = df['Tanggal'].dt.year

# Target variabel (apa yang ingin diprediksi)
target_column_name = 'Galon Terjual' # Sesuaikan dengan nama kolom di CSV

# Fitur (variabel input)
feature_column_names = ['hari_ke', 'hari_dalam_minggu', 'bulan'] # Sesuaikan dengan fitur Anda

X = df[feature_column_names]
y = df[target_column_name]

# --- Membagi Data (Train & Test) ---
# Untuk data time series, split berdasarkan waktu lebih disarankan
train_size_ratio = 0.8
train_index_end = int(len(df) * train_size_ratio)

X_train, X_test = X.iloc[:train_index_end], X.iloc[train_index_end:]
y_train, y_test = y.iloc[:train_index_end], y.iloc[train_index_end:]

print(f"\nUkuran data latih: {X_train.shape}, {y_train.shape}")
print(f"Ukuran data uji: {X_test.shape}, {y_test.shape}")

# --- Melatih Model Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2)
rf_model.fit(X_train, y_train)

# --- Evaluasi Model ---
y_pred_test = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nEvaluasi Model pada Data Uji:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- Menyimpan Model dan Metadata ---
joblib.dump(rf_model, 'random_forest_gallon_model.joblib')
model_metadata = {
    'features': feature_column_names,
    'training_start_date': df['Tanggal'].min().isoformat() # Gunakan 'Tanggal'
}
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)

print("\nModel berhasil dilatih dan disimpan sebagai 'random_forest_gallon_model.joblib'")
print("Metadata model disimpan sebagai 'model_metadata.json'")
