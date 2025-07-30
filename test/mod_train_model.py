import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- 1. PENYIAPAN DATASET ---
print("--- Langkah 1: Memuat dan Menyiapkan Dataset ---")
try:
    df_raw = pd.read_csv('galon.csv')
    df_raw.columns = ['Hari_Minggu_Raw', 'Galon_Terjual']
except FileNotFoundError:
    print("Error: File 'galon.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

# Membuat timeline kalender nyata
start_date = '2022-07-04'
print(f"Membuat linimasa data yang lengkap dimulai dari {start_date}...")
df = pd.DataFrame()
# Buat kolom tanggal berdasarkan jumlah baris di data mentah + buffer untuk prediksi masa depan
df['tanggal'] = pd.to_datetime(pd.date_range(start=start_date, periods=len(df_raw) + 100))

# Gabungkan data asli dengan timeline kalender
df_raw['hari_ke'] = range(len(df_raw))
df['hari_ke'] = range(len(df))
df = pd.merge(df, df_raw[['hari_ke', 'Galon_Terjual']], on='hari_ke', how='left')
print(f"Dataset berhasil dibuat dengan {len(df)} hari berkelanjutan.")


# --- 2. PEMBERSIHAN DATA & PENERAPAN ATURAN BISNIS (PERBAIKAN) ---
print("\n--- Langkah 2: Pembersihan Data & Winsorization ---")
# PERBAIKAN: Menggunakan batas bawah 22 sesuai permintaan
lower_bound = 25.0
upper_bound = 52.0
print(f"Menerapkan Winsorization: Penjualan dibatasi antara {lower_bound} dan {upper_bound} galon.")

# PERBAIKAN: Proses pembersihan yang lebih sederhana dan fokus pada Winsorization
# 1. Isi nilai yang hilang terlebih dahulu (menggunakan forward fill, umum untuk time series)
df['Galon_Terjual_Filled'] = df['Galon_Terjual'].fillna(method='ffill')
# Isi sisa NaN di awal (jika ada) dengan lower bound
df['Galon_Terjual_Filled'].fillna(lower_bound, inplace=True)

# 2. Terapkan Winsorization (clipping) pada data yang sudah diisi
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Filled'].clip(lower=lower_bound, upper=upper_bound)
print("Data telah di-winsorize dan nilai kosong telah ditangani.")


# --- 3. REKAYASA FITUR (PERBAIKAN) ---
print("\n--- Langkah 3: Rekayasa Fitur ---")
target_col = 'Galon_Terjual_Cleaned'
# Penting: Gunakan data yang sudah dibersihkan untuk membuat fitur lag/rolling
shifted_target = df[target_col].shift(1)

# Fitur-fitur dasar (lag, rolling) - Fokus pada jangka pendek
for lag in [1, 2, 3, 7, 14]: # Menambah lag 14 untuk melihat pola 2 mingguan
    df[f'lag_{lag}'] = shifted_target.shift(lag)
for window in [3, 7, 14]: # Menambah window 14
    df[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()

# Fitur momentum
df['lag_diff_1'] = shifted_target.diff(1)
df['lag_diff_7'] = shifted_target.diff(7) # Perbedaan dengan minggu lalu

# Fitur berbasis kalender nyata
print("Membuat fitur-fitur berbasis kalender nyata...")
df['hari_dalam_bulan'] = df['tanggal'].dt.day
df['hari_dalam_tahun'] = df['tanggal'].dt.dayofyear
df['minggu_dalam_tahun'] = df['tanggal'].dt.isocalendar().week.astype(int)
df['bulan'] = df['tanggal'].dt.month
df['hari_minggu'] = df['tanggal'].dt.dayofweek # 0=Senin, 6=Minggu
df['is_weekend'] = (df['hari_minggu'] >= 5).astype(int) # Sabtu & Minggu
df['awal_bulan'] = (df['tanggal'].dt.is_month_start).astype(int)
df['akhir_bulan'] = (df['tanggal'].dt.is_month_end).astype(int)


df.fillna(0, inplace=True)

# PERBAIKAN: Daftar fitur final tanpa 'is_low_sales'
print("Menyusun daftar fitur final...")
features = [
    'hari_minggu', 'is_weekend', 'awal_bulan', 'akhir_bulan',
    'hari_dalam_bulan', 'minggu_dalam_tahun', 'bulan', 'hari_dalam_tahun',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14',
    'rolling_std_3', 'rolling_std_7', 'rolling_std_14',
    'lag_diff_1', 'lag_diff_7'
]
X = df[features]
y = df[target_col]


# --- 4. PEMBAGIAN DATA ---
print("\n--- Langkah 4: Membagi Data Menjadi Set Latih dan Uji ---")
# Memisahkan data berdasarkan data yang ada vs data masa depan (untuk prediksi)
train_test_split_point = len(df_raw)
X_train = X.iloc[:train_test_split_point]
y_train = y.iloc[:train_test_split_point]

# Data uji yang sebenarnya adalah data yang sama dengan data latih,
# namun evaluasi dilakukan melalui cross-validation (TimeSeriesSplit)
# yang lebih robust untuk data waktu.
# Untuk plot akhir, kita akan prediksi di seluruh rentang waktu.
print(f"Ukuran data untuk melatih model: {len(X_train)} sampel")


# --- 5. MODELING (PERBAIKAN: Strategi Tuning 2 Tahap) ---
print("\n--- Langkah 5: Melatih Model XGBoost dengan Hyperparameter Tuning 2 Tahap ---")
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# --- TAHAP 1: Pencarian Luas dengan RandomizedSearchCV ---
print("\n--- Tahap 5.1: Pencarian Luas dengan RandomizedSearchCV ---")
param_dist = {
    'n_estimators': [200, 400, 600, 800],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0.1, 0.2, 0.3],
    'reg_alpha': [0.05, 0.1, 0.2]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # Mencoba 50 kombinasi acak untuk kecepatan
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
print(f"\nParameter terbaik dari RandomizedSearch: {random_search.best_params_}")


# --- TAHAP 2: Pencarian Halus dengan GridSearchCV ---
print("\n--- Tahap 5.2: Pencarian Halus dengan GridSearchCV di Sekitar Hasil Terbaik ---")
best_params_from_random = random_search.best_params_

# Buat grid yang lebih sempit di sekitar parameter terbaik dari tahap 1
param_grid = {
    'n_estimators': [best_params_from_random['n_estimators']],
    'max_depth': [best_params_from_random['max_depth']],
    'learning_rate': [best_params_from_random['learning_rate'] - 0.005, best_params_from_random['learning_rate'], best_params_from_random['learning_rate'] + 0.005],
    'subsample': [best_params_from_random['subsample']],
    'colsample_bytree': [best_params_from_random['colsample_bytree']],
    'gamma': [best_params_from_random['gamma']],
    'reg_alpha': [best_params_from_random['reg_alpha']]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nParameter terbaik final dari GridSearchCV: {grid_search.best_params_}")
best_model = grid_search.best_estimator_


# --- 6. EVALUASI MODEL ---
print("\n--- Langkah 6: Mengevaluasi Kinerja Model ---")
# Prediksi dibuat pada seluruh dataset untuk melihat performa
y_pred_full = best_model.predict(X)
y_pred_full = y_pred_full.clip(min=lower_bound, max=upper_bound)

# Evaluasi dilakukan pada data historis (set latih)
# karena tidak ada set uji terpisah (mengandalkan CV)
y_pred_train = y_pred_full[:train_test_split_point]

mae = mean_absolute_error(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2 = r2_score(y_train, y_pred_train)

print("Evaluasi Kinerja pada Data Historis (setelah Cross-Validation):")
print(f"Mean Absolute Error (MAE): {mae:.2f} galon")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")


# --- 7. PENYIMPANAN & VISUALISASI ---
print("\n--- Langkah 7: Menyimpan Model dan Membuat Visualisasi ---")
joblib.dump(best_model, 'xgboost_gallon_model.joblib')
print("Model berhasil disimpan sebagai 'xgboost_gallon_model.joblib'")

# Menggunakan parameter terbaik dari grid_search
cleaned_best_params = {
    k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
    for k, v in grid_search.best_params_.items()
}
model_metadata = {
    'features_used': features,
    'business_rules': {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    },
    'model_performance_on_historic_data': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    },
    'best_hyperparameters': cleaned_best_params
}
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("Metadata model berhasil disimpan sebagai 'model_metadata.json'")


# Visualisasi
plt.figure(figsize=(18, 8))
# Plot data aktual (hanya data historis yang ada)
plt.plot(df.loc[:train_test_split_point-1, 'tanggal'], y_train, label='Penjualan Aktual (Telah Dibersihkan)', color='royalblue', marker='.', linestyle='-')
# Plot prediksi (di seluruh rentang waktu)
plt.plot(df['tanggal'], y_pred_full, label='Prediksi Model (Historis & Masa Depan)', color='darkorange', marker='.', linestyle='--')

plt.title('Perbandingan Aktual vs. Prediksi (Model Diperbaiki)', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Jumlah Galon Terjual', fontsize=12)
plt.legend()
plt.axvline(df.loc[train_test_split_point, 'tanggal'], color='red', linestyle='--', label='Batas Prediksi Masa Depan')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("actual_vs_prediction.png")
plt.close()
print("Plot perbandingan telah disimpan sebagai 'actual_vs_prediction.png'")

plt.figure(figsize=(12, 10))
feat_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title('20 Fitur Paling Penting (Model Diperbaiki)', fontsize=16)
plt.xlabel('Tingkat Kepentingan (Importance)', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Plot feature importance telah disimpan sebagai 'feature_importance.png'")
