# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON
# Script: mod_train_model.py (Versi Disesuaikan dengan Interpolasi Outlier)
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- KONFIGURASI ---
TEST_SIZE_PERCENT = 0.40
START_DATE_TRAIN = '2022-07-04'
# Batas untuk mendefinisikan outlier
LOWER_OUTLIER_THRESHOLD = 19
UPPER_OUTLIER_THRESHOLD = 52
# Batas untuk fitur 'is_low_sale' dan 'is_spike'
LOW_SALE_FEATURE_THRESHOLD = 21
SPIKE_FEATURE_THRESHOLD = 49


print("--- TAHAP 1: PERSIAPAN DATA ---")
try:
    df_raw = pd.read_csv('galon.csv', header=0, names=['Hari_dalam_seminggu_Raw', 'Galon_Terjual'])
    print("* Berhasil memuat data mentah dari 'galon.csv'.")
except FileNotFoundError:
    print("Error: File 'galon.csv' tidak ditemukan.")
    exit()

print(f"* Membangun linimasa data latih yang benar, mulai dari {START_DATE_TRAIN}...")
dates = []
current_date = pd.to_datetime(START_DATE_TRAIN)

start_day_of_week = current_date.dayofweek + 1
first_day_in_data = df_raw['Hari_dalam_seminggu_Raw'].iloc[0]

if start_day_of_week != first_day_in_data:
    print(f"Error: Tanggal mulai {START_DATE_TRAIN} adalah hari ke-{start_day_of_week},")
    print(f"sedangkan data latih dimulai dengan hari ke-{first_day_in_data}.")
    exit()

dates.append(current_date)
for i in range(1, len(df_raw)):
    day_diff = (df_raw['Hari_dalam_seminggu_Raw'].iloc[i] - df_raw['Hari_dalam_seminggu_Raw'].iloc[i-1] + 7) % 7
    if day_diff == 0:
        day_diff = 7
    current_date += pd.Timedelta(days=day_diff)
    dates.append(current_date)

df = pd.DataFrame({'tanggal': dates, 'Galon_Terjual': df_raw['Galon_Terjual'].values})
df.to_csv('hasil_pemetaan.csv', index=False)
print("* Pemetaan tanggal ke hari selesai.")


print("\n--- TAHAP 2: PEMBERSIHAN DATA (METODE BARU) ---")
# --- LOGIKA BARU: Hapus outlier dan interpolasi ---
print("* Mengganti outlier dengan NaN untuk diinterpolasi.")
# Salin kolom asli untuk dimodifikasi
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual'].copy().astype(float)

# Ubah nilai di atas UPPER_OUTLIER_THRESHOLD menjadi NaN
df.loc[df['Galon_Terjual'] > UPPER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan
# Ubah nilai di bawah LOWER_OUTLIER_THRESHOLD menjadi NaN
df.loc[df['Galon_Terjual'] < LOWER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan

# Hitung jumlah outlier yang dihapus
outliers_removed = df['Galon_Terjual_Cleaned'].isna().sum()
print(f"* Ditemukan dan ditandai {outliers_removed} outlier untuk diinterpolasi.")
# Tampilkan hasil jumlah outlier yang dihapus dalam csv
df.to_csv('output/csv/hasil_pembersihan_outlier.csv', index=False)

# Isi nilai NaN menggunakan interpolasi linear
print("* Mengisi nilai kosong (bekas outlier) dengan interpolasi linear.")
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Cleaned'].interpolate(method='linear')

# Gunakan backfill dan forward-fill untuk menangani NaN jika ada di awal/akhir data
df['Galon_Terjual_Cleaned'].fillna(method='bfill', inplace=True)
df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)
print("* Pembersihan data dengan metode interpolasi selesai.")


print("\n--- TAHAP 3: REKAYASA FITUR ---")
target_col = 'Galon_Terjual_Cleaned'
df.set_index('tanggal', inplace=True)

# Memastikan tidak ada tanggal yang hilang dalam rentang data
df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max()), method=None)

# Isi gap pada data mentah dan data bersih setelah reindex
df['Galon_Terjual'] = df['Galon_Terjual'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Cleaned'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

for lag in [1, 2, 3, 7, 14]:
    df[f'lag_{lag}'] = df[target_col].shift(lag)

# 2. Buat target yang digeser HANYA untuk fitur rolling & diff
shifted_target = df[target_col].shift(1)

# 3. Buat fitur rolling dan diff dari target yang sudah digeser
for window in [3, 7, 14, 21]:
    df[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()

df['lag_diff_1'] = shifted_target.diff(1)
df['lag_diff_7'] = shifted_target.diff(7)
# --- AKHIR PERBAIKAN ---

# Fitur berbasis kalender
df['hari_dalam_bulan'] = df.index.day
df['hari_dalam_tahun'] = df.index.dayofyear
df['minggu_dalam_tahun'] = df.index.isocalendar().week.astype(int)
df['bulan'] = df.index.month
df['hari_dalam_seminggu'] = df.index.dayofweek + 1
df['akhir_pekan'] = (df['hari_dalam_seminggu'] >= 6).astype(int)
df['awal_bulan'] = df.index.is_month_start.astype(int)
df['akhir_bulan'] = df.index.is_month_end.astype(int)

# Fitur 'days since last spike'
df['is_spike'] = (df[target_col] > SPIKE_FEATURE_THRESHOLD).astype(int)
# Cek apakah fitur is_spike mengindikasikan kebocoran data dengan print
print(df['is_spike'].value_counts())




spike_days = df['is_spike'].copy()
spike_days[spike_days == 0] = np.nan
spike_days = spike_days.reset_index()
spike_days['day_num'] = range(len(spike_days))
spike_days.set_index(spike_days.columns[0], inplace=True)
spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
spike_days['day_num'].fillna(method='ffill', inplace=True)
df['days_since_last_spike'] = (range(len(df)) - spike_days['day_num']).fillna(0)

# Fitur 'days since last low sale'
df['is_low_sale'] = (df[target_col] < LOW_SALE_FEATURE_THRESHOLD).astype(int)
low_sale_days = df['is_low_sale'].copy()
low_sale_days[low_sale_days == 0] = np.nan
low_sale_days = low_sale_days.reset_index()
low_sale_days['day_num'] = range(len(low_sale_days))
low_sale_days.set_index(low_sale_days.columns[0], inplace=True)
low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
low_sale_days['day_num'].fillna(method='ffill', inplace=True)
df['days_since_last_low'] = (range(len(df)) - low_sale_days['day_num']).fillna(0)

df.reset_index(inplace=True)
df.rename(columns={'index': 'tanggal'}, inplace=True)
df.fillna(0, inplace=True)

features = [
    'hari_dalam_seminggu', 'akhir_pekan', 'awal_bulan', 'akhir_bulan',
    'hari_dalam_bulan', 'minggu_dalam_tahun', 'bulan', 'hari_dalam_tahun',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_21',
    'rolling_std_3', 'rolling_std_7', 'rolling_std_14', 'rolling_std_21',
    'lag_diff_1', 'lag_diff_7',
    'days_since_last_spike', 'is_spike', 'is_low_sale',
    'days_since_last_low'
]
X = df[features]
y = df[target_col]
print("* Rekayasa fitur selesai.")

print("\n--- TAHAP 4: PEMBAGIAN DATA ---")
test_size = int(len(df) * TEST_SIZE_PERCENT)
train_size = len(df) - test_size
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
print(f"* Total data (setelah diisi): {len(df)} hari.")
# Tampilkan total data dalam csv hanya menampilkan Hari, tanggal, galon terjual, dan galon terjual cleaned
df = df[['tanggal', 'Galon_Terjual', 'Galon_Terjual_Cleaned']]
df.to_csv('output/csv/hasil_rekayasa_fitur.csv', index=False)
print(f"* Data Latih: {len(X_train)} hari (~{100*(1-TEST_SIZE_PERCENT):.0f}%)")
print(f"* Data Uji  : {len(X_test)} hari (~{100*TEST_SIZE_PERCENT:.0f}%)")


print("\n--- TAHAP 5: PEMODELAN ---")
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

print("* Mencari hyperparameter terbaik dengan RandomizedSearchCV dan GridSearchCV...")
param_dist = {
    'n_estimators': [200, 400, 600, 800], 'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9], 'gamma': [0.1, 0.2, 0.3],
    'reg_alpha': [0.05, 0.1, 0.2]
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=tscv, scoring='neg_mean_absolute_error', verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_params_from_random = random_search.best_params_
param_grid = {
    'n_estimators': [best_params_from_random['n_estimators']], 'max_depth': [best_params_from_random['max_depth']],
    'learning_rate': [best_params_from_random['learning_rate'] - 0.005, best_params_from_random['learning_rate'], best_params_from_random['learning_rate'] + 0.005],
    'subsample': [best_params_from_random['subsample']], 'colsample_bytree': [best_params_from_random['colsample_bytree']],
    'gamma': [best_params_from_random['gamma']], 'reg_alpha': [best_params_from_random['reg_alpha']]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"\n* Parameter terbaik final ditemukan: {grid_search.best_params_}")
print("* Pelatihan model selesai.")


print("\n--- TAHAP 6: EVALUASI KINERJA MODEL ---")
y_pred = best_model.predict(X_test)
# Opsi: Kliping pada prediksi akhir sebagai "safety net" agar tidak ada prediksi ekstrem
y_pred = y_pred.clip(min=LOWER_OUTLIER_THRESHOLD)

mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan).dropna())) * 100
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Metrik Kinerja pada Data Uji:")
print(f"Mean Absolute Error (MAE)       : {mae:.2f} galon")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE)    : {rmse:.2f}")


print("\n--- TAHAP 7: PENYIMPANAN MODEL & METADATA ---")
joblib.dump(best_model, 'xgboost_gallon_model.joblib')
print("* Model berhasil disimpan sebagai 'xgboost_gallon_model.joblib'")

cleaned_best_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in grid_search.best_params_.items()}
model_metadata = {
    'features_used': features,
    'business_rules': {
        'lower_outlier_threshold': LOWER_OUTLIER_THRESHOLD,
        'upper_outlier_threshold': UPPER_OUTLIER_THRESHOLD,
        'imputation_method': 'linear_interpolation'
    },
    'model_performance_on_test_set': {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)},
    'best_hyperparameters': cleaned_best_params
}
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("* Metadata model berhasil disimpan sebagai 'model_metadata.json'")

# Visualisasi hasil
plt.figure(figsize=(18, 9))
plt.plot(df.loc[y_test.index, 'tanggal'], y_test, label='Data Uji (Aktual)', color='green', marker='o', markersize=4)
plt.plot(df.loc[y_test.index, 'tanggal'], y_pred, label='Prediksi Model', color='darkorange', linestyle='--')
plt.title('Perbandingan Data Uji dan Prediksi (Metode Interpolasi)', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Jumlah Galon Terjual', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("output/images/actual_vs_prediction_interpolated.png")
plt.close()
print("* Visualisasi perbandingan aktual vs prediksi disimpan.")

# Visualisasi fitur penting
plt.figure(figsize=(12, 10))
xgb.plot_importance(best_model, importance_type='weight', max_num_features=20, title='Fitur Penting Model')
plt.tight_layout()
plt.savefig("output/images/feature_importance.png")
plt.close()
print("* Visualisasi fitur penting disimpan.")

print("\n--- Proses Selesai ---")
