# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON
# Script: mod_train_model.py (Versi Final dengan Logika Fitur yang Benar)
# =============================================================================

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

TEST_SIZE_PERCENT = 0.30
START_DATE_TRAIN = '2022-07-04'

print("--- TAHAP 1: PERSIAPAN DATA ---")
try:
    df_raw = pd.read_csv('galon.csv', header=0, names=['Hari_Minggu_Raw', 'Galon_Terjual'])
    print("* Berhasil memuat data mentah dari 'galon.csv'.")
except FileNotFoundError:
    print("Error: File 'galon.csv' tidak ditemukan.")
    exit()

print(f"* Membangun linimasa data latih yang benar, mulai dari {START_DATE_TRAIN}...")
dates = []
current_date = pd.to_datetime(START_DATE_TRAIN)

start_day_of_week = current_date.dayofweek + 1
first_day_in_data = df_raw['Hari_Minggu_Raw'].iloc[0]

if start_day_of_week != first_day_in_data:
    print(f"Error: Tanggal mulai {START_DATE_TRAIN} adalah hari ke-{start_day_of_week},")
    print(f"sedangkan data latih dimulai dengan hari ke-{first_day_in_data}.")
    exit()

dates.append(current_date)
for i in range(1, len(df_raw)):
    day_diff = (df_raw['Hari_Minggu_Raw'].iloc[i] - df_raw['Hari_Minggu_Raw'].iloc[i-1] + 7) % 7
    if day_diff == 0:
        day_diff = 7
    current_date += pd.Timedelta(days=day_diff)
    dates.append(current_date)

df = pd.DataFrame({'tanggal': dates, 'Galon_Terjual': df_raw['Galon_Terjual'].values})

df.to_csv('hasil_pemetaan.csv', index=False)
print(f"* Pemetaan tanggal ke hari selesai.")

print("\n--- TAHAP 2: PEMBERSIHAN DATA ---")
lower_bound = 20.0
upper_bound = 52.0
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual'].clip(lower=lower_bound, upper=upper_bound)
# Forward Fill Method
# df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)
# df['Galon_Terjual_Cleaned'].fillna(lower_bound, inplace=True)

# Interpolate Linier Method
df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Cleaned'].interpolate(method='linear')
df['Galon_Terjual_Cleaned'].fillna(method='bfill', inplace=True)  # jaga-jaga kalau NaN di awal
df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)  # jaga-jaga kalau NaN di akhir

print("* Pembersihan data selesai.")

print("\n--- TAHAP 3: REKAYASA FITUR ---")
target_col = 'Galon_Terjual_Cleaned'
df.set_index('tanggal', inplace=True)

df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max()), method=None)
# Forward Fill Method
# df['Galon_Terjual'].fillna(method='ffill', inplace=True)
# df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)

# Interpolate Linier Method
df['Galon_Terjual'] = df['Galon_Terjual'].interpolate(method='linear')
df['Galon_Terjual'].fillna(method='bfill', inplace=True)
df['Galon_Terjual'].fillna(method='ffill', inplace=True)

df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Cleaned'].interpolate(method='linear')
df['Galon_Terjual_Cleaned'].fillna(method='bfill', inplace=True)
df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)


# =============================================================================
# --- PERBAIKAN KESALAHAN LOGIKA FITUR ---
# =============================================================================
# 1. Buat fitur lag secara langsung dari kolom target (MENGHINDARI DOUBLE SHIFT)
for lag in [1, 2, 3, 7, 14]:
    df[f'lag_{lag}'] = df[target_col].shift(lag)

# 2. Buat target yang digeser HANYA untuk fitur rolling & diff (mencegah data leakage)
shifted_target = df[target_col].shift(1)

# 3. Buat fitur rolling dan diff dari target yang sudah digeser
for window in [3, 7, 14, 21]:
    df[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()

df['lag_diff_1'] = shifted_target.diff(1)
df['lag_diff_7'] = shifted_target.diff(7)
# =============================================================================
# --- AKHIR PERBAIKAN ---
# =============================================================================

df['hari_dalam_bulan'] = df.index.day
df['hari_dalam_tahun'] = df.index.dayofyear
df['minggu_dalam_tahun'] = df.index.isocalendar().week.astype(int)
df['bulan'] = df.index.month
df['hari_minggu'] = df.index.dayofweek
df['is_weekend'] = (df['hari_minggu'] >= 5).astype(int)
df['awal_bulan'] = df.index.is_month_start.astype(int)
df['akhir_bulan'] = df.index.is_month_end.astype(int)

spike_threshold = 49
df['is_spike'] = (df[target_col] > spike_threshold).astype(int)
spike_days = df['is_spike'].copy()
spike_days[spike_days == 0] = np.nan
spike_days = spike_days.reset_index()
spike_days.rename(columns={'index': 'tanggal'}, inplace=True)
spike_days['day_num'] = range(len(spike_days))
spike_days.set_index('tanggal', inplace=True)
spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
spike_days['day_num'].fillna(method='ffill', inplace=True)
df['days_since_last_spike'] = (range(len(df)) - spike_days['day_num']).fillna(0)

low_threshold = 23
df['is_low_sale'] = (df[target_col] < low_threshold).astype(int)
low_sale_days = df['is_low_sale'].copy()
low_sale_days[low_sale_days == 0] = np.nan
low_sale_days = low_sale_days.reset_index()
low_sale_days.rename(columns={'index': 'tanggal'}, inplace=True)
low_sale_days['day_num'] = range(len(low_sale_days))
low_sale_days.set_index('tanggal', inplace=True)
low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
low_sale_days['day_num'].fillna(method='ffill', inplace=True)
df['days_since_last_low'] = (range(len(df)) - low_sale_days['day_num']).fillna(0)

df.reset_index(inplace=True)
df.rename(columns={'index': 'tanggal'}, inplace=True)
df.fillna(0, inplace=True)

features = [
    'hari_minggu', 'is_weekend', 'awal_bulan', 'akhir_bulan',
    'hari_dalam_bulan', 'minggu_dalam_tahun', 'bulan', 'hari_dalam_tahun',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_21',
    'rolling_std_3', 'rolling_std_7', 'rolling_std_14', 'rolling_std_21',
    'lag_diff_1', 'lag_diff_7',
    'days_since_last_spike',
    'days_since_last_low'
]
X = df[features]
y = df[target_col]
print("* Rekayasa fitur selesai.")

# (Sisa skrip untuk splitting, training, dan saving tidak berubah)
# ...
# =============================================================================
# TAHAP 4: PEMBAGIAN DATA (DATA SPLITTING)
# =============================================================================
print("\n--- TAHAP 4: PEMBAGIAN DATA ---")
test_size = int(len(df) * TEST_SIZE_PERCENT)
train_size = len(df) - test_size
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
print(f"* Total data (setelah diisi): {len(df)} hari.")
print(f"* Data Latih: {len(X_train)} hari (~{100*(1-TEST_SIZE_PERCENT):.0f}%)")
print(f"* Data Uji  : {len(X_test)} hari (~{100*TEST_SIZE_PERCENT:.0f}%)")


# =============================================================================
# TAHAP 5: PEMODELAN (MODELING)
# =============================================================================
print("\n--- TAHAP 5: PEMODELAN ---")
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

print("* Tahap 5.1: Pencarian Luas dengan RandomizedSearchCV...")
param_dist = {
    'n_estimators': [200, 400, 600, 800], 'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9], 'gamma': [0.1, 0.2, 0.3],
    'reg_alpha': [0.05, 0.1, 0.2]
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=tscv, scoring='neg_mean_absolute_error', verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
print(f"\n* Parameter terbaik sementara: {random_search.best_params_}")

print("\n* Tahap 5.2: Pencarian Halus dengan GridSearchCV...")
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


# =============================================================================
# TAHAP 6: EVALUASI (EVALUATION)
# =============================================================================
print("\n--- TAHAP 6: EVALUASI KINERJA MODEL ---")
y_pred = best_model.predict(X_test)
y_pred = y_pred.clip(min=lower_bound, max=upper_bound)

mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan).dropna())) * 100
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Metrik Kinerja pada Data Uji:")
print(f"  - Mean Absolute Error (MAE)       : {mae:.2f} galon")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  - Root Mean Squared Error (RMSE)    : {rmse:.2f}")
print(f"  - R-squared (R2 Score)            : {r2:.2f}")


# =============================================================================
# TAHAP 7: PENYIMPANAN (DEPLOYMENT)
# =============================================================================
print("\n--- TAHAP 7: PENYIMPANAN MODEL & METADATA ---")
joblib.dump(best_model, 'xgboost_gallon_model.joblib')
print("* Model berhasil disimpan sebagai 'xgboost_gallon_model.joblib'")

cleaned_best_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in grid_search.best_params_.items()}
model_metadata = {
    'features_used': features,
    'business_rules': {'lower_bound': lower_bound, 'upper_bound': upper_bound},
    'model_performance_on_test_set': {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'mape': float(mape)},
    'best_hyperparameters': cleaned_best_params
}
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("* Metadata model berhasil disimpan sebagai 'model_metadata.json'")

# --- Visualisasi Hasil ---
print("* Membuat visualisasi...")
plt.figure(figsize=(18, 9))
plt.plot(df.loc[y_train.index, 'tanggal'], y_train, label='Data Latih', color='royalblue')
plt.plot(df.loc[y_test.index, 'tanggal'], y_test, label='Data Uji (Aktual)', color='green', marker='o')
plt.plot(df.loc[y_test.index, 'tanggal'], y_pred, label='Prediksi Model', color='darkorange', linestyle='--')
plt.title('Perbandingan Data Latih, Uji, dan Prediksi', fontsize=16)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Jumlah Galon Terjual', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("actual_vs_prediction.png")
plt.close()

plt.figure(figsize=(12, 10))
feat_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title('20 Fitur Paling Penting (Model Dilatih pada Data Latih)', fontsize=16)
plt.xlabel('Tingkat Kepentingan (Importance)', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("* Visualisasi telah disimpan.")
print("\nProses Selesai.")
