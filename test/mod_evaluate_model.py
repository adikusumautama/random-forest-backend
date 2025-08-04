# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON
# Script: mod_evaluate_model.py (Diperbaiki untuk kompatibilitas dengan model lama)
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
INPUT_CSV_LATIH = "galon.csv"
INPUT_CSV_UJI = "mod_data_uji.csv"
OUTPUT_CSV_HASIL = "output/csv/hasil_evaluasi_final.csv"
OUTPUT_PLOT_HASIL = "output/images/plot_evaluasi_final.png"
MODEL_PATH = "xgboost_gallon_model.joblib"
METADATA_PATH = "model_metadata.json"
KONTEKS_HARI = 30 # Jumlah hari data historis yang diambil sebagai konteks

# Tanggal mulai data uji HARUS sesuai dengan hari pertama di file mod_data_uji.csv
TANGGAL_MULAI_UJI = '2025-06-10' 

# --- 1. Memuat Model dan Metadata ---
print(f"Memuat model dari {MODEL_PATH} dan metadata dari {METADATA_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError as e:
    print(f"Error: File '{e.filename}' tidak ditemukan. Pastikan Anda sudah menjalankan skrip training.")
    exit()

# Mengambil daftar fitur dan aturan bisnis dari metadata
feature_names_from_training = metadata['features_used']
business_rules = metadata['business_rules']
LOWER_OUTLIER_THRESHOLD = business_rules.get('lower_outlier_threshold', 15)
UPPER_OUTLIER_THRESHOLD = business_rules.get('upper_outlier_threshold', 55) 
SPIKE_FEATURE_THRESHOLD = 49 
LOW_SALE_FEATURE_THRESHOLD = 21

# --- 2. Memuat dan Membangun Data Historis & Data Uji ---
print("Memuat data...")
try:
    # Ganti nama kolom agar konsisten
    df_train_raw = pd.read_csv(INPUT_CSV_LATIH, header=0, names=['Hari_dalam_seminggu_Raw', 'Galon_Terjual'])
    df_test_raw = pd.read_csv(INPUT_CSV_UJI, header=0, names=['Hari_dalam_seminggu_Raw', 'Galon_Terjual'])
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan - {e.filename}")
    exit()

# --- PERBAIKAN: Menggunakan tanggal mulai yang benar untuk data training ---
START_DATE_TRAIN = '2022-07-04' 
train_dates = []
current_date_train = pd.to_datetime(START_DATE_TRAIN)
train_dates.append(current_date_train)
for i in range(1, len(df_train_raw)):
    day_diff = (df_train_raw['Hari_dalam_seminggu_Raw'].iloc[i] - df_train_raw['Hari_dalam_seminggu_Raw'].iloc[i-1] + 7) % 7
    if day_diff == 0: day_diff = 7
    current_date_train += pd.Timedelta(days=day_diff)
    train_dates.append(current_date_train)
df_train_full = pd.DataFrame({'tanggal': train_dates, 'Galon_Terjual': df_train_raw['Galon_Terjual'].values})

print(f"Membangun linimasa data uji, mulai dari {TANGGAL_MULAI_UJI}...")
test_dates = []
current_date_test = pd.to_datetime(TANGGAL_MULAI_UJI)
test_dates.append(current_date_test)
for i in range(1, len(df_test_raw)):
    day_diff = (df_test_raw['Hari_dalam_seminggu_Raw'].iloc[i] - df_test_raw['Hari_dalam_seminggu_Raw'].iloc[i-1] + 7) % 7
    if day_diff == 0: day_diff = 7
    current_date_test += pd.Timedelta(days=day_diff)
    test_dates.append(current_date_test)
df_test_full = pd.DataFrame({'tanggal': test_dates, 'Galon_Terjual': df_test_raw['Galon_Terjual'].values})

# --- 3. Menggabungkan data & Rekayasa Fitur ---
print("Menggabungkan data historis dan data uji untuk rekayasa fitur...")
df_konteks = df_train_full.tail(KONTEKS_HARI).copy()
df_combined = pd.concat([df_konteks, df_test_full], ignore_index=True)

df_combined.set_index('tanggal', inplace=True)
df_combined = df_combined.reindex(pd.date_range(start=df_combined.index.min(), end=df_combined.index.max()), method=None)
df_combined['Galon_Terjual'].interpolate(method='linear', inplace=True)

print("Membersihkan data dengan metode interpolasi...")
df_combined['Galon_Terjual_Cleaned'] = df_combined['Galon_Terjual'].copy().astype(float)
df_combined.loc[df_combined['Galon_Terjual'] > UPPER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan
df_combined.loc[df_combined['Galon_Terjual'] < LOWER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan
df_combined['Galon_Terjual_Cleaned'] = df_combined['Galon_Terjual_Cleaned'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')



# Tampilkan data yang dijadikan konteks
print("Data yang digunakan sebagai konteks:")
print(df_combined.head(KONTEKS_HARI))
print("Data yang digunakan untuk uji:")
print(df_combined.tail(len(df_test_full)))



print("Melakukan rekayasa fitur agar sesuai dengan model yang dilatih...")
target_col = 'Galon_Terjual_Cleaned'
shifted_target = df_combined[target_col].shift(1)

# --- PERBAIKAN: Membuat SEMUA fitur yang diharapkan oleh model lama ---
for lag in [1, 2, 3, 7, 14]: # Mengembalikan lag 14
    df_combined[f'lag_{lag}'] = df_combined[target_col].shift(lag)

for window in [3, 7, 14, 21]: # Mengembalikan window 14 & 21
    df_combined[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df_combined[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()

df_combined['lag_diff_1'] = shifted_target.diff(1)
df_combined['lag_diff_7'] = shifted_target.diff(7)

# Mengembalikan fitur kalender dan menggunakan nama kolom yang benar
df_combined['hari_dalam_bulan'] = df_combined.index.day
df_combined['hari_dalam_tahun'] = df_combined.index.dayofyear
df_combined['minggu_dalam_tahun'] = df_combined.index.isocalendar().week.astype(int)
df_combined['bulan'] = df_combined.index.month
df_combined['hari_dalam_seminggu'] = df_combined.index.dayofweek + 1 # Nama kolom sesuai error
df_combined['akhir_pekan'] = (df_combined['hari_dalam_seminggu'] >= 6).astype(int) # Nama kolom sesuai error
df_combined['awal_bulan'] = df_combined.index.is_month_start.astype(int)
df_combined['akhir_bulan'] = df_combined.index.is_month_end.astype(int)

# Membuat fitur is_spike dan is_low_sale langsung (mereplikasi kebocoran data dari training lama)
df_combined['is_spike'] = (df_combined[target_col] > SPIKE_FEATURE_THRESHOLD).astype(int)
df_combined['is_low_sale'] = (df_combined[target_col] < LOW_SALE_FEATURE_THRESHOLD).astype(int)

# Fitur days_since...
spike_days = df_combined['is_spike'].copy().shift(1) # Digeser untuk evaluasi
spike_days[spike_days == 0] = np.nan
spike_days = spike_days.reset_index()
spike_days['day_num'] = range(len(spike_days))
spike_days.set_index(spike_days.columns[0], inplace=True)
spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
spike_days['day_num'].fillna(method='ffill', inplace=True)
df_combined['days_since_last_spike'] = (range(len(df_combined)) - spike_days['day_num']).fillna(0)

low_sale_days = df_combined['is_low_sale'].copy().shift(1) # Digeser untuk evaluasi
low_sale_days[low_sale_days == 0] = np.nan
low_sale_days = low_sale_days.reset_index()
low_sale_days['day_num'] = range(len(low_sale_days))
low_sale_days.set_index(low_sale_days.columns[0], inplace=True)
low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
low_sale_days['day_num'].fillna(method='ffill', inplace=True)
df_combined['days_since_last_low'] = (range(len(df_combined)) - low_sale_days['day_num']).fillna(0)

df_combined.reset_index(inplace=True)
df_combined.rename(columns={'index': 'tanggal'}, inplace=True)
df_combined.fillna(0, inplace=True)

# --- 4. Melakukan Prediksi ---
print("Memisahkan data uji dan melakukan prediksi...")
df_test_final = df_combined[df_combined['tanggal'].isin(df_test_full['tanggal'])].copy()

# Baris ini sekarang seharusnya berhasil karena semua kolom sudah dibuat
X_test = df_test_final[feature_names_from_training]
y_actual = df_test_final[target_col]

y_pred = model.predict(X_test)
y_pred = y_pred.clip(min=LOWER_OUTLIER_THRESHOLD)

# --- 5. Menyimpan Hasil ---
os.makedirs('output/csv', exist_ok=True)
os.makedirs('output/images', exist_ok=True)
print(f"Menyimpan hasil ke {OUTPUT_CSV_HASIL}...")
df_result = df_test_final[['tanggal']].copy()
df_result['Aktual_Asli'] = df_test_final['Galon_Terjual'].values
df_result['Aktual_Dibersihkan'] = y_actual.values
df_result['Prediksi'] = y_pred
df_result['Prediksi_Bulat'] = np.round(y_pred)
df_result['Selisih'] = abs(df_result['Aktual_Dibersihkan'] - df_result['Prediksi_Bulat'])
df_result.to_csv(OUTPUT_CSV_HASIL, index=False)

# --- 6. Evaluasi & 7. Plotting ---
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / (y_actual.replace(0, np.nan).dropna()))) * 100

print("\nEvaluasi Kinerja pada Data Uji Baru:")
print(f"  - MAE : {mae:.2f}")
print(f"  - MAPE: {mape:.2f}%")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - R2  : {r2:.2f}")

# Tampilkan data yang digunakan sebagai uji dan hasil prediksi  dalam csv
df_result[['tanggal', 'Aktual_Asli', 'Aktual_Dibersihkan', 'Prediksi', 'Prediksi_Bulat', 'Selisih']].to_csv(OUTPUT_CSV_HASIL, index=False)
print(f"Hasil evaluasi disimpan ke {OUTPUT_CSV_HASIL}")


plt.figure(figsize=(15, 7))
plt.plot(df_result['tanggal'], df_result['Aktual_Dibersihkan'], label='Aktual (Dibersihkan)', color='royalblue', marker='o', linestyle='-')
plt.plot(df_result['tanggal'], df_result['Prediksi'], label='Prediksi', color='darkorange', marker='x', linestyle='--')
plt.title('Perbandingan Penjualan Aktual vs. Prediksi pada Data Uji Baru')
plt.xlabel('Tanggal')
plt.ylabel('Galon Terjual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_HASIL)
plt.close()
print(f"Plot evaluasi disimpan ke {OUTPUT_PLOT_HASIL}")
print("\n--- Proses Selesai ---")
