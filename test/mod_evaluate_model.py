# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON
# Script: mod_evaluate_model.py (Versi Final dengan Logika Pemetaan Hari)
# Deskripsi: Script ini membangun linimasa tanggal yang benar berdasarkan
#            urutan hari di file input, menjaga integritas data time series.
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
INPUT_CSV_LATIH = "galon.csv"
INPUT_CSV_UJI = "mod_data_uji.csv"
OUTPUT_CSV_HASIL = "hasil_evaluasi_final.csv"
OUTPUT_PLOT_HASIL = "plot_evaluasi_final.png"
MODEL_PATH = "xgboost_gallon_model.joblib"
METADATA_PATH = "model_metadata.json"
KONTEKS_HARI = 30

# --- TANGGAL MULAI DIKETAHUI ---
TANGGAL_MULAI_UJI = '2025-06-10'

# --- 1. Memuat Model dan Metadata ---
print(f"Memuat model dari {MODEL_PATH} dan metadata dari {METADATA_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError as e:
    print(f"Error: File '{e.filename}' tidak ditemukan.")
    exit()

feature_names_from_training = metadata['features_used']
business_rules = metadata['business_rules']
lower_bound = business_rules['lower_bound']
upper_bound = business_rules['upper_bound']

# --- 2. Memuat dan Membangun Linimasa Data Uji yang Benar ---
print(f"Memuat data latih dari {INPUT_CSV_LATIH}...")
try:
    # Memastikan kolom dibaca dengan benar meskipun tidak ada header di file
    df_train_raw = pd.read_csv(INPUT_CSV_LATIH, header=0, names=['Hari_Minggu_Raw', 'Galon_Terjual'])
    print(f"Memuat data uji dari {INPUT_CSV_UJI}...")
    df_test_raw = pd.read_csv(INPUT_CSV_UJI, header=0, names=['Hari_Minggu_Raw', 'Galon_Terjual'])
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan - {e.filename}")
    exit()
except Exception as e:
    print(f"Error saat membaca file CSV: {e}")
    exit()


# =============================================================================
# --- PERUBAHAN KUNCI: Membuat tanggal yang sesuai dengan hari ---
# Logika ini akan membuat tanggal yang benar untuk setiap baris di data uji,
# berdasarkan tanggal mulai dan urutan hari yang tidak sekuensial.
# =============================================================================
print(f"Membangun linimasa data uji yang benar, mulai dari {TANGGAL_MULAI_UJI}...")
dates = []
current_date = pd.to_datetime(TANGGAL_MULAI_UJI)

# Validasi: Cek apakah hari pada tanggal mulai cocok dengan hari pertama di data uji
start_day_of_week = current_date.dayofweek + 1 # Konversi ke format 1=Senin, ..., 7=Minggu
first_day_in_data = df_test_raw['Hari_Minggu_Raw'].iloc[0]

if start_day_of_week != first_day_in_data:
    print(f"Error: Tanggal mulai {TANGGAL_MULAI_UJI} adalah hari ke-{start_day_of_week},")
    print(f"sedangkan data uji dimulai dengan hari ke-{first_day_in_data}.")
    print("Harap sesuaikan TANGGAL_MULAI_UJI agar harinya cocok.")
    exit()

# Iterasi untuk membangun daftar tanggal yang benar
dates.append(current_date)
for i in range(1, len(df_test_raw)):
    # Hitung selisih hari dari baris sebelumnya ke baris sekarang
    day_diff = (df_test_raw['Hari_Minggu_Raw'].iloc[i] - df_test_raw['Hari_Minggu_Raw'].iloc[i-1] + 7) % 7
    if day_diff == 0:
        day_diff = 7 # Jika harinya sama, berarti satu minggu kemudian
    
    # Tambahkan selisih hari ke tanggal terakhir untuk mendapatkan tanggal baru
    current_date += pd.Timedelta(days=day_diff)
    dates.append(current_date)

# Buat dataframe pengujian yang sudah benar tanggalnya
df_test_full = pd.DataFrame({
    'tanggal': dates,
    'Galon_Terjual': df_test_raw['Galon_Terjual'].values,
    'Hari_Minggu_Raw': df_test_raw['Hari_Minggu_Raw'].values
})
print("* Pemetaan tanggal ke hari selesai. Linimasa yang benar telah dibuat.")
# =============================================================================
# --- AKHIR PERUBAHAN KUNCI ---
# =============================================================================


# --- 3. Menggabungkan data & Melanjutkan Proses Seperti Biasa ---
# Bagian ini dan seterusnya tidak diubah karena sudah benar.
# Logika ini akan bekerja dengan benar karena linimasa tanggal sudah diperbaiki di atas.

# Buat dataframe latih lengkap dengan tanggal untuk mengambil konteks
start_date_train = '2022-07-04' # Tanggal mulai data latih
df_train_full = pd.DataFrame()
df_train_full['tanggal'] = pd.to_datetime(pd.date_range(start=start_date_train, periods=len(df_train_raw)))
df_train_full['Galon_Terjual'] = df_train_raw['Galon_Terjual'].values
df_train_full['Hari_Minggu_Raw'] = df_train_raw['Hari_Minggu_Raw'].values

# Ambil konteks dari data latih dan gabungkan dengan data uji yang sudah benar
df_konteks = df_train_full.tail(KONTEKS_HARI).copy()
df_combined = pd.concat([df_konteks, df_test_full], ignore_index=True)

# Lakukan reindexing untuk mengisi tanggal yang hilang (jika ada) demi integritas fitur
df_combined.set_index('tanggal', inplace=True)
df_combined = df_combined.reindex(pd.date_range(start=df_combined.index.min(), end=df_combined.index.max()), method=None)

print("* Melakukan pembersihan dan rekayasa fitur...")
# Isi data pada tanggal yang kosong (gap) menggunakan data terakhir yang valid
df_combined['Galon_Terjual'].fillna(method='ffill', inplace=True) 
df_combined['Galon_Terjual_Cleaned'] = df_combined['Galon_Terjual'].clip(lower=lower_bound, upper=upper_bound)
df_combined['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)
df_combined['Galon_Terjual_Cleaned'].fillna(lower_bound, inplace=True)

target_col = 'Galon_Terjual_Cleaned'
shifted_target = df_combined[target_col].shift(1)

# Membuat semua fitur time series
for lag in [1, 2, 3, 7, 14]:
    df_combined[f'lag_{lag}'] = shifted_target.shift(lag)
for window in [3, 7, 14, 21]:
    df_combined[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df_combined[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()
df_combined['lag_diff_1'] = shifted_target.diff(1)
df_combined['lag_diff_7'] = shifted_target.diff(7)

# Membuat semua fitur kalender
df_combined['hari_dalam_bulan'] = df_combined.index.day
df_combined['hari_dalam_tahun'] = df_combined.index.dayofyear
df_combined['minggu_dalam_tahun'] = df_combined.index.isocalendar().week.astype(int)
df_combined['bulan'] = df_combined.index.month
df_combined['hari_minggu'] = df_combined.index.dayofweek # Ini akan selalu sesuai kalender
df_combined['is_weekend'] = (df_combined['hari_minggu'] >= 5).astype(int)
df_combined['awal_bulan'] = df_combined.index.is_month_start.astype(int)
df_combined['akhir_bulan'] = df_combined.index.is_month_end.astype(int)

# Membuat fitur spesifik domain
# Fitur Siklus Lonjakan
spike_threshold = 49
df_combined['is_spike'] = (df_combined[target_col] > spike_threshold).astype(int)
spike_days = df_combined['is_spike'].copy()
spike_days[spike_days == 0] = np.nan
# --- PERBAIKAN ERROR ---
spike_days = spike_days.reset_index() # Dulu: reset_index(names='tanggal')
spike_days.rename(columns={'index': 'tanggal'}, inplace=True) # Baris tambahan untuk ganti nama
# --- AKHIR PERBAIKAN ---
spike_days['day_num'] = range(len(spike_days))
spike_days.set_index('tanggal', inplace=True)
spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
spike_days['day_num'].fillna(method='ffill', inplace=True)
df_combined['days_since_last_spike'] = (range(len(df_combined)) - spike_days['day_num']).fillna(0)

# Fitur Siklus Penjualan Rendah
low_threshold = 23
df_combined['is_low_sale'] = (df_combined[target_col] < low_threshold).astype(int)
low_sale_days = df_combined['is_low_sale'].copy()
low_sale_days[low_sale_days == 0] = np.nan
# --- PERBAIKAN ERROR ---
low_sale_days = low_sale_days.reset_index() # Dulu: reset_index(names='tanggal')
low_sale_days.rename(columns={'index': 'tanggal'}, inplace=True) # Baris tambahan untuk ganti nama
# --- AKHIR PERBAIKAN ---
low_sale_days['day_num'] = range(len(low_sale_days))
low_sale_days.set_index('tanggal', inplace=True)
low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
low_sale_days['day_num'].fillna(method='ffill', inplace=True)
df_combined['days_since_last_low'] = (range(len(df_combined)) - low_sale_days['day_num']).fillna(0)

df_combined.reset_index(inplace=True)
df_combined.rename(columns={'index': 'tanggal'}, inplace=True)
df_combined.fillna(0, inplace=True)

# --- 4. Melakukan Prediksi ---
print("Memisahkan data uji dan melakukan prediksi...")
# Kita ambil data uji berdasarkan tanggal asli yang sudah kita buat
df_test_final = df_combined[df_combined['tanggal'].isin(df_test_full['tanggal'])].copy()

X_test = df_test_final[feature_names_from_training]
y_actual = df_test_final[target_col]

y_pred = model.predict(X_test)
y_pred = y_pred.clip(min=lower_bound, max=upper_bound)

# --- 5. Menyimpan Hasil ---
print(f"Menyimpan hasil ke {OUTPUT_CSV_HASIL}...")
df_result = df_test_final[['tanggal', 'Galon_Terjual', 'hari_minggu']].copy()
df_result.rename(columns={'Galon_Terjual': 'Aktual_Asli'}, inplace=True)
df_result['Aktual_Dibersihkan'] = y_actual.values
df_result['Prediksi'] = y_pred
df_result['Prediksi_Bulat'] = np.round(y_pred)
df_result['Selisih'] = abs(df_result['Aktual_Dibersihkan'] - df_result['Prediksi_Bulat'])
df_result['Hari_Kalender'] = df_result['hari_minggu'] + 1
df_result.drop(columns=['hari_minggu'], inplace=True)
df_result.to_csv(OUTPUT_CSV_HASIL, index=False)

# --- 6. Evaluasi & 7. Plotting ---
print("\nEvaluasi selesai.")
# (Kode evaluasi dan plotting tidak diubah)
# ... (sisa kode untuk evaluasi dan plotting) ...
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / (y_actual.replace(0, np.nan).dropna()))) * 100

print(f"  - MAE : {mae:.2f}")
print(f"  - MAPE: {mape:.2f}%")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - R2  : {r2:.2f}")

plt.figure(figsize=(15, 7))
plt.plot(df_result['tanggal'], df_result['Aktual_Dibersihkan'], label='Aktual (Dibersihkan)', color='royalblue', marker='o', linestyle='-')
plt.plot(df_result['tanggal'], df_result['Prediksi'], label='Prediksi', color='darkorange', marker='x', linestyle='--')
plt.title('Perbandingan Penjualan Aktual vs. Prediksi pada Data Uji')
plt.xlabel('Tanggal')
plt.ylabel('Galon Terjual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_HASIL)
plt.close()
print(f"Plot disimpan ke {OUTPUT_PLOT_HASIL}")
