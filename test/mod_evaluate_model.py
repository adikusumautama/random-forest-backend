import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
INPUT_CSV_LATIH = "galon.csv" # File data latih asli untuk konteks
INPUT_CSV_UJI = "mod_data_uji.csv"
OUTPUT_CSV_HASIL = "hasil_evaluasi_final.csv"
OUTPUT_PLOT_HASIL = "plot_evaluasi_final.png"
MODEL_PATH = "xgboost_gallon_model.joblib"
METADATA_PATH = "model_metadata.json"
KONTEKS_HARI = 30 # Jumlah hari dari data latih yang akan digunakan sebagai konteks

# --- 1. Memuat Model dan Metadata ---
print(f"Memuat model dari {MODEL_PATH} dan metadata dari {METADATA_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    print(f"Error: Pastikan file '{MODEL_PATH}', '{METADATA_PATH}', dan '{INPUT_CSV_LATIH}' ada di direktori yang sama.")
    exit()

feature_names_from_training = metadata['features_used']
business_rules = metadata['business_rules']
lower_bound = business_rules['lower_bound']
upper_bound = business_rules['upper_bound']

# --- 2. Memuat dan Menyiapkan Data dengan Tanggal yang Benar ---
print(f"Memuat data latih dari {INPUT_CSV_LATIH} untuk konteks...")
try:
    df_train_raw = pd.read_csv(INPUT_CSV_LATIH)
    df_train_raw.columns = ['Hari_Minggu_Raw', 'Galon_Terjual']
    
    print(f"Memuat data uji dari {INPUT_CSV_UJI}...")
    df_test_raw = pd.read_csv(INPUT_CSV_UJI)
    df_test_raw.columns = ['Hari_Minggu_Raw', 'Galon_Terjual']
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan - {e.filename}")
    exit()

# Buat dataframe latih lengkap dengan tanggal untuk mengambil konteks
start_date_train = '2022-07-04'
df_train_full = pd.DataFrame()
df_train_full['tanggal'] = pd.to_datetime(pd.date_range(start=start_date_train, periods=len(df_train_raw)))
df_train_full['Galon_Terjual'] = df_train_raw['Galon_Terjual'].values

# Gunakan tanggal mulai spesifik untuk data uji
start_date_test = '2025-07-02'
print(f"Tanggal mulai spesifik untuk data uji telah ditetapkan: {start_date_test}")
df_test_full = pd.DataFrame()
df_test_full['tanggal'] = pd.to_datetime(pd.date_range(start=start_date_test, periods=len(df_test_raw)))
df_test_full['Galon_Terjual'] = df_test_raw['Galon_Terjual'].values

# Ambil konteks dari data latih yang sudah memiliki tanggal
df_konteks = df_train_full.tail(KONTEKS_HARI).copy()

# Gabungkan data konteks dengan data uji
df_combined = pd.concat([df_konteks, df_test_full], ignore_index=True)


# --- 3. Pembersihan dan Rekayasa Fitur pada Data Gabungan ---
print("Melakukan pembersihan dan rekayasa fitur pada data gabungan...")
df_combined.set_index('tanggal', inplace=True)

df_combined['Galon_Terjual_Filled'] = df_combined['Galon_Terjual'].fillna(method='ffill')
df_combined['Galon_Terjual_Filled'].fillna(lower_bound, inplace=True)
df_combined['Galon_Terjual_Cleaned'] = df_combined['Galon_Terjual_Filled'].clip(lower=lower_bound, upper=upper_bound)

target_col = 'Galon_Terjual_Cleaned'
shifted_target = df_combined[target_col].shift(1)

# Fitur lag, rolling, dan kalender (sama seperti saat training)
for lag in [1, 2, 3, 7, 14]:
    df_combined[f'lag_{lag}'] = shifted_target.shift(lag)
for window in [3, 7, 14, 21]:
    df_combined[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df_combined[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()
df_combined['lag_diff_1'] = shifted_target.diff(1)
df_combined['lag_diff_7'] = shifted_target.diff(7)
df_combined['hari_dalam_bulan'] = df_combined.index.day
df_combined['hari_dalam_tahun'] = df_combined.index.dayofyear
df_combined['minggu_dalam_tahun'] = df_combined.index.isocalendar().week.astype(int)
df_combined['bulan'] = df_combined.index.month
df_combined['hari_minggu'] = df_combined.index.dayofweek
df_combined['is_weekend'] = (df_combined['hari_minggu'] >= 5).astype(int)
df_combined['awal_bulan'] = df_combined.index.is_month_start.astype(int)
df_combined['akhir_bulan'] = df_combined.index.is_month_end.astype(int)

# PENYESUAIAN: Tambahkan logika fitur siklus lonjakan yang sama persis
print("Membuat fitur siklus lonjakan penjualan pada data gabungan...")
spike_threshold = 49
df_combined['is_spike'] = (df_combined[target_col] > spike_threshold).astype(int)

spike_days = df_combined['is_spike'].copy()
spike_days[spike_days == 0] = np.nan
spike_days = spike_days.reset_index()
spike_days['day_num'] = range(len(spike_days))
spike_days.set_index('tanggal', inplace=True)
spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
spike_days['day_num'].fillna(method='ffill', inplace=True)
df_combined['days_since_last_spike'] = (range(len(df_combined)) - spike_days['day_num']).fillna(0)

df_combined.reset_index(inplace=True)
df_combined.fillna(0, inplace=True)


# --- 4. Melakukan Prediksi ---
print("Memisahkan data uji dan melakukan prediksi...")
df_test_final = df_combined.tail(len(df_test_raw))

# Pastikan urutan kolom sama persis dengan saat pelatihan
X_test = df_test_final[feature_names_from_training]
y_actual = df_test_final[target_col]

y_pred = model.predict(X_test)
y_pred = y_pred.clip(min=lower_bound, max=upper_bound)


# --- 5. Menyimpan Hasil ---
print(f"Menyimpan hasil ke {OUTPUT_CSV_HASIL}...")
df_result = df_test_final[['tanggal', 'Galon_Terjual']].copy()
df_result.rename(columns={'Galon_Terjual': 'Aktual_Asli'}, inplace=True)
df_result['Aktual_Dibersihkan'] = y_actual
df_result['Prediksi'] = y_pred
df_result['Prediksi_Bulat'] = np.round(y_pred)
df_result['Selisih'] = abs(df_result['Aktual_Dibersihkan'] - df_result['Prediksi_Bulat'])
df_result.to_csv(OUTPUT_CSV_HASIL, index=False)


# --- 6. Evaluasi Metrik ---
print("\n--- HASIL EVALUASI MODEL PADA DATA UJI ---")
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} galon")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print("-----------------------------------------")


# --- 7. Membuat Plot Perbandingan ---
print(f"Menyimpan plot perbandingan ke {OUTPUT_PLOT_HASIL}...")
plt.figure(figsize=(15, 7))
plt.plot(df_result['tanggal'], df_result['Aktual_Dibersihkan'], label='Aktual (Dibersihkan)', color='royalblue', marker='o', linestyle='-')
plt.plot(df_result['tanggal'], df_result['Prediksi'], label='Prediksi', color='darkorange', marker='x', linestyle='--')
plt.plot(df_result['tanggal'], df_result['Aktual_Asli'], label='Aktual (Asli)', color='green', marker='.', linestyle=':', alpha=0.5)
plt.title('Perbandingan Penjualan Aktual vs. Prediksi pada Data Uji')
plt.xlabel('Tanggal')
plt.ylabel('Galon Terjual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_HASIL)
plt.close()

print("\nEvaluasi selesai.")
