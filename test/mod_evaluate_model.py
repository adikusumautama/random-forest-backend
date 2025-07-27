import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
# Pastikan file ini ada dan memiliki kolom 'Hari' dan 'Galon_Terjual'
INPUT_CSV_UJI = "mod_data_uji.csv" 
OUTPUT_CSV_HASIL = "hasil_evaluasi.csv"
OUTPUT_PLOT_HASIL = "plot_evaluasi.png"
MODEL_PATH = "xgboost_gallon_model.joblib"
METADATA_PATH = "model_metadata.json"

# --- 1. Memuat Model dan Metadata ---
print(f"Memuat model dari {MODEL_PATH} dan metadata dari {METADATA_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("Error: Pastikan file model dan metadata ada di direktori yang sama.")
    exit()

# Ekstrak informasi penting dari metadata
feature_names_from_training = metadata['features']
stats = metadata.get('transformation_stats', {})
day_stats_map = metadata.get('day_of_week_stats', {})

# --- 2. Memuat Data Uji ---
print(f"Memuat data uji dari {INPUT_CSV_UJI}...")
try:
    df_test = pd.read_csv(INPUT_CSV_UJI)
    df_test.columns = ['Hari', 'Galon_Terjual']
except FileNotFoundError:
    print(f"Error: File data uji '{INPUT_CSV_UJI}' tidak ditemukan.")
    exit()

# --- 3. Rekayasa Fitur (Sama Persis dengan Skrip Pelatihan) ---
print("Melakukan rekayasa fitur pada data uji...")

# Konversi tipe data dan buat 'hari_ke'
df_test['hari_ke'] = range(1, len(df_test) + 1)
df_test['Galon_Terjual'] = pd.to_numeric(df_test['Galon_Terjual'], errors='coerce')
df_test['Hari'] = pd.to_numeric(df_test['Hari'], errors='coerce')
df_test.dropna(inplace=True)

# Terapkan aturan bisnis untuk membuat target yang disesuaikan
df_test['Galon_Terjual_Adjusted'] = df_test['Galon_Terjual'].copy()

NOISE_THRESHOLD = stats.get('noise_threshold', 10)
SPIKE_THRESHOLD = stats.get('spike_threshold', 50)
URGENT_DELIVERY_VALUE = stats.get('urgent_delivery_value', 40)

df_test['is_noise_day'] = (df_test['Galon_Terjual'] < NOISE_THRESHOLD).astype(int)
df_test.loc[df_test['Galon_Terjual'] < NOISE_THRESHOLD, 'Galon_Terjual_Adjusted'] = 0

df_test['is_spike_day'] = (df_test['Galon_Terjual'] > SPIKE_THRESHOLD).astype(int)
for index, row in df_test.iterrows():
    if row['Galon_Terjual'] > SPIKE_THRESHOLD:
        excess = row['Galon_Terjual'] - SPIKE_THRESHOLD
        df_test.at[index, 'Galon_Terjual_Adjusted'] = SPIKE_THRESHOLD
        prev_day_index = index - 1
        while excess > 0 and prev_day_index >= 0:
            if prev_day_index in df_test.index:
                fill_amount = SPIKE_THRESHOLD - df_test.at[prev_day_index, 'Galon_Terjual_Adjusted']
                if fill_amount > 0:
                    add_amount = min(excess, fill_amount)
                    df_test.at[prev_day_index, 'Galon_Terjual_Adjusted'] += add_amount
                    excess -= add_amount
            prev_day_index -= 1

df_test['is_urgent_delivery'] = (df_test['Galon_Terjual'] == URGENT_DELIVERY_VALUE).astype(int)

# Buat fitur penanda ambang batas
df_test['penjualan_lebih_25'] = (df_test['Galon_Terjual_Adjusted'] > 25).astype(int)
df_test['penjualan_kurang_25'] = (df_test['Galon_Terjual_Adjusted'] < 25).astype(int)
df_test['penjualan_lebih_35'] = (df_test['Galon_Terjual_Adjusted'] > 35).astype(int)
df_test['penjualan_kurang_35'] = (df_test['Galon_Terjual_Adjusted'] < 35).astype(int)
df_test['penjualan_lebih_20'] = (df_test['Galon_Terjual_Adjusted'] > 20).astype(int)
df_test['penjualan_kurang_20'] = (df_test['Galon_Terjual_Adjusted'] < 20).astype(int)

# Buat fitur statistik dan waktu
static_avg = stats.get('static_default_avg', df_test['Galon_Terjual_Adjusted'].mean())
static_std = stats.get('static_default_std', df_test['Galon_Terjual_Adjusted'].std())

df_test['rata2_per_hari'] = df_test['Hari'].astype(str).map(lambda x: day_stats_map.get(x, {}).get('mean', static_avg))
df_test['std_per_hari'] = df_test['Hari'].astype(str).map(lambda x: day_stats_map.get(x, {}).get('std', static_std))

df_test['sin_minggu'] = np.sin(2 * np.pi * df_test['Hari'] / 7)
df_test['cos_minggu'] = np.cos(2 * np.pi * df_test['Hari'] / 7)

df_test['penjualan_kemarin'] = df_test['Galon_Terjual_Adjusted'].shift(1).fillna(static_avg)
df_test['penjualan_2hari_lalu'] = df_test['Galon_Terjual_Adjusted'].shift(2).fillna(static_avg)
df_test['rata2_3hari'] = df_test['Galon_Terjual_Adjusted'].shift(1).rolling(3, min_periods=1).mean().fillna(static_avg)
df_test['rata2_7hari'] = df_test['Galon_Terjual_Adjusted'].shift(1).rolling(7, min_periods=1).mean().fillna(static_avg)
df_test['rata2_14hari'] = df_test['Galon_Terjual_Adjusted'].shift(1).rolling(14, min_periods=1).mean().fillna(static_avg)
df_test['std_7hari'] = df_test['Galon_Terjual_Adjusted'].shift(1).rolling(7, min_periods=1).std().fillna(static_std)
df_test['delta_penjualan'] = df_test['Galon_Terjual_Adjusted'].shift(1).diff().fillna(0)


# --- 4. Melakukan Prediksi ---
print("Melakukan prediksi pada data uji...")
# Pastikan urutan kolom sama persis dengan saat pelatihan
X_test = df_test[feature_names_from_training]
# Target aktual adalah data yang sudah disesuaikan, sama seperti saat pelatihan
y_actual_adjusted = df_test['Galon_Terjual_Adjusted']

y_pred = model.predict(X_test)

# --- 5. Menyimpan Hasil ---
print(f"Menyimpan hasil ke {OUTPUT_CSV_HASIL}...")
df_result = df_test[['hari_ke', 'Hari', 'Galon_Terjual']].copy()
df_result.rename(columns={'Galon_Terjual': 'Aktual_Asli'}, inplace=True)
df_result['Aktual_Disesuaikan'] = y_actual_adjusted
df_result['Prediksi'] = y_pred
df_result['Prediksi_Bulat'] = np.round(y_pred)
df_result['Selisih'] = abs(df_result['Aktual_Disesuaikan'] - df_result['Prediksi_Bulat'])
df_result.to_csv(OUTPUT_CSV_HASIL, index=False)

# --- 6. Evaluasi Metrik ---
print("\n--- HASIL EVALUASI MODEL ---")
mae = mean_absolute_error(y_actual_adjusted, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual_adjusted, y_pred))
r2 = r2_score(y_actual_adjusted, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} galon")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print("----------------------------")

# --- 7. Membuat Plot Perbandingan ---
print(f"Menyimpan plot perbandingan ke {OUTPUT_PLOT_HASIL}...")
plt.figure(figsize=(15, 7))
plt.plot(df_result['hari_ke'], df_result['Aktual_Disesuaikan'], label='Aktual (Disesuaikan)', color='blue', marker='o', linestyle='-')
plt.plot(df_result['hari_ke'], df_result['Prediksi'], label='Prediksi', color='red', marker='x', linestyle='--')
plt.plot(df_result['hari_ke'], df_result['Aktual_Asli'], label='Aktual (Asli)', color='green', marker='.', linestyle=':', alpha=0.5)
plt.title('Perbandingan Penjualan Aktual vs. Prediksi pada Data Uji')
plt.xlabel('Indeks Hari pada Data Uji')
plt.ylabel('Galon Terjual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_HASIL)
plt.close()

print("\nEvaluasi selesai.")
