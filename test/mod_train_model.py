import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- Konfigurasi Model ---
# Sesuaikan rasio pembagian data di sini. 0.8 berarti 80% data untuk pelatihan dan 20% untuk pengujian.
TRAIN_SIZE_RATIO = 0.8

# --- 1. Pemuatan dan Pra-pemrosesan Data ---
print("Memuat dan memproses data dari galon.csv...")
try:
    df = pd.read_csv('galon.csv', header=0)
    df.columns = ['Hari', 'Galon_Terjual'] 
except FileNotFoundError:
    print("Error: File 'galon.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

df['hari_ke'] = range(1, len(df) + 1)
df['Galon_Terjual'] = pd.to_numeric(df['Galon_Terjual'], errors='coerce')
df['Hari'] = pd.to_numeric(df['Hari'], errors='coerce')
df.dropna(inplace=True)


# --- 2. Rekayasa Fitur (Feature Engineering) ---
print("Melakukan rekayasa fitur...")

# Kolom target akan disesuaikan berdasarkan aturan bisnis.
df['Galon_Terjual_Adjusted'] = df['Galon_Terjual'].copy()

# Penyesuaian untuk data noise (penjualan < 10 dianggap bukan penjualan riil).
NOISE_THRESHOLD = 10
df['is_noise_day'] = (df['Galon_Terjual'] < NOISE_THRESHOLD).astype(int)
df.loc[df['Galon_Terjual'] < NOISE_THRESHOLD, 'Galon_Terjual_Adjusted'] = 0

# Penyesuaian untuk lonjakan penjualan (dianggap laporan gabungan).
SPIKE_THRESHOLD = 50
df['is_spike_day'] = (df['Galon_Terjual'] > SPIKE_THRESHOLD).astype(int)

# Mendistribusikan ulang nilai penjualan dari hari lonjakan ke hari-hari sebelumnya.
for index, row in df.iterrows():
    if row['Galon_Terjual'] > SPIKE_THRESHOLD:
        excess = row['Galon_Terjual'] - SPIKE_THRESHOLD
        df.at[index, 'Galon_Terjual_Adjusted'] = SPIKE_THRESHOLD
        
        prev_day_index = index - 1
        while excess > 0 and prev_day_index >= 0:
            fill_amount = SPIKE_THRESHOLD - df.at[prev_day_index, 'Galon_Terjual_Adjusted']
            if fill_amount > 0:
                add_amount = min(excess, fill_amount)
                df.at[prev_day_index, 'Galon_Terjual_Adjusted'] += add_amount
                excess -= add_amount
            prev_day_index -= 1

# Penanda untuk kebutuhan mendadak (penjualan tepat 40 galon).
URGENT_DELIVERY_VALUE = 40
df['is_urgent_delivery'] = (df['Galon_Terjual'] == URGENT_DELIVERY_VALUE).astype(int)

# Penanda untuk kategori penjualan berdasarkan ambang batas.
df['penjualan_lebih_25'] = (df['Galon_Terjual_Adjusted'] > 25).astype(int)
df['penjualan_kurang_25'] = (df['Galon_Terjual_Adjusted'] < 25).astype(int)
df['penjualan_lebih_35'] = (df['Galon_Terjual_Adjusted'] > 35).astype(int)
df['penjualan_kurang_35'] = (df['Galon_Terjual_Adjusted'] < 35).astype(int)
df['penjualan_lebih_20'] = (df['Galon_Terjual_Adjusted'] > 20).astype(int)
df['penjualan_kurang_20'] = (df['Galon_Terjual_Adjusted'] < 20).astype(int)


# --- Pra-pembagian data untuk statistik ---
train_index_end = int(len(df) * TRAIN_SIZE_RATIO)
df_train_for_stats = df.iloc[:train_index_end].copy()

# Hitung statistik (rata-rata, std) dari data latih untuk mengisi nilai NaN nanti.
static_default_avg = df_train_for_stats['Galon_Terjual_Adjusted'].mean()
static_default_std = df_train_for_stats['Galon_Terjual_Adjusted'].std()

# Fitur statistik berdasarkan hari (rata-rata & std penjualan untuk setiap hari).
day_stats = df_train_for_stats.groupby('Hari')['Galon_Terjual_Adjusted'].agg(['mean', 'std']).rename(
    columns={'mean': 'rata2_per_hari', 'std': 'std_per_hari'}
)
df = pd.merge(df, day_stats, on='Hari', how='left')
df['rata2_per_hari'].fillna(static_default_avg, inplace=True)
df['std_per_hari'].fillna(static_default_std, inplace=True)

# Fitur musiman mingguan.
df['sin_minggu'] = np.sin(2 * np.pi * df['Hari'] / 7)
df['cos_minggu'] = np.cos(2 * np.pi * df['Hari'] / 7)

# Fitur Lag dan Rolling Window (berdasarkan data yang sudah disesuaikan).
df['penjualan_kemarin'] = df['Galon_Terjual_Adjusted'].shift(1).fillna(static_default_avg)
df['penjualan_2hari_lalu'] = df['Galon_Terjual_Adjusted'].shift(2).fillna(static_default_avg)
df['rata2_3hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(window=3, min_periods=1).mean().fillna(static_default_avg)
df['rata2_7hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(window=7, min_periods=1).mean().fillna(static_default_avg)
df['rata2_14hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(window=14, min_periods=1).mean().fillna(static_default_avg)
df['std_7hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(window=7, min_periods=1).std().fillna(static_default_std)
df['delta_penjualan'] = df['Galon_Terjual_Adjusted'].shift(1).diff().fillna(0)

# Mendefinisikan fitur dan target.
feature_column_names = [
    'Hari', 'hari_ke',
    'penjualan_kemarin', 'penjualan_2hari_lalu', 'rata2_3hari', 'rata2_7hari',
    'rata2_14hari', 'std_7hari', 'delta_penjualan',
    'sin_minggu', 'cos_minggu',
    'rata2_per_hari', 'std_per_hari',
    'is_spike_day',
    'is_urgent_delivery',
    'is_noise_day',
    'penjualan_lebih_25',
    'penjualan_kurang_25',
    'penjualan_lebih_35',
    'penjualan_kurang_35',
    'penjualan_lebih_20',
    'penjualan_kurang_20'
]
target_column_name = 'Galon_Terjual_Adjusted'

X = df[feature_column_names]
y = df[target_column_name]

# --- 3. Pembagian Data ---
print(f"Membagi data: {train_index_end} untuk pelatihan, {len(df) - train_index_end} untuk pengujian.")
X_train, X_test = X.iloc[:train_index_end], X.iloc[train_index_end:]
y_train, y_test = y.iloc[:train_index_end], y.iloc[train_index_end:]

tscv = TimeSeriesSplit(n_splits=5)

# --- 4. Pelatihan dan Penyetelan Model XGBoost ---

# Langkah 1: RandomizedSearchCV untuk pencarian area yang luas
print("\nLangkah 1: Menjalankan RandomizedSearchCV...")
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2]
}

xgb_model = XGBRegressor(random_state=42)
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=50, 
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)
xgb_random_search.fit(X_train, y_train)
print(f"Parameter terbaik dari RandomizedSearch: {xgb_random_search.best_params_}")

# Langkah 2: GridSearchCV untuk penyetelan halus di sekitar parameter terbaik
print("\nLangkah 2: Menjalankan GridSearchCV untuk penyetelan halus...")
best_params_rs = xgb_random_search.best_params_
param_grid_gs = {
    'n_estimators': [best_params_rs['n_estimators'] - 50, best_params_rs['n_estimators'], best_params_rs['n_estimators'] + 50],
    'learning_rate': [best_params_rs['learning_rate'] * 0.8, best_params_rs['learning_rate'], best_params_rs['learning_rate'] * 1.2],
    'max_depth': [best_params_rs['max_depth'] - 1, best_params_rs['max_depth'], best_params_rs['max_depth'] + 1],
    'subsample': [best_params_rs['subsample']],
    'colsample_bytree': [best_params_rs['colsample_bytree']]
}
# Pastikan nilai parameter tidak negatif
param_grid_gs['n_estimators'] = [n for n in param_grid_gs['n_estimators'] if n > 0]
param_grid_gs['max_depth'] = [d for d in param_grid_gs['max_depth'] if d > 0]


xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_gs,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)
print(f"Parameter terbaik final dari GridSearchCV: {xgb_grid_search.best_params_}")

best_xgb_model = xgb_grid_search.best_estimator_

# --- 5. Evaluasi Model ---
print("\nEvaluasi Model XGBoost pada Data Uji:")
y_pred = best_xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} galon")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- 6. Penyimpanan Model dan Metadata ---
print("\nMenyimpan model dan metadata...")
joblib.dump(best_xgb_model, 'xgboost_gallon_model.joblib')

# Memastikan serialisasi JSON berhasil dengan mengonversi tipe data NumPy.
cleaned_best_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in xgb_grid_search.best_params_.items()}
cleaned_day_stats = day_stats.astype(float).to_dict('index')

model_metadata = {
    'features': feature_column_names,
    'recent_14_days_sales_for_prediction': [float(x) for x in df['Galon_Terjual_Adjusted'].iloc[-14:].values],
    'transformation_stats': {
        'spike_threshold': SPIKE_THRESHOLD,
        'urgent_delivery_value': URGENT_DELIVERY_VALUE,
        'noise_threshold': NOISE_THRESHOLD,
        'static_default_avg': float(static_default_avg),
        'static_default_std': float(static_default_std)
    },
    'day_of_week_stats': cleaned_day_stats,
    'model_performance_on_test_set': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    },
    'model_best_params': cleaned_best_params
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

print("Model dan metadata berhasil disimpan.")

# --- 7. Analisis dan Visualisasi ---
print("\nMembuat visualisasi hasil prediksi...")

# Plot perbandingan nilai aktual (setelah disesuaikan) vs. prediksi.
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label='Aktual (Disesuaikan)', color='blue', marker='o', linestyle='-')
plt.plot(y_test.index, y_pred, label='Prediksi', color='red', marker='x', linestyle='--')
plt.plot(df.loc[y_test.index, 'Galon_Terjual'], label='Data Asli', color='green', marker='.', linestyle=':', alpha=0.5)
plt.title('Perbandingan Penjualan Aktual (Disesuaikan) vs. Prediksi')
plt.xlabel('Indeks Data Uji')
plt.ylabel('Galon Terjual')
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_prediction.png")
plt.close()
print("Plot perbandingan telah disimpan.")

# Plot feature importance.
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(best_xgb_model.feature_importances_, index=X.columns)
feat_importances.nlargest(len(feature_column_names)).plot(kind='barh')
plt.title('Feature Importance dari Model XGBoost')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Plot feature importance telah disimpan.")
