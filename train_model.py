import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Tambahkan GradientBoostingRegressor jika ingin mencoba
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
import holidays # Tambahkan import holidays
from datetime import datetime

# --- Load Data ---
# Membaca CSV tanpa header dan memberikan nama kolom secara eksplisit
# Kolom kedua dari CSV ('AngkaHari' dari damiu.csv) akan dinamai '_day_num_from_csv'
# dan tidak secara langsung digunakan setelah 'Tanggal' diparsing,
# kecuali jika Anda memiliki logika khusus untuknya.
df = pd.read_csv('data/damiu.csv', header=None, names=['Tanggal', '_day_num_from_csv', 'Galon Terjual'])

# --- Parsing & Urutkan Tanggal ---
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m-%d', errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df = df.sort_values(by='Tanggal')

# --- Reindex ke Kalender Lengkap (isi tanggal bolong) ---
all_dates = pd.date_range(start=df['Tanggal'].min(), end=df['Tanggal'].max(), freq='D')
df = df.set_index('Tanggal').reindex(all_dates).rename_axis('Tanggal').reset_index()

# --- Fitur Hari Libur ---
indo_holidays = holidays.Indonesia()
df['is_holiday'] = df['Tanggal'].apply(lambda date: 1 if date in indo_holidays else 0).astype(int)

# --- Fitur Waktu Dasar ---
df['hari_ke'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days
df['hari_dalam_minggu'] = df['Tanggal'].apply(lambda x: x.isoweekday()) # Senin=1, Minggu=7
df['bulan'] = df['Tanggal'].dt.month
df['minggu_ke'] = df['Tanggal'].dt.isocalendar().week.astype(int) # Pastikan integer
df['tahun'] = df['Tanggal'].dt.year
df['is_weekend'] = df['hari_dalam_minggu'].isin([6, 7]).astype(int) # Sabtu=6, Minggu=7

# --- Fitur Awal/Akhir Bulan ---
df['is_awal_bulan'] = (df['Tanggal'].dt.day <= 3).astype(int)
df['is_akhir_bulan'] = (df['Tanggal'].dt.day >= 28).astype(int)

# --- Fitur Interaksi ---
df['hari_minggu_x_holiday'] = df['hari_dalam_minggu'] * df['is_holiday']

# --- Fourier Terms untuk Musiman ---
# Musiman Tahunan (periode P = 365.25)
P_year = 365.25
df['sin_tahun_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_year)
df['cos_tahun_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_year)
df['sin_tahun_2'] = np.sin(2 * np.pi * 2 * df['hari_ke'] / P_year)
df['cos_tahun_2'] = np.cos(2 * np.pi * 2 * df['hari_ke'] / P_year)

# Musiman Mingguan (periode P = 7)
P_week = 7
df['sin_minggu_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_week)
df['cos_minggu_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_week)

# --- Pastikan kolom target adalah numerik ---
df['Galon Terjual'] = pd.to_numeric(df['Galon Terjual'], errors='coerce')

# --- Penanganan Outlier Sederhana (Winsorization) sebelum membuat fitur lag/rolling ---
lower_bound = df['Galon Terjual'].quantile(0.05) # Batas bawah 5%
upper_bound = df['Galon Terjual'].quantile(0.95) # Batas atas 95%
df['Galon Terjual_winsorized'] = df['Galon Terjual'].clip(lower=lower_bound, upper=upper_bound)

# --- Hitung nilai default untuk fillna dari data winsorized SEBELUM membuat fitur lag/rolling ---
static_default_avg_winsorized_train = df['Galon Terjual_winsorized'].mean()
static_default_std_winsorized_train = df['Galon Terjual_winsorized'].std()

# --- Lag Fitur & Rolling ---
# Gunakan 'Galon Terjual_winsorized' untuk membuat fitur turunan agar lebih stabil
df['penjualan_kemarin'] = df['Galon Terjual_winsorized'].shift(1).fillna(static_default_avg_winsorized_train)
df['penjualan_2hari_lalu'] = df['Galon Terjual_winsorized'].shift(2).fillna(static_default_avg_winsorized_train)

df['rata2_3hari'] = df['Galon Terjual_winsorized'].rolling(window=3, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)
df['rata2_7hari'] = df['Galon Terjual_winsorized'].rolling(window=7, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)
df['rata2_14hari'] = df['Galon Terjual_winsorized'].rolling(window=14, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)

df['std_7hari'] = df['Galon Terjual_winsorized'].rolling(window=7, min_periods=1).std().fillna(static_default_std_winsorized_train)

df['delta_penjualan'] = df['Galon Terjual_winsorized'].diff().fillna(0)

# --- Bersihkan NaN akibat rolling & shift ---
# dropna() di sini akan membersihkan baris jika 'Galon Terjual' (target) adalah NaN
# atau jika ada NaN lain yang tidak tertangani oleh fillna di atas.
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
    'is_holiday',
    'hari_minggu_x_holiday', # Fitur interaksi
    'penjualan_kemarin',
    'penjualan_2hari_lalu',
    'rata2_3hari',
    'rata2_7hari',
    'rata2_14hari',
    'std_7hari',
    'delta_penjualan',
    'sin_tahun_1', # Fourier terms
    'cos_tahun_1',
    'sin_tahun_2',
    'cos_tahun_2',
    'sin_minggu_1',
    'cos_minggu_1'
]

# --- Fitur Terkait Hari Nol Penjualan ---
# Identifikasi hari dengan penjualan nol
df['is_zero_sale_day'] = (df['Galon Terjual'] == 0).astype(int)

# Hitung hari sejak penjualan nol terakhir
# Ini sedikit tricky karena perlu forward fill tanggal, lalu hitung selisih
zero_sale_dates = df[df['is_zero_sale_day'] == 1]['Tanggal']
if not zero_sale_dates.empty:
    # Buat series dengan tanggal penjualan nol, lalu reindex ke semua tanggal
    zero_sale_series = pd.Series(zero_sale_dates.values, index=zero_sale_dates)
    # Forward fill tanggal penjualan nol terakhir
    last_zero_sale_date = zero_sale_series.reindex(df['Tanggal'], method='ffill') # type: ignore
    df['days_since_last_zero_sale'] = (df['Tanggal'] - last_zero_sale_date).dt.days.fillna(df['hari_ke'])
else:
    # Jika tidak ada hari penjualan nol, isi dengan hari_ke sebagai fallback
    # untuk memastikan kolom selalu ada.
    df['days_since_last_zero_sale'] = df['hari_ke']

# Fitur biner: apakah hari ini setelah hari penjualan nol
df['is_day_after_zero_sale'] = df['is_zero_sale_day'].shift(1).fillna(0).astype(int)

# --- Perbarui feature_column_names ---
feature_column_names.extend(['is_zero_sale_day', 'days_since_last_zero_sale', 'is_day_after_zero_sale'])


target_column_name = 'Galon Terjual'

# --- Transformasi Target (Log) ---
# df['Galon Terjual'] = np.log1p(df['Galon Terjual']) # Uncomment jika ingin mencoba transformasi log

X = df[feature_column_names]
y = df[target_column_name]

# --- Split Data Train & Test (berdasarkan urutan waktu) ---
# Split data menjadi training dan testing untuk evaluasi akhir
# Train set digunakan untuk tuning hyperparameter dengan TimeSeriesSplit CV
train_size_ratio = 0.8
train_index_end = int(len(df) * train_size_ratio)

X_train, X_test = X.iloc[:train_index_end], X.iloc[train_index_end:]
y_train, y_test = y.iloc[:train_index_end], y.iloc[train_index_end:]

# --- Inverse Transformasi untuk Evaluasi (jika menggunakan transformasi log) ---
# def inverse_transform(y_transformed):
#     return np.expm1(y_transformed)

# --- Konfigurasi TimeSeriesSplit ---
# n_splits: Jumlah split. Menentukan berapa banyak "fold" yang akan dibuat.
#           Dengan n_splits=5, akan ada 5 iterasi:
#           Fold 1: Train [0..~0.2), Test [~0.2..~0.4)
#           Fold 2: Train [0..~0.4), Test [~0.4..~0.6)
#           ...
#           Fold 5: Train [0..~0.8), Test [~0.8..1.0) (Ini akan menggunakan seluruh X_train)
# max_train_size: Batas ukuran maksimum untuk setiap fold training set (opsional)
# gap: Jumlah sampel yang akan dikecualikan antara train dan test set (opsional)
tscv = TimeSeriesSplit(n_splits=5) # Sesuaikan n_splits sesuai kebutuhan dan ukuran data

# --- Hyperparameter Tuning dengan RandomizedSearchCV ---

# Parameter grid untuk Random Forest
param_dist_rf = {
    'n_estimators': [200, 300, 500, 800], # Diperluas sedikit di sekitar 500
    'max_depth': [15, 20, 25, None], # Diperluas sedikit di sekitar None
    'min_samples_split': [2, 3, 5], # Diperluas sedikit di sekitar 2
    'min_samples_leaf': [1, 2], # Diperluas sedikit di sekitar 1
    'bootstrap': [True, False]
}

# Parameter grid untuk XGBoost
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 500, 800, 1000], # Diperluas
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], # Diperluas
    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15], # Diperluas
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Diperluas
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Diperluas
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5], # Diperluas
    'reg_alpha': [0.5, 1, 2, 5], # Diperluas sedikit di sekitar 2
    'reg_lambda': [0.5, 1, 2, 5] # Diperluas sedikit di sekitar 2
}

# --- Latih Model Random Forest ---
rf_model = RandomForestRegressor(
    # Hyperparameter ini sebaiknya di-tune menggunakan GridSearchCV atau RandomizedSearchCV
    # dengan TimeSeriesSplit sebagai cv strategy.
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

print("\nMelakukan tuning hyperparameter untuk Random Forest...")
rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist_rf,
    n_iter=50, # Jumlah kombinasi parameter yang akan dicoba. Sesuaikan!
    cv=tscv,
    scoring='neg_mean_squared_error', # Menggunakan MSE sebagai metrik, neg_ karena GridSearchCV memaksimalkan skor
    random_state=42,
    n_jobs=-1) # Gunakan semua core CPU
rf_random_search.fit(X_train, y_train)

# Gunakan model terbaik dari tuning
best_rf_model = rf_random_search.best_estimator_
print(f"Best parameters for Random Forest: {rf_random_search.best_params_}")

# --- Evaluasi Random Forest ---
y_pred_test = best_rf_model.predict(X_test) # Gunakan best_rf_model
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"\nEvaluasi Model pada Data Uji:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- Tampilkan Feature Importances untuk Random Forest ---
print("\nFeature Importances (Random Forest):")
importances_rf = best_rf_model.feature_importances_ # Gunakan best_rf_model
sorted_indices_rf = np.argsort(importances_rf)[::-1]
for i in sorted_indices_rf:
    print(f"{feature_column_names[i]:<30}: {importances_rf[i]:.4f}")



# --- Simpan Model Random Forest ---
joblib.dump(best_rf_model, 'random_forest_gallon_model.joblib')

# --- Latih Model XGBoost ---
xgb_model = XGBRegressor(
    # Hyperparameter ini juga sebaiknya di-tune.
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1, # type: ignore
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    # Pertimbangkan parameter regularisasi untuk data fluktuatif:
    # reg_alpha=0.1, # L1 regularization
    # reg_lambda=0.1  # L2 regularization
)

print("\nMelakukan tuning hyperparameter untuk XGBoost...")
xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist_xgb, n_iter=100, cv=tscv, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1) # n_iter ditingkatkan menjadi 100
xgb_random_search.fit(X_train, y_train)

# Gunakan model terbaik dari tuning
best_xgb_model = xgb_random_search.best_estimator_
print(f"Best parameters for XGBoost: {xgb_random_search.best_params_}")


# --- Evaluasi XGBoost ---
y_pred_xgb = best_xgb_model.predict(X_test) # Gunakan best_xgb_model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"\nEvaluasi XGBoost pada Data Uji:")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.2f}")
print(f"R-squared (R2): {r2_xgb:.2f}")

# --- Tampilkan Feature Importances untuk XGBoost ---
print("\nFeature Importances (XGBoost):")
importances_xgb = best_xgb_model.feature_importances_ # Gunakan best_xgb_model
sorted_indices_xgb = np.argsort(importances_xgb)[::-1]
for i in sorted_indices_xgb:
    print(f"{feature_column_names[i]:<30}: {importances_xgb[i]:.4f}")

# --- GridSearchCV untuk Fine-tuning (Opsional) ---
# Anda bisa menambahkan langkah ini untuk fine-tuning di sekitar parameter terbaik
# yang ditemukan oleh RandomizedSearchCV. Ini akan memakan waktu lebih lama.

from sklearn.model_selection import GridSearchCV

# Contoh Grid Parameter untuk GridSearchCV (sesuaikan berdasarkan hasil RandomizedSearchCV)
# Ini hanya contoh, Anda perlu menyesuaikannya dengan best_params_ yang Anda dapatkan
param_grid_rf = {
    'n_estimators': sorted(list(set([rf_random_search.best_params_['n_estimators'], max(100, rf_random_search.best_params_['n_estimators'] - 100), rf_random_search.best_params_['n_estimators'] + 100]))),
    'max_depth': sorted(list(set([rf_random_search.best_params_['max_depth'], None if rf_random_search.best_params_['max_depth'] is None else max(1, rf_random_search.best_params_['max_depth'] - 2), None if rf_random_search.best_params_['max_depth'] is None else rf_random_search.best_params_['max_depth'] + 2]))),
    'bootstrap': [rf_random_search.best_params_['bootstrap']]
}

# Refined dynamic generation for min_samples_split and min_samples_leaf
best_min_samples_split = rf_random_search.best_params_['min_samples_split']
if isinstance(best_min_samples_split, int):
    # Ensure integer min_samples_split is >= 2
    grid_min_samples_split = sorted(list(set([best_min_samples_split, max(2, best_min_samples_split - 1), best_min_samples_split + 1])))
else: # Assume float (0.0, 1.0]
     grid_min_samples_split = sorted(list(set([best_min_samples_split, max(0.01, best_min_samples_split - 0.05), min(1.0, best_min_samples_split + 0.05)])))

best_min_samples_leaf = rf_random_search.best_params_['min_samples_leaf']
if isinstance(best_min_samples_leaf, int):
    # Ensure integer min_samples_leaf is >= 1
    grid_min_samples_leaf = sorted(list(set([best_min_samples_leaf, max(1, best_min_samples_leaf - 1), best_min_samples_leaf + 1])))
else: # Assume float (0.0, 0.5]
     grid_min_samples_leaf = sorted(list(set([best_min_samples_leaf, max(0.01, best_min_samples_leaf - 0.02), min(0.5, best_min_samples_leaf + 0.02)])))

# Add the refined parameters to the grid
param_grid_rf['min_samples_split'] = grid_min_samples_split
param_grid_rf['min_samples_leaf'] = grid_min_samples_leaf

# --- Parameter Grid untuk GridSearchCV XGBoost (Fokus pada parameter kunci) ---
# Ambil parameter terbaik dari RandomizedSearchCV untuk XGBoost
best_params_xgb_random = xgb_random_search.best_params_

# Definisikan grid hanya untuk parameter yang ingin di-fine-tune lebih lanjut
# dan gunakan nilai terbaik dari RandomizedSearch untuk parameter lainnya.
param_grid_xgb = {
    'n_estimators': sorted(list(set([
        best_params_xgb_random['n_estimators'],
        max(50, best_params_xgb_random['n_estimators'] - 25), # Kurangi delta, min 50
        best_params_xgb_random['n_estimators'] + 25
    ]))),
    'learning_rate': sorted(list(set([
        best_params_xgb_random['learning_rate'],
        max(0.001, best_params_xgb_random['learning_rate'] - 0.01), # Kurangi delta, min 0.001
        best_params_xgb_random['learning_rate'] + 0.01
    ]))),
    'max_depth': sorted(list(set([
        best_params_xgb_random['max_depth'],
        max(1, best_params_xgb_random['max_depth'] - 1), # Pastikan min depth adalah 1
        best_params_xgb_random['max_depth'] + 1
    ]))),
    # Parameter lain diatur sebagai list dengan satu nilai (fixed dari hasil RandomizedSearchCV)
    'subsample': [best_params_xgb_random['subsample']],
    'colsample_bytree': [best_params_xgb_random['colsample_bytree']],
    'gamma': [best_params_xgb_random['gamma']],
    'reg_alpha': [best_params_xgb_random['reg_alpha']],
    'reg_lambda': [best_params_xgb_random['reg_lambda']]
}

print("\nMelakukan fine-tuning dengan GridSearchCV untuk Random Forest...")
rf_grid_search = GridSearchCV(estimator=rf_random_search.best_estimator_, param_grid=param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf_model_tuned = rf_grid_search.best_estimator_
print(f"Best parameters for Random Forest (GridSearchCV): {rf_grid_search.best_params_}")

print("\nMelakukan fine-tuning dengan GridSearchCV untuk XGBoost...")
xgb_grid_search = GridSearchCV(estimator=xgb_random_search.best_estimator_, param_grid=param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model_tuned = xgb_grid_search.best_estimator_
print(f"Best parameters for XGBoost (GridSearchCV): {xgb_grid_search.best_params_}")

# --- Evaluasi Model Terbaik dari GridSearchCV ---
print("\nEvaluasi Model Random Forest (GridSearchCV) pada Data Uji:")
y_pred_rf_tuned = best_rf_model_tuned.predict(X_test)
mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)
mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
rmse_rf_tuned = np.sqrt(mse_rf_tuned)
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)
print(f"Mean Squared Error (MSE): {mse_rf_tuned:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf_tuned:.2f}")
print(f"Mean Absolute Error (MAE): {mae_rf_tuned:.2f}")
print(f"R-squared (R2): {r2_rf_tuned:.2f}")

print("\nEvaluasi Model XGBoost (GridSearchCV) pada Data Uji:")
y_pred_xgb_tuned = best_xgb_model_tuned.predict(X_test)
mse_xgb_tuned = mean_squared_error(y_test, y_pred_xgb_tuned)
mae_xgb_tuned = mean_absolute_error(y_test, y_pred_xgb_tuned)
rmse_xgb_tuned = np.sqrt(mse_xgb_tuned)
r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)
print(f"Mean Squared Error (MSE): {mse_xgb_tuned:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb_tuned:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb_tuned:.2f}")
print(f"R-squared (R2): {r2_xgb_tuned:.2f}")

# --- Evaluasi Model Ensemble (Rata-rata) ---
print("\nEvaluasi Model Ensemble (Rata-rata) pada Data Uji:")
y_pred_ensemble = (y_pred_rf_tuned + y_pred_xgb_tuned) / 2
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mse_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)
print(f"Mean Squared Error (MSE): {mse_ensemble:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_ensemble:.2f}")
print(f"Mean Absolute Error (MAE): {mae_ensemble:.2f}")
print(f"R-squared (R2): {r2_ensemble:.2f}")

# --- Simpan Model Terbaik (dari GridSearchCV) dan Metadata ---
joblib.dump(best_rf_model_tuned, 'random_forest_gallon_model.joblib')
joblib.dump(best_xgb_model_tuned, 'xgboost_gallon_model.joblib')

model_metadata = {
    'features': feature_column_names,
    'training_start_date': df['Tanggal'].min().isoformat(),
    'training_end_date': df['Tanggal'].max().isoformat(),
    'model_types': ['RandomForestRegressor', 'XGBRegressor'], # type: ignore
    'rf_rmse': rmse_rf_tuned,
    'rf_mae': mae_rf_tuned,
    'rf_r2': r2_rf_tuned,
    'xgb_rmse': rmse_xgb_tuned,
    'xgb_mae': mae_xgb_tuned,
    'xgb_r2': r2_xgb_tuned,
    'ensemble_rmse': rmse_ensemble,
    'ensemble_mae': mae_ensemble,
    'ensemble_r2': r2_ensemble,
    'rf_best_params_randomized': rf_random_search.best_params_,
    'xgb_best_params_randomized': xgb_random_search.best_params_,
    'rf_best_params_grid': rf_grid_search.best_params_,
    'xgb_best_params_grid': xgb_grid_search.best_params_
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)

print("\nModel dan metadata berhasil disimpan.")


#  ---- SHAP Analysis ----
import matplotlib.pyplot as plt # Import matplotlib
try:
    import shap
    print("\nMelakukan SHAP analysis...")

    # Gunakan model terbaik dari tuning
    # Untuk tree-based models (RF, XGBoost), gunakan TreeExplainer
    explainer_rf = shap.TreeExplainer(best_rf_model_tuned) # Gunakan model dari GridSearchCV
    explainer_xgb = shap.TreeExplainer(best_xgb_model_tuned) # Gunakan model dari GridSearchCV

    # Hitung SHAP values pada data uji
    # Perhatikan: shap_values untuk TreeExplainer bisa berupa array atau list of arrays
    # tergantung apakah modelnya regressor atau classifier, dan apakah itu ensemble
    # Untuk regressor, biasanya array tunggal.
    shap_values_rf = explainer_rf.shap_values(X_test)
    shap_values_xgb = explainer_xgb.shap_values(X_test)

    # SHAP Summary Plot (Global Feature Importance)
    print("\nSHAP Summary Plot (Global Feature Importance):")
        # Try setting a non-GUI backend for matplotlib
    try:
        plt.switch_backend('Agg')
        print("DEBUG: Matplotlib backend set to Agg.")
    except Exception as e:
        print(f"DEBUG: Failed to set matplotlib backend to Agg: {e}")
    shap.summary_plot(shap_values_rf, X_test, feature_names=feature_column_names, show=False) # Use show=False to prevent interactive plot
    shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_column_names, show=False) # Use show=False
    # Simpan plot Random Forest
    shap.summary_plot(shap_values_rf, X_test, feature_names=feature_column_names, show=False)
    plt.savefig("shap_summary_rf.png", bbox_inches='tight') # Simpan ke file
    plt.close() # Tutup plot agar tidak tumpang tindih

    # Simpan plot XGBoost
    shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_column_names, show=False)
    plt.savefig("shap_summary_xgb.png", bbox_inches='tight') # Simpan ke file
    plt.close() # Tutup plot

    print("Plot SHAP telah disimpan sebagai shap_summary_rf.png dan shap_summary_xgb.png")
except ImportError:
    print("\nSHAP library not installed. Skipping SHAP analysis. Install with: pip install shap")
except Exception as e:
    print(f"\nAn error occurred during SHAP analysis: {e}")
    import traceback
    traceback.print_exc()
