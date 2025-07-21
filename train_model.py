import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
import holidays
import matplotlib.pyplot as plt

# --- 1. Pemuatan dan Pra-pemrosesan Data ---
print("Memuat dan memproses data...")
# Membaca CSV tanpa header dan memberikan nama kolom secara eksplisit
df = pd.read_csv('data/damiu.csv', header=None, names=['Tanggal', '_day_num_from_csv', 'Galon Terjual'])

# Mengurai dan mengurutkan tanggal
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%Y-%m-%d', errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df = df.sort_values(by='Tanggal')

# Mengisi tanggal yang hilang untuk memastikan data runtun waktu kontinu
all_dates = pd.date_range(start=df['Tanggal'].min(), end=df['Tanggal'].max(), freq='D')
df = df.set_index('Tanggal').reindex(all_dates).rename_axis('Tanggal').reset_index()

# --- 2. Rekayasa Fitur (Feature Engineering) ---
print("Melakukan rekayasa fitur...")
# Fitur Hari Libur Nasional
indo_holidays = holidays.Indonesia()
df['is_holiday'] = df['Tanggal'].apply(lambda date: 1 if date in indo_holidays else 0).astype(int)

# Fitur berbasis waktu dasar
df['hari_ke'] = (df['Tanggal'] - df['Tanggal'].min()).dt.days
df['hari_dalam_minggu'] = df['Tanggal'].apply(lambda x: x.isoweekday()) # Senin=1, Minggu=7
df['bulan'] = df['Tanggal'].dt.month
df['minggu_ke'] = df['Tanggal'].dt.isocalendar().week.astype(int)
df['tahun'] = df['Tanggal'].dt.year
df['is_weekend'] = df['hari_dalam_minggu'].isin([6, 7]).astype(int)

# Fitur Awal/Akhir Bulan
df['is_awal_bulan'] = (df['Tanggal'].dt.day <= 3).astype(int)
df['is_akhir_bulan'] = (df['Tanggal'].dt.day >= 28).astype(int)

# Fitur Interaksi
df['hari_minggu_x_holiday'] = df['hari_dalam_minggu'] * df['is_holiday']

# Fitur Musiman menggunakan Fourier Terms
P_year = 365.25
df['sin_tahun_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_year)
df['cos_tahun_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_year)
df['sin_tahun_2'] = np.sin(2 * np.pi * 2 * df['hari_ke'] / P_year)
df['cos_tahun_2'] = np.cos(2 * np.pi * 2 * df['hari_ke'] / P_year)

P_week = 7
df['sin_minggu_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_week)
df['cos_minggu_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_week)

# Memastikan kolom target adalah numerik
df['Galon Terjual'] = pd.to_numeric(df['Galon Terjual'], errors='coerce')

# Penanganan Outlier Sederhana (Winsorization)
lower_bound = df['Galon Terjual'].quantile(0.05)
upper_bound = df['Galon Terjual'].quantile(0.95)
df['Galon Terjual_winsorized'] = df['Galon Terjual'].clip(lower=lower_bound, upper=upper_bound)

# Menghitung nilai default untuk mengisi data yang hilang
static_default_avg_winsorized_train = df['Galon Terjual_winsorized'].mean()
static_default_std_winsorized_train = df['Galon Terjual_winsorized'].std()

# Fitur Lag dan Rolling Window
df['penjualan_kemarin'] = df['Galon Terjual_winsorized'].shift(1).fillna(static_default_avg_winsorized_train)
df['penjualan_2hari_lalu'] = df['Galon Terjual_winsorized'].shift(2).fillna(static_default_avg_winsorized_train)
df['rata2_3hari'] = df['Galon Terjual_winsorized'].rolling(window=3, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)
df['rata2_7hari'] = df['Galon Terjual_winsorized'].rolling(window=7, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)
df['rata2_14hari'] = df['Galon Terjual_winsorized'].rolling(window=14, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized_train)
df['std_7hari'] = df['Galon Terjual_winsorized'].rolling(window=7, min_periods=1).std().fillna(static_default_std_winsorized_train)
df['delta_penjualan'] = df['Galon Terjual_winsorized'].diff().fillna(0)

# Membersihkan baris dengan nilai NaN pada kolom target
df.dropna(subset=['Galon Terjual'], inplace=True)

# Daftar nama kolom fitur
feature_column_names = [
    'hari_ke', 'hari_dalam_minggu', 'bulan', 'minggu_ke', 'tahun', 'is_weekend',
    'is_awal_bulan', 'is_akhir_bulan', 'is_holiday', 'hari_minggu_x_holiday',
    'penjualan_kemarin', 'penjualan_2hari_lalu', 'rata2_3hari', 'rata2_7hari',
    'rata2_14hari', 'std_7hari', 'delta_penjualan', 'sin_tahun_1', 'cos_tahun_1',
    'sin_tahun_2', 'cos_tahun_2', 'sin_minggu_1', 'cos_minggu_1'
]

# Fitur terkait hari dengan penjualan nol
df['is_zero_sale_day'] = (df['Galon Terjual'] == 0).astype(int)
zero_sale_dates = df[df['is_zero_sale_day'] == 1]['Tanggal']
if not zero_sale_dates.empty:
    zero_sale_series = pd.Series(zero_sale_dates.values, index=zero_sale_dates)
    last_zero_sale_date = zero_sale_series.reindex(df['Tanggal'], method='ffill')
    df['days_since_last_zero_sale'] = (df['Tanggal'] - last_zero_sale_date).dt.days.fillna(df['hari_ke'])
else:
    df['days_since_last_zero_sale'] = df['hari_ke']
df['is_day_after_zero_sale'] = df['is_zero_sale_day'].shift(1).fillna(0).astype(int)

# Menambahkan fitur baru ke daftar kolom
feature_column_names.extend(['is_zero_sale_day', 'days_since_last_zero_sale', 'is_day_after_zero_sale'])

target_column_name = 'Galon Terjual'

X = df[feature_column_names]
y = df[target_column_name]

# --- 3. Pembagian Data ---
print("Membagi data menjadi set pelatihan dan pengujian...")
train_size_ratio = 0.8
train_index_end = int(len(df) * train_size_ratio)
X_train, X_test = X.iloc[:train_index_end], X.iloc[train_index_end:]
y_train, y_test = y.iloc[:train_index_end], y.iloc[train_index_end:]

# Setup validasi silang untuk data runtun waktu
tscv = TimeSeriesSplit(n_splits=5)

# --- 4. Pelatihan dan Penyetelan Model XGBoost ---
# Parameter grid untuk RandomizedSearchCV
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 500, 800, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'reg_alpha': [0.5, 1, 2, 5],
    'reg_lambda': [0.5, 1, 2, 5]
}

# Inisialisasi model XGBoost dasar
xgb_model = XGBRegressor(random_state=42)

print("\nMelakukan tuning hyperparameter untuk XGBoost dengan RandomizedSearchCV...")
xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=100, # Jumlah iterasi pencarian acak
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
xgb_random_search.fit(X_train, y_train)
print(f"Parameter terbaik dari RandomizedSearchCV: {xgb_random_search.best_params_}")

# Parameter grid untuk GridSearchCV (fine-tuning)
best_params_xgb_random = xgb_random_search.best_params_
param_grid_xgb = {
    'n_estimators': sorted(list(set([best_params_xgb_random['n_estimators'], max(50, best_params_xgb_random['n_estimators'] - 25), best_params_xgb_random['n_estimators'] + 25]))),
    'learning_rate': sorted(list(set([best_params_xgb_random['learning_rate'], max(0.001, best_params_xgb_random['learning_rate'] - 0.01), best_params_xgb_random['learning_rate'] + 0.01]))),
    'max_depth': sorted(list(set([best_params_xgb_random['max_depth'], max(1, best_params_xgb_random['max_depth'] - 1), best_params_xgb_random['max_depth'] + 1]))),
    'subsample': [best_params_xgb_random['subsample']],
    'colsample_bytree': [best_params_xgb_random['colsample_bytree']],
    'gamma': [best_params_xgb_random['gamma']],
    'reg_alpha': [best_params_xgb_random['reg_alpha']],
    'reg_lambda': [best_params_xgb_random['reg_lambda']]
}

print("\nMelakukan fine-tuning dengan GridSearchCV untuk XGBoost...")
xgb_grid_search = GridSearchCV(
    estimator=xgb_random_search.best_estimator_,
    param_grid=param_grid_xgb,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model_tuned = xgb_grid_search.best_estimator_
print(f"Parameter terbaik dari GridSearchCV: {xgb_grid_search.best_params_}")

# --- 5. Evaluasi Model ---
print("\nEvaluasi Model XGBoost (setelah GridSearchCV) pada Data Uji:")
y_pred_xgb_tuned = best_xgb_model_tuned.predict(X_test)
mse_xgb_tuned = mean_squared_error(y_test, y_pred_xgb_tuned)
mae_xgb_tuned = mean_absolute_error(y_test, y_pred_xgb_tuned)
rmse_xgb_tuned = np.sqrt(mse_xgb_tuned)
r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)
print(f"Mean Squared Error (MSE): {mse_xgb_tuned:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb_tuned:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb_tuned:.2f}")
print(f"R-squared (R2): {r2_xgb_tuned:.2f}")

# --- 6. Penyimpanan Model dan Metadata ---
print("\nMenyimpan model dan metadata...")
# Simpan model XGBoost yang sudah di-tune
joblib.dump(best_xgb_model_tuned, 'xgboost_gallon_model.joblib')

# Simpan metadata model
model_metadata = {
    'features': feature_column_names,
    'training_start_date': df['Tanggal'].min().isoformat(),
    'training_end_date': df['Tanggal'].max().isoformat(),
    'model_type': 'XGBRegressor',
    'xgb_rmse': rmse_xgb_tuned,
    'xgb_mae': mae_xgb_tuned,
    'xgb_r2': r2_xgb_tuned,
    'xgb_best_params_randomized': xgb_random_search.best_params_,
    'xgb_best_params_grid': xgb_grid_search.best_params_
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

# --- 7. Menampilkan Feature Importance ---
print("\nFeature Importance dari Model XGBoost:")
for name, importance in zip(feature_column_names, best_xgb_model_tuned.feature_importances_):
    print(f"  - {name:25s}: {importance:.4f}")


print("Model dan metadata berhasil disimpan.")

# --- 7. Analisis SHAP ---
try:
    import shap
    print("\nMelakukan analisis SHAP untuk XGBoost...")

    # Gunakan TreeExplainer untuk model XGBoost
    explainer_xgb = shap.TreeExplainer(best_xgb_model_tuned)
    shap_values_xgb = explainer_xgb.shap_values(X_test)

    # Simpan plot SHAP Summary
    print("Menyimpan plot SHAP Summary...")
    shap.summary_plot(shap_values_xgb, X_test, feature_names=feature_column_names, show=False)
    plt.savefig("shap_summary_xgb.png", bbox_inches='tight')
    plt.close()

    print("Plot SHAP telah disimpan sebagai shap_summary_xgb.png")

except ImportError:
    print("\nPerpustakaan SHAP tidak terinstal. Melewatkan analisis SHAP. Instal dengan: pip install shap")
except Exception as e:
    print(f"\nTerjadi kesalahan saat analisis SHAP: {e}")
