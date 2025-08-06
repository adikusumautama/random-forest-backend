# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON - BACKEND API
# Script: mod_app.py (Versi Disesuaikan dengan Interpolasi Outlier)
# Deskripsi: Flask API yang efisien untuk prediksi, menggunakan logika
#            rekayasa fitur yang 100% konsisten dengan skrip training.
# =============================================================================

from flask import Flask, jsonify, request
import joblib, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import warnings, traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Pengaturan Firebase ---
try:
    # Ganti dengan path kredensial Anda yang sebenarnya
    cred = credentials.Certificate("firebase/damiu-app-ad9f7-firebase-adminsdk-fbsvc-9be6b99017.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("* Terhubung ke Firestore.")
except Exception as e:
    db = None
    print(f"* Gagal koneksi Firebase: {e}")

# Memuat model dan metadata
try:
    MODEL_PATH = "xgboost_gallon_model.joblib"
    METADATA_PATH = "model_metadata.json"

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    FEATURES = metadata["features_used"]
    BUSINESS_RULES = metadata["business_rules"]
    # Mengambil threshold dari metadata yang baru
    LOWER_OUTLIER_THRESHOLD = BUSINESS_RULES['lower_outlier_threshold']
    UPPER_OUTLIER_THRESHOLD = BUSINESS_RULES['upper_outlier_threshold']
    LOW_SALE_FEATURE_THRESHOLD = 21
    SPIKE_FEATURE_THRESHOLD = 49
    
    print(f"* Model '{MODEL_PATH}' dan metadata berhasil dimuat.")
except Exception as e:
    model, FEATURES, BUSINESS_RULES, LOWER_OUTLIER_THRESHOLD, UPPER_OUTLIER_THRESHOLD = None, [], {}, 0, 0
    print(f"* Gagal memuat model atau metadata: {e}")

def get_sales_history_from_firestore(days_of_context=30):
    """
    Mengambil data historis dari Firestore secara efisien dan TERURUT.
    """
    NAMA_KOLEKSI = "daily_sales"
    NAMA_FIELD_JUMLAH = "quantity"
    NAMA_FIELD_TANGGAL = "date"

    if db is None:
        raise Exception("Firestore belum terhubung.")
    
    start_date = datetime.now() - timedelta(days=days_of_context)
    
    sales_data = []
    
    query = db.collection(NAMA_KOLEKSI) \
              .where(NAMA_FIELD_TANGGAL, '>=', start_date) \
              .order_by(NAMA_FIELD_TANGGAL) \
              .stream()
    
    count = 0
    for doc in query:
        count += 1
        try:
            data = doc.to_dict()
            sale_date = data.get(NAMA_FIELD_TANGGAL)
            if not isinstance(sale_date, datetime):
                sale_date = pd.to_datetime(sale_date)

            quantity = float(data.get(NAMA_FIELD_JUMLAH, np.nan))
            sales_data.append({"tanggal": sale_date, "Galon_Terjual": quantity})
        except Exception as e:
            print(f"* WARNING: Dokumen {doc.id} dilewati karena error: {e}")
    
    print(f"* Query selesai. Ditemukan {count} dokumen.")
    if not sales_data:
        return pd.DataFrame(columns=["tanggal", "Galon_Terjual"]).set_index('tanggal')
        
    df = pd.DataFrame(sales_data).sort_values(by="tanggal").set_index('tanggal')
    df.index = df.index.normalize()
    return df

def build_features_for_prediction(historical_df, target_date):

    if historical_df.empty:
        raise ValueError("Data historis tidak boleh kosong untuk membangun fitur.")

    df = historical_df.copy()

    # Pastikan rentang tanggal lengkap hingga tanggal target
    full_date_range = pd.date_range(start=df.index.min(), end=target_date)
    df = df.reindex(full_date_range)
    
    df['Galon_Terjual_Cleaned'] = df['Galon_Terjual'].copy().astype(float)
    df.loc[df['Galon_Terjual'] > UPPER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan
    df.loc[df['Galon_Terjual'] < LOWER_OUTLIER_THRESHOLD, 'Galon_Terjual_Cleaned'] = np.nan
    
    # Isi nilai NaN menggunakan interpolasi linear
    df['Galon_Terjual_Cleaned'] = df['Galon_Terjual_Cleaned'].interpolate(method='linear')

    # Gunakan backfill dan forward-fill untuk menangani NaN jika ada di awal/akhir data
    df['Galon_Terjual_Cleaned'].fillna(method='bfill', inplace=True)
    df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)

    target_col = 'Galon_Terjual_Cleaned'
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    shifted_target = df[target_col].shift(1)
    for window in [3, 7, 14, 21]:
        # Rolling
        df[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()
    
    # Fitur Lag
    df['lag_diff_1'] = shifted_target.diff(1)
    df['lag_diff_7'] = shifted_target.diff(7)

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
    spike_days = df['is_spike'].copy()
    spike_days[spike_days == 0] = np.nan
    spike_days = spike_days.reset_index(name='is_spike').rename(columns={'index': 'tanggal'})
    spike_days['day_num'] = range(len(spike_days))
    spike_days.set_index('tanggal', inplace=True)
    spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
    spike_days['day_num'].fillna(method='ffill', inplace=True)
    df['days_since_last_spike'] = (range(len(df)) - spike_days['day_num']).fillna(0)

    # Fitur 'days since last low sale'
    df['is_low_sale'] = (df[target_col] < LOW_SALE_FEATURE_THRESHOLD).astype(int)
    low_sale_days = df['is_low_sale'].copy()
    low_sale_days[low_sale_days == 0] = np.nan
    low_sale_days = low_sale_days.reset_index(name='is_low_sale').rename(columns={'index': 'tanggal'})
    low_sale_days['day_num'] = range(len(low_sale_days))
    low_sale_days.set_index('tanggal', inplace=True)
    low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
    low_sale_days['day_num'].fillna(method='ffill', inplace=True)
    df['days_since_last_low'] = (range(len(df)) - low_sale_days['day_num']).fillna(0)

    df.fillna(0, inplace=True)
    
    return df.tail(1)[FEATURES]

# Cek ukuran request (buat lihat seberapa besar data yang dikirim)
@app.before_request
def log_request_info():
    print(f"Request size: {request.content_length} bytes")



@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not FEATURES:
        return jsonify({"error": "Model atau metadata belum siap."}), 500
    try:
        df_hist_context = get_sales_history_from_firestore(days_of_context=60)
        
        if df_hist_context.empty:
            return jsonify({"error": "Tidak ada data historis yang cukup dalam 60 hari terakhir untuk membuat prediksi."}), 400
        
        last_date = df_hist_context.index.max()
        target_date = last_date + timedelta(days=1)
        
        feature_df = build_features_for_prediction(df_hist_context, target_date)
        
        raw_pred = model.predict(feature_df)[0]
        
        final_pred = round(float(raw_pred.clip(min=LOWER_OUTLIER_THRESHOLD)))
        
        print("\nFitur Input Model:")
        if not feature_df.empty:
            for col, val in feature_df.iloc[0].items():
                print(f"  - {col:25s}: {val:.4f}")
        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print(f"\nPrediksi Mentah dari Model: {raw_pred:.2f}")
        print(f"Prediksi Akhir (setelah pembulatan & kliping): {final_pred} galon")
        
                
        # Perlihatkan nilai kosong yang terisi
        print("\nNilai kosong yang terisi:")
        for col in feature_df.columns:
            if feature_df[col].isnull().any():
                print(f"  - {col:25s}: {feature_df[col].isnull().sum()} nilai kosong terisi")
        
        return jsonify({
            "last_known_data_date": last_date.strftime("%Y-%m-%d"),
            "prediction_for_next_day": {
                "Tanggal": target_date.strftime("%Y-%m-%d"),
                "Prediksi Galon": final_pred
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
