# =============================================================================
# PROYEK PREDIKSI PENJUALAN GALON - BACKEND API
# Script: mod_app.py
# Deskripsi: Flask API untuk memberikan prediksi penjualan galon untuk
#            hari berikutnya berdasarkan data historis dari Firestore.
# =============================================================================

from flask import Flask, request, jsonify
import joblib, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import warnings, traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Pengaturan Firebase ---
try:
    # Ganti dengan path ke file kredensial Firebase Anda
    cred = credentials.Certificate("firebase/damiu-app-ad9f7-firebase-adminsdk-fbsvc-9be6b99017.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("* Terhubung ke Firestore.")
except Exception as e:
    db = None
    print(f"* Gagal koneksi Firebase: {e}")

# --- Memuat Model & Metadata ---
try:
    MODEL_PATH = "xgboost_gallon_model.joblib"
    METADATA_PATH = "model_metadata.json"
    
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    # Mengambil semua informasi yang diperlukan dari metadata
    FEATURES = metadata["features_used"]
    BUSINESS_RULES = metadata["business_rules"]
    LOWER_BOUND = BUSINESS_RULES['lower_bound']
    UPPER_BOUND = BUSINESS_RULES['upper_bound']
    
    print(f"* Model '{MODEL_PATH}' dan metadata berhasil dimuat.")
except Exception as e:
    model, FEATURES, BUSINESS_RULES, LOWER_BOUND, UPPER_BOUND = None, [], {}, 0, 0
    print(f"* Gagal memuat model atau metadata: {e}")

def get_sales_history_from_firestore():
    """
    Mengambil data historis dari Firestore, mengurutkan berdasarkan tanggal,
    dan mengembalikannya sebagai DataFrame yang diindeks oleh tanggal.
    """
    if db is None:
        raise Exception("Firestore belum terhubung.")
    
    sales_data = []
    query = db.collection("daily_sales").stream()
    
    for doc in query:
        try:
            sale_date = pd.to_datetime(doc.id, format="%Y-%m-%d")
            data = doc.to_dict()
            quantity = float(data.get("quantity", np.nan))
            sales_data.append({"tanggal": sale_date, "Galon_Terjual": quantity})
        except (ValueError, TypeError):
            print(f"* WARNING: Dokumen {doc.id} dilewati karena format tidak valid.")
    
    if not sales_data:
        return pd.DataFrame(columns=["tanggal", "Galon_Terjual"]).set_index('tanggal')
        
    df = pd.DataFrame(sales_data).sort_values(by="tanggal").set_index('tanggal')
    return df

def build_features_for_prediction(historical_df, target_date):
    """
    Membangun fitur untuk satu tanggal target, menggunakan data historis sebagai konteks.
    Logika ini HARUS MEREPLIKASI mod_train_model.py dengan TEPAT.
    """
    new_row = pd.DataFrame([{'Galon_Terjual': np.nan}], index=[target_date])
    df = pd.concat([historical_df, new_row])
    
    # --- 1. Pembersihan Data ---
    df['Galon_Terjual_Cleaned'] = df['Galon_Terjual'].clip(lower=LOWER_BOUND, upper=UPPER_BOUND)
    df['Galon_Terjual_Cleaned'].fillna(method='ffill', inplace=True)
    df['Galon_Terjual_Cleaned'].fillna(LOWER_BOUND, inplace=True)
    
    target_col = 'Galon_Terjual_Cleaned'
    
    # --- 2. Rekayasa Fitur (sama persis seperti saat training) ---
    shifted_target = df[target_col].shift(1)
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = shifted_target.shift(lag)
    for window in [3, 7, 14, 21]:
        df[f'rolling_mean_{window}'] = shifted_target.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = shifted_target.rolling(window=window).std()

    df['lag_diff_1'] = shifted_target.diff(1)
    df['lag_diff_7'] = shifted_target.diff(7)
    df['hari_dalam_bulan'] = df.index.day
    df['hari_dalam_tahun'] = df.index.dayofyear
    df['minggu_dalam_tahun'] = df.index.isocalendar().week.astype(int)
    df['bulan'] = df.index.month
    df['hari_minggu'] = df.index.dayofweek
    df['is_weekend'] = (df['hari_minggu'] >= 5).astype(int)
    df['awal_bulan'] = df.index.is_month_start.astype(int)
    df['akhir_bulan'] = df.index.is_month_end.astype(int)

    # Fitur Siklus Lonjakan
    spike_threshold = 49
    df['is_spike'] = (df[target_col] > spike_threshold).astype(int)
    spike_days = df['is_spike'].copy()
    spike_days[spike_days == 0] = np.nan
    spike_days = spike_days.reset_index()
    spike_days['day_num'] = range(len(spike_days))
    spike_days.set_index('tanggal', inplace=True)
    spike_days['day_num'] = spike_days['day_num'] * spike_days['is_spike']
    spike_days['day_num'].fillna(method='ffill', inplace=True)
    df['days_since_last_spike'] = (range(len(df)) - spike_days['day_num']).fillna(0)
    
    # PENYESUAIAN: Fitur Siklus Penjualan Rendah
    low_threshold = 23
    df['is_low_sale'] = (df[target_col] < low_threshold).astype(int)
    low_sale_days = df['is_low_sale'].copy()
    low_sale_days[low_sale_days == 0] = np.nan
    low_sale_days = low_sale_days.reset_index()
    low_sale_days['day_num'] = range(len(low_sale_days))
    low_sale_days.set_index('tanggal', inplace=True)
    low_sale_days['day_num'] = low_sale_days['day_num'] * low_sale_days['is_low_sale']
    low_sale_days['day_num'].fillna(method='ffill', inplace=True)
    df['days_since_last_low'] = (range(len(df)) - low_sale_days['day_num']).fillna(0)
    
    df.fillna(0, inplace=True)
    
    # Ambil baris terakhir yang berisi fitur untuk hari prediksi
    return df.tail(1)[FEATURES]


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not FEATURES:
        return jsonify({"error": "Model atau metadata belum siap."}), 500
    try:
        # 1. Ambil seluruh riwayat penjualan dari Firestore
        df_hist = get_sales_history_from_firestore()
        if df_hist.empty:
            return jsonify({"error": "Tidak ada data historis di Firestore."}), 400
        
        # 2. Tentukan tanggal yang akan diprediksi (H+1 dari data terakhir)
        last_date = df_hist.index.max()
        target_date = last_date + timedelta(days=1)

        # 3. Bangun fitur untuk tanggal target menggunakan seluruh riwayat
        feature_df = build_features_for_prediction(df_hist, target_date)
        
        # 4. Lakukan prediksi
        raw_pred = model.predict(feature_df)[0]

        # 5. Bulatkan dan pastikan hasil sesuai aturan bisnis (clipping)
        final_pred = round(float(raw_pred), 2)
        
        # --- Blok Logging untuk Debugging ---
        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print("\nFitur Input Model:")
        if not feature_df.empty:
            for col, val in feature_df.iloc[0].items():
                print(f"  - {col:25s}: {val:.4f}")
        print(f"\nPrediksi Mentah dari Model: {raw_pred:.2f}")
        print(f"Prediksi Akhir (setelah pembulatan): {final_pred:.2f} galon")
        print("==============================")
        
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
