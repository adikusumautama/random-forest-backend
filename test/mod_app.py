# mod_app.py - Disesuaikan untuk model dengan rekayasa fitur tingkat lanjut

from flask import Flask, request, jsonify
import joblib, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import warnings, traceback

warnings.filterwarnings("ignore", category=FutureWarning)

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
    model = joblib.load("xgboost_gallon_model.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Mengambil semua informasi yang diperlukan dari metadata
    FEATURES = metadata["features"]
    TRANSFORMATION_STATS = metadata.get("transformation_stats", {})
    DAY_OF_WEEK_STATS = metadata.get("day_of_week_stats", {})
    print("* Model XGBoost dan metadata berhasil dimuat.")
except Exception as e:
    model, FEATURES, TRANSFORMATION_STATS, DAY_OF_WEEK_STATS = None, [], {}, {}
    print(f"* Gagal memuat model atau metadata: {e}")

def get_sales_history():
    """Mengambil data historis dari Firestore dan menambahkan kolom 'Hari' dan 'hari_ke'."""
    if db is None:
        raise Exception("Firestore belum terhubung.")
    
    parsed = []
    query = db.collection("daily_sales").order_by("__name__")
    
    for doc in query.stream():
        d = doc.to_dict()
        try:
            tgl = pd.to_datetime(doc.id, format="%Y-%m-%d")
            qty = float(d.get("quantity", np.nan))
            
            # Menambahkan 'Hari' (1=Senin, 7=Minggu) dan 'hari_ke'
            parsed.append({
                "Tanggal": tgl, 
                "Galon Terjual": qty, 
                "Hari": tgl.isoweekday() 
            })
        except (ValueError, TypeError):
            print(f"* WARNING: Dokumen {doc.id} dilewati karena format tidak valid.")
    
    if not parsed:
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual", "Hari", "hari_ke"])
        
    df = pd.DataFrame(parsed)
    df['hari_ke'] = range(1, len(df) + 1)
    return df

def build_features(df_hist, target_hari_ke, target_date, stats, day_stats_map):
    """
    Membangun fitur untuk prediksi, mereplikasi proses dari skrip pelatihan.
    """
    # Salin data historis untuk menghindari modifikasi data asli
    df = df_hist.copy()
    
    # Ambil nilai ambang batas dari metadata
    NOISE_THRESHOLD = stats.get('noise_threshold', 10)
    SPIKE_THRESHOLD = stats.get('spike_threshold', 50)
    URGENT_DELIVERY_VALUE = stats.get('urgent_delivery_value', 40)
    
    # --- Terapkan Aturan Bisnis yang Sama Seperti Saat Pelatihan ---
    df['Galon_Terjual_Adjusted'] = df['Galon Terjual'].copy()
    df.loc[df['Galon Terjual'] < NOISE_THRESHOLD, 'Galon_Terjual_Adjusted'] = 0
    
    for index, row in df.iterrows():
        if row['Galon Terjual'] > SPIKE_THRESHOLD:
            excess = row['Galon Terjual'] - SPIKE_THRESHOLD
            df.at[index, 'Galon_Terjual_Adjusted'] = SPIKE_THRESHOLD
            prev_day_index = index - 1
            while excess > 0 and prev_day_index >= 0:
                fill_amount = SPIKE_THRESHOLD - df.at[prev_day_index, 'Galon_Terjual_Adjusted']
                if fill_amount > 0:
                    add_amount = min(excess, fill_amount)
                    df.at[prev_day_index, 'Galon_Terjual_Adjusted'] += add_amount
                    excess -= add_amount
                prev_day_index -= 1

    # Buat baris baru untuk hari yang akan diprediksi
    new_row_dict = {
        "Tanggal": target_date, 
        "Galon Terjual": np.nan, 
        "Hari": target_date.isoweekday(), 
        "hari_ke": target_hari_ke
    }
    new_row = pd.DataFrame([new_row_dict])
    df = pd.concat([df, new_row], ignore_index=True)

    # --- Buat Fitur Berdasarkan Data Asli dan Data yang Disesuaikan ---
    df['is_noise_day'] = (df['Galon Terjual'] < NOISE_THRESHOLD).astype(int)
    df['is_spike_day'] = (df['Galon Terjual'] > SPIKE_THRESHOLD).astype(int)
    df['is_urgent_delivery'] = (df['Galon Terjual'] == URGENT_DELIVERY_VALUE).astype(int)
    
    df['penjualan_lebih_25'] = (df['Galon_Terjual_Adjusted'] > 25).astype(int)
    df['penjualan_kurang_25'] = (df['Galon_Terjual_Adjusted'] < 25).astype(int)
    df['penjualan_lebih_35'] = (df['Galon_Terjual_Adjusted'] > 35).astype(int)
    df['penjualan_kurang_35'] = (df['Galon_Terjual_Adjusted'] < 35).astype(int)
    df['penjualan_lebih_20'] = (df['Galon_Terjual_Adjusted'] > 20).astype(int)
    df['penjualan_kurang_20'] = (df['Galon_Terjual_Adjusted'] < 20).astype(int)

    # --- Buat Fitur Statistik dan Runtun Waktu ---
    static_avg = stats.get('static_default_avg', df['Galon_Terjual_Adjusted'].mean())
    static_std = stats.get('static_default_std', df['Galon_Terjual_Adjusted'].std())

    df['rata2_per_hari'] = df['Hari'].astype(str).map(lambda x: day_stats_map.get(x, {}).get('mean', static_avg))
    df['std_per_hari'] = df['Hari'].astype(str).map(lambda x: day_stats_map.get(x, {}).get('std', static_std))
    
    df['sin_minggu'] = np.sin(2 * np.pi * df['Hari'] / 7)
    df['cos_minggu'] = np.cos(2 * np.pi * df['Hari'] / 7)
    
    df['penjualan_kemarin'] = df['Galon_Terjual_Adjusted'].shift(1).fillna(static_avg)
    df['penjualan_2hari_lalu'] = df['Galon_Terjual_Adjusted'].shift(2).fillna(static_avg)
    df['rata2_3hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(3, min_periods=1).mean().fillna(static_avg)
    df['rata2_7hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(7, min_periods=1).mean().fillna(static_avg)
    df['rata2_14hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(14, min_periods=1).mean().fillna(static_avg)
    df['std_7hari'] = df['Galon_Terjual_Adjusted'].shift(1).rolling(7, min_periods=1).std().fillna(static_std)
    df['delta_penjualan'] = df['Galon_Terjual_Adjusted'].shift(1).diff().fillna(0)
    
    # Ambil baris terakhir yang berisi fitur untuk hari prediksi
    row = df[df["hari_ke"] == target_hari_ke]
    if row.empty:
        raise Exception("Baris prediksi tidak dapat dibuat.")
    return row[FEATURES]


@app.route("/predict", methods=["POST"])
def predict():
    if None in (model, TRANSFORMATION_STATS):
        return jsonify({"error": "Model atau metadata belum siap."}), 500
    try:
        df_hist = get_sales_history()
        if df_hist.empty:
            return jsonify({"error": "Tidak ada data historis di Firestore."}), 400
        
        last_row = df_hist.iloc[-1]
        last_date = last_row['Tanggal']
        last_hari_ke = last_row['hari_ke']

        target_date = last_date + timedelta(days=1)
        target_hari_ke = last_hari_ke + 1

        feature_df = build_features(df_hist, target_hari_ke, target_date, TRANSFORMATION_STATS, DAY_OF_WEEK_STATS)
        raw_pred = model.predict(feature_df)[0]

        # Logika bisnis pasca-prediksi (dipertahankan sesuai permintaan)
        min_ratio = 0.60
        min_allowed = df_hist["Galon Terjual"].iloc[-1] * min_ratio
        final_pred = max(0, round(max(raw_pred, min_allowed), 2))
        
        # --- Blok Logging untuk Debugging ---
        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print(f"(Prediksi untuk hari_ke: {target_hari_ke})")
        print("\nFitur Input Model:")
        if not feature_df.empty:
            for col in feature_df.columns:
                print(f"  - {col:25s}: {feature_df.iloc[0][col]:.4f}")
        print(f"\nPrediksi Mentah dari Model: {raw_pred:.2f}")
        print(f"Prediksi Akhir (setelah penyesuaian): {final_pred:.2f} galon")
        print("==============================")
        
        return jsonify({
            "last_known_data_date": last_date.strftime("%Y-%m-%d"),
            "prediction_for_next_day": {
                "Tanggal": target_date.strftime("%Y-%m-%d"),
                "Prediksi Galon": float(final_pred)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
