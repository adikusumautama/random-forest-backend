from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import holidays
import warnings

# Mengabaikan peringatan FutureWarning dari library pandas/numpy
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# --- Pengaturan Firebase ---
try:
    # Ganti dengan path ke file kunci layanan Firebase Anda
    cred = credentials.Certificate("firebase/damiu-app-ad9f7-firebase-adminsdk-fbsvc-9be6b99017.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("* Terhubung ke Firestore.")
except Exception as e:
    db = None
    print(f"* Gagal koneksi Firebase: {e}")

# --- Memuat Model dan Metadata ---
try:
    # Hanya memuat model XGBoost
    model = joblib.load("xgboost_gallon_model.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    features = metadata["features"]
    # Memuat statistik transformasi yang disimpan saat pelatihan
    TRANSFORMATION_STATS = metadata.get("transformation_stats")
    training_start_date = datetime.fromisoformat(metadata["training_start_date"]).replace(tzinfo=None)
    print("* Model XGBoost dan metadata berhasil dimuat.")
except Exception as e:
    model = None
    features = []
    TRANSFORMATION_STATS = None
    training_start_date = None
    print(f"* Gagal memuat model atau metadata: {e}")

# --- Fungsi untuk Mengambil Data dari Firestore ---
def get_sales_history():
    if db is None:
        raise Exception("Firestore belum terhubung.")
    
    docs = db.collection("daily_sales").stream()
    parsed_data = []
    for doc in docs:
        d = doc.to_dict()
        doc_id = doc.id
        if "quantity" in d:
            try:
                tanggal = pd.to_datetime(doc_id, format='%Y-%m-%d')
                jumlah = float(d["quantity"])
                parsed_data.append({"Tanggal": tanggal, "Galon Terjual": jumlah})
            except (ValueError, TypeError):
                print(f"* WARNING: Melewati dokumen {doc_id} karena format tidak valid.")
    
    if not parsed_data:
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])
    
    df = pd.DataFrame(parsed_data)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values(by="Tanggal").reset_index(drop=True)
    return df

# --- Fungsi untuk Mengambil Data dari CSV ---
def get_sales_history_from_csv(csv_path="data/damiu.csv"):
    try:
        df = pd.read_csv(csv_path, header=None, names=['Tanggal', '_day_num_from_csv', 'Galon Terjual'])
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df.dropna(subset=['Tanggal'], inplace=True)
        df['Galon Terjual'] = pd.to_numeric(df['Galon Terjual'], errors='coerce').astype(float)
        df.dropna(subset=['Galon Terjual'], inplace=True)
        df = df.sort_values(by="Tanggal").reset_index(drop=True)
        return df[["Tanggal", "Galon Terjual"]]
    except Exception as e:
        print(f"* Gagal membaca atau memproses CSV {csv_path}: {e}")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])

# --- Fungsi untuk Membangun Fitur untuk Prediksi ---
def build_features(df_full, predict_date, transform_stats):
    # Ekstrak statistik yang sudah dihitung saat pelatihan
    lower_bound = transform_stats['winsor_lower_bound']
    upper_bound = transform_stats['winsor_upper_bound']
    static_default_avg = transform_stats['static_default_avg_winsorized']
    static_default_std = transform_stats['static_default_std_winsorized']

    # Terapkan winsorisasi pada data historis menggunakan batas yang sama saat pelatihan
    df_full['Galon Terjual_winsorized'] = df_full['Galon Terjual'].clip(lower=lower_bound, upper=upper_bound)

    df = pd.concat([df_full, pd.DataFrame([{"Tanggal": predict_date, "Galon Terjual": np.nan}])], ignore_index=True)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values(by="Tanggal").reset_index(drop=True)

    # Rekayasa fitur (harus sama persis dengan skrip pelatihan, kecuali winsorisasi)
    indo_holidays = holidays.Indonesia()
    df["hari_ke"] = (df["Tanggal"] - training_start_date).dt.days
    df["hari_dalam_minggu"] = df["Tanggal"].apply(lambda x: x.isoweekday())
    df["bulan"] = df["Tanggal"].dt.month
    df["minggu_ke"] = df["Tanggal"].dt.isocalendar().week.astype(int)
    df["tahun"] = df["Tanggal"].dt.year
    df["is_weekend"] = df["hari_dalam_minggu"].isin([6, 7]).astype(int)
    df["is_awal_bulan"] = (df["Tanggal"].dt.day <= 3).astype(int)
    df["is_akhir_bulan"] = (df["Tanggal"].dt.day >= 28).astype(int)
    df["is_holiday"] = df["Tanggal"].apply(lambda date: 1 if date in indo_holidays else 0).astype(int)
    df['hari_minggu_x_holiday'] = df['hari_dalam_minggu'] * df['is_holiday']

    P_year = 365.25
    df['sin_tahun_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_year)
    df['cos_tahun_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_year)
    df['sin_tahun_2'] = np.sin(2 * np.pi * 2 * df['hari_ke'] / P_year)
    df['cos_tahun_2'] = np.cos(2 * np.pi * 2 * df['hari_ke'] / P_year)
    P_week = 7
    df['sin_minggu_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_week)
    df['cos_minggu_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_week)

    # Buat fitur lag/rolling dari kolom yang sudah di-winsorize
    df["penjualan_kemarin"] = df["Galon Terjual_winsorized"].shift(1).fillna(static_default_avg)
    df["penjualan_2hari_lalu"] = df["Galon Terjual_winsorized"].shift(2).fillna(static_default_avg)
    df["rata2_3hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(window=3, min_periods=1).mean().fillna(static_default_avg)
    df["rata2_7hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(window=7, min_periods=1).mean().fillna(static_default_avg)
    df["rata2_14hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(window=14, min_periods=1).mean().fillna(static_default_avg)
    df["std_7hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(window=7, min_periods=1).std().fillna(static_default_std)
    df["delta_penjualan"] = df["Galon Terjual_winsorized"].diff().fillna(0)

    df['is_zero_sale_day'] = (df['Galon Terjual'] == 0).astype(int)
    zero_sale_dates = df[df['is_zero_sale_day'] == 1]['Tanggal']
    if not zero_sale_dates.empty:
        zero_sale_series = pd.Series(zero_sale_dates.values, index=zero_sale_dates)
        last_zero_sale_date = zero_sale_series.reindex(df["Tanggal"], method="ffill")
        df["days_since_last_zero_sale"] = (df["Tanggal"] - last_zero_sale_date).dt.days.fillna(df["hari_ke"])
    else:
        df["days_since_last_zero_sale"] = df["hari_ke"]
    df["is_day_after_zero_sale"] = df["is_zero_sale_day"].shift(1).fillna(0).astype(int)

    row = df[df["Tanggal"] == predict_date]
    if row.empty:
        raise Exception("Baris prediksi tidak dapat dibuat.")
    
    return row[features]

# --- Endpoint Prediksi ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or training_start_date is None or TRANSFORMATION_STATS is None:
        return jsonify({"error": "Model belum siap atau metadata/statistik transformasi tidak lengkap."}), 500

    try:
        # Langsung ambil data historis dari Firestore (karena hanya ini yang dipakai)
        df_hist_initial = get_sales_history()
        if df_hist_initial.empty:
            return jsonify({"error": "Tidak ada data historis di Firestore."}), 400
        print("* Menggunakan data historis dari Firestore.")

        last_historical_date = df_hist_initial["Tanggal"].max()
        target_date = last_historical_date + timedelta(days=1)

        feature_df = build_features(df_hist_initial, target_date, TRANSFORMATION_STATS)

        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print("\nFitur Input Model:")
        for col in feature_df.columns:
            print(f"  - {col:25s}: {feature_df.iloc[0][col]:.4f}")

        # Lakukan prediksi
        prediction = model.predict(feature_df)[0]

        # Aturan pasca-pemrosesan
        max_daily_decrease_ratio = 0.60
        previous_day_sale = df_hist_initial["Galon Terjual"].iloc[-1]
        minimum_allowed_prediction = previous_day_sale * max_daily_decrease_ratio

        # Penyesuaian prediksi
        pred_adjusted = max(prediction, minimum_allowed_prediction)
        pred_final = max(0, round(pred_adjusted, 2))

        print(f"\nPrediksi dari Model XGBoost: {prediction:.2f}")
        print(f"Prediksi Akhir (setelah penyesuaian): {pred_final:.2f} galon")
        print("==============================")

        prediction_output = {
            "Tanggal": target_date.strftime("%Y-%m-%d"),
            "Prediksi Galon": float(pred_final)
        }

        return jsonify({
            "last_known_data_date": last_historical_date.strftime("%Y-%m-%d"),
            "prediction_for_next_day": prediction_output
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan di server: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)