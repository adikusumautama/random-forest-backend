from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# --- Firebase Setup ---
try:
    cred = credentials.Certificate("firebase/damiu-app-ad9f7-firebase-adminsdk-fbsvc-19f7bfa236.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("* Terhubung ke Firestore.")
except Exception as e:
    db = None
    print(f"* Gagal koneksi Firebase: {e}")

# --- Load Model dan Metadata ---
try:
    model = joblib.load("random_forest_gallon_model.joblib")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    features = metadata["features"]
    training_start_date = datetime.fromisoformat(metadata["training_start_date"]).replace(tzinfo=None)
    print("* Model dan metadata berhasil dimuat.")
except Exception as e:
    model = None
    features = []
    training_start_date = None
    print(f"* Gagal load model atau metadata: {e}")

# --- Ambil Data Firestore ---
def get_sales_history():
    if db is None:
        raise Exception("Firestore belum terhubung.")

    docs = db.collection("daily_sales").stream()
    parsed_data = []
    for doc in docs:
        d = doc.to_dict()
        doc_id = doc.id

        if "date" in d and "quantity" in d:
            try:
                firestore_date = d["date"]
                if not isinstance(firestore_date, datetime):
                    tanggal = pd.to_datetime(firestore_date).tz_localize(None)
                else:
                    tanggal = firestore_date.replace(tzinfo=None)

                jumlah = float(d["quantity"])
                parsed_data.append({"Tanggal": tanggal, "Galon Terjual": jumlah})
            except Exception as e:
                print(f"* Gagal parsing data dari dokumen {doc_id}: {e}. Data: {d}")
        else:
            print(f"* Dokumen {doc_id} dilewati: field 'date' atau 'quantity' tidak ditemukan. Data: {d}")

    if not parsed_data:
        print("* Tidak ada data valid yang berhasil diparsing dari Firestore.")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])

    df = pd.DataFrame(parsed_data)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values(by="Tanggal").reset_index(drop=True)
    return df

# --- Bangun Fitur Ideal ---
def build_features(df_full, predict_date):
    df = df_full.copy()

    # Tambahkan baris prediksi
    df = pd.concat([
        df,
        pd.DataFrame([{
            "Tanggal": predict_date,
            "Galon Terjual": np.nan
        }])
    ], ignore_index=True)

    df = df.sort_values(by="Tanggal")

    # Hitung fallback untuk nilai default
    default_avg = df_full["Galon Terjual"].mean() if not df_full.empty else 0
    default_std = df_full["Galon Terjual"].std() if not df_full.empty else 0

    # Fitur waktu
    df["hari_ke"] = (df["Tanggal"] - training_start_date).dt.days
    # df["hari_dalam_minggu"] = df["Tanggal"].dt.dayofweek  # 0-6 untuk Senin-Minggu
    df["hari_dalam_minggu"] = df["Tanggal"].apply(lambda x: x.isoweekday())
    df["bulan"] = df["Tanggal"].dt.month
    df["minggu_ke"] = df["Tanggal"].dt.isocalendar().week
    df["tahun"] = df["Tanggal"].dt.year
    df["is_weekend"] = df["hari_dalam_minggu"].isin([5, 6]).astype(int)
    df["is_awal_bulan"] = (df["Tanggal"].dt.day <= 3).astype(int)
    df["is_akhir_bulan"] = (df["Tanggal"].dt.day >= 28).astype(int)

    # Lag & Rolling
    df["penjualan_kemarin"] = df["Galon Terjual"].shift(1).fillna(default_avg)
    df["penjualan_2hari_lalu"] = df["Galon Terjual"].shift(2).fillna(default_avg)
    df["rata2_3hari"] = df["Galon Terjual"].rolling(window=3).mean().fillna(method='backfill').fillna(default_avg)
    df["rata2_7hari"] = df["Galon Terjual"].rolling(window=7).mean().fillna(method='backfill').fillna(default_avg)
    df["rata2_14hari"] = df["Galon Terjual"].rolling(window=14).mean().fillna(method='backfill').fillna(default_avg)
    df["std_7hari"] = df["Galon Terjual"].rolling(window=7).std().fillna(default_std)
    df["delta_penjualan"] = df["Galon Terjual"].diff().fillna(0)

    # Ambil baris target prediksi
    row = df[df["Tanggal"] == predict_date]

    if row.empty:
        raise Exception("Baris prediksi tidak ditemukan.")
    row = row.fillna(default_avg)
    return row[features]

# --- Endpoint Prediksi ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or training_start_date is None:
        return jsonify({"error": "Model belum siap atau metadata tidak lengkap."}), 500

    try:
        req = request.get_json()
        days_to_predict = req.get("days_to_predict", 7)

        if not isinstance(days_to_predict, int) or days_to_predict <= 0:
            return jsonify({"error": "days_to_predict harus angka positif."}), 400

        df_hist = get_sales_history()

        if df_hist.empty:
            return jsonify({"error": "Tidak ada data historis penjualan yang valid ditemukan di Firestore."}), 400

        last_date = df_hist["Tanggal"].max()
        predictions = []

        for i in range(1, days_to_predict + 1):
            target_date = last_date + timedelta(days=1)
            feature_df = build_features(df_hist, target_date)

            print(f"\n--- Prediksi untuk tanggal: {target_date.strftime('%Y-%m-%d')} ---")
            print(feature_df.to_string(index=False))

            pred = model.predict(feature_df)[0]
            pred = max(0, round(pred, 1))
            predictions.append({
                "Tanggal": target_date.strftime("%Y-%m-%d"),
                "Prediksi Galon": pred
            })

            df_hist = pd.concat([
                df_hist,
                pd.DataFrame([{
                    "Tanggal": target_date,
                    "Galon Terjual": pred
                }])
            ], ignore_index=True)

            last_date = target_date

        return jsonify({
            "last_known_data": df_hist["Tanggal"].max().strftime("%Y-%m-%d"),
            "predictions": predictions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
