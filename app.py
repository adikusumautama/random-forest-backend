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
warnings.filterwarnings("ignore", category=FutureWarning)

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
    ensemble_models = [
        joblib.load("random_forest_gallon_model.joblib"),
        joblib.load("xgboost_gallon_model.joblib")
    ]
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    features = metadata["features"]
    training_start_date = datetime.fromisoformat(metadata["training_start_date"]).replace(tzinfo=None)
    print("* Model dan metadata berhasil dimuat.")
except Exception as e:
    ensemble_models = []
    features = []
    training_start_date = None
    print(f"* Gagal load model atau metadata: {e}")

# --- Ambil Data Firestore ---
def get_sales_history():
    if db is None:
        raise Exception("Firestore belum terhubung.")

    docs = db.collection("daily_sales").stream()
    parsed_data = []
    processed_dates = [] # Tambahkan ini untuk melacak tanggal yang berhasil diproses
    print("\n--- DEBUG: Memulai pengambilan data dari Firestore ---")
    for doc in docs:
        d = doc.to_dict()
        doc_id = doc.id

        # Gunakan ID dokumen sebagai sumber kebenaran untuk tanggal.
        # Ini menghindari masalah zona waktu karena ID dibuat dari tanggal lokal di Flutter.
        if "quantity" in d:
            try:
                # Coba parse tanggal dari ID dokumen, yang seharusnya YYYY-MM-DD
                tanggal = pd.to_datetime(doc_id, format='%Y-%m-%d')

                jumlah = float(d["quantity"])
                parsed_data.append({"Tanggal": tanggal, "Galon Terjual": jumlah})
                processed_dates.append(tanggal) # Tambahkan tanggal ke daftar yang diproses
                print(f"  DEBUG: Berhasil memparsing dokumen {doc_id} (Tanggal dari ID: {tanggal.strftime('%Y-%m-%d')}, Galon: {jumlah})")
            except ValueError:
                print(f"* WARNING: ID Dokumen '{doc_id}' bukan format tanggal yang valid. Dokumen dilewati.")
            except Exception as e:
                print(f"* ERROR: Gagal parsing data dari dokumen {doc_id}: {e}. Data mentah: {d}")
        else:
            print(f"* WARNING: Dokumen {doc_id} dilewati: field 'quantity' tidak ditemukan. Data mentah: {d}")

    if not parsed_data:
        print("--- DEBUG: Tidak ada data valid yang berhasil diparsing dari Firestore. ---")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])
    
    print(f"--- DEBUG: Selesai pengambilan data dari Firestore. Total dokumen diproses: {len(parsed_data)} ---")
    if processed_dates:
        print(f"--- DEBUG: Tanggal terawal yang diproses: {min(processed_dates).strftime('%Y-%m-%d')} ---")
        print(f"--- DEBUG: Tanggal terakhir yang diproses: {max(processed_dates).strftime('%Y-%m-%d')} ---")

    df = pd.DataFrame(parsed_data)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values(by="Tanggal").reset_index(drop=True)
    return df

# --- Ambil Data dari CSV ---
def get_sales_history_from_csv(csv_path="data/damiu.csv"):
    """
    Membaca data historis penjualan dari file CSV.
    Diasumsikan CSV tidak memiliki header dan kolomnya adalah: Tanggal, AngkaHari, GalonTerjual.
    """
    try:
        df = pd.read_csv(csv_path, header=None, names=['Tanggal', '_day_num_from_csv', 'Galon Terjual'])
        
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df.dropna(subset=['Tanggal'], inplace=True) # Hapus baris dengan tanggal yang tidak valid

        df['Galon Terjual'] = pd.to_numeric(df['Galon Terjual'], errors='coerce')
        df.dropna(subset=['Galon Terjual'], inplace=True) # Hapus baris dengan kuantitas yang tidak valid
        df['Galon Terjual'] = df['Galon Terjual'].astype(float) # Konsisten dengan parsing Firestore

        df = df.sort_values(by="Tanggal").reset_index(drop=True)
        
        # Kembalikan hanya kolom yang relevan
        return df[["Tanggal", "Galon Terjual"]]
    except FileNotFoundError:
        print(f"* File CSV {csv_path} tidak ditemukan.")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])
    except pd.errors.EmptyDataError:
        print(f"* File CSV {csv_path} kosong.")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])
    except Exception as e:
        print(f"* Gagal membaca atau memproses CSV {csv_path}: {e}")
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])


# --- Bangun Fitur Ideal ---
def build_features(df_full, predict_date, 
                   static_lower_bound, static_upper_bound, 
                   static_default_avg_winsorized, static_default_std_winsorized):
    # df_full adalah data historis

    # Buat DataFrame baru yang mencakup tanggal prediksi
    df = pd.concat([df_full, pd.DataFrame([{"Tanggal": predict_date, "Galon Terjual": np.nan}])], ignore_index=True)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values(by="Tanggal").reset_index(drop=True)

    # Buat kolom 'Galon Terjual_winsorized'
    df["Galon Terjual_winsorized"] = df["Galon Terjual"].copy()

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

    # --- Fourier Terms ---
    P_year = 365.25
    df['sin_tahun_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_year)
    df['cos_tahun_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_year)
    df['sin_tahun_2'] = np.sin(2 * np.pi * 2 * df['hari_ke'] / P_year)
    df['cos_tahun_2'] = np.cos(2 * np.pi * 2 * df['hari_ke'] / P_year)
    P_week = 7
    df['sin_minggu_1'] = np.sin(2 * np.pi * 1 * df['hari_ke'] / P_week)
    df['cos_minggu_1'] = np.cos(2 * np.pi * 1 * df['hari_ke'] / P_week)

    # --- Lag & Rolling Features ---
    df["penjualan_kemarin"] = df["Galon Terjual"].shift(1).fillna(static_default_avg_winsorized)
    df["penjualan_2hari_lalu"] = df["Galon Terjual"].shift(2).fillna(static_default_avg_winsorized)
    df["rata2_3hari"] = df["Galon Terjual"].rolling(window=3, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized)
    df["rata2_7hari"] = df["Galon Terjual"].rolling(window=7, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized)
    df["rata2_14hari"] = df["Galon Terjual"].rolling(window=14, min_periods=1).mean().bfill().fillna(static_default_avg_winsorized)
    df["std_7hari"] = df["Galon Terjual"].rolling(window=7, min_periods=1).std().fillna(static_default_std_winsorized)
    df["delta_penjualan"] = df["Galon Terjual_winsorized"].diff().fillna(0)

    # --- Fitur Terkait Hari Nol Penjualan (INI BAGIAN YANG DIPERBAIKI) ---
    df['is_zero_sale_day'] = (df['Galon Terjual'] == 0).astype(int)
    zero_sale_dates = df[df['is_zero_sale_day'] == 1]['Tanggal']
    if not zero_sale_dates.empty:
        zero_sale_series = pd.Series(zero_sale_dates.values, index=zero_sale_dates)
        last_zero_sale_date = zero_sale_series.reindex(df["Tanggal"], method="ffill")
        df["days_since_last_zero_sale"] = (df["Tanggal"] - last_zero_sale_date).dt.days.fillna(df["hari_ke"])
    else:
        # Jika tidak ada hari penjualan nol, isi dengan hari_ke sebagai fallback
        # untuk memastikan kolom selalu ada.
        df["days_since_last_zero_sale"] = df["hari_ke"]
    df["is_day_after_zero_sale"] = df["is_zero_sale_day"].shift(1).fillna(0).astype(int)

    row = df[df["Tanggal"] == predict_date]
    if row.empty:
        raise Exception("Baris prediksi tidak ditemukan.")
    
    return row[features]

# --- Endpoint Prediksi ---
@app.route("/predict", methods=["POST"])
def predict():
    if not ensemble_models or training_start_date is None:
        return jsonify({"error": "Model belum siap atau metadata tidak lengkap."}), 500

    try:
        # Sesuai permintaan, prediksi hanya untuk 1 hari ke depan (besok).
        # Parameter days_to_predict dari request diabaikan jika ada.
        req = request.get_json()
        data_source = req.get("data_source", "firestore") # Default ke 'firestore' jika tidak dispesifikasikan

        if data_source == "firestore":
            df_hist_initial = get_sales_history()
            if df_hist_initial.empty:
                return jsonify({"error": "Tidak ada data historis penjualan yang valid ditemukan di Firestore."}), 400
            print("* Menggunakan data historis dari Firestore.")
        elif data_source == "csv":
            df_hist_initial = get_sales_history_from_csv()
            if df_hist_initial.empty:
                return jsonify({"error": "Tidak ada data historis penjualan yang valid ditemukan di damiu.csv."}), 400
            print("* Menggunakan data historis dari damiu.csv.")
        else:
            return jsonify({"error": "Nilai 'data_source' tidak valid. Gunakan 'firestore' atau 'csv'."}), 400

        # Hitung batas winsorisasi dan default statis dari data historis awal
        static_lower_bound = df_hist_initial["Galon Terjual"].quantile(0.05) if not df_hist_initial.empty else 0 # type: ignore
        static_upper_bound = df_hist_initial["Galon Terjual"].quantile(0.95) if not df_hist_initial.empty else float('inf')
        
        df_hist_initial_temp_winsorized_col = df_hist_initial["Galon Terjual"].clip(lower=static_lower_bound, upper=static_upper_bound)
        static_default_avg_winsorized = df_hist_initial_temp_winsorized_col.mean() if not df_hist_initial.empty else 0
        static_default_std_winsorized = df_hist_initial_temp_winsorized_col.std() if not df_hist_initial.empty else 0

        # Data historis yang akan digunakan untuk membangun fitur
        df_hist_for_features = df_hist_initial.copy()
        last_historical_date = df_hist_for_features["Tanggal"].max() # type: ignore
        print(f"DEBUG: Last historical date (max date in fetched data): {last_historical_date.strftime('%Y-%m-%d')}")
        
        target_date = last_historical_date + timedelta(days=1) # Prediksi untuk besok
        print(f"DEBUG: Calculated prediction target date: {target_date.strftime('%Y-%m-%d')}")
        
        feature_df = build_features(df_hist_for_features, target_date,
                                      static_lower_bound, static_upper_bound,
                                      static_default_avg_winsorized, static_default_std_winsorized)

        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print("\nFitur Input Model:")
        for col in feature_df.columns:
            print(f"  - {col:25s}: {feature_df.iloc[0][col]:.4f}")

        all_preds = [model.predict(feature_df)[0] for model in ensemble_models]
        pred_mean = np.mean(all_preds)

        if len(all_preds) > 1:
            pred_std = np.std(all_preds)
            lower_ci = pred_mean - 1.96 * pred_std
            upper_ci = pred_mean + 1.96 * pred_std
        else:
            lower_ci = pred_mean - 5 # Fallback sederhana jika hanya satu model
            upper_ci = pred_mean + 5

        pred_candidate = max(0, round(pred_mean, 1))

        # --- Post-processing: Batasi penurunan maksimal harian ---
        max_daily_decrease_ratio = 0.60 # Batasi penurunan maksimal 40% (prediksi tidak boleh < 60% dari hari sebelumnya)
        # Gunakan penjualan aktual terakhir dari data historis
        previous_day_sale = df_hist_initial["Galon Terjual"].iloc[-1] if not df_hist_initial.empty else pred_candidate
        
        minimum_allowed_prediction = round(previous_day_sale * max_daily_decrease_ratio, 1)
        
        pred_final = pred_candidate
        if pred_candidate < minimum_allowed_prediction:
            print(f"  INFO: Prediksi awal {pred_candidate:.1f} disesuaikan menjadi {minimum_allowed_prediction:.1f} (min {max_daily_decrease_ratio*100:.0f}% dari {previous_day_sale:.1f})")
            pred_final = max(0, minimum_allowed_prediction) # Pastikan tetap non-negatif

        print("\nPrediksi dari Semua Model:")
        for idx, pred_val in enumerate(all_preds):
            print(f"  Model-{idx+1}: {pred_val:.2f}")

        print(f"\nRata-rata Prediksi: {pred_mean:.2f}")
        print(f"Confidence Interval : [{lower_ci:.2f} - {upper_ci:.2f}]")
        print(f"Prediksi Akhir      : {pred_final:.2f} galon")
        print("==============================")

        prediction_output = {
            "Tanggal": target_date.strftime("%Y-%m-%d"),
            "Prediksi Galon": pred_final,
            "Confidence Range": [round(lower_ci, 1), round(upper_ci, 1)]
        }
        print(f"DEBUG: API response 'prediction_for_next_day' Tanggal: {prediction_output['Tanggal']}")

        return jsonify({
            "last_known_data_date": last_historical_date.strftime("%Y-%m-%d"), # type: ignore
            "prediction_for_next_day": prediction_output
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
