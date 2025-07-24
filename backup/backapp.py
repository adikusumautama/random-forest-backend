from flask import Flask, request, jsonify
import joblib, json, pandas as pd, numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import holidays, warnings, traceback

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# --- Pengaturan Firebase ---
try:
    cred = credentials.Certificate("firebase/damiu-app-ad9f7-firebase-adminsdk-fbsvc-9be6b99017.json")
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
    features = metadata["features"]
    TRANSFORMATION_STATS = metadata.get("transformation_stats")
    training_start_date = datetime.fromisoformat(metadata["training_start_date"]).replace(tzinfo=None)
    print("* Model XGBoost dan metadata berhasil dimuat.")
except Exception as e:
    model, features, TRANSFORMATION_STATS, training_start_date = None, [], None, None
    print(f"* Gagal memuat model atau metadata: {e}")

# --- Ambil Data Historis dari Firestore ---
def get_sales_history():
    if db is None:
        raise Exception("Firestore belum terhubung.")
    parsed = []
    for doc in db.collection("daily_sales").stream():
        d = doc.to_dict()
        try:
            tgl = pd.to_datetime(doc.id, format="%Y-%m-%d")
            qty = float(d.get("quantity", np.nan))
            parsed.append({"Tanggal": tgl, "Galon Terjual": qty})
        except (ValueError, TypeError):
            print(f"* WARNING: Dokumen {doc.id} dilewati karena format tidak valid.")
    if not parsed:
        return pd.DataFrame(columns=["Tanggal", "Galon Terjual"])
    df = pd.DataFrame(parsed).sort_values("Tanggal").reset_index(drop=True)
    return df

# --- Build Fitur (harus identik dengan training) ---
def build_features(df_full, predict_date, stats):
    lb, ub = stats['winsor_lower_bound'], stats['winsor_upper_bound']
    def_avg, def_std = stats['static_default_avg_winsorized'], stats['static_default_std_winsorized']

    df_full['Galon Terjual_winsorized'] = df_full['Galon Terjual'].clip(lower=lb, upper=ub)
    df = pd.concat([df_full, pd.DataFrame([{"Tanggal": predict_date, "Galon Terjual": np.nan}])])
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    indo_holidays = holidays.Indonesia()
    df["hari_ke"] = (df["Tanggal"] - training_start_date).dt.days
    df["hari_dalam_minggu"] = df["Tanggal"].apply(lambda x: x.isoweekday())
    df["bulan"] = df["Tanggal"].dt.month
    df["minggu_ke"] = df["Tanggal"].dt.isocalendar().week.astype(int)
    df["tahun"] = df["Tanggal"].dt.year
    df["is_weekend"] = df["hari_dalam_minggu"].isin([6, 7]).astype(int)
    df["is_awal_bulan"] = (df["Tanggal"].dt.day <= 3).astype(int)
    df["is_akhir_bulan"] = (df["Tanggal"].dt.day >= 28).astype(int)
    df["is_holiday"] = df["Tanggal"].apply(lambda d: int(d in indo_holidays))
    df["hari_minggu_x_holiday"] = df["hari_dalam_minggu"] * df["is_holiday"]

    P_year, P_week = 365.25, 7
    df['sin_tahun_1'] = np.sin(2 * np.pi * df['hari_ke'] / P_year)
    df['cos_tahun_1'] = np.cos(2 * np.pi * df['hari_ke'] / P_year)
    df['sin_tahun_2'] = np.sin(4 * np.pi * df['hari_ke'] / P_year)
    df['cos_tahun_2'] = np.cos(4 * np.pi * df['hari_ke'] / P_year)
    df['sin_minggu_1'] = np.sin(2 * np.pi * df['hari_ke'] / P_week)
    df['cos_minggu_1'] = np.cos(2 * np.pi * df['hari_ke'] / P_week)

    df["penjualan_kemarin"] = df["Galon Terjual_winsorized"].shift(1).fillna(def_avg)
    df["penjualan_2hari_lalu"] = df["Galon Terjual_winsorized"].shift(2).fillna(def_avg)
    df["rata2_3hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(3).mean().fillna(def_avg)
    df["rata2_7hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(7).mean().fillna(def_avg)
    df["rata2_14hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(14).mean().fillna(def_avg)
    df["std_7hari"] = df["Galon Terjual_winsorized"].shift(1).rolling(7).std().fillna(def_std)
    df["delta_penjualan"] = df["Galon Terjual_winsorized"].diff().fillna(0)

    df['is_zero_sale_day'] = (df['Galon Terjual'] == 0).astype(int)
    last_zero = df[df['is_zero_sale_day'] == 1]['Tanggal']
    if not last_zero.empty:
        ffill = pd.Series(last_zero.values, index=last_zero).reindex(df["Tanggal"], method="ffill")
        df["days_since_last_zero_sale"] = (df["Tanggal"] - ffill).dt.days.fillna(df["hari_ke"])
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
    if None in (model, training_start_date, TRANSFORMATION_STATS):
        return jsonify({"error": "Model atau metadata belum siap."}), 500
    try:
        df_hist = get_sales_history()
        if df_hist.empty:
            return jsonify({"error": "Tidak ada data historis di Firestore."}), 400
        last_date = df_hist["Tanggal"].max()
        target_date = last_date + timedelta(days=1)

        feature_df = build_features(df_hist, target_date, TRANSFORMATION_STATS)
        raw_pred = model.predict(feature_df)[0]

        min_ratio = 0.60
        min_allowed = df_hist["Galon Terjual"].iloc[-1] * min_ratio
        final_pred = max(0, round(max(raw_pred, min_allowed), 2))

        # --- LOGGING TAMBAHAN UNTUK DEBUGGING ---
        print("\n==============================")
        print(f"Prediksi Tanggal: {target_date.strftime('%A, %d %B %Y')}")
        print("\nFitur Input Model:")
        for col in feature_df.columns:
            print(f"  - {col:25s}: {feature_df.iloc[0][col]:.4f}")
        print(f"\nPrediksi dari Model XGBoost: {raw_pred:.2f}")
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
