from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Muat model dan metadata saat aplikasi dimulai
try:
    model = joblib.load('random_forest_gallon_model.joblib')
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    features_from_training = model_metadata['features']
    training_start_date = datetime.fromisoformat(model_metadata['training_start_date'])
    print("* Model Random Forest dan metadata berhasil dimuat.")
except FileNotFoundError:
    model = None
    model_metadata = None
    features_from_training = []
    training_start_date = None
    print("* Peringatan: File model 'random_forest_gallon_model.joblib' atau 'model_metadata.json' tidak ditemukan.")
except Exception as e:
    model = None
    model_metadata = None
    features_from_training = []
    training_start_date = None
    print(f"* Error saat memuat model atau metadata: {e}")


def create_features_for_prediction(target_date_obj, base_date_for_index, feature_list):
    """
    Fungsi untuk membuat fitur untuk tanggal prediksi.
    Ini HARUS konsisten dengan feature engineering saat training.
    """
    features_dict = {}
    if 'hari_ke' in feature_list:
        features_dict['hari_ke'] = (target_date_obj - base_date_for_index).days
    if 'hari_dalam_minggu' in feature_list:
        features_dict['hari_dalam_minggu'] = target_date_obj.weekday()
    if 'bulan' in feature_list:
        features_dict['bulan'] = target_date_obj.month
    # Tambahkan fitur lain sesuai yang ada di feature_list dan digunakan saat training
    
    # Pastikan urutan kolom sama dengan saat training
    ordered_features = [features_dict.get(feat_name) for feat_name in feature_list]
    return pd.DataFrame([ordered_features], columns=feature_list)


@app.route('/predict', methods=['POST'])
def predict_sales():
    if model is None or training_start_date is None:
        return jsonify({'error': 'Model atau metadata tidak tersedia atau gagal dimuat.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body tidak boleh kosong.'}), 400

        days_to_predict = data.get('days_to_predict')
        last_historical_date_str = data.get('last_historical_date') # Format YYYY-MM-DD dari Flutter

        if days_to_predict is None or last_historical_date_str is None:
            return jsonify({'error': 'Parameter "days_to_predict" dan "last_historical_date" dibutuhkan.'}), 400
        
        if not isinstance(days_to_predict, int) or days_to_predict <= 0:
            return jsonify({'error': '"days_to_predict" harus integer positif.'}), 400

        try:
            # Tanggal terakhir data historis yang diketahui oleh Flutter
            last_known_date_from_flutter = datetime.strptime(last_historical_date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Format "last_historical_date" tidak valid. Gunakan YYYY-MM-DD.'}), 400

        predictions = []

        for i in range(1, days_to_predict + 1):
            # Tanggal yang akan diprediksi
            predict_date_obj = last_known_date_from_flutter + timedelta(days=i)
            
            # Buat fitur untuk tanggal prediksi
            # `training_start_date` adalah tanggal paling awal dari data yang digunakan untuk melatih model
            features_df = create_features_for_prediction(predict_date_obj, training_start_date, features_from_training)
            
            prediction = model.predict(features_df)[0]
            predictions.append(max(0, round(prediction, 1))) # Pastikan non-negatif dan bulatkan

        return jsonify({'predictions': predictions})

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc() # Cetak traceback untuk debugging lebih detail
        return jsonify({'error': f'Terjadi kesalahan internal: {str(e)}'}), 500

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    # host='0.0.0.0' agar bisa diakses dari luar localhost (misalnya dari emulator/device)
    # debug=True hanya untuk pengembangan, jangan gunakan di produksi
    app.run(host='0.0.0.0', port=5000, debug=True)
