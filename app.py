import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# Load Model Safely
# ----------------------------
MODEL_PATH = "model.h5"
model = None

if os.path.exists(MODEL_PATH):
    print("--- Loading model ---")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    print(f"--- WARNING: {MODEL_PATH} not found. Please run train_model.py first! ---")

# ----------------------------
# Load Nutrition Data
# ----------------------------
NUTRITION_FILE = "food_data.json"

if os.path.exists(NUTRITION_FILE):
    with open(NUTRITION_FILE, "r") as f:
        nutrition_data = json.load(f)
else:
    print("⚠ food_data.json not found!")
    nutrition_data = {}

# The class names must match dataset folders alphabetically
class_names = [
    'adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch',
    'aloo_tikki', 'anarsa', 'ariselu', 'bandar_laddu', 'basundi', 'bhatura',
    'bhindi_masala', 'biryani', 'boondi', 'butter_chicken', 'chak_hao_kheer',
    'cham_cham', 'chana_masala', 'chapati', 'chhena_kheeri', 'chicken_razala',
    'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'daal_baati_churma',
    'daal_puri', 'dal_makhani', 'dal_tadka', 'dharwad_pedha', 'doodhpak',
    'double_ka_meetha', 'dum_aloo', 'gajar_ka_halwa', 'gavvalu', 'ghevar',
    'gulab_jamun', 'imarti', 'jalebi', 'kachori', 'kadai_paneer', 'kadhi_pakoda',
    'kajjikaya', 'kakinada_khaja', 'kalakand', 'karela_bharta', 'kofta',
    'kuzhi_paniyaram', 'lassi', 'ledikeni', 'litti_chokha', 'lyangcha',
    'maach_jhol', 'makki_di_roti_sarson_da_saag', 'malapua', 'misi_roti',
    'misti_doi', 'modak', 'mysore_pak', 'naan', 'navrattan_korma', 'palak_paneer',
    'paneer_butter_masala', 'phirni', 'pithe', 'poha', 'poornalu', 'pootharekulu',
    'qubani_ka_meetha', 'rabri', 'ras_malai', 'rasgulla', 'sandesh', 'shankarpali',
    'sheer_korma', 'sheera', 'shrikhand', 'sohan_halwa', 'sohan_papdi', 'sutar_feni',
    'unni_appam'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained or found on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess image (same as training)
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        # Get nutrition info
        nutrition_info = nutrition_data.get(predicted_class, {
            "calories": "N/A",
            "protein": "N/A",
            "carbs": "N/A",
            "fats": "N/A",
            "benefits": "No nutrition data available."
        })

        return jsonify({
            "class_name": predicted_class.replace("_", " ").title(),
            "confidence": round(confidence, 2),
            "nutrition": nutrition_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)