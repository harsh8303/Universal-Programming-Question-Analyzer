import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import keras

# Mac par memory crash rokne ke liye TF ko sirf CPU par limit kar rahe hain
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ==========================================
#  FIXED REGISTRATION (Matching evaluate.py exactly)
# ==========================================
@keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='glorot_uniform', 
                                 trainable=True)
        
        # 100% working logic from evaluate.py
        bias_shape = (248, 1) if "Self_Attention_Mechanism" in self.name else (1,)
            
        self.b = self.add_weight(name='attention_bias', 
                                 shape=bias_shape, 
                                 initializer='zeros', 
                                 trainable=True)
                                 
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.bool), axis=-1)
            e = tf.where(mask, e, -1e9)  
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        return None
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================

from ocr_utils import extract_text

app = Flask(__name__)

print("Loading models and files... wait a sec")

# loading label encoder
encoder_path = "artifacts/difficulty_encoder.pkl"
if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    labels = list(encoder.classes_)
else:
    labels = ["Easy", "Medium", "Hard"]

# loading tokenizer
tokenizer_path = "artifacts/tokenizer.pkl"
rnn_tokenizer = None
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        rnn_tokenizer = pickle.load(f)

# loading my trained deep learning models
custom_objects = {"AttentionLayer": AttentionLayer}

bigru_path = "models/GRU_model/best_bigru_model.keras"
bilstm_path = "models/LSTM_model/best_manual_tuned_bilstm.keras"

with keras.saving.custom_object_scope(custom_objects):
    bigru_model = keras.saving.load_model(bigru_path, compile=False) if os.path.exists(bigru_path) else None
    bilstm_model = keras.saving.load_model(bilstm_path, compile=False) if os.path.exists(bilstm_path) else None

# loading distilbert artifacts
distilbert_dir = "models/best_distilbert/best_distilbert_model"
if os.path.exists(distilbert_dir):
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
    distilbert_model = TFAutoModelForSequenceClassification.from_pretrained(distilbert_dir)
else:
    distilbert_tokenizer, distilbert_model = None, None

print("Done loading models!")

# -----------------
# Flask API Routes
# -----------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = ""
    model_type = "bigru" # setting default
    
    if 'images' in request.files:
        image_files = request.files.getlist('images')
        extracted_texts = []
        
        for i, image_file in enumerate(image_files):
            if image_file.filename != '':
                temp_path = f"temp_uploaded_image_{i}.png"
                image_file.save(temp_path)
                try:
                    extracted_text = extract_text(temp_path)
                    extracted_texts.append(extracted_text)
                except Exception as e:
                    return jsonify({"error": f"OCR failed on image {i+1}: {str(e)}"}), 500
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if extracted_texts:
            text = " ".join(extracted_texts)
        
        model_type = request.form.get('model_type', 'bigru')
        
    elif request.is_json:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON format"}), 400
        text = data.get("text", "")
        model_type = data.get("model_type", "bigru")
        
    else:
        text = request.form.get("text", "")
        model_type = request.form.get("model_type", "bigru")

    if not text.strip():
        return jsonify({"error": "No text or images provided"}), 400

    model_type = model_type.lower()
    if model_type not in ["bigru", "bilstm", "distilbert"]:
        return jsonify({"error": "Invalid model_type. Choose from 'bigru', 'bilstm', 'distilbert'"}), 400

    try:
        if model_type in ["bigru", "bilstm"]:
            model = bigru_model if model_type == "bigru" else bilstm_model
            
            if rnn_tokenizer is None or model is None:
                return jsonify({"error": f"{model_type.upper()} model or tokenizer not loaded."}), 500
            
            #  FIXED MAXLEN: Set to 248 as required by both models
            max_len = 248
            
            sequences = rnn_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            
            preds = model(padded, training=False).numpy()
            pred_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds))
            
        elif model_type == "distilbert":
            if distilbert_tokenizer is None or distilbert_model is None:
                return jsonify({"error": "DistilBERT model artifacts not loaded."}), 500
            
            inputs = distilbert_tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
            outputs = distilbert_model(inputs)
            preds = tf.nn.softmax(outputs.logits, axis=-1).numpy()
            
            pred_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds))
            
        return jsonify({
            "status": "success",
            "model_used": model_type.upper(),
            "extracted_text": text[:200] + "..." if len(text) > 200 else text,
            "prediction": labels[pred_idx] if pred_idx < len(labels) else "Medium",
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)