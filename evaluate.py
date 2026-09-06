import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Mac par memory crash rokne ke liye TF ko sirf CPU par limit kar rahe hain
tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings("ignore")

# ==========================================
#  REGISTER CUSTOM ATTENTION LAYER
# ==========================================
@keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        bias_shape = (248, 1) if "Self_Attention_Mechanism" in self.name else (1,)
        self.b = self.add_weight(name='attention_bias', shape=bias_shape, initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.bool), axis=-1)
            e = tf.where(mask, e, -1e9)  
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)

# ==========================================
#  DIRECTORIES & DATA LOADING
# ==========================================
BASE_DIR = "."
CLEAN_DIR = os.path.join(BASE_DIR, "data/clean")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("Loading Dataset...")
df = pd.read_csv(os.path.join(CLEAN_DIR, "cleaned_programming_problems.csv"), low_memory=False).dropna(subset=['model_text', 'difficulty'])
df = df[df['difficulty'].astype(str).str.lower() != 'unknown']

with open(os.path.join(ARTIFACTS_DIR, "difficulty_encoder.pkl"), 'rb') as f:
    encoder = pickle.load(f)
y = encoder.transform(df['difficulty'].values)

_, X_temp, _, y_temp = train_test_split(df['model_text'].astype(str).tolist(), y, test_size=0.20, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Test Set Size: {len(X_test)} questions\n")

# ==========================================
#  MODEL LOADING (PURE TENSORFLOW)
# ==========================================
print("Loading Models...")
with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), 'rb') as f:
    rnn_tokenizer = pickle.load(f)

custom_objects = {"AttentionLayer": AttentionLayer}
with keras.saving.custom_object_scope(custom_objects):
    bigru_model = keras.saving.load_model(os.path.join(MODELS_DIR, "GRU_model", "best_bigru_model.keras"), compile=False)
    bilstm_model = keras.saving.load_model(os.path.join(MODELS_DIR, "LSTM_model", "best_manual_tuned_bilstm.keras"), compile=False)

distilbert_dir = os.path.join(MODELS_DIR, "best_distilbert", "best_distilbert_model")
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
# Tera puraana apna TensorFlow wala DistilBERT
distilbert_model = TFAutoModelForSequenceClassification.from_pretrained(distilbert_dir)

# ==========================================
#  EVALUATION (BATCHED TO PREVENT MAC FREEZE)
# ==========================================
def evaluate_rnn(model, max_len, name):
    seqs = rnn_tokenizer.texts_to_sequences(X_test)
    padded = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    all_preds = []
    
    # 32 ke chote batch taaki Mac saans le sake
    for i in tqdm(range(0, len(padded), 32), desc=name):
        batch = padded[i:i+32]
        preds = model(batch, training=False)
        all_preds.extend(np.argmax(preds, axis=1))
    return np.array(all_preds)

def evaluate_distilbert():
    all_preds = []
    # 16 ke chote batch taaki DistilBERT Mac ko hang na kare
    for i in tqdm(range(0, len(X_test), 16), desc="DistilBERT"):
        inputs = distilbert_tokenizer(X_test[i:i+16], return_tensors="tf", truncation=True, padding=True, max_length=128)
        outputs = distilbert_model(inputs)
        preds = tf.nn.softmax(outputs.logits, axis=-1).numpy()
        all_preds.extend(np.argmax(preds, axis=1))
    return np.array(all_preds)

print("\nStarting Evaluations...")
y_pred_bigru = evaluate_rnn(bigru_model, 248, "BiGRU")
y_pred_bilstm = evaluate_rnn(bilstm_model, 248, "BiLSTM")
y_pred_distilbert = evaluate_distilbert()

# ==========================================
#  RESULTS & SAVING ARTIFACTS
# ==========================================
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "="*50)
print("  FINAL MODEL COMPARISON REPORT (SAVING TO RESULTS FOLDER)")
print("="*50)

models_info = [
    ("final_kerastuner_bigru", y_pred_bigru),
    ("final_kerastuner_bilstm", y_pred_bilstm),
    ("final_distilbert", y_pred_distilbert)
]

for model_name, preds in models_info:
    print(f"\nEvaluating and Saving metrics for {model_name}...")
    
    # 1. Terminal Print & Save TXT Report
    acc = accuracy_score(y_test, preds) * 100
    report_str = classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0)
    
    print(f"{model_name} Accuracy: {acc:.2f}%")
    
    with open(os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
        f.write(report_str)

    # 2. Save CSV Report
    report_dict = classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(RESULTS_DIR, f"classification_report_{model_name}.csv"))

    # 3. Generate & Save Confusion Matrix (PNG & CSV)
    cm = confusion_matrix(y_test, preds)
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.csv"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name} ({acc:.2f}%)')
    plt.ylabel('Actual Difficulty')
    plt.xlabel('Predicted Difficulty')
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png"), bbox_inches='tight')
    plt.close()

print("\n🚀 All new reports and confusion matrices successfully saved in the 'results/' folder!")