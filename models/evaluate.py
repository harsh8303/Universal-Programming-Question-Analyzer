# models/evaluate.py
import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Apple Silicon (M1/M2/M3) GPU Hang Fix ---
tf.config.set_visible_devices([], 'GPU')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from attention_layer import AttentionLayer

# --- Universal Keras 3 Deserialization Safe Patch ---
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform

if not getattr(Layer, '_is_patched', False):
    _original_layer_init = Layer.__init__
    def _patched_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _original_layer_init(self, *args, **kwargs)
    Layer.__init__ = _patched_layer_init
    Layer._is_patched = True

if not getattr(GlorotUniform, '_is_patched', False):
    _original_glorot_init = GlorotUniform.__init__
    def _patched_glorot_init(self, *args, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        _original_glorot_init(self, *args, **kwargs)
    GlorotUniform.__init__ = _patched_glorot_init
    GlorotUniform._is_patched = True
# ---------------------------------------------

def evaluate_best_model(model_filename):
    print(f"Starting evaluation for {model_filename}...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    models_dir = os.path.join(ROOT_DIR, "models")
    results_dir = os.path.join(ROOT_DIR, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    model_tag = Path(model_filename).stem.removeprefix("best_")
    
    # 1. Load Test Data
    print("Loading test data...")
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # 2. Load Class Labels Encoder
    encoder_path = os.path.join(artifacts_dir, "difficulty_encoder.pkl")
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found at {encoder_path}")
        sys.exit(1)
        
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    class_names = encoder.classes_
    
    # 3. Load Best Model
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    print(f"Loading model checkpoint from {model_path}...")
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    
    # 4. Generate Predictions
    print(f"Total test samples: {len(X_test)}")
    y_pred_probs = model(X_test, training=False).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print(f"PERFORMANCE METRICS ({model_tag.upper()})")
    print("="*50)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("="*50)
    
    # 6. Generate Classification Report & Save inside results/
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    csv_path = os.path.join(results_dir, f"classification_report_{model_tag}.csv")
    txt_path = os.path.join(results_dir, f"classification_report_{model_tag}.txt")
    
    report_df.to_csv(csv_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy    : {acc:.4f}\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n")
        f.write(f"Weighted F1 : {weighted_f1:.4f}\n\n")
        f.write(report_text)
        
    print(f"Report saved to results folder.")
    
    # 7. Generate Confusion Matrix & Save inside results/
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    cm_csv_path = os.path.join(results_dir, f"confusion_matrix_{model_tag}.csv")
    cm_df.to_csv(cm_csv_path)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_tag.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Difficulty', fontsize=12)
    plt.xlabel('Predicted Difficulty', fontsize=12)
    
    cm_plot_path = os.path.join(results_dir, f"confusion_matrix_{model_tag}.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix plot saved to results folder.")
    print("Evaluation finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deep learning models")
    parser.add_argument(
        "--model", 
        default="best_bigru_attention_v1.keras", 
        help="Model filename inside models/ folder"
    )
    args = parser.parse_args()
    evaluate_best_model(args.model)