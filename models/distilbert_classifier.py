# models/distilbert_classifier.py
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def plot_training_history(history, models_dir):
    print("\n Generating Training History Plot...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (DistilBERT)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (DistilBERT)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(models_dir, "training_history_distilbert.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Training plot saved at: {plot_path}")

def train_distilbert():
    print(" Starting DistilBERT Fine-Tuning Pipeline (COLAB GPU VERSION)...")
    
    clean_csv_path = os.path.join(config.DATA_DIR, "clean", "cleaned_programming_problems.csv")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(" Loading Cleaned Dataset...")
    df = pd.read_csv(clean_csv_path)
    df = df.dropna(subset=['model_text', 'difficulty'])
    df = df[df['difficulty'].astype(str).str.lower() != 'unknown']
    
    texts = df['model_text'].astype(str).tolist()
    labels = df['difficulty'].values
    
    encoder_path = os.path.join(artifacts_dir, "difficulty_encoder.pkl")
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
        
    encoded_labels = encoder.transform(labels)
    num_classes = len(encoder.classes_)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, encoded_labels, test_size=0.20, random_state=42, stratify=encoded_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(" Initializing DistilBERT Tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    max_len = 128
    
    print(" Tokenizing Datasets...")
    train_encodings = tokenizer(X_train, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
    val_encodings = tokenizer(X_val, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
    test_encodings = tokenizer(X_test, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
    
    train_x = dict(train_encodings)
    val_x = dict(val_encodings)
    test_x = dict(test_encodings)
    
    print(" Loading Pre-trained TFDistilBertForSequenceClassification...")
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_classes
    )
    
    #  Standard Adam Optimizer for GPU (Removed legacy Mac optimizer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    )
    
    print("\n Commencing DistilBERT Fine-Tuning on GPU...\n")
    
    # Batch size increased to 32 because GPU can easily handle it
    history = model.fit(
        x=train_x,
        y=y_train,
        validation_data=(val_x, y_val),
        batch_size=32, 
        epochs=3,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print("\n Evaluating Final Model on Test Set...")
    test_loss, test_acc = model.evaluate(test_x, y_test, batch_size=32, verbose=1)
    print(f" Final Test Accuracy: {test_acc:.4f}")
    
    best_model_dir = os.path.join(models_dir, "best_distilbert")
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f" Best DistilBERT model & tokenizer saved at: {best_model_dir}")
    
    with open(os.path.join(models_dir, "training_history_distilbert.json"), "w") as f:
        history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=4)
        
    with open(os.path.join(models_dir, "distilbert_config.json"), "w") as f:
        json.dump(model.config.to_dict(), f, indent=4)
        
    plot_training_history(history, models_dir)
    print("\n Pipeline Executed Successfully on GPU!")

if __name__ == "__main__":
    train_distilbert()