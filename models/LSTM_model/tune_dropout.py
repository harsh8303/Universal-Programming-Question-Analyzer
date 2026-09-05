# models/LSTM_model/tune_dropout.py
import os
import sys
import json
import numpy as np
import tensorflow as tf

# ==========================================
#  SEEDS FOR REPRODUCIBILITY (Global)
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ==========================================
#  FOLDER PATH SETUP 
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(CURRENT_DIR)                 
ROOT_DIR = os.path.dirname(MODELS_DIR)                    

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MODELS_DIR)

import config
from attention_layer import AttentionLayer 

# ==========================================
#  EXPERIMENT SETUP
# ==========================================
DROPOUT_RATES_TO_TEST = [0.2, 0.3, 0.4]

# LOCKED PARAMETERS FROM ALL PREVIOUS STEPS
LEARNING_RATE = 5e-4
EMBEDDING_DIM = 128    # Locked from Embedding Tuning
LSTM_1_UNITS = 128     # Locked from LSTM Tuning
LSTM_2_UNITS = 64      # Locked from LSTM Tuning
DENSE_UNITS = 128      # 👈 Locked from Dense Tuning!
# ==========================================

def build_model(vocab_size, max_seq_length, num_classes, dropout_rate):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)
    
    #  Explicit Dropouts for Keras 3 Bug Bypass
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=LSTM_1_UNITS, return_sequences=True))(x)
    
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units=LSTM_2_UNITS, return_sequences=True))(x)
    
    x = AttentionLayer()(x)
    
    x = Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_dropout_experiments():
    print(" STARTING AUTOMATED DROPOUT TUNING LOOP...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    
    with open(os.path.join(artifacts_dir, "sequence_config.json"), 'r') as f:
        seq_config = json.load(f)
        
    vocab_size = seq_config["Vocabulary Size"]
    max_seq_length = seq_config["Maximum Sequence Length"]
    num_classes = len(seq_config["Difficulty Classes"])
    
    print("Loading Validation & Training Data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    experiment_results = {}
    
    for dropout_rate in DROPOUT_RATES_TO_TEST:
        config_name = f"Dropout_{dropout_rate}"
        print("\n" + "="*60)
        print(f" TESTING DROPOUT RATE: {dropout_rate}")
        print("="*60)
        
        # Memory and Seed resets omitted to bypass Keras 3 LSTM Masking bug
        
        model = build_model(vocab_size, max_seq_length, num_classes, dropout_rate)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        best_epoch_idx = int(np.argmin(history.history['val_loss']))
        best_val_loss = float(history.history['val_loss'][best_epoch_idx])
        best_val_acc = float(history.history['val_accuracy'][best_epoch_idx])
        best_epoch = best_epoch_idx + 1
        
        experiment_results[config_name] = {
            'Dropout_Rate': dropout_rate,
            'Best_Validation_Loss': best_val_loss,
            'Best_Validation_Accuracy': best_val_acc,
            'Best_Epoch': best_epoch
        }
    
    json_path = os.path.join(CURRENT_DIR, "dropout_tuning_results.json")
    with open(json_path, 'w') as json_file:
        json.dump(experiment_results, json_file, indent=4)
        
    print("\n\n" + "*"*25)
    print("      FINAL DROPOUT REPORT")
    print("*"*25)
    print(f"{'Dropout Config':<15} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Best Val Loss':<15}")
    print("-" * 65)
    for cfg, metrics in experiment_results.items():
        print(f"{cfg:<15} | {metrics['Best_Epoch']:<12} | {metrics['Best_Validation_Accuracy']:<15.4f} | {metrics['Best_Validation_Loss']:<15.4f}")
    print("-" * 65)
    print(f"\n Results successfully saved to: {json_path}\n")

if __name__ == "__main__":
    run_dropout_experiments()