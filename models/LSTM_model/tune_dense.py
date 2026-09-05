# models/LSTM_model/tune_dense.py
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
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
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
DENSE_UNITS_TO_TEST = [128, 64, 32]

# LOCKED PARAMETERS FROM PREVIOUS STEPS
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.3
EMBEDDING_DIM = 128    # Locked from Embedding Tuning
LSTM_1_UNITS = 128     # Locked from LSTM Tuning
LSTM_2_UNITS = 64      # Locked from LSTM Tuning
# ==========================================

def build_model(vocab_size, max_seq_length, num_classes, dense_units):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)
    
    #  Explicit Dropouts for Keras 3 Bug Bypass
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(units=LSTM_1_UNITS, return_sequences=True))(x)
    
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(units=LSTM_2_UNITS, return_sequences=True))(x)
    
    x = AttentionLayer()(x)
    
    #  TUNING THIS DENSE LAYER
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_dense_experiments():
    print(" STARTING AUTOMATED DENSE UNITS TUNING LOOP...")
    
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
    
    for dense_units in DENSE_UNITS_TO_TEST:
        config_name = f"Dense_{dense_units}"
        print("\n" + "="*60)
        print(f" TESTING DENSE CAPACITY: Units={dense_units}")
        print("="*60)
        
        # Memory and Seed resets omitted to bypass Keras 3 LSTM Masking bug
        
        model = build_model(vocab_size, max_seq_length, num_classes, dense_units)
        
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
            'Dense_Units': dense_units,
            'Best_Validation_Loss': best_val_loss,
            'Best_Validation_Accuracy': best_val_acc,
            'Best_Epoch': best_epoch
        }
    
    json_path = os.path.join(CURRENT_DIR, "dense_tuning_results.json")
    with open(json_path, 'w') as json_file:
        json.dump(experiment_results, json_file, indent=4)
        
    print("\n\n" + "*"*25)
    print("      FINAL DENSE CAPACITY REPORT")
    print("*"*25)
    print(f"{'Dense Config':<15} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Best Val Loss':<15}")
    print("-" * 65)
    for cfg, metrics in experiment_results.items():
        print(f"{cfg:<15} | {metrics['Best_Epoch']:<12} | {metrics['Best_Validation_Accuracy']:<15.4f} | {metrics['Best_Validation_Loss']:<15.4f}")
    print("-" * 65)
    print(f"\n Results successfully saved to: {json_path}\n")

if __name__ == "__main__":
    run_dense_experiments()