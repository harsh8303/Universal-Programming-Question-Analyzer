# models/GRU_model/tune_dense.py
import os
import sys
import json
import numpy as np
import tensorflow as tf

# ==========================================
#  SEEDS FOR REPRODUCIBILITY 
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, Layer
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
#  EXPERIMENT SETUP: THE LOOP VALUES
# ==========================================
# CHANGE: Testing different Dense Layer Units
DENSE_UNITS_TO_TEST = [32, 64, 128]

# FIXED PARAMETERS (Strictly untouched! The All-Star Team 🏆)
LEARNING_RATE = 5e-4   # Locked!
GRU_UNITS_1 = 128      # Locked!
GRU_UNITS_2 = 64       # Locked!
DROPOUT_RATE = 0.5     # Locked!
EMBEDDING_DIM = 128    # Locked!
# ==========================================

def build_model(vocab_size, max_seq_length, num_classes, dense_units):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)
    x = Bidirectional(GRU(units=GRU_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = Bidirectional(GRU(units=GRU_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = AttentionLayer()(x)
    
    #  TUNING THE DENSE LAYER HERE
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_dense_experiments():
    print(" STARTING AUTOMATED DENSE LAYER TUNING LOOP...")
    
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
        print("\n" + "="*60)
        print(f" TESTING DENSE UNITS: {dense_units}")
        print("="*60)
        
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        
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
        
        best_val_loss = float(min(history.history['val_loss']))
        best_val_acc = float(max(history.history['val_accuracy']))
        best_epoch = int(np.argmin(history.history['val_loss']) + 1)
        
        experiment_results[str(dense_units)] = {
            'Dense_Units': dense_units,
            'Best_Validation_Loss': best_val_loss,
            'Best_Validation_Accuracy': best_val_acc,
            'Best_Epoch': best_epoch
        }
    
    # ==========================================
    # SAVE RESULTS TO JSON
    # ==========================================
    json_path = os.path.join(CURRENT_DIR, "dense_tuning_results.json")
    with open(json_path, 'w') as json_file:
        json.dump(experiment_results, json_file, indent=4)
        
    # ==========================================
    #  FINAL SUMMARY REPORT
    # ==========================================
    print("\n\n" + "*"*25)
    print("      FINAL DENSE LAYER TUNING REPORT")
    print("*"*25)
    print(f"{'Dense Units':<12} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Best Val Loss':<15}")
    print("-" * 65)
    for dense_val, metrics in experiment_results.items():
        print(f"{dense_val:<12} | {metrics['Best_Epoch']:<12} | {metrics['Best_Validation_Accuracy']:<15.4f} | {metrics['Best_Validation_Loss']:<15.4f}")
    print("-" * 65)
    print(f"\n Results successfully saved to: {json_path}\n")

if __name__ == "__main__":
    run_dense_experiments()