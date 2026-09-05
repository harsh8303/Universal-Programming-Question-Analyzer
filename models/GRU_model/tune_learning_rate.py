# models/GRU_model/tune_learning_rate.py
import os
import sys
import json
import numpy as np
import tensorflow as tf

# ==========================================
#  SEEDS FOR REPRODUCIBILITY (Crucial for scientific tuning)
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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # models/GRU_model/
MODELS_DIR = os.path.dirname(CURRENT_DIR)                 # models/
ROOT_DIR = os.path.dirname(MODELS_DIR)                    # UPQA_Project main folder/

# Add paths so Python can find config.py and attention_layer.py
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MODELS_DIR)

import config
from attention_layer import AttentionLayer 

# ==========================================
#  EXPERIMENT SETUP: THE LOOP VALUES
# ==========================================
# CHANGE: Learning Rate
LEARNING_RATES_TO_TEST = [1e-3, 5e-4, 3e-4, 1e-4]

# FIXED PARAMETERS (Strictly untouched for this experiment)
GRU_UNITS_1 = 64
GRU_UNITS_2 = 32
DROPOUT_RATE = 0.3
EMBEDDING_DIM = 64
# ==========================================

def build_model(vocab_size, max_seq_length, num_classes, current_lr):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)
    x = Bidirectional(GRU(units=GRU_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = Bidirectional(GRU(units=GRU_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = AttentionLayer()(x)
    
    x = Dense(GRU_UNITS_2, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_lr_experiments():
    print("🚀 STARTING AUTOMATED LEARNING RATE TUNING LOOP...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    
    with open(os.path.join(artifacts_dir, "sequence_config.json"), 'r') as f:
        seq_config = json.load(f)
        
    vocab_size = seq_config["Vocabulary Size"]
    max_seq_length = seq_config["Maximum Sequence Length"]
    num_classes = len(seq_config["Difficulty Classes"])
    
    print(" Loading Validation & Training Data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    # Note: Test set intentionally NOT loaded.
    
    # Dictionary to store final results for JSON export
    experiment_results = {}
    
    for lr in LEARNING_RATES_TO_TEST:
        print("\n" + "="*60)
        print(f" TESTING LEARNING RATE: {lr}")
        print("="*60)
        
        # Clear session to prevent memory leak
        tf.keras.backend.clear_session()
        
        # Reset seeds inside the loop to ensure exactly same weight initialization for every LR!
        np.random.seed(42)
        tf.random.set_seed(42)
        
        model = build_model(vocab_size, max_seq_length, num_classes, lr)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate best metrics (converting to native Python types for JSON saving)
        best_val_loss = float(min(history.history['val_loss']))
        best_val_acc = float(max(history.history['val_accuracy']))
        best_epoch = int(np.argmin(history.history['val_loss']) + 1)
        
        # Store in dict
        lr_key = str(lr)
        experiment_results[lr_key] = {
            'Best_Validation_Loss': best_val_loss,
            'Best_Validation_Accuracy': best_val_acc,
            'Best_Epoch': best_epoch
        }
    
    # ==========================================
    #  SAVE RESULTS TO JSON
    # ==========================================
    json_path = os.path.join(CURRENT_DIR, "learning_rate_tuning_results.json")
    with open(json_path, 'w') as json_file:
        json.dump(experiment_results, json_file, indent=4)
        
    # ==========================================
    #  FINAL SUMMARY REPORT
    # ==========================================
    print("\n\n" + "*"*25)
    print("      FINAL EXPERIMENT REPORT")
    print("*"*25)
    print(f"{'Learning Rate':<15} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Best Val Loss':<15}")
    print("-" * 65)
    for lr_key, metrics in experiment_results.items():
        print(f"{lr_key:<15} | {metrics['Best_Epoch']:<12} | {metrics['Best_Validation_Accuracy']:<15.4f} | {metrics['Best_Validation_Loss']:<15.4f}")
    print("-" * 65)
    print(f"\n Results successfully saved to: {json_path}\n")

if __name__ == "__main__":
    run_lr_experiments()