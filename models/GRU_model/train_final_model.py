# models/GRU_model/train_final_model.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
#  SEEDS FOR REPRODUCIBILITY 
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
#  THE FINAL "ALL-STAR" HYPERPARAMETERS
# ==========================================
LEARNING_RATE = 5e-4
EMBEDDING_DIM = 128
GRU_UNITS_1 = 128
GRU_UNITS_2 = 64
DROPOUT_RATE = 0.5
DENSE_UNITS = 64
# ==========================================

def build_final_model(vocab_size, max_seq_length, num_classes):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)
    x = Bidirectional(GRU(units=GRU_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = Bidirectional(GRU(units=GRU_UNITS_2, return_sequences=True, dropout=DROPOUT_RATE))(x)
    x = AttentionLayer()(x)
    
    x = Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate():
    print(" STARTING FINAL MODEL TRAINING...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    
    with open(os.path.join(artifacts_dir, "sequence_config.json"), 'r') as f:
        seq_config = json.load(f)
        
    vocab_size = seq_config["Vocabulary Size"]
    max_seq_length = seq_config["Maximum Sequence Length"]
    num_classes = len(seq_config["Difficulty Classes"])
    target_names = seq_config["Difficulty Classes"]
    
    print(" Loading ALL Data (Train, Val, TEST)...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    #  LOADING THE UNSEEN TEST SET FOR THE FIRST TIME 
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    model = build_final_model(vocab_size, max_seq_length, num_classes)
    model.summary()
    
    # Callbacks: Early Stopping + Saving the Best Model
    model_save_path = os.path.join(CURRENT_DIR, "best_bigru_model.keras")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    
    print("\n TRAINING IN PROGRESS...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, # Let it run until early stopping kicks in
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    print("\n" + "="*50)
    print(" TRAINING COMPLETE! EVALUATING ON UNSEEN TEST SET...")
    print("="*50)
    
    # Evaluate on Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f" FINAL TEST LOSS: {test_loss:.4f}\n")
    
    # Generate Predictions for Classification Report
    print(" Generating Detailed Classification Report & Confusion Matrix...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    print("\n" + "-"*50)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("-"*50)
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    print("\n" + "-"*50)
    print("CONFUSION MATRIX")
    print("-"*50)
    print(confusion_matrix(y_test, y_pred_classes))
    
    print(f"\n Model successfully saved at: {model_save_path}")
    print(" You are now ready to build the inference API!")

if __name__ == "__main__":
    train_and_evaluate()