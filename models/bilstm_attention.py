# models/bilstm_attention.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

# ==========================================
# CUSTOM ATTENTION MECHANISM
# ==========================================
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='glorot_uniform', 
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================
# ADVANCED NEURAL NETWORK ARCHITECTURE
# ==========================================
def build_model(vocab_size, max_seq_length, num_classes, embedding_dim=128, lstm_units=64):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, 
                  output_dim=embedding_dim, 
                  mask_zero=True, 
                  name="Word_Embedding_Matrix")(inputs)
                  
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True, dropout=0.2), 
                      name="Deep_Bi_LSTM")(x)
                      
    x = AttentionLayer(name="Self_Attention_Mechanism")(x)
    
    x = Dense(64, activation='relu', name="Dense_Feature_Extractor")(x)
    x = Dropout(0.3, name="Dropout_Regularization")(x)
    
    outputs = Dense(num_classes, activation='softmax', name="Difficulty_Classifier")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Universal_Problem_Analyzer")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# ==========================================
# TRAINING VISUALIZATION
# ==========================================
def plot_training_history(history, models_dir):
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_path = os.path.join(models_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f" Training plot saved at: {plot_path}")

# ==========================================
# MAIN TRAINING PIPELINE & FINE-TUNING
# ==========================================
def train_model():
    print(" Starting Deep Learning Model Training...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    models_dir = os.path.join(ROOT_DIR, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load Configurations
    with open(os.path.join(artifacts_dir, "sequence_config.json"), 'r') as f:
        seq_config = json.load(f)
        
    vocab_size = seq_config["Vocabulary Size"]
    max_seq_length = seq_config["Maximum Sequence Length"]
    num_classes = len(seq_config["Difficulty Classes"])
    
    print("✓ Loading NumPy Arrays...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    print(f"   - Training Samples: {X_train.shape[0]}")
    print(f"   - Validation Samples: {X_val.shape[0]}")
    print(f"   - Test Samples: {X_test.shape[0]}")
    
    # Build Model
    model = build_model(vocab_size, max_seq_length, num_classes)
    model.summary()
    
    # Save Model Summary to TXT
    with open(os.path.join(models_dir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("✓ Model summary saved to model_summary.txt")
    
    # Callbacks for Automated Fine-Tuning
    best_model_path = os.path.join(models_dir, "best_bilstm_attention_model.keras")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    
    print("\n Commencing Neural Network Training (with automatic checkpointing)...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    print("\n Training Complete!")
    
    # Save Training History to JSON
    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(history.history, f)
    print("✓ Training history saved to training_history.json")
    
    # Plot History
    plot_training_history(history, models_dir)
    
    # Evaluate on Unseen Test Data
    print("\n Evaluating Model on Unseen Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"   - Final Test Accuracy: {test_acc:.4f}")
    print(f"   - Final Test Loss: {test_loss:.4f}")
    print("\n Model Training and Evaluation Fully Complete (85% Project Complete)!")

if __name__ == "__main__":
    train_model()