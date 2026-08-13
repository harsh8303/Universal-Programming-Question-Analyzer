# models/bilstm_attention_v2.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
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
# CUSTOM ATTENTION MECHANISM (Fixed Bias Shape)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='glorot_uniform', 
                                 trainable=True)
        # Fixed shape to (1,) for proper broadcasting
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(1,), 
                                 initializer='zeros', 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            e = e - (1.0 - mask) * 1e9  
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        return None
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================
# OPTIMIZED V2 ARCHITECTURE
# ==========================================
def build_model(vocab_size, max_seq_length, num_classes, embedding_dim=64, lstm_units=32):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="Word_Embedding_Matrix")(inputs)
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True, dropout=0.4), name="Deep_Bi_LSTM")(x)
    x = AttentionLayer(name="Self_Attention_Mechanism")(x)
    
    # Dense reduced to 32, Dropout increased to 0.5, L2 set to 0.001
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001), name="Dense_Feature_Extractor")(x)
    x = Dropout(0.5, name="Dropout_Regularization")(x)
    
    outputs = Dense(num_classes, activation='softmax', name="Difficulty_Classifier")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Universal_Problem_Analyzer_V2")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history, models_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (V2)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (V2)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_path = os.path.join(models_dir, "training_history_v2.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f" Training plot saved at: {plot_path}")

def train_model():
    print(" Starting Deep Learning Model Training (V2 Optimized)...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
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
    
    model = build_model(vocab_size, max_seq_length, num_classes)
    model.summary()
    
    with open(os.path.join(models_dir, "model_summary_v2.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    best_model_path = os.path.join(models_dir, "best_bilstm_attention_v2.keras")
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    
    print("\n Commencing Neural Network Training (V2)...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    print("\n Training Complete!")
    
    with open(os.path.join(models_dir, "training_history_v2.json"), "w") as f:
        json.dump(history.history, f)
        
    plot_training_history(history, models_dir)
    
    print("\n Loading Best Model for Final Evaluation...")
    best_model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    
    print(" Evaluating V2 Model on Unseen Test Set...")
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"   - Final Test Accuracy (V2): {test_acc:.4f}")
    print(f"   - Final Test Loss (V2): {test_loss:.4f}")

if __name__ == "__main__":
    train_model()