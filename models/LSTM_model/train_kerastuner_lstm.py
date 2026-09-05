# models/LSTM_model/train_kerastuner_lstm.py
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

# ==========================================
#  SEEDS FOR REPRODUCIBILITY 
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
#  FOLDER PATH SETUP 
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.dirname(CURRENT_DIR)                 
ROOT_DIR = os.path.dirname(MODELS_DIR)                    

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

import config

# ==========================================
# 1. MAC-SAFE CUSTOM ATTENTION MECHANISM
# ==========================================
@tf.keras.utils.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='glorot_uniform', 
                                 trainable=True)
        seq_len = input_shape[1]
        bias_shape = (seq_len, 1) if seq_len is not None else (1,)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=bias_shape, 
                                 initializer='zeros', 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)
            # The Keras 3 safe fix to prevent BroadcastTo errors on Mac
            e = tf.where(mask, e, -1e9)  
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        return None
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================
# 2. COLAB WINNING HYPERPARAMETERS
# ==========================================
BEST_EMBEDDING_DIM = 64
BEST_LSTM_1 = 128
BEST_LSTM_2 = 128
BEST_DENSE = 32
BEST_DROPOUT = 0.2
LEARNING_RATE = 0.001
# ==========================================

def build_model(vocab_size, max_seq_length, num_classes):
    inputs = Input(shape=(max_seq_length,), name="Input_Sequence")
    
    x = Embedding(input_dim=vocab_size, output_dim=BEST_EMBEDDING_DIM, mask_zero=True, name="Word_Embedding_Matrix")(inputs)
    
    # Explicit Dropouts to bypass Keras 3 RNN masking bugs
    x = Dropout(BEST_DROPOUT, name="Dropout_1")(x)
    x = Bidirectional(LSTM(units=BEST_LSTM_1, return_sequences=True), name="BiLSTM_Layer_1")(x)
    
    x = Dropout(BEST_DROPOUT, name="Dropout_2")(x)
    x = Bidirectional(LSTM(units=BEST_LSTM_2, return_sequences=True), name="BiLSTM_Layer_2")(x)
    
    x = AttentionLayer(name="Self_Attention_Mechanism")(x)
    
    x = Dense(BEST_DENSE, activation='relu', kernel_regularizer=l2(0.001), name="Dense_Extractor")(x)
    x = Dropout(BEST_DROPOUT, name="Dropout_Regularization_Final")(x)
    
    outputs = Dense(num_classes, activation='softmax', name="Difficulty_Classifier")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Universal_Problem_Analyzer_KerasTuner_BiLSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('KerasTuner BiLSTM Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
    plt.title('KerasTuner BiLSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(save_dir, "kerastuner_bilstm_training_history.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f" Training plot saved at: {plot_path}")

def train_final_lstm():
    print("\n STARTING FINAL DEEP BiLSTM TRAINING (Colab Hyperparameters)...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    save_dir = CURRENT_DIR
    
    with open(os.path.join(artifacts_dir, "sequence_config.json"), 'r') as f:
        seq_config = json.load(f)
        
    vocab_size = seq_config["Vocabulary Size"]
    max_seq_length = seq_config["Maximum Sequence Length"]
    num_classes = len(seq_config["Difficulty Classes"])
    
    print(" Loading Numpy Arrays...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    model = build_model(vocab_size, max_seq_length, num_classes)
    model.summary()
    
    with open(os.path.join(save_dir, "kerastuner_bilstm_model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    best_model_path = os.path.join(save_dir, "best_kerastuner_bilstm.keras")
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, mode='min', verbose=1)
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    print("\n Commencing Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, 
        batch_size=64, 
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    print("\n Training Complete!")
    
    with open(os.path.join(save_dir, "kerastuner_bilstm_training_history.json"), "w") as f:
        json.dump(history.history, f)
        
    plot_training_history(history, save_dir)
    
    print("\n Evaluating Final KerasTuner BiLSTM Model on Unseen Test Set...")
    best_model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=1)
    print("\n" + "*"*40)
    print(f" Final Test Accuracy : {test_acc:.4f}")
    print(f" Final Test Loss     : {test_loss:.4f}")
    print("*"*40)

if __name__ == "__main__":
    train_final_lstm()