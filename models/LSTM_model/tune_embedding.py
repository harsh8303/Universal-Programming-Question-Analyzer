import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. SETUP PATHS
# ==========================================
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
SAVE_DIR = "models/LSTM_model"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. THE ATTENTION LAYER (Keras 3 Safe for Mac)
# ==========================================
@tf.keras.utils.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                 initializer='glorot_uniform', trainable=True)
        seq_len = input_shape[1]
        bias_shape = (seq_len, 1) if seq_len is not None else (1,)
        self.b = self.add_weight(name='attention_bias', shape=bias_shape, 
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)
            
            e = tf.where(mask, e, -1e9)  
            
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        return None

# ==========================================
# 3. LOAD DATA
# ==========================================
print(" Loading Data...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

with open(os.path.join(ARTIFACTS_DIR, "sequence_config.json"), 'r') as f:
    seq_config = json.load(f)

VOCAB_SIZE = seq_config["Vocabulary Size"]
MAX_SEQ_LENGTH = seq_config["Maximum Sequence Length"] 
NUM_CLASSES = len(seq_config["Difficulty Classes"])

# ==========================================
# 4. FIXED DEFAULTS & TUNING VALUES
# ==========================================
# Constant values while we tune Embedding
DEFAULT_LSTM1 = 64
DEFAULT_LSTM2 = 32
DEFAULT_DENSE = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_LR = 5e-4
EPOCHS = 10
BATCH_SIZE = 64

# Values to test
embedding_dims_to_test = [64, 128, 256]
results = {}

# ==========================================
# 5. TUNING LOOP
# ==========================================
for emb_dim in embedding_dims_to_test:
    print(f"\n{'='*50}")
    print(f" Testing Embedding Dimension: {emb_dim}")
    print(f"{'='*50}")
    
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="Input_Sequence")
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=emb_dim, mask_zero=True)(inputs)
    x = Bidirectional(LSTM(units=DEFAULT_LSTM1, return_sequences=True, dropout=DEFAULT_DROPOUT))(x)
    x = Bidirectional(LSTM(units=DEFAULT_LSTM2, return_sequences=True, dropout=DEFAULT_DROPOUT))(x)
    x = AttentionLayer()(x)
    x = Dense(DEFAULT_DENSE, activation='relu')(x)
    x = Dropout(DEFAULT_DROPOUT)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LR),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        callbacks=[early_stopping],
                        verbose=1)
                        
    val_acc = max(history.history['val_accuracy'])
    results[f"Embedding_{emb_dim}"] = float(val_acc)
    print(f" Best Val Accuracy for Embedding {emb_dim}: {val_acc:.4f}")

# ==========================================
# 6. SAVE RESULTS TO JSON
# ==========================================
json_path = os.path.join(SAVE_DIR, "embedding_tuning_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\n Tuning results saved to {json_path}")