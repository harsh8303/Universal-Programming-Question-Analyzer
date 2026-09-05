import os
import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. SETUP PATHS (Direct Colab Path)
# ==========================================
BASE_PATH = "/content/UPQA_Data"
PROCESSED_DIR = BASE_PATH
ARTIFACTS_DIR = BASE_PATH

# ==========================================
# 2. THE ATTENTION LAYER (Safe Setup)
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
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)
            e = tf.where(mask, e, -1e9)  
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ==========================================
# 3. LOAD DATA
# ==========================================
print("Loading Data from UPQA_Data folder...")
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
# 4. BUILD LSTM MODEL FOR KERASTUNER
# ==========================================
def build_model(hp):
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name="Input_Sequence")

    hp_embedding = hp.Choice('embedding_dim', values=[64, 128, 256])
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=hp_embedding, mask_zero=True)(inputs)

    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    hp_lstm1 = hp.Choice('lstm_units_1', values=[64, 128, 256])
    hp_lstm2 = hp.Choice('lstm_units_2', values=[32, 64, 128])

    # Explicit Dropout for stability
    x = Dropout(hp_dropout)(x)
    x = Bidirectional(LSTM(units=hp_lstm1, return_sequences=True))(x)
    
    x = Dropout(hp_dropout)(x)
    x = Bidirectional(LSTM(units=hp_lstm2, return_sequences=True))(x)
    
    x = AttentionLayer()(x)

    hp_dense = hp.Choice('dense_units', values=[32, 64, 128])
    x = Dense(hp_dense, activation='relu')(x)
    x = Dropout(hp_dropout)(x)

    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 3e-4, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 5. START HYPERBAND TUNING
# ==========================================
print(" Starting KerasTuner Hyperband Search for BiLSTM...")
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=12,
    factor=3,
    directory='/content',
    project_name='bilstm_tuning_project'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tuner.search(X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=12,
             batch_size=64,
             callbacks=[early_stopping])

print("\n" + "="*50)
print(" BEST LSTM HYPERPARAMETERS FOUND:")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Embedding Dim: {best_hps.get('embedding_dim')}")
print(f"LSTM 1 Units : {best_hps.get('lstm_units_1')}")
print(f"LSTM 2 Units : {best_hps.get('lstm_units_2')}")
print(f"Dense Units  : {best_hps.get('dense_units')}")
print(f"Dropout Rate : {best_hps.get('dropout')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print("="*50)

best_model = tuner.get_best_models(num_models=1)[0]
save_path = "/content/best_colab_bilstm.keras"
best_model.save(save_path)
print(f"Best LSTM Model saved at: {save_path}")