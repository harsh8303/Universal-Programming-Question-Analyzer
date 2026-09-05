import tensorflow as tf

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
            # Keras 3 Safe Masking (Works for both GRU & LSTM)
            e = tf.where(mask, e, -1e9)  
            
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
        
    def compute_mask(self, inputs, mask=None):
        return None
        
    def get_config(self):
        return super(AttentionLayer, self).get_config()