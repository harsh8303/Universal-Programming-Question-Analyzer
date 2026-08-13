import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification



# Folders ka path
MODEL_PATH = os.path.join("models", "best_distilbert")
ENCODER_PATH = os.path.join("artifacts", "difficulty_encoder.pkl")

# 1. Trained Model aur Tokenizer load karna
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# 2. Label Encoder load karna (taaki numbers wapas text me convert ho sakein)
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

def ask_ai(question_text):
    # Text ko model ke samajhne layag (tokens) banana
    inputs = tokenizer(question_text, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
    
    # Model se prediction lena
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    
    # Sabse high probability wala label nikalna
    class_id = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    predicted_label = encoder.inverse_transform([class_id])[0]
    return predicted_label, confidence

# ==========================================
# YAHAN APNA QUESTION DAAL KAR TEST KAR!
# ==========================================
sample_dsa_question = "Write a function to find the longest common substring in two given strings using dynamic programming."

print("\n" + "="*50)
print(f" QUESTION: {sample_dsa_question}")
print("="*50)

label, conf = ask_ai(sample_dsa_question)

print(f" AI PREDICTION: {label}")
print(f" CONFIDENCE SCORE: {conf*100:.2f}%\n")