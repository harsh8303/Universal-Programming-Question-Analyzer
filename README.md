<p align="center">
  <img src="images/Banner.png" width="100%" alt="Universal Programming Question Analyzer">
</p>

<h1 align="center">🤖 Universal Programming Question Analyzer (UPQA)</h1>

<p align="center">
An End-to-End Multimodal AI Pipeline (NLP + Computer Vision) for Understanding Algorithmic Complexity
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/Hugging_Face-FFAA00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face">
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
</p>

---

## 📌 Project Overview

The **Universal Programming Question Analyzer (UPQA)** is an enterprise-grade, end-to-end **Multimodal AI system** designed to comprehend programming problems and instantly predict their algorithmic difficulty (Easy, Medium, Hard). 

Breaking past standard text classifiers, this project accepts both **raw text and coding screenshots**. It leverages a robust OCR engine for image processing, deep linguistic tokenization, and state-of-the-art Deep Learning models (including Transformers and Custom Attention Mechanisms) to capture the true complexity of algorithmic constraints.

---

## 🚀 Project Highlights

* **Massive Research-Grade Data:** Aggregated programming datasets from 9 global platforms (LeetCode, Codeforces, CodeChef, GeeksforGeeks) and advanced AI research datasets (DeepMind, TACO, APPS, Alpaca, Evol-Instruct).
* **Multimodal Input (Computer Vision):** Integrated OpenCV and EasyOCR to parse code screenshots in real-time, optimized with 1000px down-scaling and grayscale conversion for a ~20x inference speed boost.
* **Advanced Deep Learning Architectures:** Fine-tuned **DistilBERT (88.27% Accuracy)** alongside custom-built BiLSTM and BiGRU architectures optimized via KerasTuner.
* **Custom Mathematical Attention:** Engineered a bespoke Keras `AttentionLayer` to isolate and weigh critical algorithmic keywords and constraints within long problem descriptions.
* **Production-Ready API:** Packaged the entire NLP and CV pipeline into a dynamic Flask backend with a responsive web interface.

---

## 📊 Dataset Summary

| Feature | Value |
|---------|---------|
| Total Platforms/Sources | **9 Global Datasets** |
| Raw Problems Collected | **100,000+** |
| High-Quality Filtered Problems | **72,000+** |
| Unified Dataset Features | **19** |
| Vocabulary Size | **10,000+ Specialized Tokens** |
| Maximum Token Sequence Length | **128 - 512** |

---

## 🏗️ System Architecture Pipeline

<p align="center">
<img src="images/pipeline.png" width="100%" alt="Pipeline Architecture">
</p>

---

## 🔄 The 10-Step Pipeline Flow

### Phase 1: Data Engineering & Preprocessing
* **01. Data Collection:** Scraped and merged 100,000+ raw challenges via multi-source API connectors.
* **02. Data Engineering:** Unified 9 diverse schema structures into a single 19-feature master dataset.
* **03. NLP Cleaning:** Stripped HTML/Markdown noise while strictly preserving code logic, equations, and mathematical constraints.
* **04. Quality Analysis:** Audited semantic integrity to finalize 72,000+ high-quality problems.

### Phase 2: Sequence & Model Preparation
* **05. Tokenization:** Engineered a custom NLP tokenizer mapped to a 10,000+ word technical vocabulary.
* **06. Sequence Generation:** Converted textual problems into NumPy arrays with dynamic padding and truncation.
* **07. Train/Val/Test Split:** Applied stratified splitting (80/10/10) to maintain uniform difficulty distributions across the massive dataset.

### Phase 3: AI Modeling & Deployment
* **08. Deep Learning Models:** Trained DistilBERT, BiLSTM, and BiGRU networks. Achieved peak performance (88.27%) using Transformer architectures and custom attention mechanisms.
* **09. Image Processing (OCR):** Built a Computer Vision layer (`ocr_utils.py`) to extract clean text from UI screenshots.
* **10. Production API:** Deployed the full multimodal inference system locally via a Flask server (`app.py`).

---

## 💻 Tech Stack

* **Deep Learning:** TensorFlow, Keras, Hugging Face (Transformers)
* **Computer Vision:** OpenCV, EasyOCR
* **Data Engineering:** Pandas, NumPy, Scikit-Learn
* **Backend & Web:** Python, Flask, HTML/CSS
* **Visualization:** Matplotlib, Seaborn

---

> **Note on Git Best Practices:** To maintain repository health and comply with GitHub's 100MB file limits, massive binary files (including `.keras` checkpoints, `.npy` arrays, and DistilBERT weights) are intentionally ignored via `.gitignore`. This repository focuses purely on production code, training logic, and architectural design.

---

## 👨‍💻 Author

**Harshit Sahu**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/harsh8303)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harshit-sahu-67119530a/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:harshitsahu8303@gmail.com)

---
<p align="center">
 If you found this enterprise-scale AI architecture useful, consider giving it a ⭐.
</p>