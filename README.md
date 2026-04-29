# 🇮🇳 Hindi Tweet Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Gradio](https://img.shields.io/badge/Deployment-Gradio-yellow?logo=gradio)

---

## 🚀 Project Overview

This project performs **sentiment analysis on Hindi tweets**, classifying them as:

* 😊 **Positive**
* 😡 **Negative**

It leverages **Natural Language Processing (NLP)** and **Machine Learning models** to deliver accurate predictions.

💡 The best-performing model is deployed using an **interactive Gradio web interface** for real-time predictions.

---

## ✨ Features

* 🔍 Binary sentiment classification (Positive vs Negative)
* 🧹 Text preprocessing pipeline
* 📊 TF-IDF vectorization for feature extraction
* 🤖 Multiple ML models for comparison:

  * Support Vector Machine (SVM)
  * Random Forest
  * Naive Bayes
  * Logistic Regression
* 📈 Model evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC Curve
  * Confusion Matrix
* 🔁 Cross-validation for best model selection
* 🌐 Gradio deployment for real-time predictions

---

## 📂 Dataset Description

| Column       | Description                 |
| ------------ | --------------------------- |
| 📝 tweet     | Hindi tweet text            |
| 🎯 sentiment | Label (Positive / Negative) |

---

## ⚙️ Methodology

### 🧹 Data Preprocessing

1. Remove neutral tweets
2. Convert labels:

   * Positive → **1**
   * Negative → **0**
3. Train-test split:

   * 80% Training
   * 20% Testing

---

### 🧠 Feature Extraction

* TF-IDF Vectorization
* `max_features = 5000`

---

### 🤖 Model Training & Evaluation

Models trained:

![SVM](https://img.shields.io/badge/SVM-Model-purple?style=for-the-badge\&logo=scikitlearn)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Model-green?style=for-the-badge\&logo=tree)
![Naive Bayes](https://img.shields.io/badge/Naive%20Bayes-Model-blue?style=for-the-badge\&logo=python)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Model-orange?style=for-the-badge\&logo=databricks)

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve
* Confusion Matrix

---

### 🏆 Best Model Selection

* Cross-validation performed
* Best-performing model selected
* Final model trained on full dataset

---

## 🌐 Deployment

Interactive **Gradio interface**:

* ✍️ Input a Hindi tweet
* ⚡ Get instant sentiment prediction

---

## ▶️ How to Run

```bash
# Clone the repository
git clone <repository_link>

# Install dependencies
pip install -r requirements.txt

# Run the notebook / app
python Final_Hindi_Sentiment_Analysis1.ipynb
```

---

## 🛠️ Tech Stack

* 🐍 Python 3.x
* 📊 Pandas, NumPy
* 🤖 Scikit-learn
* 📉 Matplotlib, Seaborn
* 🧠 TF-IDF Vectorizer
* 🌐 Gradio

---

## 🌟 Key Takeaways

* NLP can effectively analyze regional language sentiment 🇮🇳
* Model comparison helps identify the best approach 🎯
* Deployment bridges the gap between ML and real-world use 🚀

---

## 🔮 Future Improvements

* 🤖 Deep Learning models (LSTM, BERT)
* 🌍 Multilingual sentiment analysis
* 📡 Real-time streaming tweet analysis

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ and sharing it!
