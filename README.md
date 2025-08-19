# Hindi-Sentiment-Analysis

Hindi Tweet Sentiment Analysis
Project Overview

This project predicts the sentiment of Hindi tweets as Positive or Negative using machine learning techniques. It combines text preprocessing, feature extraction with TF-IDF, and model training using multiple classifiers, including SVM, Random Forest, Naive Bayes, and Logistic Regression. The best performing model is deployed with a Gradio web interface for real-time predictions.

Features

Binary sentiment classification: Positive vs Negative

Text preprocessing with TF-IDF vectorization

Model evaluation using accuracy, precision, recall, F1-score, ROC curve, and confusion matrix

Comparison of multiple classifiers for performance analysis

Cross-validation to select the best model

Deployment using Gradio for interactive predictions

Dataset

Dataset: Custom Hindi sentiment analysis CSV

Columns:

tweet – Text of the Hindi tweet

sentiment – Sentiment label (positive or negative)

Neutral tweets are excluded to focus on binary classification

Methodology

Data Preprocessing

Remove neutral sentiments

Convert labels to binary: Positive = 1, Negative = 0

Split dataset into 80% training and 20% testing

Feature Extraction

TF-IDF vectorization (max_features=5000)

Model Training & Evaluation

Train multiple classifiers:

Support Vector Machine (SVM)

Random Forest

Multinomial Naive Bayes

Logistic Regression

Evaluate using accuracy, precision, recall, F1-score

ROC curves and confusion matrices plotted

Best Model Selection

Perform cross-validation to select the best classifier

Train the best model on full training data

Deployment

Gradio interface allows input of a Hindi tweet and outputs predicted sentiment

How to Run

Clone the repository

git clone <repository_link>


Install required packages

pip install -r requirements.txt


Launch Gradio interface

python Final_Hindi_Sentiment_Analysis1.ipynb


Enter a Hindi tweet and get sentiment prediction

Technologies & Libraries

Python 3.x

Pandas, NumPy

Scikit-learn (SVM, Random Forest, Naive Bayes, Logistic Regression)

Matplotlib, Seaborn

TF-IDF Vectorizer for feature extraction

Gradio for deployment

Future Improvements

Include neutral sentiment classification

Support code-mixed Hindi-English tweets
