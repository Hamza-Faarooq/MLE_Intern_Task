# MLE_Intern_Task

# Twitter Sentiment Analysis – ML Engineer Internship Assignment

## 📌 Overview
This project classifies tweets from the Sentiment140 dataset as **Positive**, **Negative**, or **Neutral** using multiple machine learning and deep learning models. It is built as part of an internship application for the ML Engineer role.

## 📁 Dataset
- Source: [Sentiment140](http://help.sentiment140.com/for-students)
- Contains 1.6 million tweets labeled as:
  - `0` → Negative
  - `2` → Neutral
  - `4` → Positive

## 🧹 Data Preprocessing
- Lowercasing
- URL, hashtag, and mention removal
- Non-alphabetic character removal
- Stopword removal
- Lemmatization

## 📊 EDA
- Sentiment distribution plot
- Tweet length distribution
- Most common words in each sentiment class

## 🧠 Models Used
| Model               | Description                          |
|--------------------|--------------------------------------|
| Logistic Regression| Baseline linear classifier           |
| Naive Bayes        | Probabilistic model                  |
| Random Forest (Fast)| Optimized for large dataset training|
| DNN (TensorFlow)   | Deep Neural Network with Dropout     |

## 🧪 Evaluation
- Accuracy, confusion matrix, and classification report for each model
- Side-by-side model comparison plot

## 🖼️ Visualizations
- Accuracy bar chart
- Confusion matrix heatmaps

## 📈 Sample Result
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 82.4%    |
| Random Forest      | 79.2%    |
| Naive Bayes        | 77.5%    |
| DNN                | 81.8%    |

> *Note: Replace with actual results from your run.*

## 🚀 How to Run

### Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
