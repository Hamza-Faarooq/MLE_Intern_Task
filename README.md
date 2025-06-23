# 🐦 Twitter Sentiment Analysis – ML Engineer Internship Assignment



## Files attached in the Repository
- Jupyter Notebook (.ipynb file) having my code with output
- Python code file
- Evaluation report



## 📌 Problem Statement
Social media platforms like Twitter are filled with opinions, feedback, and emotions. The task is to **build a model that can classify the sentiment of tweets** as:

- **Positive**
- **Negative**
- **Neutral**

This can help businesses monitor their brand, gather real-time feedback, and make informed decisions based on public opinion.

---

## 📂 Dataset Used
**Sentiment140** dataset:  
- ✅ [Download link]((https://www.kaggle.com/datasets/kazanova/sentiment140))
- ✅ Contains **1.6 million tweets**  
- ✅ Each tweet is labeled as:
  - `0`: Negative
  - `2`: Neutral
  - `4`: Positive

We map them to `Negative`, `Neutral`, and `Positive` classes respectively for easier understanding.

---

## ⚙️ Project Pipeline Overview

1. **Data Loading**  
   Reads the large CSV file with correct encoding and column naming.

2. **Data Preprocessing**  
   Each tweet is cleaned to make it ready for machine learning. Steps include:
   - Lowercasing text
   - Removing links, hashtags, usernames
   - Removing non-alphabetic characters
   - Tokenizing words
   - Removing common stopwords like "the", "is", etc.
   - Lemmatizing words (e.g., "running" → "run")

3. **Exploratory Data Analysis (EDA)**  
   - Bar chart showing number of tweets in each sentiment class
   - Distribution of tweet lengths
   - Top 10 most frequent words in each sentiment category

4. **Feature Extraction**  
   - Uses **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
   - Reduces input space to top 5000 most informative words.

5. **Model Building**  
   ✅ Trained and evaluated **four models**:
   - **Logistic Regression**: Simple yet effective linear model
   - **Naive Bayes**: Fast and great for text
   - **Random Forest (Fast Mode)**: Ensemble tree model optimized for speed
   - **Deep Neural Network (DNN)**: Memory-optimized deep learning model with dropout layers

6. **Evaluation**
   - Accuracy score for each model
   - Confusion matrices
   - Classification reports (precision, recall, F1-score)
   - Bar chart comparing all models

---

## 🤖 Models in Detail

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Logistic Regression** | A linear classifier for multi-class problems | Fast, interpretable | May underfit complex data |
| **Naive Bayes** | Uses probabilities and Bayes' theorem | Very fast | Assumes feature independence |
| **Random Forest (Fast Mode)** | Ensemble of decision trees with shallow depth | Good performance, optimized speed | Less accurate than full version |
| **Deep Neural Network (DNN)** | Multi-layer perceptron with dropout | Captures complex patterns | Slower, memory intensive |



### 🧠 Deep Neural Network (DNN)

We trained a memory-efficient feedforward neural network using **TensorFlow** and `tf.data` API to handle the 1.6 million tweet records from Sentiment140.

#### ✅ Features:
- Dense architecture with dropout regularization
- Uses `tf.data.Dataset.from_generator()` to avoid loading entire dataset into RAM
- Converts sparse TF-IDF vectors to dense only in batches, preserving memory
- Runs 10x faster than earlier row-wise implementations

#### 🔧 Model Architecture:
Input → Dense(128, relu) → Dropout(0.3)
→ Dense(64, relu) → Dropout(0.2)
→ Output Layer (Softmax)


#### 🔁 Optimizations:
- Batch size = 64
- Early stopping on validation accuracy
- GPU memory growth configured
- Only `clear_session()` before training (not after every batch)

> This DNN model balances speed, accuracy, and memory usage for large-scale sentiment classification.


---

## 🧪 Model Evaluation Summary

| Model               | Accuracy (%) |
|--------------------|--------------|
| Logistic Regression| 77.47%        |
| Random Forest      | 64.09%        |
| Naive Bayes        | 75.77%        |
| DNN                | 78.56%        |

Also includes:
- Confusion Matrix: See how well each class is predicted.
- Classification Report: Includes **precision, recall, F1-score** for each class.

---

---

## 💻 How to Run

### 🛠 Requirements
Install required Python packages with:

```bash
pip install -r requirements.txt
