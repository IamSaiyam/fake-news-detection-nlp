# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview
The spread of fake news across digital media platforms has become a significant challenge, impacting public opinion and trust in journalism. This project implements a **cost-free, explainable fake news detection system** using **classical Natural Language Processing (NLP)** techniques and **machine learning models** to classify news articles as **Fake** or **Factual** based solely on their textual content.

The focus is on building a solution that is:
- Fully open-source
- Reproducible and leakage-safe
- Interpretable and easy to extend

---

## 🎯 Problem Statement
Fake news articles often imitate legitimate journalism while subtly differing in:
- language tone and emotional intensity,
- word choice and repetition,
- use of named entities and political framing.

The challenge is to **automatically identify these linguistic patterns** without relying on external fact-checking services, paid APIs, or proprietary models.

---

## 🧠 Solution Approach
This project combines multiple NLP and ML techniques into a single end-to-end pipeline:

- **Exploratory Text Analysis**
  - Part-of-Speech (POS) tagging
  - Named Entity Recognition (NER)
  - Token and n-gram frequency analysis

- **Text Preprocessing**
  - Safe normalization and cleaning
  - Lemmatization and stopword removal using spaCy

- **Sentiment Analysis**
  - Rule-based sentiment scoring using VADER

- **Topic Modeling**
  - Latent Dirichlet Allocation (LDA) with coherence-based topic selection

- **Supervised Classification**
  - TF-IDF feature extraction (unigrams & bigrams)
  - Logistic Regression and linear SVM (SGDClassifier)
  - Stratified cross-validation and hyperparameter tuning

All steps are implemented using **scikit-learn pipelines** to prevent data leakage and ensure reliable evaluation.

---

## ⚙️ How It Works
1. Raw news article text is cleaned and normalized  
2. Text is tokenized and lemmatized using spaCy  
3. Features are extracted using TF-IDF  
4. Models are trained and tuned via stratified cross-validation  
5. The final model predicts whether an article is *Fake* or *Factual*

---

## 📊 Results
- **Best Model**: TF-IDF + Logistic Regression  
- **Evaluation Strategy**: Stratified k-fold cross-validation + held-out test set  
- **Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix

The final model achieves **strong and stable performance**, demonstrating that well-engineered classical NLP pipelines can effectively detect fake news without expensive or opaque models.

---

## 🧪 Tech Stack
- **Python**
- **spaCy** – tokenization, lemmatization, POS, NER
- **NLTK** – auxiliary NLP utilities
- **VADER** – sentiment analysis
- **Gensim** – topic modeling (LDA, LSI)
- **Scikit-learn** – feature extraction, modeling, evaluation
- **Matplotlib / Seaborn** – data visualization

---

## 📁 Project Structure
```text
├── data/
│   └── fake_news_data.csv
├── notebooks/
│   └── fake_news_detection.ipynb
├── models/
│   └── best_fake_news_pipeline.joblib
├── .gitignore
├── README.md
