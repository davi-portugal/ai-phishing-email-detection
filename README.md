# ai-phishing-email-detection
Phishing email detection using NLP, Logistic Regression, and explainable AI
# AI-Powered Phishing Email Detection and Explanation System

A machine learning project for detecting phishing emails using natural language processing and explainable AI techniques.

## Overview

This project builds a phishing email classifier using traditional NLP pipelines and compares multiple text-classification approaches:

- CountVectorizer + Multinomial Naive Bayes
- CountVectorizer + Logistic Regression
- TF-IDF + Logistic Regression

The final model also includes:

- global feature importance analysis
- single-email explanation output
- LIME-based local explanations

This project was developed as a final project for **CAP4630 – Introduction to Artificial Intelligence**.

## Problem

Phishing emails are a common cybersecurity threat. Attackers often use urgent language, suspicious links, and deceptive wording to trick users into sharing credentials or financial information.

The goal of this project is to classify emails as either:

- **Safe**
- **Phishing**

while also providing interpretable explanations for why a message was flagged.

## Dataset

The project uses the public Hugging Face dataset:

- `zefang-liu/phishing-email-dataset`

After preprocessing, the dataset used in the project contained:

- **18,629 total emails**
- **11,321 safe emails**
- **7,308 phishing emails**

## Methods

### Preprocessing

The email text is cleaned by:

- converting text to lowercase
- replacing URLs with `URL`
- replacing email addresses with `EMAIL`
- removing most punctuation
- removing empty rows after cleaning

### Models

Three pipelines are trained and compared:

1. **CountVectorizer + MultinomialNB**
2. **CountVectorizer + LogisticRegression**
3. **TfidfVectorizer + LogisticRegression**

### Explainability

The project includes two explanation layers:

- **global feature importance** using Logistic Regression coefficients
- **local explanations** using token-level contributions and **LIME**

## Results

### Experiment 1 – Classifier Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Count + Logistic Regression | 0.9659 | 0.9394 | 0.9761 | 0.9574 |
| Count + MultinomialNB | 0.9173 | 0.8893 | 0.9015 | 0.8954 |

### Experiment 2 – Vectorizer Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9681 | 0.9480 | 0.9720 | 0.9598 |
| Count + Logistic Regression | 0.9659 | 0.9394 | 0.9761 | 0.9574 |

### Best Model

**TF-IDF + Logistic Regression** achieved the best overall performance:

- **Accuracy:** 96.81%
- **Precision:** 94.80%
- **Recall:** 97.20%
- **F1-score:** 95.98%

## Example Phishing Indicators Found by the Model

Top features pushing predictions toward phishing included words such as:

- `click`
- `http`
- `money`
- `free`
- `email`
- `info`

This makes the model behavior easier to interpret in a cybersecurity context.

## Project Structure

```bash
ai-phishing-email-detection/
├── phishing_email_classifier.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python phishing_email_classifier.py
```

## Main Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- datasets
- lime

## Future Improvements

- test on additional phishing datasets for stronger generalization
- add ROC-AUC and precision-recall curves
- build a small web interface with Streamlit or Flask
- experiment with transformer-based models such as BERT
- reduce dataset-specific bias in the safe-email class

## Resume Value

This project demonstrates experience with:

- machine learning for cybersecurity
- NLP preprocessing and text classification
- model comparison and evaluation
- explainable AI techniques
- Python and scikit-learn pipelines

## Author

**Davi Portugal**  
CAP4630 Final Project
