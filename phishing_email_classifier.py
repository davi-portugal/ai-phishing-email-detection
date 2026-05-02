# AI-Powered Phishing Email Detection and Explanation System
# CAP4630 Final Project Davi Portugal

# !pip install -q datasets pandas numpy scikit-learn matplotlib lime

# Imports
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore")


# Load dataset
print("Loading dataset...")
ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
df = ds.to_pandas()

print("\nColumns:", df.columns.tolist())
print("Original shape:", df.shape)
print(df.head())


# Auto-detect text and label columns
def choose_column(columns, keywords):
    """Return the first column whose name contains any of the given keywords."""
    for keyword in keywords:
        for col in columns:
            if keyword in col.lower():
                return col
    return None

text_col  = choose_column(df.columns, ["text", "email", "body", "message", "content"])
label_col = choose_column(df.columns, ["label", "type", "class", "category"])

if text_col is None or label_col is None:
    raise ValueError(f"Could not detect text/label columns. Found: {df.columns.tolist()}")

print(f"\nDetected text column : {text_col}")
print(f"Detected label column: {label_col}")


# Label normalization
# Maps heterogeneous label representations → 1 (phishing) / 0 (safe).

PHISHING_VALUES = {"phishing", "phishing email", "spam", "malicious", "1", "true", "yes"}
SAFE_VALUES     = {"safe", "safe email", "ham", "legitimate", "0", "false", "no"}

def map_label(value):
    s = str(value).strip().lower()
    if s in PHISHING_VALUES:
        return 1
    if s in SAFE_VALUES:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return np.nan


# Text cleaning
# Intentionally keeps `!`, `?`, `$` — these carry phishing signal.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ",  text)   # anonymise URLs
    text = re.sub(r"\S+@\S+",          " EMAIL ", text)   # anonymise addresses
    text = re.sub(r"[^a-z0-9\s!?$]",  " ",       text)   # drop other punctuation
    text = re.sub(r"\s+",              " ",       text).strip()
    return text


# Build clean DataFrame
df = df[[text_col, label_col]].copy()
df.columns = ["text_raw", "label_raw"]

df["label"] = df["label_raw"].apply(map_label)

# Log and drop rows with unrecognised labels
unknown_mask = df["label"].isna()
if unknown_mask.any():
    unknown_vals = df.loc[unknown_mask, "label_raw"].unique()
    print(f"\nDropping {unknown_mask.sum()} rows with unrecognized labels: {unknown_vals}")

df = df.dropna(subset=["text_raw", "label"]).copy()
df["label"]      = df["label"].astype(int)
df["text_clean"] = df["text_raw"].apply(clean_text)

# Remove rows that are empty after cleaning (e.g. symbol-only emails)
df = df[df["text_clean"].str.len() > 0].copy()

print("\nCleaned dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())


# Train / test split
# Split raw and clean together to guarantee index alignment.

X_raw   = df["text_raw"]
X_clean = df["text_clean"]
y       = df["label"]

X_train_raw, X_test_raw, X_train, X_test, y_train, y_test = train_test_split(
    X_raw, X_clean, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# Evaluation helper
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Fit pipeline and return a metrics dict plus the trained pipeline."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        "model":     name,
        "accuracy":  accuracy_score (y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score   (y_test, y_pred, zero_division=0),
        "f1":        f1_score       (y_test, y_pred, zero_division=0),
    }, pipeline


# Define all pipelines once
# Each pipeline is constructed and trained exactly once,
# then shared across all experiment result tables.

PIPELINES = {
    "Count + MultinomialNB": Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english", max_features=5000)),
        ("classifier", MultinomialNB()),
    ]),
    "Count + LogisticRegression": Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "TFIDF + LogisticRegression": Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
}


# Train all models once
trained_models = {}
all_results    = []

for name, pipeline in PIPELINES.items():
    result, trained = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
    trained_models[name] = trained
    all_results.append(result)

# Master results table (sorted once; experiment tables filter from this)
results_df = (
    pd.DataFrame(all_results)
    .sort_values("f1", ascending=False)
    .reset_index(drop=True)
)


# EXPERIMENT 1 — Classifier comparison (CountVec)
# Hypothesis: LR should outperform NB by modelling feature interactions.

exp1_names = ["Count + MultinomialNB", "Count + LogisticRegression"]
exp1_df = results_df[results_df["model"].isin(exp1_names)].reset_index(drop=True)

print("\n================ EXPERIMENT 1 =================")
print("Classifier Comparison with CountVectorizer")
print("Hypothesis: LR outperforms NB by modelling feature interactions")
print(exp1_df.to_string(index=False))


# EXPERIMENT 2 — Vectorizer comparison (LR)
# Hypothesis: TF-IDF outperforms raw counts by down-weighting
# high-frequency words that appear equally in both classes.

exp2_names = ["Count + LogisticRegression", "TFIDF + LogisticRegression"]
exp2_df = results_df[results_df["model"].isin(exp2_names)].reset_index(drop=True)

print("\n================ EXPERIMENT 2 =================")
print("Vectorizer Comparison with Logistic Regression")
print("Hypothesis: TF-IDF outperforms counts by penalising common words")
print(exp2_df.to_string(index=False))


# Best overall model
print("\n================ ALL RESULTS =================")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
best_model      = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")


# Confusion matrix + classification report
y_pred_best = best_model.predict(X_test)

print("\n================ CLASSIFICATION REPORT =================")
print(f"Model: {best_model_name}")
print(classification_report(y_test, y_pred_best, target_names=["Safe", "Phishing"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, display_labels=["Safe", "Phishing"]
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()


# EXPERIMENT 3 — Global feature importance
# Always uses TFIDF + LR for interpretability regardless of which
# model won the competition above.

lr_model_name = "TFIDF + LogisticRegression"
lr_model      = trained_models[lr_model_name]

vectorizer_lr = lr_model.named_steps["vectorizer"]
classifier_lr = lr_model.named_steps["classifier"]
feature_names = np.array(vectorizer_lr.get_feature_names_out())
coefs         = classifier_lr.coef_[0]

top_phishing_idx = np.argsort(coefs)[-15:][::-1]
top_safe_idx     = np.argsort(coefs)[:15]

print("\n================ EXPERIMENT 3 =================")
print(f"Global Feature Importance — {lr_model_name}")

print("\nTop words pushing toward PHISHING:")
for word, weight in zip(feature_names[top_phishing_idx], coefs[top_phishing_idx]):
    print(f"  {word:<20} {weight:+.4f}")

print("\nTop words pushing toward SAFE:")
for word, weight in zip(feature_names[top_safe_idx], coefs[top_safe_idx]):
    print(f"  {word:<20} {weight:+.4f}")


# Individual email explanation
def explain_prediction(text, pipeline, top_n=10):
    """
    Return the top contributing tokens for a single email.
    Accepts raw text — cleans internally so callers never need to pre-clean.
    Uses sparse operations; never densifies the full vocabulary matrix.
    """
    cleaned       = clean_text(text)
    vectorizer    = pipeline.named_steps["vectorizer"]
    classifier    = pipeline.named_steps["classifier"]
    x             = vectorizer.transform([cleaned])       # stays sparse
    coef          = classifier.coef_[0]

    scored = sorted(
        [(vectorizer.get_feature_names_out()[i], float(x[0, i] * coef[i]))
         for i in x.indices if x[0, i] * coef[i] > 0],
        key=lambda t: t[1], reverse=True,
    )
    return scored[:top_n]


# Sample email predictions
sample_emails = [
    "Urgent: Your account has been suspended. Click here immediately to verify your information.",
    "Hi team, attached is the updated meeting agenda for tomorrow morning.",
    "We detected unusual login activity. Confirm your banking details now to avoid permanent closure.",
]

print("\n================ SAMPLE PREDICTIONS =================")
for i, email in enumerate(sample_emails, start=1):
    cleaned = clean_text(email)                           # clean once, reuse
    proba   = lr_model.predict_proba([cleaned])[0][1]
    pred    = "Phishing" if proba >= 0.5 else "Safe"

    print(f"\nSample {i}")
    print("Email:", email)
    print("Prediction:", pred)
    print("Phishing probability:", round(float(proba), 4))
    print("Top contributing words:", explain_prediction(email, lr_model, top_n=8))


# LIME explanation
# LIME receives the already-cleaned text so its perturbations match
# the same token space the model was trained on.

explainer    = LimeTextExplainer(class_names=["Safe", "Phishing"])
sample_raw   = X_test_raw.iloc[0]
sample_clean = X_test.iloc[0]

exp = explainer.explain_instance(
    sample_clean,
    lr_model.predict_proba,
    num_features=10,
)

print("\n================ LIME EXPLANATION =================")
print("Original email preview:", str(sample_raw)[:200], "...")
print("Cleaned email preview :", str(sample_clean)[:200], "...\n")

for feature, weight in exp.as_list():
    print(f"  {feature:<25} {weight:+.4f}")


# Custom email prediction helper
def predict_email(email_text, model=None):
    """
    Predict and explain a single custom email.

    Args:
        email_text: Raw email string.
        model: Trained sklearn pipeline. Defaults to TFIDF + LR.
               Uses None-default to avoid capturing a mutable object
               at definition time.
    """
    if model is None:
        model = lr_model

    cleaned = clean_text(email_text)
    proba   = model.predict_proba([cleaned])[0][1]
    pred    = "Phishing" if proba >= 0.5 else "Safe"

    print("\n================ CUSTOM EMAIL PREDICTION =================")
    print("Email:", email_text)
    print("Prediction:", pred)
    print("Phishing probability:", round(float(proba), 4))
    print("Top contributing words:", explain_prediction(email_text, model, top_n=8))


# Example usage:
# predict_email("Your mailbox storage is full. Click this link now to keep your account active.")
# predict_email("Hey, just checking in on the project status for next week's demo.")
