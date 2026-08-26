"""
phishing_detector.py
---------------------
Trains a Scikit-learn model to classify emails as "phishing" or "safe"
using a mix of:
  1. TF-IDF features from the email text itself
  2. Hand-engineered URL features (IP-based links, "@" tricks, suspicious
     TLDs, HTTPS usage, etc.)
  3. Hand-engineered keyword/style features (urgency language, exclamation
     marks, dollar signs, etc.)

Usage:
    python3 phishing_detector.py                    # generate data, train, evaluate
    python3 phishing_detector.py --data emails.csv  # use your own CSV (text,label)
    python3 phishing_detector.py --classify "some email text here"
"""

import argparse
import re
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score
)

MODEL_PATH = "phishing_model.joblib"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"

# -----------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------

URGENCY_WORDS = [
    "urgent", "immediately", "verify", "suspended", "act now", "limited time",
    "click here", "confirm", "password", "account will be closed", "winner",
    "free", "gift card", "security alert", "final notice", "expire",
    "restricted", "unusual activity", "unauthorized", "locked", "reset",
]

SUSPICIOUS_URL_KEYWORDS = ["verify", "login", "secure", "account", "update",
                            "confirm", "signin", "banking", "billing"]

FAKE_TLDS = [".tk", ".ml", ".ga", ".cf", ".info", ".xyz", ".top"]

URL_PATTERN = re.compile(r"https?://[^\s,]+")
IP_HOST_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


def extract_row_features(text):
    text_lower = text.lower()
    urls = URL_PATTERN.findall(text)

    num_urls = len(urls)
    has_ip_url = int(any(IP_HOST_PATTERN.match(u) for u in urls))
    has_at_symbol = int(any("@" in u for u in urls))
    has_https = int(any(u.lower().startswith("https") for u in urls))
    avg_url_length = float(np.mean([len(u) for u in urls])) if urls else 0.0
    has_fake_tld = int(any(any(u.lower().endswith(tld) or tld in u.lower() for tld in FAKE_TLDS) for u in urls))
    suspicious_url_kw_count = sum(
        1 for u in urls for kw in SUSPICIOUS_URL_KEYWORDS if kw in u.lower()
    )

    urgency_word_count = sum(1 for w in URGENCY_WORDS if w in text_lower)
    exclamation_count = text.count("!")
    dollar_sign_count = text.count("$")
    uppercase_word_count = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    text_length = len(text)

    return {
        "num_urls": num_urls,
        "has_ip_url": has_ip_url,
        "has_at_symbol": has_at_symbol,
        "has_https": has_https,
        "avg_url_length": avg_url_length,
        "has_fake_tld": has_fake_tld,
        "suspicious_url_kw_count": suspicious_url_kw_count,
        "urgency_word_count": urgency_word_count,
        "exclamation_count": exclamation_count,
        "dollar_sign_count": dollar_sign_count,
        "uppercase_word_count": uppercase_word_count,
        "text_length": text_length,
    }


NUMERIC_FEATURE_NAMES = [
    "num_urls", "has_ip_url", "has_at_symbol", "has_https", "avg_url_length",
    "has_fake_tld", "suspicious_url_kw_count", "urgency_word_count",
    "exclamation_count", "dollar_sign_count", "uppercase_word_count",
    "text_length",
]


def build_feature_frame(texts):
    """Turn a list/Series of raw email text into a DataFrame with a 'text'
    column plus one column per engineered numeric feature."""
    feats = pd.DataFrame([extract_row_features(t) for t in texts])
    feats.insert(0, "text", list(texts))
    return feats


# -----------------------------------------------------------------------
# Pipeline construction
# -----------------------------------------------------------------------

def build_pipeline(classifier="logreg"):
    """ColumnTransformer combines TF-IDF text features with the scaled
    numeric URL/keyword features into a single feature matrix."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2),
                                       stop_words="english"), "text"),
            ("numeric", StandardScaler(), NUMERIC_FEATURE_NAMES),
        ]
    )

    if classifier == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif classifier == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    elif classifier == "naive_bayes":
        # Naive Bayes needs non-negative features; numeric StandardScaler output
        # can go negative, so this option only really works well on TF-IDF alone.
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return Pipeline([
        ("features", preprocessor),
        ("clf", clf),
    ])


# -----------------------------------------------------------------------
# Train / evaluate
# -----------------------------------------------------------------------

def load_data(data_path=None):
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError("CSV must have 'text' and 'label' columns.")
        return df
    # fall back to generating a synthetic dataset on the fly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_dataset import generate_dataset
    df = generate_dataset(n_per_class=300)
    df.to_csv("sample_emails.csv", index=False)
    print("No dataset provided — generated a synthetic one at sample_emails.csv")
    return df


def train_and_evaluate(df, classifier="logreg"):
    feature_df = build_feature_frame(df["text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    report = classification_report(y_test, y_pred)

    # 5-fold cross-validation for a more robust accuracy estimate
    cv_scores = cross_val_score(pipeline, feature_df, y, cv=5, scoring="accuracy")

    print("=" * 60)
    print(f"Classifier: {classifier}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("-" * 60)
    print("Classification report:")
    print(report)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_))
    print("=" * 60)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {classifier} (accuracy={accuracy:.2%})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {CONFUSION_MATRIX_PATH}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")

    return pipeline, {"accuracy": accuracy, "cv_mean": cv_scores.mean(), "cm": cm}


def show_top_features(pipeline, top_n=15):
    """Only meaningful for the logistic regression classifier, whose
    coefficients are directly interpretable."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        print("Feature importances not available for this classifier type.")
        return
    tfidf = pipeline.named_steps["features"].named_transformers_["tfidf"]
    text_feature_names = list(tfidf.get_feature_names_out())
    all_feature_names = text_feature_names + NUMERIC_FEATURE_NAMES

    # For binary LogisticRegression, sklearn stores a single coefficient row
    # that points toward classes_[1] (positive class), not classes_[0].
    # So a *positive* weight pushes toward classes_[1], and a *negative*
    # weight pushes toward classes_[0].
    positive_class = pipeline.classes_[1]
    negative_class = pipeline.classes_[0]

    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    print(f"\nTop {top_n} features pushing toward {positive_class.upper()}:")
    for i in order[-top_n:][::-1]:
        print(f"  {all_feature_names[i]:<30} weight={coefs[i]:.3f}")
    print(f"\nTop {top_n} features pushing toward {negative_class.upper()}:")
    for i in order[:top_n]:
        print(f"  {all_feature_names[i]:<30} weight={coefs[i]:.3f}")


# -----------------------------------------------------------------------
# Inference on a new email
# -----------------------------------------------------------------------

def classify_email(pipeline, text):
    feature_row = build_feature_frame([text])
    prediction = pipeline.predict(feature_row)[0]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(feature_row)[0]
        confidence = dict(zip(pipeline.classes_, proba))
    else:
        confidence = None
    return prediction, confidence


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phishing email detector (Scikit-learn).")
    parser.add_argument("--data", default=None, help="Path to a CSV with 'text','label' columns")
    parser.add_argument("--classifier", default="logreg",
                         choices=["logreg", "random_forest", "naive_bayes"],
                         help="Which model to train")
    parser.add_argument("--classify", default=None,
                         help="Classify a single piece of email text using a saved model")
    parser.add_argument("--show-features", action="store_true",
                         help="Print the most phishing-indicative and safe-indicative features")
    args = parser.parse_args()

    if args.classify:
        if not os.path.exists(MODEL_PATH):
            print("No trained model found. Train one first by running without --classify.")
            sys.exit(1)
        pipeline = joblib.load(MODEL_PATH)
        prediction, confidence = classify_email(pipeline, args.classify)
        print(f"Prediction: {prediction.upper()}")
        if confidence:
            for label, prob in confidence.items():
                print(f"  P({label}) = {prob:.3f}")
        return

    df = load_data(args.data)
    pipeline, metrics = train_and_evaluate(df, classifier=args.classifier)

    if args.show_features or args.classifier == "logreg":
        show_top_features(pipeline)

    # Demo: classify a couple of hand-written example emails
    demo_emails = [
        "Dear customer, your account has been locked due to suspicious activity. "
        "Verify your identity immediately at http://192.168.4.22/secure-login or "
        "your account will be permanently deleted.",
        "Hi Sam, just checking if you're free for a quick call tomorrow afternoon "
        "to go over the project timeline. Let me know what works.",
    ]
    print("\n" + "=" * 60)
    print("Demo predictions on new, unseen emails:")
    for email in demo_emails:
        prediction, confidence = classify_email(pipeline, email)
        conf_str = f" (P(phishing)={confidence['phishing']:.2f})" if confidence else ""
        print(f"\n  \"{email[:70]}...\"")
        print(f"  -> {prediction.upper()}{conf_str}")


if __name__ == "__main__":
    main()
