# Phishing Email Detection Model

A Scikit-learn pipeline that classifies emails as **phishing** or **safe**
using a combination of TF-IDF text features and hand-engineered URL/keyword
features.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds a synthetic-but-realistic labeled dataset (since real datasets require internet access this environment doesn't have). Produces `sample_emails.csv`. |
| `phishing_detector.py` | Feature extraction, model training, evaluation, and a CLI for classifying new emails. |
| `sample_emails.csv` | The generated dataset (600 emails, balanced 50/50). |
| `phishing_model.joblib` | The trained pipeline, saved after running the script. |
| `confusion_matrix.png` | Confusion matrix plot from the most recent training run. |

## How it works

**Features extracted per email:**
- *Text* — TF-IDF over unigrams and bigrams (up to 2000 terms, English stop words removed).
- *URL features* — number of URLs, whether any URL uses a raw IP address instead of a domain, the `user@fakehost` trick, use of HTTPS, average URL length, suspicious TLDs (`.tk`, `.ml`, `.xyz`, etc.), and count of suspicious words inside URLs (`verify`, `login`, `secure`, ...).
- *Keyword/style features* — count of urgency phrases ("act now", "suspended", "verify", "winner", ...), exclamation marks, dollar signs, all-caps words, and overall length.

These are combined with a `ColumnTransformer` (TF-IDF on the text column, `StandardScaler` on the numeric columns) feeding into a classifier — Logistic Regression by default, with Random Forest and Naive Bayes also available.

**Evaluation:** train/test split (80/20, stratified) plus 5-fold cross-validation, accuracy, a full precision/recall/F1 classification report, and a plotted confusion matrix.

## Usage

```bash
# Train on the synthetic dataset (generates it automatically if missing)
python3 phishing_detector.py

# Try a different classifier
python3 phishing_detector.py --classifier random_forest

# Train on your own data instead (CSV needs 'text' and 'label' columns,
# labels should be exactly "phishing" / "safe")
python3 phishing_detector.py --data your_emails.csv

# Classify a single new email with the saved model
python3 phishing_detector.py --classify "Verify your account now: http://192.168.1.5/login"

# See which features most strongly indicate phishing vs. safe
python3 phishing_detector.py --show-features
```

## About the dataset — please read

There's no internet access in this environment, so this project ships with
a **synthetic** dataset generated from templates that mirror well-known
phishing patterns (urgency, fake account alerts, prize scams, IP-address
links, lookalike domains) alongside plausible everyday emails. It's genuinely
useful for learning the pipeline end-to-end, but:

- **The 100% accuracy you'll see is expected and not impressive** — synthetic
  templates are far more separable than real-world email, where phishing
  authors deliberately mimic legitimate style. Treat this as a working
  skeleton, not a production-ready detector.
- To get a meaningful accuracy number, swap in a real labeled dataset, e.g.
  the public **Nazario phishing corpus**, **SpamAssassin public corpus**, or
  a phishing dataset from Kaggle — just make sure the CSV has `text` and
  `label` columns and re-run with `--data your_file.csv`.
- Real phishing datasets are also class-imbalanced (far more legitimate mail
  than phishing) — you'll want to check precision/recall separately per
  class rather than relying on accuracy alone, which the classification
  report already gives you.

## Extending this further

- Add header-based features (SPF/DKIM failures, sender-domain mismatch with
  reply-to, display-name spoofing) if you have raw `.eml` files to parse.
- Try `GridSearchCV` to tune `TfidfVectorizer(max_features=...)` and the
  classifier's regularization strength.
- Swap Logistic Regression for a gradient-boosted tree model (e.g. XGBoost/
  LightGBM) once you're working with a larger, real dataset — TF-IDF plus
  boosted trees is a common strong baseline for this exact task.
- Wrap `classify_email()` in a small Flask/FastAPI endpoint to demo live
  scoring of incoming email text.
