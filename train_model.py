import re
import json
import pickle
import numpy as np
<<<<<<< HEAD
=======
import pandas as pd
from pathlib import Path

>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
<<<<<<< HEAD
=======
from sklearn.ensemble import RandomForestClassifier
>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: Load datasets
# ---------------------------------------------------------------
print("Loading datasets from dataset/Fake.csv and dataset/True.csv...")
fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

fake["label"] = 0
true["label"] = 1

# ---------------------------------------------------------------
# STEP 2: Scrub Source Leakage and Web Boilerplate Artifacts
# ---------------------------------------------------------------
# 1. Reuters datelines like 'WASHINGTON (Reuters) -'
dateline_pattern = re.compile(r'^[A-Z][A-Za-z\.\s/]{2,40}\(Reuters\)\s*-\s*', re.UNICODE)
reuters_word_pattern = re.compile(r'\bReuters\b', re.IGNORECASE)

# 2. Web boilerplate words that leak blog metadata
boilerplate_patterns = [
    re.compile(r'\bfeatured\s+image\b', re.IGNORECASE),
    re.compile(r'\bgetty\s+images\b', re.IGNORECASE),
    re.compile(r'\bread\s+more\b', re.IGNORECASE),
    re.compile(r'\bwatch\s+video\b', re.IGNORECASE),
    re.compile(r'\bfollow\s+us\s+on\s+twitter\b', re.IGNORECASE),
    re.compile(r'\blike\s+us\s+on\s+facebook\b', re.IGNORECASE),
    re.compile(r'\bsubscribe\s+now\b', re.IGNORECASE),
    re.compile(r'\bclick\s+here\b', re.IGNORECASE),
]

def clean_source_and_boilerplate(text):
    text = str(text)
    text = dateline_pattern.sub('', text, count=1)
    text = reuters_word_pattern.sub('', text)
    for pat in boilerplate_patterns:
        text = pat.sub('', text)
    return text

print("Scrubbing source datelines and web boilerplate artifacts...")
true["title"] = true["title"].apply(clean_source_and_boilerplate)
true["text"] = true["text"].apply(clean_source_and_boilerplate)
fake["title"] = fake["title"].apply(clean_source_and_boilerplate)
fake["text"] = fake["text"].apply(clean_source_and_boilerplate)

true["combined"] = true["title"] + " " + true["text"]
fake["combined"] = fake["title"] + " " + fake["text"]

# ---------------------------------------------------------------
# STEP 3: Combine and Deduplicate
# ---------------------------------------------------------------
data = pd.concat([fake, true], ignore_index=True)
before = len(data)
data = data.drop_duplicates(subset="combined")
after = len(data)
print(f"Removed {before - after} duplicate articles ({before} -> {after})")

# Filter out articles that became empty or near-empty
data = data[data["combined"].str.strip().str.len() > 30]

X = data["combined"]
y = data["label"]

# ---------------------------------------------------------------
<<<<<<< HEAD
# STEP 4: Train/test split FIRST to prevent TF-IDF data leakage
# ---------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# STEP 5: TF-IDF vectorization (fit ON X_train ONLY, transform X_test)
# ---------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=3)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Calculate mean feature vector over training corpus for exact Linear SHAP baseline
bg_mean = np.asarray(X_train.mean(axis=0)).ravel()
=======
# STEP 4: TF-IDF Vectorization with Enhanced Stop-Words
# ---------------------------------------------------------------
# Remove metadata words that cause false-positive bias on modern web pages
custom_stop_words = [
    "said", "told", "says", "reporting", "reports", "spokesman", "spokeswoman", "statement", "added",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "video", "image", "images", "featured", "getty", "read", "watch", "twitter", "facebook", "youtube",
    "com", "https", "http", "click", "share", "pic", "photo", "subscribe", "subscriber", "subscription"
]
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

print("Fitting TF-IDF Vectorizer (8,000 max features, unigrams and bigrams)...")
vectorizer = TfidfVectorizer(
    stop_words=all_stop_words,
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=5,
    max_features=8000
)
X_vectorized = vectorizer.fit_transform(X)

# ---------------------------------------------------------------
# STEP 5: Stratified Train/Test Split
# ---------------------------------------------------------------
print("Splitting dataset into 80% train and 20% test (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)
>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b

# ---------------------------------------------------------------
# STEP 6: Multi-Algorithm Benchmarking
# ---------------------------------------------------------------
candidates = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(C=0.5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=25, random_state=42, n_jobs=-1)
}

results = {}
print("\n" + "=" * 65)
print("BENCHMARKING CLASSIFIERS ON HELD-OUT TEST SPLIT (20%)")
print("=" * 65)

for name, clf in candidates.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds))
    rec = float(recall_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))

    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")

# ---------------------------------------------------------------
# STEP 7: Select Best Model and Wrap with Probability Calibration
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
print("\n" + "=" * 65)
print(f"BEST PERFORMING ALGORITHM: {best_name} (F1: {results[best_name]['f1']*100:.2f}%)")
print("=" * 65)

<<<<<<< HEAD
# Wrap SVM with probability calibration if it's the best model
if best_name == "SVM (LinearSVC)":
    print("\nWrapping SVM with probability calibration for probability outputs...")
    best_model = CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)
    best_model.fit(X_train, y_train)
=======
if "SVM" in best_name:
    print("Calibrating SVM with Platt Sigmoid Scaling (CalibratedClassifierCV) for true probabilities...")
    final_model = CalibratedClassifierCV(LinearSVC(C=0.5, random_state=42), cv=3)
    final_model.fit(X_train, y_train)
else:
    final_model = results[best_name]["model"]
>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b

test_preds = final_model.predict(X_test)
cm = confusion_matrix(y_test, test_preds)
print("\nFinal Test Confusion Matrix [Rows: Actual (0=Fake, 1=Real), Cols: Predicted]:")
print(cm)

<<<<<<< HEAD
# Calculate and save reliability calibration curve coordinates
probs = best_model.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
pickle.dump((fraction_of_positives, mean_predicted_value), open("model/calibration.pkl", "wb"))
pickle.dump(bg_mean, open("model/bg_mean.pkl", "wb"))

# ---------------------------------------------------------------
# STEP 8: Save artifacts, vectorizer, and dynamic metrics object
=======
# Calibration curve points
if hasattr(final_model, "predict_proba"):
    probs = final_model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
    with open(MODEL_DIR / "calibration.pkl", "wb") as f:
        pickle.dump((fraction_of_positives, mean_predicted_value), f)

# ---------------------------------------------------------------
# STEP 8: Save Model Assets and Actual Empirical Metrics
>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b
# ---------------------------------------------------------------
print("\nSaving trained models, vectorizer, and test metrics...")
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(final_model, f)

<<<<<<< HEAD
metrics_artifact = {
    "models": {
        name: {
            "accuracy": round(r["accuracy"] * 100, 2),
            "precision": round(r["precision"] * 100, 2),
            "recall": round(r["recall"] * 100, 2),
            "f1": round(r["f1"] * 100, 2),
        }
        for name, r in results.items()
    },
    "best_model_name": best_name,
    "confusion_matrix": cm.tolist(),
}
pickle.dump(metrics_artifact, open("model/metrics.pkl", "wb"))

with open("model/metrics.txt", "w") as f:
    f.write(f"Model used: {best_name}\n")
    f.write(f"Accuracy: {results[best_name]['accuracy']*100:.2f}%\n")
    f.write(f"Precision: {results[best_name]['precision']*100:.2f}%\n")
    f.write(f"Recall: {results[best_name]['recall']*100:.2f}%\n")
    f.write(f"F1 Score: {results[best_name]['f1']*100:.2f}%\n")
    f.write(f"Confusion Matrix [0=Fake,1=Real]:\n{cm}\n")
    f.write(f"\nAll models compared:\n")
    for name, r in results.items():
        f.write(f"  {name}: acc={r['accuracy']*100:.2f}% prec={r['precision']*100:.2f}% "
                f"rec={r['recall']*100:.2f}% f1={r['f1']*100:.2f}%\n")

print("\nModel trained and saved successfully without TF-IDF data leakage! (artifacts updated in model/)")
=======
with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

np.save(MODEL_DIR / "confusion_matrix.npy", cm)

metrics_summary = {
    "best_model": best_name,
    "evaluated_test_samples": int(len(y_test)),
    "best_metrics": {
        "accuracy": round(results[best_name]["accuracy"] * 100, 2),
        "precision": round(results[best_name]["precision"] * 100, 2),
        "recall": round(results[best_name]["recall"] * 100, 2),
        "f1_score": round(results[best_name]["f1"] * 100, 2)
    },
    "confusion_matrix": cm.tolist(),
    "comparison": [
        {
            "classifier": name,
            "accuracy": round(r["accuracy"] * 100, 2),
            "precision": round(r["precision"] * 100, 2),
            "recall": round(r["recall"] * 100, 2),
            "f1_score": round(r["f1"] * 100, 2)
        }
        for name, r in results.items()
    ]
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

with open(MODEL_DIR / "metrics.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Accuracy: {metrics_summary['best_metrics']['accuracy']}%\n")
    f.write(f"Precision: {metrics_summary['best_metrics']['precision']}%\n")
    f.write(f"Recall: {metrics_summary['best_metrics']['recall']}%\n")
    f.write(f"F1 Score: {metrics_summary['best_metrics']['f1_score']}%\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write("Full Comparison:\n")
    for row in metrics_summary["comparison"]:
        f.write(f"  - {row['classifier']}: Acc={row['accuracy']}% Prec={row['precision']}% Rec={row['recall']}% F1={row['f1_score']}%\n")

print("\nTraining and Evaluation complete! Artifacts saved to model/ directory.")
>>>>>>> c835feb6e1726276462368d63e3f97617350ab1b
