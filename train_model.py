import re
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------
# STEP 1: Load datasets
# ---------------------------------------------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

# ---------------------------------------------------------------
# STEP 2: Remove Reuters source leakage
# Strip datelines and occurrences of "Reuters" to prevent style leakage
# ---------------------------------------------------------------
dateline_pattern = re.compile(r'^[A-Z][A-Za-z\.\s/]{2,40}\(Reuters\)\s*-\s*')
reuters_word_pattern = re.compile(r'\bReuters\b', re.IGNORECASE)

def strip_source_leakage(text):
    text = str(text)
    text = dateline_pattern.sub('', text, count=1)
    text = reuters_word_pattern.sub('', text)
    return text

true["title"] = true["title"].apply(strip_source_leakage)
true["text"] = true["text"].apply(strip_source_leakage)
fake["title"] = fake["title"].apply(strip_source_leakage)
fake["text"] = fake["text"].apply(strip_source_leakage)

# Combine title and text to train on both headlines and body text styles
true["combined"] = true["title"] + " " + true["text"]
fake["combined"] = fake["title"] + " " + fake["text"]

# ---------------------------------------------------------------
# STEP 3: Combine and remove duplicate/near-duplicate articles
# so the same article can't leak across train and test splits.
# ---------------------------------------------------------------
data = pd.concat([fake, true], ignore_index=True)
before = len(data)
data = data.drop_duplicates(subset="combined")
after = len(data)
print(f"Removed {before - after} duplicate articles ({before} -> {after})")

# Drop rows that became empty or near-empty
data = data[data["combined"].str.strip().str.len() > 20]

X = data["combined"]
y = data["label"]

# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# STEP 6: Train and evaluate Naive Bayes, SVM, and Logistic Regression
# ---------------------------------------------------------------
candidates = {
    "Naive Bayes": MultinomialNB(),
    "SVM (LinearSVC)": LinearSVC(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = {}
print("\n" + "=" * 55)
print("MODEL COMPARISON (on held-out 20% test set)")
print("=" * 55)

for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {"model": clf, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")

# ---------------------------------------------------------------
# STEP 7: Pick the best model by F1 score
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]

# Wrap SVM with probability calibration if it's the best model
if best_name == "SVM (LinearSVC)":
    print("\nWrapping SVM with probability calibration for probability outputs...")
    best_model = CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)
    best_model.fit(X_train, y_train)

print("\n" + "=" * 55)
print(f"FINAL CHOICE: {best_name}")
print(f"  Accuracy : {results[best_name]['accuracy']*100:.2f}%")
print(f"  Precision: {results[best_name]['precision']*100:.2f}%")
print(f"  Recall   : {results[best_name]['recall']*100:.2f}%")
print(f"  F1 Score : {results[best_name]['f1']*100:.2f}%")
print("=" * 55)

cm = confusion_matrix(y_test, best_model.predict(X_test))
print("\nConfusion Matrix (rows=actual, cols=predicted) [0=Fake, 1=Real]")
print(cm)

# Calculate and save reliability calibration curve coordinates
probs = best_model.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
pickle.dump((fraction_of_positives, mean_predicted_value), open("model/calibration.pkl", "wb"))
pickle.dump(bg_mean, open("model/bg_mean.pkl", "wb"))

# ---------------------------------------------------------------
# STEP 8: Save artifacts, vectorizer, and dynamic metrics object
# ---------------------------------------------------------------
pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

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
