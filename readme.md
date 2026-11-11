# TruthLens: AI-Powered Fake News Detection & Verification System
### *A Publication-Grade B.Tech Final-Year Capstone Project in Applied Machine Learning & NLP*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Project Abstract

The rapid proliferation of digital media and decentralized content distribution has exacerbated the dissemination of misinformation, digital hoaxes, and political propaganda. Automatic detection of deceptive news presents a foundational challenge in Natural Language Processing (NLP). 

**TruthLens** is an end-to-end, multi-modal verification system developed for academic rigor and real-world deployment. Unlike naive black-box classifiers that force binary "Real" or "Fake" labels, TruthLens implements a **3-Tier Calibrated Decision Synthesis Architecture**:
- 🟢 **Likely Reliable**: High statistical probability of journalistic structure ($P(\text{Real}) \ge 0.65$), verified domain reputation, and multi-source corroboration.
- 🔴 **Likely Misleading**: High statistical probability of fabrication ($P(\text{Fake}) \ge 0.65$), excessive sensationalism/clickbait indicators, or matching verified debunks.
- 🟡 **Needs Further Verification**: Ambiguous probabilistic boundary zones ($0.40 < P < 0.65$), brief texts (< 50 words), or out-of-domain linguistic constructs requiring human discernment.

A central scientific contribution of this project is the **remediation of dataset shortcut learning**: standard models trained on the ISOT dataset score artificially high (>99%) by memorizing publisher datelines (e.g. `(Reuters) -`) and web blogging artifacts (e.g. `video`, `getty images`, `read more`). TruthLens implements comprehensive source scrubbing and web boilerplate filtering, forcing the classifier to learn **genuine stylistic and structural patterns**.

---

## 🚀 Key Innovations & Features

1. **Multi-Algorithm Benchmarking**:
   - Evaluated across **Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machine (LinearSVC), and Random Forest** on a held-out 20% stratified test split.
   - Best-performing model (**Calibrated Linear SVM**, **98.20% Accuracy**, **98.34% F1-Score**) calibrated via Platt Sigmoid Scaling (`CalibratedClassifierCV`) to output mathematically sound posterior probabilities.
2. **Transparent Explainable AI (XAI)**:
   - **Linear SHAP Attributions**: Computes feature importance scores ($w_i \cdot x_i$) to render horizontal bar plots illustrating top words driving the prediction.
   - **Semantic Token Highlighting**: Color-codes document text in real time (soft green for reliable markers, soft red for misleading markers).
3. **Paywall-Aware URL Web Scraper**:
   - Parses news articles using `requests` and `BeautifulSoup4` with automated detection for paywalls, subscription banners, and JSON-LD structured schema.
4. **Image OCR Analysis**:
   - Supports screenshot uploads (PNG, JPG, JPEG) with dual-engine Optical Character Recognition (local Tesseract + cloud OCR fallback).
5. **Multi-Source Fact-Checking & Corroboration**:
   - Queries **ClaimReview** verified debunks (IFCN signatories, PolitiFact, Snopes, BoomLive) and live multi-source Google News feeds.
6. **Downloadable PDF Verification Dossier**:
   - Automatically generates publication-grade PDF reports with audit IDs, calibrated confidence, SHAP breakdowns, and academic disclaimers using `fpdf2`.
7. **Empirical Analytics Dashboard**:
   - Dynamically visualizes real test confusion matrices, multi-model comparative bar charts, and SQLite telemetry distributions without hardcoded values.
8. **Multilingual Guardrails**:
   - Detects English, Hindi, and Odia scripts. Enforces academic honesty by notifying users when an English-trained model encounters non-English syntax, redirecting to fact-checking corroboration.

---

## 📐 System Architecture

```mermaid
graph TD
    User([User / Evaluator]) -->|Streamlit Interface| TruthLens[TruthLens Web Portal]
    
    subgraph Input_Channels [Multi-Modal Ingestion]
        T[Text Claims]
        U[URL Web Scraper + Paywall Parser]
        I[Image Screenshot + OCR Engine]
    end
    User --> Input_Channels
    Input_Channels --> Prep[Regex Leakage Scrubbing & Multilingual Detection]
    
    subgraph Intelligence_Pipeline [Verification Intelligence Core]
        Prep --> ML[Calibrated SVM Model + TF-IDF Vectorizer]
        Prep --> XAI[Linear SHAP Attribution Engine]
        Prep --> Heur[Clickbait Index & Domain Reputation Rater]
        Prep --> FC[ClaimReview Fact-Check & Live News Feed]
    end
    
    ML & Heur & FC --> Synth[3-Tier Decision Synthesizer]
    Synth --> Verdict[Likely Reliable | Likely Misleading | Needs Verification]
    Verdict --> UI[High-Contrast Telemetry Cards & Token Visualizer]
    Verdict --> PDF[Downloadable PDF Verification Report]
    Verdict --> DB[(SQLite: news_logs.db)]
```

---

## 📊 Empirical Model Benchmarking Results

Evaluated on the held-out 20% stratified test split (7,681 unseen articles) from the cleaned ISOT News Dataset:

| Classifier Algorithm | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| Multinomial Naive Bayes | 93.19% | 94.63% | 92.69% | 93.65% |
| Logistic Regression | 97.47% | 97.15% | 98.21% | 97.68% |
| **Linear SVM (Calibrated)** | **98.20%** | **98.01%** | **98.68%** | **98.34%** |
| Random Forest | 93.94% | 92.41% | 96.77% | 94.54% |

### Final Test Confusion Matrix (Linear SVM)
*Total held-out test articles: 7,681*

```
               Predicted Fake    Predicted Real
Actual Fake         3506               76
Actual Real           65             4174
```

---

## 🗂️ Project Directory Structure

```
FAKE-NEWS-DETECTION/
├── dataset/
│   ├── Fake.csv                     # ISOT Fake news corpus (~23.4k articles)
│   └── True.csv                     # ISOT Real news corpus (~21.4k articles)
├── model/
│   ├── model.pkl                    # Best performing Calibrated Linear SVM
│   ├── vectorizer.pkl               # Fitted TF-IDF Vectorizer (8,000 features)
│   ├── calibration.pkl              # Probability calibration curve points
│   ├── metrics.json                 # Machine-readable test split evaluation metrics
│   ├── metrics.txt                  # Human-readable benchmark report
│   └── confusion_matrix.npy         # Test confusion matrix array
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constants, paths, and thresholds
│   ├── preprocessor.py              # Regex dateline/boilerplate scrubbing & language guardrails
│   ├── heuristics.py                # Clickbait scoring and domain reputation rater
│   ├── explainability.py            # Linear SHAP attribution & semantic token highlighter
│   ├── scraper.py                   # Robust article scraper with paywall detection
│   ├── ocr_engine.py                # OCR engine for image screenshot analysis
│   ├── fact_checker.py              # ClaimReview search & live multi-source news feed
│   ├── classifier.py                # 3-tier probabilistic decision synthesizer
│   ├── pdf_report.py                # Professional PDF dossier generator (fpdf2)
│   ├── database.py                  # SQLite logging and telemetry analytics
│   └── ui_theme.py                  # Slate & Indigo UI theme styling
├── app.py                           # Multi-page Streamlit application entry point
├── train_model.py                   # Rigorous multi-algorithm training & benchmarking script
├── verify_generalization.py         # Multi-topic and domain shift verification tests
├── requirements.txt                 # Project dependencies
├── .env.example                     # Environment template for optional API keys
└── README.md                        # Capstone project documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Kashish995/FAKE-NEWS-DETECTOR.git
cd FAKE-NEWS-DETECTOR
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain and benchmark models
```bash
python train_model.py
```

### 5. Run TruthLens
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Streamlit Community Cloud Deployment

1. Push your repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. (Optional) Under **Advanced Settings > Secrets**, configure any desired API keys:
   ```toml
   NEWSAPI_KEY = "your_key"
   FACTCHECK_API_KEY = "your_google_key"
   OCR_API_KEY = "helloworld"
   ```
5. Click **Deploy**. TruthLens will install dependencies from `requirements.txt` and launch automatically.

---

## ⚖️ Academic & Methodological Disclaimer

TruthLens uses statistical natural language processing and probability-calibrated machine learning models trained on benchmark journalistic corpuses. Predictions evaluate stylistic, structural, and lexical conventions. **They do not constitute an infallible claim to objective real-world truth.** Multi-source journalistic corroboration, institutional fact-checking, and critical human discernment remain indispensable.

---
**TruthLens System** • *B.Tech Final-Year Capstone Major Project in Computer Science & Artificial Intelligence*.
