# TruthLens — Capstone Defense Presentation Guide

**Target Audience**: Head of Department (HOD), Academic Project Committee, External Examiners  
**Project**: TruthLens — AI-Powered Fake News Detection & Multi-Source Verification System  
**Presentation Deck File**: [TruthLens_Final_Presentation.pptx](file:///c:/Users/swain/OneDrive/Desktop/FAKE-NEWS-DETECTION/TruthLens_Final_Presentation.pptx)  
**Layout Format**: 16:9 Modern Widescreen Executive Dark Theme

---

## Slide-by-Slide Structure & Speaker Script

### Slide 1: Title & Introduction
* **Header / Badge**: FINAL YEAR MAJOR PROJECT PRESENTATION | B.TECH CSE
* **Title**: TruthLens: AI-Powered Fake News Detection & Multi-Source Verification System
* **Subtitle**: A Multi-Modal, Explainable NLP Framework Bridging Linguistic Machine Learning with Real-Time Journalistic Corroboration
* **Presented To**: Head of Department & Project Evaluation Committee
* **🎤 What to Say to Your HOD**:
  > *"Good morning respected HOD sir and committee members. Today, we present TruthLens, our final-year capstone project. While traditional fake news detectors rely solely on isolated statistical models that struggle with real-time news, TruthLens is an end-to-end multi-modal verification platform. It fuses linguistic machine learning with live journalistic corroboration and explainable AI to provide transparent, trustworthy assessments."*

---

### Slide 2: Problem Statement & Motivation
* **Three Pillar Cards**:
  1. **The Misinformation Epidemic**: Viral velocity (falsehoods spread 6x faster than truth), electoral and financial disruption, and billions of daily unvetted social media posts.
  2. **Why Existing Systems Fail**: Pure "black-box" models offer no reasoning; static datasets cannot verify events happening today; binary Real/Fake models force 51/49 predictions into false certainty.
  3. **The TruthLens Innovation**: Synthesizes stylistic NLP with live ClaimReview debunks, provides multi-modal ingestion (Text, URL, Screenshot OCR), and implements honest 3-tier uncertainty reporting.
* **🎤 What to Say to Your HOD**:
  > *"The core research problem we identified is that a static machine learning model cannot determine truth alone. If an earthquake happens today, a 2021 model cannot know the facts. Furthermore, existing tools are black boxes that force ambiguous claims into an overconfident 'Real' or 'Fake' label. TruthLens addresses this through a multi-signal approach that evaluates writing style, source credibility, and live external reporting simultaneously."*

---

### Slide 3: System Architecture (Four Independent Components)
* **Four Component Cards**:
  * **Component A (ML Core)**: Calibrated Linear Support Vector Classifier evaluating stylistic syntax and vocabulary.
  * **Component B (Source Credibility)**: Domain authority tracker and sensationalism index.
  * **Component C (External Verification)**: Real-time IFCN ClaimReview search + Google News RSS semantic cross-referencing.
  * **Component D (Evidence Synthesis)**: 3-tier verdict synthesis (`Likely Reliable`, `Likely Misleading`, `Needs Further Verification`) with automated conflict detection.
* **🎤 What to Say to Your HOD**:
  > *"Our architecture is decoupled into four independent components. Component A evaluates linguistic style; Component B profiles the publisher; Component C cross-references live news archives; and Component D synthesizes these signals. Crucially, no single component can override another: if an article is written in formal style but has been debunked by fact-checkers, Component D immediately flags the conflict."*

---

### Slide 4: Machine Learning & NLP Pipeline
* **Card 1: NLP Preprocessing & Feature Engineering**:
  * Unicode cleaning, regex sanitation, selective stopword preservation for journalistic syntax.
  * Bi-gram TF-IDF vectorization (10,000 features, sublinear term-frequency scaling).
* **Card 2: Model Comparison & Calibration**:
  * Evaluated across Multinomial Naive Bayes, Logistic Regression, Random Forest, and Linear SVM.
  * **Linear SVM selected as champion (93.8% test accuracy)** due to superior margin separation in high-dimensional sparse text vectors.
  * **Platt Sigmoid Scaling**: Calibrated decision boundaries ($P(\text{Real}) = \frac{1}{1 + e^{-f(x)}}$) ensuring mathematically sound probabilities without index inversion.
* **🎤 What to Say to Your HOD**:
  > *"For our machine learning pipeline, we benchmarked multiple algorithms. Linear SVM outperformed Naive Bayes and Logistic Regression with 93.8% test accuracy. To transform raw hyperplane distances into reliable probabilistic confidence, we applied Platt Sigmoid scaling, ensuring our probabilities are well-calibrated and mathematically consistent."*

---

### Slide 5: Multi-Modal Ingestion Capabilities
* **Card 1: Direct Text Engine**: Dropdown test samples, minimum word count threshold (>= 25 words) to avoid guessing on single phrases, script detection for non-Latin text.
* **Card 2: News URL Web Scraper**: SSRF security guard blocking RFC-1918 private IPs, multi-stage fallback parser (Trafilatura $\to$ JSON-LD Schema $\to$ HTML paragraphs), and paywall detection.
* **Card 3: Screenshot Image OCR**: Dual-pipeline architecture (Local Tesseract OCR engine + Cloud OCR), exponential backoff on HTTP 429, and SHA-256 hash synchronization to bind predictions directly to the uploaded image.
* **🎤 What to Say to Your HOD**:
  > *"TruthLens is not limited to plain text. We built three input modalities: a secure web scraper with SSRF protection against private network injection, a direct text claim analyzer, and a dual-pipeline OCR engine that extracts text from news screenshots using local Tesseract OCR with automatic cloud fallback."*

---

### Slide 6: Ground-Truth Fact-Checking & Corroboration
* **Card 1: ClaimReview & News Feeds**:
  * Real-time querying of certified IFCN debunks (Reuters, BoomLive, PolitiFact, AltNews).
  * Live Google News RSS multi-source feed.
  * Anti-circular filter: Automatically disqualifies the article's own publisher to prevent circular self-confirmation.
* **Card 2: Semantic Relevance Scoring**:
  * Replaced token overlap with normalized Sørensen–Dice similarity: $\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$.
  * Acronym and entity weighting (e.g., 'GRAP', 'ISRO', 'Supreme Court').
  * Strict $\ge 25\%$ relevance threshold to eliminate keyword noise.
* **🎤 What to Say to Your HOD**:
  > *"To verify real-world facts, we query live ClaimReview databases and mainstream news feeds. To ensure retrieved articles genuinely discuss the same event, we developed a normalized Sørensen–Dice semantic relevance algorithm. If an event has no matching coverage, the system honestly states that no corroboration was found rather than inventing an answer."*

---

### Slide 7: Explainable AI (XAI) & Interpretability
* **Card 1: Linear SHAP Methodology**:
  * Game-theoretic Shapley values calculated for individual n-gram tokens.
  * Visualizes the exact positive and negative contribution of words towards the classification.
  * Demystifies black-box machine learning for academic scrutiny.
* **Card 2: Visual Diagnostics**:
  * Green bars highlight objective journalistic tokens (*'spokesperson'*, *'confirmed'*, *'official'*).
  * Red bars highlight sensational clickbait cues (*'shocking'*, *'miracle cure'*, *'secret documents'*).
* **🎤 What to Say to Your HOD**:
  > *"One of our primary contributions is Explainable AI. Using SHAP—Shapley Additive exPlanations—we unpack the model's decision. The system visualizes exactly which words pushed the prediction towards reliable or misleading, providing transparent evidence rather than an unexplained score."*

---

### Slide 8: Empirical Validation & Generalization Benchmark
* **Card 1: Held-Out 20% Test Evaluation**:
  * Test Accuracy: **93.8%**
  * Precision: **94.2%** | Recall: **93.1%** | F1-Score: **93.6%**
  * Inference Latency: **< 35 ms** per article.
* **Card 2: 30-Case Real-World Generalization Benchmark**:
  * Genuine News (The Hindu, BBC, Reuters): 100% alignment.
  * Viral Fabrications: 100% correctly flagged with debunks.
  * Ambiguous / Hyperlocal News: 80% assigned to `Needs Further Verification` rather than forcing false certainty.
* **🎤 What to Say to Your HOD**:
  > *"Beyond standard held-out test accuracy of 93.8%, we designed a 30-case real-world generalization stress test spanning breaking news, local events, and viral hoaxes. This proved that TruthLens maintains high accuracy on known news, while appropriately outputting 'Needs Further Verification' on unfamiliar local events."*

---

### Slide 9: Technical Stack & Software Architecture
* **Core ML & NLP**: Python 3.13, Scikit-Learn, NumPy, Pandas, SHAP, NLTK, Tesseract OCR.
* **Data Pipeline & Security**: Trafilatura, BeautifulSoup4, lxml, Requests, Socket SSRF validator, ReportLab PDF exporter.
* **Frontend & DevOps**: Streamlit, SQLite relational database, Pytest (20/20 tests passing), Streamlit Community Cloud (Debian Linux container).
* **🎤 What to Say to Your HOD**:
  > *"Our software engineering adheres to industry standards. The backend is written in Python 3.13 using Scikit-Learn and SHAP, backed by SQLite for relational audit logging. The entire codebase is verified by an automated Pytest suite of 20 unit and integration tests with a 100% pass rate."*

---

### Slide 10: Conclusion & Future Scope
* **Key Achievements**:
  * Unified stylistic linguistic NLP with live ground-truth fact-checking.
  * Eliminated binary overconfidence with a calibrated 3-tier verdict model.
  * Delivered a zero-downtime, production-ready Streamlit Cloud deployment.
* **Future Roadmap**:
  * Fine-tuning multilingual Transformers (IndicBERT / mBERT) for vernacular regional Indian languages.
  * Multimodal Vision Transformers (ViT) for synthetic image and deepfake detection.
  * Browser extension for real-time social media feed verification.
* **🎤 What to Say to Your HOD**:
  > *"In conclusion, TruthLens provides a complete, scalable, and explainable defense against digital misinformation. In future iterations, we plan to incorporate fine-tuned IndicBERT models for Indian regional languages and Vision Transformers for deepfake image analysis. Thank you, and we are now open to your questions."*

---

## Top 5 Likely HOD Questions & Winning Defense Answers

#### Q1: "Why did you use Linear SVM instead of deep learning or large language models like BERT/GPT?"
> **Answer**:  
> *"Sir, we evaluated multiple algorithms. While LLMs and heavy Transformers are powerful, they suffer from high inference latency (several seconds per article), costly GPU requirements, and a known tendency to hallucinate. Linear SVM with TF-IDF provides 93.8% accuracy, sub-35ms inference on standard CPUs, and deterministic mathematical margins that allow exact SHAP feature attribution. Furthermore, to verify facts, we query real-time external databases rather than relying on an LLM's static training cutoff."*

#### Q2: "Can your system detect if a piece of breaking news from today is true or fake?"
> **Answer**:  
> *"Yes, sir. That is the exact reason we implemented Component C (Independent Corroboration). If breaking news happens today, our system queries Google News RSS and ClaimReview feeds in real time. If multiple independent publishers (e.g. The Hindu, Indian Express, BBC) report the same event with high semantic Dice similarity, TruthLens identifies external corroboration. If no reliable publisher is covering it, the system transparently outputs 'Needs Further Verification' rather than guessing."*

#### Q3: "What happens if someone uploads a manipulated screenshot?"
> **Answer**:  
> *"Our Image Analysis engine uses a dual OCR pipeline with local Tesseract OCR and cloud fallback. It binds each uploaded image to a unique SHA-256 hash. Once the text is extracted, the user can review and edit it, and TruthLens runs the claim through linguistic classification and live news corroboration. We also include an explicit methodological disclaimer that TruthLens performs semantic text extraction and fact verification, not EXIF photo-forensics."*

#### Q4: "Why 3 verdict tiers instead of just Real or Fake?"
> **Answer**:  
> *"In real-world journalism, forcing binary labels is dangerous. If a model calculates a 51% probability of being Real, calling it 'Real News' gives false assurance to the public. By introducing 'Needs Further Verification' for predictions between 35% and 65%, or for articles under 25 words, TruthLens upholds academic and journalistic integrity by communicating honest uncertainty."*

#### Q5: "How does your SSRF protection in URL verification work?"
> **Answer**:  
> *"When a user inputs a URL, an attacker might try to input `http://localhost:8501` or internal cloud metadata IPs (`169.254.169.254`) to inspect server internals. Our `is_safe_url()` function resolves the target host's IP and strictly verifies that it does not belong to RFC-1918 private subnets, loopback addresses, or cloud metadata endpoints before issuing any HTTP request."*
