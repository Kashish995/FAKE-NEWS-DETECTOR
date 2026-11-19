import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_truthlens_presentation(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    # Theme Colors
    BG_DARK = RGBColor(15, 23, 42)       # Slate 900
    CARD_BG = RGBColor(30, 41, 59)       # Slate 800
    BORDER_COLOR = RGBColor(51, 65, 85)  # Slate 700
    TEXT_WHITE = RGBColor(248, 250, 252) # White/Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184) # Slate 400
    ACCENT_CYAN = RGBColor(14, 165, 233) # Sky 500
    ACCENT_GREEN = RGBColor(16, 185, 129)# Emerald 500
    ACCENT_AMBER = RGBColor(245, 158, 11)# Amber 500

    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, title, category="TRUTHLENS | B.TECH CAPSTONE"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN
        
        p_title = tf.add_paragraph()
        p_title.text = title
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE
        p_title.space_before = Pt(4)

    def add_card(slide, left, top, width, height, title, items, badge="", accent=ACCENT_CYAN):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = BORDER_COLOR
        card.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        if badge:
            p0.text = badge.upper()
            p0.font.size = Pt(10)
            p0.font.bold = True
            p0.font.color.rgb = accent
            
            p_title = tf.add_paragraph()
            p_title.text = title
            p_title.font.size = Pt(16)
            p_title.font.bold = True
            p_title.font.color.rgb = TEXT_WHITE
            p_title.space_before = Pt(2)
        else:
            p0.text = title
            p0.font.size = Pt(16)
            p0.font.bold = True
            p0.font.color.rgb = TEXT_WHITE

        for item in items:
            p = tf.add_paragraph()
            p.space_before = Pt(6)
            if isinstance(item, tuple):
                lead, rest = item
                r1 = p.add_run()
                r1.text = f"• {lead}: "
                r1.font.bold = True
                r1.font.size = Pt(12)
                r1.font.color.rgb = TEXT_WHITE
                
                r2 = p.add_run()
                r2.text = rest
                r2.font.size = Pt(12)
                r2.font.color.rgb = TEXT_MUTED
            else:
                r = p.add_run()
                r.text = f"• {item}"
                r.font.size = Pt(12)
                r.font.color.rgb = TEXT_MUTED

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p_badge = tf.paragraphs[0]
    p_badge.text = "FINAL YEAR MAJOR PROJECT PRESENTATION | B.TECH CSE"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_CYAN

    p_title = tf.add_paragraph()
    p_title.text = "TruthLens: AI-Powered Fake News Detection & Multi-Source Verification System"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.space_before = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "A Multi-Modal, Explainable NLP Framework Bridging Linguistic Machine Learning with Real-Time Journalistic Corroboration"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = TEXT_MUTED
    p_sub.space_before = Pt(10)

    p_meta = tf.add_paragraph()
    p_meta.text = "Guided By: Academic Project Committee  |  Presented To: Head of Department & Examiners\nDomain: Machine Learning, Natural Language Processing, Explainable AI (XAI)"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = ACCENT_GREEN
    p_meta.space_before = Pt(28)

    # ==========================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Problem Statement & Industry Motivation", "BACKGROUND & MOTIVATION")

    add_card(s2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
        "The Misinformation Epidemic",
        [
            ("Viral Velocity", "False news spreads 6x faster than factual reporting across social media algorithms."),
            ("Societal Impact", "Electoral manipulation, public health panic, financial stock fraud, and community polarization."),
            ("Content Scale", "Billions of daily unstructured articles, WhatsApp forwards, and screenshot graphics render manual policing impossible.")
        ],
        badge="THE CRISIS", accent=ACCENT_AMBER
    )

    add_card(s2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
        "Why Existing Systems Fail",
        [
            ("Black-Box Fallacy", "Traditional ML outputs raw probabilities without explainability, leading to skepticism."),
            ("Static Datasets", "Models trained in 2020 cannot verify events happening today (lack of temporal real-time grounding)."),
            ("False Certainty", "Binary Real/Fake models force 51/49 predictions into confident Real News, misleading users.")
        ],
        badge="EXISTING GAPS", accent=RGBColor(239, 68, 68)
    )

    add_card(s2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
        "The TruthLens Paradigm",
        [
            ("Multi-Signal Synthesis", "Combines stylistic linguistic NLP with independent journalistic fact-check databases."),
            ("Multi-Modal Ingestion", "Handles raw text, live website URLs, and optical screenshot OCR seamlessly."),
            ("Honest Uncertainty", "Incorporates a rigorous 3-tier verdict model that explicitly flags unverified claims.")
        ],
        badge="OUR INNOVATION", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 3: SYSTEM ARCHITECTURE
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Four Independent Analysis Components", "ARCHITECTURAL BLUEPRINT")

    col_w = Inches(2.75)
    gap = Inches(0.22)
    left_m = Inches(0.8)

    add_card(s3, left_m, Inches(1.8), col_w, Inches(5.0),
        "Component A: ML Core",
        [
            ("Algorithm", "Calibrated Linear Support Vector Classifier (SVM)."),
            ("Stylistic Analysis", "Evaluates formal journalistic syntax vs sensational vocabulary."),
            ("Platt Scaling", "Converts raw distance to hyperplane into calibrated probabilities.")
        ],
        badge="COMPONENT A", accent=ACCENT_CYAN
    )

    add_card(s3, left_m + (col_w + gap), Inches(1.8), col_w, Inches(5.0),
        "Component B: Source Profiler",
        [
            ("Domain Authority", "Tracks 60+ verified news domains and flagged clickbait networks."),
            ("Sensationalism Index", "Detects sensational clickbait markers and exaggerated claims."),
            ("Independence", "Reputation alone never overrides ML evidence.")
        ],
        badge="COMPONENT B", accent=RGBColor(168, 85, 247)
    )

    add_card(s3, left_m + 2*(col_w + gap), Inches(1.8), col_w, Inches(5.0),
        "Component C: Corroboration",
        [
            ("ClaimReview API", "Live IFCN fact-check database (Reuters, BoomLive, PolitiFact)."),
            ("Multi-Source News", "Live Google News RSS cross-referencing."),
            ("Semantic Dice Metric", "Calculates genuine claim alignment to prevent keyword mismatches.")
        ],
        badge="COMPONENT C", accent=ACCENT_AMBER
    )

    add_card(s3, left_m + 3*(col_w + gap), Inches(1.8), col_w, Inches(5.0),
        "Component D: Synthesis",
        [
            ("3-Tier Taxonomy", "Likely Reliable, Likely Misleading, Needs Further Verification."),
            ("Conflict Detection", "Detects when style is formal but external debunks exist."),
            ("Honest Uncertainty", "Reports when data is insufficient.")
        ],
        badge="COMPONENT D", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 4: MACHINE LEARNING & NLP PIPELINE
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Machine Learning Modeling & Text Preprocessing", "NLP & CLASSIFICATION")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
        "NLP Preprocessing & Feature Extraction",
        [
            ("Text Normalization", "Lowercasing, regex URL/mention stripping, and Unicode punctuation cleansing."),
            ("Stopword Preservation", "Selectively retains linguistic stylistic markers that distinguish factual reporting from emotion-driven text."),
            ("TF-IDF Vectorization", "10,000 top n-grams (unigrams & bigrams) with sublinear term-frequency scaling (1 + log(tf))."),
            ("Sparsity Optimization", "Sparse matrix representation achieves sub-10ms inference latency during real-time user requests.")
        ],
        badge="FEATURE ENGINEERING", accent=ACCENT_CYAN
    )

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
        "Model Selection & Empirical Benchmark",
        [
            ("Linear SVM (Champion)", "Achieved highest test accuracy (93.8%) and superior separation in high-dimensional text space."),
            ("Multinomial Naive Bayes", "Baseline model (89.2% accuracy), suffered from strong independence assumptions."),
            ("Logistic Regression", "Strong baseline (92.1% accuracy), slightly less robust on ambiguous vocabulary boundaries."),
            ("Platt Sigmoid Scaling", "Calibrated decision boundaries: P(Real) = 1 / (1 + exp(-decision)), ensuring mathematically sound probability outputs.")
        ],
        badge="MODEL EVALUATION", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 5: MULTI-MODAL INGESTION
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Multi-Modal Verification Capabilities", "INPUT MODES & ROBUSTNESS")

    add_card(s5, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
        "1. Direct Text Claim Engine",
        [
            ("Quick Testing", "Curated dropdown samples for instant HOD demonstration."),
            ("Word Count Guard", "Requires >= 25 words to prevent arbitrary predictions on single phrases."),
            ("Language Support", "Detects non-Latin scripts and displays explicit notice rather than guessing.")
        ],
        badge="MODULE 1", accent=ACCENT_CYAN
    )

    add_card(s5, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
        "2. News URL Web Scraper",
        [
            ("SSRF Security Shield", "Blocks RFC-1918 private IPs, AWS/GCP metadata endpoints, and loopback attacks."),
            ("Multi-Stage Parser", "Trafilatura -> JSON-LD Schema -> Fallback HTML paragraph aggregator."),
            ("Paywall Detection", "Detects subscription modals and alerts user to paste text manually.")
        ],
        badge="MODULE 2", accent=RGBColor(168, 85, 247)
    )

    add_card(s5, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
        "3. Screenshot Image OCR",
        [
            ("Dual OCR Architecture", "Local Tesseract OCR engine + Cloud OCR fallback for high reliability."),
            ("Rate Limit Protection", "Exponential backoff with Retry-After header parsing on HTTP 429 errors."),
            ("SHA-256 Hash Sync", "Binds predictions to image hash, automatically purging old results upon new uploads.")
        ],
        badge="MODULE 3", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 6: FACT CHECKING & CORROBORATION
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "External Corroboration & Fact-Checking", "GROUND-TRUTH VERIFICATION")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
        "ClaimReview & Journalistic News Feeds",
        [
            ("IFCN Fact-Check Archive", "Cross-references claims against global debunk records from certified organizations."),
            ("Live Google News RSS", "Queries live journalistic coverage across The Hindu, Indian Express, BBC, Reuters, etc."),
            ("Anti-Circulation Filter", "Disqualifies the article's own publisher domain to prevent self-referential confirmation bias."),
            ("Direct Hyperlinks", "Provides validated web hyperlinks to original articles and source newspapers.")
        ],
        badge="EXTERNAL EVIDENCE", accent=ACCENT_AMBER
    )

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
        "Semantic Relevance Scoring (Dice Metric)",
        [
            ("Mathematical Formula", "Dice(A, B) = 2|A n B| / (|A| + |B|). Normalizes token length to prevent false 100% matches."),
            ("Acronym & Entity Boosting", "Awards priority weighting for critical entities (e.g. 'ISRO', 'GRAP', 'Supreme Court')."),
            ("Calibrated Thresholds", "Requires >= 25% semantic alignment to qualify as genuine corroboration."),
            ("Nuanced Rating Precedence", "Classifies nuanced claims (e.g. 'Partly False', 'Misleading Context') before binary False.")
        ],
        badge="SEMANTIC ALIGNMENT", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 7: EXPLAINABLE AI (XAI)
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Explainable AI (XAI) & Interpretability", "TRANSPARENCY & TRUST")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
        "SHAP (SHapley Additive exPlanations)",
        [
            ("Game-Theoretic Foundation", "Calculates the exact marginal contribution of each word towards the final verdict."),
            ("Linear Attribution", "SHAP weights accurately reflect how the linear SVM hyperplane divides lexical patterns."),
            ("Demystifying Black-Box ML", "Explains *why* the algorithm arrived at its score rather than asserting an arbitrary output."),
            ("Educational Utility", "Allows journalists and researchers to audit bias in editorial writing styles.")
        ],
        badge="METHODOLOGY", accent=ACCENT_CYAN
    )

    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
        "Visual Diagnostic Dashboard",
        [
            ("Linguistic Impact Plot", "Horizontal bar chart showing top 10 most influential positive & negative tokens."),
            ("Green Markers (Factual)", "Attribution verbs ('reported', 'confirmed', 'spokesperson', 'official data')."),
            ("Red Markers (Sensational)", "Emotive adjectives, urgent clickbait ('shocking', 'miracle cure', 'unbelievable', 'secret')."),
            ("Interactive Telemetry", "Probability distribution bar charts and source credibility scorecards.")
        ],
        badge="VISUAL EVIDENCE", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 8: RESULTS & 30-CASE BENCHMARK
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Empirical Validation & Generalization Benchmark", "PERFORMANCE METRICS")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
        "Held-Out 20% Test Evaluation",
        [
            ("Test Accuracy", "93.8% on standardized held-out real vs fake news test dataset."),
            ("Precision", "94.2% (minimizes false accusations against legitimate journalists)."),
            ("Recall", "93.1% (detects synthesized fabrications and disinformation campaigns)."),
            ("F1-Score", "93.6% harmonic balance between precision and sensitivity."),
            ("Inference Latency", "< 35 milliseconds per article on standard commodity hardware.")
        ],
        badge="METRICS SUMMARY", accent=ACCENT_CYAN
    )

    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
        "30-Case Real-World Generalization Benchmark",
        [
            ("Mainstream Factual News", "100% alignment with real-world reporting (The Hindu, BBC, Reuters)."),
            ("Viral Fabrications", "100% correctly flagged as Likely Misleading with debunk citations."),
            ("Ambiguous & Local Events", "Honest uncertainty: 80% assigned to Needs Further Verification rather than false certainty."),
            ("Multilingual Fallback", "Detects Hindi and non-Latin scripts, advising external corroboration rather than erroneous classification.")
        ],
        badge="REAL-WORLD STRESS TEST", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 9: FULL TECH STACK & ARCHITECTURE
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Technical Stack & Software Architecture", "ENGINEERING SPECIFICATIONS")

    col_w3 = Inches(3.6)
    gap3 = Inches(0.4)

    add_card(s9, Inches(0.8), Inches(1.8), col_w3, Inches(5.0),
        "Machine Learning & NLP",
        [
            ("Scikit-Learn", "Model training, TF-IDF vectorization, Platt probability calibration."),
            ("NumPy & Pandas", "Array manipulations, metrics calculation, and telemetry storage."),
            ("SHAP", "Model explainability and token contribution calculation."),
            ("Tesseract OCR", "Optical character extraction from news screenshot images.")
        ],
        badge="CORE ML", accent=ACCENT_CYAN
    )

    add_card(s9, Inches(0.8) + col_w3 + gap3, Inches(1.8), col_w3, Inches(5.0),
        "Web Scraping & Pipelines",
        [
            ("Trafilatura", "State-of-the-art main-text article extraction from web pages."),
            ("BeautifulSoup4 & lxml", "XML/RSS parsing and JSON-LD schema metadata extraction."),
            ("Requests & Socket", "HTTP network calls with SSRF IP validation and timeouts."),
            ("ReportLab", "Automated PDF forensic verification dossier export.")
        ],
        badge="DATA PIPELINE", accent=RGBColor(168, 85, 247)
    )

    add_card(s9, Inches(0.8) + 2*(col_w3 + gap3), Inches(1.8), col_w3, Inches(5.0),
        "Frontend & Deployment",
        [
            ("Streamlit Runtime", "Reactive, clean responsive UI with dark engineering theme."),
            ("SQLite Relational DB", "Persistent analysis audit logs and feedback storage."),
            ("Pytest Suite", "20/20 automated unit and integration tests passing."),
            ("Streamlit Cloud", "Containerized live production deployment on Debian Linux.")
        ],
        badge="APPLICATION & DEVOPS", accent=ACCENT_GREEN
    )

    # ==========================================
    # SLIDE 10: CONCLUSION & FUTURE WORK
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Summary of Contributions & Future Scope", "CONCLUSION & ROADMAP")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
        "Key Project Contributions",
        [
            ("Multi-Signal Synthesis", "Successfully resolved the limitations of isolated ML models by unifying style with live facts."),
            ("Honest Uncertainty", "Eliminated false binary certainty, establishing an academic-standard 3-tier verdict model."),
            ("Explainable AI", "Integrated SHAP interpretability to give users transparent insight into model reasoning."),
            ("Production-Ready Stability", "Zero-downtime deployment with dual OCR fallbacks and automated regression tests.")
        ],
        badge="ACADEMIC OUTCOMES", accent=ACCENT_CYAN
    )

    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
        "Future Enhancements & Roadmap",
        [
            ("Multilingual Transformers", "Fine-tuning IndicBERT and mBERT for comprehensive vernacular language classification."),
            ("Multimodal Vision (ViT)", "Direct visual forensic tampering and deepfake image analysis."),
            ("Browser Extension", "Real-time client-side TruthLens banner while browsing social media feeds."),
            ("Knowledge Graphs", "Constructing automated temporal entity graphs for geopolitical claim tracking.")
        ],
        badge="FUTURE RESEARCH", accent=ACCENT_GREEN
    )

    # Save presentation
    prs.save(output_path)
    print(f"Presentation successfully generated at: {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "TruthLens_Presentation.pptx"
    create_truthlens_presentation(out_file)
