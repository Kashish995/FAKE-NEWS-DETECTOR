import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    MODEL_PATH, VECTORIZER_PATH,
    VERDICT_RELIABLE, VERDICT_MISLEADING, VERDICT_VERIFY,
    MIN_WORDS_FOR_RELIABLE_ML, CONFIDENCE_HIGH_THRESHOLD
)
from .preprocessor import clean_text_for_inference, detect_language
from .heuristics import analyze_clickbait, evaluate_source_credibility

_MODEL = None
_VECTORIZER = None

def load_ml_assets():
    """Caches and returns the trained classifier and vectorizer."""
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        if not Path(MODEL_PATH).exists() or not Path(VECTORIZER_PATH).exists():
            raise FileNotFoundError(f"Model or vectorizer not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _VECTORIZER = pickle.load(f)
    return _MODEL, _VECTORIZER

def synthesize_verdict(
    text: str,
    domain_or_url: str = "",
    fact_check_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synthesizes a robust 3-tier verdict by combining:
    1. Language safety guardrails
    2. Length validation
    3. Calibrated ML probabilistic inference
    4. Clickbait & sensationalism index
    5. Source domain credibility
    6. External Fact-Checking ClaimReviews & live news corroboration
    
    Returns structured diagnostic telemetry.
    """
    model, vectorizer = load_ml_assets()
    
    cleaned_text = clean_text_for_inference(text)
    words = cleaned_text.split()
    word_count = len(words)
    
    # 1. Language Guardrails
    lang_info = detect_language(cleaned_text)
    
    # 2. Heuristics: Clickbait & Domain
    clickbait_score, clickbait_triggers = analyze_clickbait(text)
    credibility_score, source_name, source_classification = evaluate_source_credibility(
        domain_or_url if domain_or_url else text
    )
    
    # 3. Fact Checking signals
    has_debunk = False
    fact_check_claims = []
    related_news = []
    if fact_check_result:
        has_debunk = fact_check_result.get("has_debunk", False)
        fact_check_claims = fact_check_result.get("verified_claims", [])
        related_news = fact_check_result.get("related_news", [])
        
    # 4. ML Probabilistic Prediction
    vec = vectorizer.transform([cleaned_text])
    
    # Handle calibrated probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]  # [P(Fake), P(Real)]
    else:
        # Fallback for raw linear model
        decision = model.decision_function(vec)[0]
        prob_real = 1.0 / (1.0 + np.exp(-decision))
        probs = np.array([1.0 - prob_real, prob_real])
        
    p_fake = float(probs[0])
    p_real = float(probs[1])
    
    # 5. Scientific 3-Tier Decision Synthesis
    verdict = VERDICT_VERIFY
    confidence = max(p_fake, p_real)
    explanation = ""
    badge_color = "#eab308"  # Amber
    
    # Rule A: Multilingual limitation
    if not lang_info["supported_for_ml"]:
        verdict = VERDICT_VERIFY
        confidence = 0.50
        badge_color = "#eab308"
        explanation = (
            f"ℹ️ {lang_info['notice']} "
            f"Cross-referencing live databases identified {len(related_news)} related reports."
        )
        
    # Rule B: Explicit verified debunk from fact-checking database
    elif has_debunk:
        verdict = VERDICT_MISLEADING
        confidence = max(confidence, 0.90)
        badge_color = "#ef4444"
        top_fc = fact_check_claims[0]
        explanation = (
            f"⚠️ Prior Fact-Check Debunk Found: Rated '{top_fc.get('rating', 'False')}' "
            f"by {top_fc.get('fact_checker', 'independent fact-checker')}."
        )

    # Rule C: Verified Reputable Media Publisher (e.g. The Hindu, BBC, Reuters, etc.)
    elif credibility_score >= 85 and clickbait_score < 50:
        verdict = VERDICT_RELIABLE
        confidence = max(p_real, 0.90)
        badge_color = "#22c55e"
        explanation = (
            f"✅ Verified Mainstream Publisher: Source originates from {source_name} ({credibility_score}/100 credibility). "
            f"Corroborated by journalistic reporting standards and low sensationalism indicators."
        )
        
    # Rule D: Text too short for reliable standalone stylistic ML
    elif word_count < MIN_WORDS_FOR_RELIABLE_ML:
        verdict = VERDICT_VERIFY
        confidence = 0.50
        badge_color = "#eab308"
        explanation = (
            f"🔍 Content is brief ({word_count} words). "
            f"Stylistic NLP classification requires sufficient paragraph structure (at least {MIN_WORDS_FOR_RELIABLE_ML} words) "
            f"to evaluate syntax reliably. Showing multi-source corroboration and fact-checking checks below."
        )
        
    # Rule E: Machine Learning high-confidence classification
    elif p_real >= CONFIDENCE_HIGH_THRESHOLD and clickbait_score < 50:
        verdict = VERDICT_RELIABLE
        confidence = p_real
        badge_color = "#22c55e"
        explanation = (
            f"✅ Reliable Journalistic Style: Calibrated statistical models identify formal reporting conventions, "
            f"neutral syntactic structure, and standard attribution patterns."
        )
    elif p_fake >= CONFIDENCE_HIGH_THRESHOLD or (clickbait_score >= 60 and p_fake > 0.50):
        verdict = VERDICT_MISLEADING
        confidence = p_fake
        badge_color = "#ef4444"
        explanation = (
            f"⚠️ Potential Misinformation Indicators: Stylistic vectors reflect sensationalist formatting, "
            f"high clickbait indicators, or structural deviations from verified journalistic standards."
        )
    else:
        # Ambiguous / Boundary Zone (35% to 65% probability)
        verdict = VERDICT_VERIFY
        confidence = max(p_fake, p_real)
        badge_color = "#eab308"
        explanation = (
            "🔍 Ambiguous / Mixed Signals: The text displays balanced or conflicting lexical indicators. "
            "Neither high reliability nor clear fabrication could be established stylistically. "
            "Consult the multi-source evidence below."
        )
        
    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "probability": [p_fake, p_real],
        "badge_color": badge_color,
        "explanation": explanation,
        "language_info": lang_info,
        "clickbait_score": clickbait_score,
        "clickbait_triggers": clickbait_triggers,
        "credibility_score": credibility_score,
        "source_name": source_name,
        "source_classification": source_classification,
        "word_count": word_count,
        "fact_checks": fact_check_claims,
        "related_news": related_news
    }
