import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .config import (
    MODEL_PATH, VECTORIZER_PATH,
    VERDICT_RELIABLE, VERDICT_MISLEADING, VERDICT_VERIFY,
    MIN_WORDS_FOR_RELIABLE_ML, CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD
)
from .preprocessor import clean_text_for_inference, detect_language
from .heuristics import analyze_clickbait, evaluate_source_credibility
from .explainability import explain_shap_linear

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

def get_class_probabilities(model, vec_text) -> Tuple[float, float]:
    """
    Safely retrieves probability of Fake and Real using model.classes_
    to prevent class index inversion bugs.
    
    Returns:
        Tuple[p_fake, p_real]
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_text)[0]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(vec_text)[0]
        prob_real = 1.0 / (1.0 + np.exp(-decision))
        probs = np.array([1.0 - prob_real, prob_real])
    else:
        probs = np.array([0.5, 0.5])

    classes = list(getattr(model, "classes_", [0, 1]))
    
    # Label 0 represents Fake, Label 1 represents Real
    fake_idx = classes.index(0) if 0 in classes else (classes.index("Fake") if "Fake" in classes else 0)
    real_idx = classes.index(1) if 1 in classes else (classes.index("Real") if "Real" in classes else 1)
    
    p_fake = float(probs[fake_idx])
    p_real = float(probs[real_idx])
    
    # Normalize to ensure exact mathematical consistency
    total = p_fake + p_real
    if total > 0:
        p_fake /= total
        p_real /= total
    else:
        p_fake, p_real = 0.5, 0.5
        
    return p_fake, p_real

def analyze_four_components(
    text: str,
    domain_or_url: str = "",
    fact_check_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes four independent analysis components:
    A. MACHINE LEARNING CLASSIFICATION (Unbiased, never altered)
    B. SOURCE CREDIBILITY (Independent domain authority)
    C. EXTERNAL NEWS VERIFICATION (Strictly relevant ClaimReviews & news)
    D. FINAL EVIDENCE ASSESSMENT (3-tier synthesis with honest uncertainty)
    """
    model, vectorizer = load_ml_assets()
    
    cleaned_text = clean_text_for_inference(text)
    words = cleaned_text.split()
    word_count = len(words)
    lang_info = detect_language(cleaned_text)
    
    # -------------------------------------------------------------
    # COMPONENT A: MACHINE LEARNING CLASSIFICATION
    # (Evaluates linguistic/stylistic patterns. Probabilities are NEVER altered)
    # -------------------------------------------------------------
    vec = vectorizer.transform([cleaned_text])
    p_fake, p_real = get_class_probabilities(model, vec)
    
    # Raw ML stylistic decision
    if word_count < MIN_WORDS_FOR_RELIABLE_ML:
        ml_prediction = "Insufficient Text Length"
        ml_confidence = round(max(p_fake, p_real), 4)
        ml_summary = (
            f"Content is too brief ({word_count} words). Stylistic NLP models require at least "
            f"{MIN_WORDS_FOR_RELIABLE_ML} words to evaluate structural patterns reliably."
        )
    elif not lang_info["supported_for_ml"]:
        ml_prediction = f"Unsupported Script ({lang_info['language']})"
        ml_confidence = 0.50
        ml_summary = lang_info["notice"]
    elif p_real >= CONFIDENCE_HIGH_THRESHOLD:
        ml_prediction = "Likely Reliable Style"
        ml_confidence = round(p_real, 4)
        ml_summary = (
            f"Linguistic vectors reflect standard formal journalistic syntax, neutral attribution "
            f"patterns, and factual wire conventions (P(Real): {p_real*100:.1f}%)."
        )
    elif p_fake >= CONFIDENCE_HIGH_THRESHOLD:
        ml_prediction = "Likely Misleading Style"
        ml_confidence = round(p_fake, 4)
        ml_summary = (
            f"Linguistic vectors reflect structural patterns typical of sensationalist or non-standard "
            f"fabricated articles (P(Fake): {p_fake*100:.1f}%)."
        )
    else:
        # Ambiguous / closely balanced probabilities (e.g. 46.5% vs 53.5%)
        ml_prediction = "Uncertain / Neutral Style"
        ml_confidence = round(max(p_fake, p_real), 4)
        ml_summary = (
            f"Stylistic probabilities are closely balanced (Fake: {p_fake*100:.1f}%, Real: {p_real*100:.1f}%). "
            f"Linguistic structure exhibits neither strong wire-journalism nor clear clickbait markers."
        )
        
    shap_features = explain_shap_linear(cleaned_text, model, vectorizer, top_k=10)
    
    component_a_ml = {
        "prediction_label": ml_prediction,
        "raw_prediction": "Real" if p_real >= 0.50 else "Fake",
        "confidence": float(ml_confidence),
        "confidence_pct": round(ml_confidence * 100, 1),
        "probability_fake": round(p_fake, 4),
        "probability_real": round(p_real, 4),
        "shap_features": shap_features,
        "word_count": word_count,
        "summary": ml_summary
    }
    
    # -------------------------------------------------------------
    # COMPONENT B: SOURCE CREDIBILITY
    # (Independent domain assessment. Never determines truth alone)
    # -------------------------------------------------------------
    cred_score, source_name, source_classification = evaluate_source_credibility(
        domain_or_url if domain_or_url else text
    )
    cb_score, cb_triggers = analyze_clickbait(text)
    
    component_b_source = {
        "source_name": source_name,
        "domain": domain_or_url,
        "credibility_score": cred_score,
        "classification": source_classification,
        "clickbait_score": cb_score,
        "clickbait_triggers": cb_triggers,
        "summary": (
            f"Source matched '{source_name}' ({cred_score}/100 credibility index). "
            f"Sensationalism rating: {int(cb_score)}/100."
        )
    }
    
    # -------------------------------------------------------------
    # COMPONENT C: EXTERNAL NEWS VERIFICATION
    # (ClaimReview debunks and strictly relevant news matches)
    # -------------------------------------------------------------
    if fact_check_result:
        fc_claims = fact_check_result.get("verified_claims", [])
        rel_news = fact_check_result.get("relevant_news", [])
        has_debunk = fact_check_result.get("has_debunk", False)
        ext_status = fact_check_result.get("external_status", "No sufficiently relevant matching articles found.")
        ext_summary = fact_check_result.get("external_summary", "No sufficiently relevant matching articles found.")
    else:
        fc_claims = []
        rel_news = []
        has_debunk = False
        ext_status = "No sufficiently relevant matching articles found."
        ext_summary = "No sufficiently relevant matching articles found."
        
    component_c_external = {
        "verified_claims": fc_claims,
        "relevant_news": rel_news,
        "has_debunk": has_debunk,
        "has_relevant_news": len(rel_news) > 0,
        "status": ext_status,
        "summary": ext_summary
    }
    
    # -------------------------------------------------------------
    # COMPONENT D: FINAL EVIDENCE ASSESSMENT
    # (Transparent 3-Tier Synthesis with Honest Confidence & Conflict Detection)
    # Tiers: LIKELY RELIABLE | LIKELY MISLEADING | NEEDS FURTHER VERIFICATION
    # -------------------------------------------------------------
    has_conflict = False
    conflict_message = None
    final_label = "NEEDS FURTHER VERIFICATION"
    verdict = VERDICT_VERIFY
    badge_color = "#eab308"
    final_confidence = round(max(p_fake, p_real), 4)
    final_explanation = ""
    
    # Check if any fact-check record is affirmatively "True"
    has_verified_true = False
    for fc in fc_claims:
        rating_lower = fc.get("rating", "").lower()
        if any(tr in rating_lower for tr in ["true", "correct", "accurate", "verified"]):
            if not any(fl in rating_lower for fl in ["false", "misleading", "partly", "half", "fake"]):
                has_verified_true = True
                break

    # Rule 1: Verified Fact-Check Debunk present -> LIKELY MISLEADING
    if has_debunk:
        final_label = "LIKELY MISLEADING"
        verdict = VERDICT_MISLEADING
        top_fc = fc_claims[0]
        final_confidence = max(p_fake, 0.85)
        badge_color = "#ef4444"
        final_explanation = (
            f"Independent fact-checking records evaluate this claim as '{top_fc.get('rating', 'False')}'. "
            f"Evaluated by {top_fc.get('fact_checker', 'Independent Fact Checker')}."
        )
        if p_real > 0.60:
            has_conflict = True
            conflict_message = "Conflicting Evidence: Stylistic ML identified formal writing style, but factual archives contradict the claim."
            
    # Rule 2: Multi-source external news corroboration from independent publishers
    elif len(rel_news) >= 1 and rel_news[0].get("relevance_score", 0) >= 35:
        top_art = rel_news[0]
        if p_fake >= CONFIDENCE_HIGH_THRESHOLD:
            # Conflict: Model detected fake style, but external coverage exists
            final_label = "NEEDS FURTHER VERIFICATION"
            verdict = VERDICT_VERIFY
            has_conflict = True
            badge_color = "#eab308"
            conflict_message = "Conflicting Evidence: Stylistic model detected sensational markers, but external news reports cover this event."
            final_explanation = (
                f"Conflicting Evidence: Stylistic ML flagged non-standard markers ({p_fake*100:.1f}% fake probability), "
                f"yet matching reports were located from {top_art['publisher']}."
            )
        elif p_real >= 0.50 or has_verified_true:
            final_label = "LIKELY RELIABLE"
            verdict = VERDICT_RELIABLE
            final_confidence = round(p_real, 4)
            badge_color = "#22c55e"
            final_explanation = (
                f"Reporting is corroborated by independent news coverage from {top_art['publisher']} "
                f"(relevance: {top_art.get('relevance_score')}%) and is consistent with factual reporting standards."
            )
        else:
            final_label = "NEEDS FURTHER VERIFICATION"
            verdict = VERDICT_VERIFY
            badge_color = "#eab308"
            final_explanation = (
                f"External coverage was located from {top_art['publisher']}, but stylistic analysis remains ambiguous. "
                f"Independent verification recommended."
            )
            
    # Rule 3: High-confidence stylistic ML Real with no debunks and low clickbait
    elif p_real >= 0.75 and cb_score < 40 and word_count >= MIN_WORDS_FOR_RELIABLE_ML:
        final_label = "LIKELY RELIABLE"
        verdict = VERDICT_RELIABLE
        final_confidence = round(p_real, 4)
        badge_color = "#22c55e"
        final_explanation = (
            f"The article reflects neutral journalistic styling, standard attribution syntax, "
            f"and factual wire structural conventions (P(Real): {p_real*100:.1f}%)."
        )
        
    # Rule 4: High-confidence stylistic ML Fake with sensational clickbait markers
    elif (p_fake >= 0.75 or cb_score >= 65) and word_count >= MIN_WORDS_FOR_RELIABLE_ML:
        final_label = "LIKELY MISLEADING"
        verdict = VERDICT_MISLEADING
        final_confidence = round(max(p_fake, cb_score / 100.0), 4)
        badge_color = "#ef4444"
        final_explanation = (
            f"Text exhibits significant structural patterns typical of fabricated content "
            f"or sensationalism ({p_fake*100:.1f}% stylistic fake probability)."
        )
        
    # Rule 5: Insufficient text length, ambiguous probability (e.g. 46.5% vs 53.5%), or uncorroborated claims
    else:
        final_label = "NEEDS FURTHER VERIFICATION"
        verdict = VERDICT_VERIFY
        badge_color = "#eab308"
        if word_count < MIN_WORDS_FOR_RELIABLE_ML:
            final_explanation = (
                f"Input length ({word_count} words) is below the reliable threshold ({MIN_WORDS_FOR_RELIABLE_ML}+ words). "
                f"Insufficient independent evidence found to confirm or refute the claim."
            )
        else:
            final_explanation = (
                f"Available evidence is inconclusive (Model: {p_fake*100:.1f}% fake, {p_real*100:.1f}% real) "
                f"with no prior fact-check debunks found. Independent corroboration is required."
            )
            
    component_d_assessment = {
        "final_label": final_label,
        "verdict": verdict,
        "confidence": float(final_confidence),
        "confidence_pct": round(final_confidence * 100, 1),
        "fake_probability_pct": round(p_fake * 100, 1),
        "real_probability_pct": round(p_real * 100, 1),
        "has_conflict": has_conflict,
        "conflict_message": conflict_message,
        "badge_color": badge_color,
        "explanation": final_explanation
    }
    
    return {
        "component_a_ml": component_a_ml,
        "component_b_source": component_b_source,
        "component_c_external": component_c_external,
        "component_d_assessment": component_d_assessment,
        "language_info": lang_info,
        "word_count": word_count,
        # Flat keys for PDF and backward compatibility
        "final_label": final_label,
        "verdict": verdict,
        "confidence": float(final_confidence),
        "fake_probability": round(p_fake * 100, 1),
        "probability": [p_fake, p_real],
        "explanation": final_explanation,
        "clickbait_score": cb_score,
        "credibility_score": cred_score,
        "fact_checks": fc_claims,
        "related_news": rel_news
    }

def synthesize_verdict(text: str, domain_or_url: str = "", fact_check_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compatibility wrapper around analyze_four_components."""
    return analyze_four_components(text, domain_or_url, fact_check_result)
