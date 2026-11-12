import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH, VECTORIZER_PATH, METRICS_JSON_PATH
from src.classifier import load_ml_assets, analyze_four_components
from src.preprocessor import clean_text_for_inference, detect_language, strip_source_leakage
from src.heuristics import analyze_clickbait, evaluate_source_credibility
from src.scraper import is_safe_url, is_paywall_or_boilerplate
from src.ocr_engine import preprocess_image_for_ocr
from src.fact_checker import (
    compute_semantic_relevance,
    extract_entities_and_keywords,
    search_local_claim_database,
    search_live_news_corroboration
)

# -------------------------------------------------------------
# 1. APPLICATION & ARTIFACT INTEGRITY TESTS
# -------------------------------------------------------------
def test_model_and_vectorizer_exist():
    assert Path(MODEL_PATH).exists(), f"Model missing at {MODEL_PATH}"
    assert Path(VECTORIZER_PATH).exists(), f"Vectorizer missing at {VECTORIZER_PATH}"
    assert Path(METRICS_JSON_PATH).exists(), f"Metrics JSON missing at {METRICS_JSON_PATH}"

def test_load_ml_assets():
    model, vectorizer = load_ml_assets()
    assert model is not None
    assert vectorizer is not None
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1], f"Expected classes [0, 1], got {model.classes_}"

# -------------------------------------------------------------
# 2. MACHINE LEARNING & PROBABILITY CALIBRATION TESTS
# -------------------------------------------------------------
def test_ml_probabilities_validity():
    model, vectorizer = load_ml_assets()
    sample_text = "The government announced a new municipal infrastructure project today in the capital."
    vec = vectorizer.transform([sample_text])
    probs = model.predict_proba(vec)[0]
    
    assert len(probs) == 2
    assert 0.0 <= probs[0] <= 1.0, f"Invalid P(Fake): {probs[0]}"
    assert 0.0 <= probs[1] <= 1.0, f"Invalid P(Real): {probs[1]}"
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3), f"Probabilities do not sum to 1.0: {probs}"

def test_empty_and_short_input_handling():
    res_empty = analyze_four_components("")
    assert res_empty["component_d_assessment"]["final_label"] == "UNABLE TO VERIFY"
    
    res_short = analyze_four_components("Short sentence here.")
    assert res_short["component_d_assessment"]["final_label"] == "UNABLE TO VERIFY"
    assert "brief" in res_short["component_d_assessment"]["explanation"].lower()

def test_language_detection():
    eng_res = detect_language("This is a standard English news article reporting on finance.")
    assert eng_res["language"] == "English"
    assert eng_res["supported_for_ml"] is True
    
    hindi_res = detect_language("यह एक हिंदी समाचार लेख है जो नई दिल्ली से प्रकाशित हुआ है।")
    assert hindi_res["language"] == "Hindi"
    assert hindi_res["supported_for_ml"] is False

# -------------------------------------------------------------
# 3. SECURITY & SSRF PROTECTION TESTS
# -------------------------------------------------------------
def test_ssrf_blocks_localhost_and_private_ips():
    is_safe, msg = is_safe_url("http://localhost:8501")
    assert not is_safe
    assert "localhost" in msg.lower() or "blocked" in msg.lower()

    is_safe, msg = is_safe_url("http://127.0.0.1:8000/admin")
    assert not is_safe
    assert "blocked" in msg.lower() or "denied" in msg.lower()

    is_safe, msg = is_safe_url("ftp://example.com/file.txt")
    assert not is_safe
    assert "protocol" in msg.lower()

def test_ssrf_allows_legitimate_urls():
    is_safe, msg = is_safe_url("https://www.thehindu.com/news/national/")
    assert is_safe
    assert msg == ""

# -------------------------------------------------------------
# 4. OCR & IMAGE PREPROCESSING TESTS
# -------------------------------------------------------------
def test_ocr_image_size_and_format_validation():
    # Empty bytes
    out, err = preprocess_image_for_ocr(b"")
    assert out is None
    assert "no image data" in err.lower()

    # Oversized bytes (>10MB)
    huge_bytes = b"0" * (11 * 1024 * 1024)
    out, err = preprocess_image_for_ocr(huge_bytes)
    assert out is None
    assert "too large" in err.lower()

# -------------------------------------------------------------
# 5. FACT-CHECKING & CIRCULAR CORROBORATION TESTS
# -------------------------------------------------------------
def test_local_claim_database_matches_and_preserves_ratings():
    debunks = search_local_claim_database("COVID-19 vaccine contains 5G microchips")
    assert len(debunks) >= 1
    assert "False" in debunks[0]["rating"]
    assert "microchip" in debunks[0]["claim"].lower()

def test_circular_corroboration_prevention():
    # When excluding 'thehindu.com', search results from The Hindu must be rejected
    results = search_live_news_corroboration(
        headline="Delhi HC directs MCD to monitor unsafe buildings",
        exclude_domain_or_publisher="thehindu.com"
    )
    for art in results:
        assert "the hindu" not in art["publisher"].lower()
        assert "thehindu.com" not in art["url"].lower()

# -------------------------------------------------------------
# 6. VERDICT SYNTHESIS & HONEST CONFIDENCE TESTS
# -------------------------------------------------------------
def test_verdict_three_tier_labels():
    # Fabricated claim with known debunk -> FAKE NEWS
    fc_mock = {
        "verified_claims": [{
            "claim": "Secret documents cure aging",
            "rating": "False / Fabricated",
            "fact_checker": "Snopes"
        }],
        "relevant_news": [],
        "has_debunk": True,
        "has_relevant_news": False
    }
    res = analyze_four_components(
        text="Secret documents leaked from government archives prove that scientists developed a universal cure for human aging.",
        fact_check_result=fc_mock
    )
    assert res["component_d_assessment"]["final_label"] == "FAKE NEWS"
    assert res["component_d_assessment"]["badge_color"] == "#ef4444"

def test_source_mention_in_text_does_not_grant_domain_credibility():
    # Ordinary text mentioning "Reuters" without an actual URL domain
    score, name, classification = evaluate_source_credibility("Reuters reported something strange today in city.")
    assert score == 70
    assert "Independent" in name or "Unlisted" in name
    assert "Verified Reputable Publisher" not in classification
