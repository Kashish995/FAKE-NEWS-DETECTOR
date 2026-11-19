import io
import sys
import json
import pytest
import requests
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODEL_PATH, VECTORIZER_PATH, METRICS_JSON_PATH,
    VERDICT_RELIABLE, VERDICT_MISLEADING, VERDICT_VERIFY
)
from src.classifier import load_ml_assets, analyze_four_components, get_class_probabilities
from src.preprocessor import clean_text_for_inference, detect_language, strip_source_leakage
from src.heuristics import analyze_clickbait, evaluate_source_credibility
from src.scraper import is_safe_url, is_paywall_or_boilerplate, scrape_article
from src.ocr_engine import (
    preprocess_image_for_ocr, extract_text_from_image, compute_image_hash
)
from src.fact_checker import (
    compute_semantic_relevance,
    extract_entities_and_keywords,
    search_local_claim_database,
    search_live_news_corroboration,
    classify_fact_check_rating,
    query_fact_checking_pipeline
)

def create_dummy_png_bytes(text: str = "Test Headline") -> bytes:
    img = Image.new('RGB', (300, 60), color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 20), text, fill='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# =============================================================
# 1. APPLICATION & ARTIFACT INTEGRITY TESTS
# =============================================================
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

# =============================================================
# 2. MACHINE LEARNING & PROBABILITY CALIBRATION TESTS
# =============================================================
def test_get_class_probabilities_mapping():
    model, vectorizer = load_ml_assets()
    sample = "The prime minister held a bilateral summit today discussing international trade policy."
    vec = vectorizer.transform([sample])
    p_fake, p_real = get_class_probabilities(model, vec)
    
    assert 0.0 <= p_fake <= 1.0
    assert 0.0 <= p_real <= 1.0
    assert np.isclose(p_fake + p_real, 1.0, atol=1e-3)

def test_uncertain_prediction_yields_needs_verification():
    """When probabilities are closely balanced (e.g. 52% Real / 48% Fake), verdict must be NEEDS FURTHER VERIFICATION."""
    # Mocking model probabilities to be closely balanced
    with patch("src.classifier.get_class_probabilities", return_value=(0.465, 0.535)):
        res = analyze_four_components(
            text="The regional transportation authority scheduled a maintenance meeting for tomorrow afternoon to review rail signals.",
            domain_or_url="independent-blog.com"
        )
        verdict = res["component_d_assessment"]["final_label"]
        assert verdict == "NEEDS FURTHER VERIFICATION", f"Expected NEEDS FURTHER VERIFICATION for balanced probabilities, got {verdict}"
        assert res["component_d_assessment"]["verdict"] == VERDICT_VERIFY

def test_conflicting_external_evidence():
    """When model detects fake style but external news confirms the event, conflict is raised."""
    with patch("src.classifier.get_class_probabilities", return_value=(0.78, 0.22)):
        fc_mock = {
            "verified_claims": [],
            "relevant_news": [{
                "title": "Regional transport meeting scheduled for tomorrow",
                "publisher": "Reuters",
                "relevance_score": 75,
                "url": "https://reuters.com/article/123"
            }],
            "has_debunk": False,
            "has_relevant_news": True
        }
        res = analyze_four_components(
            text="Shocking disaster! Regional transport meeting scheduled for tomorrow in dramatic turn of events!",
            fact_check_result=fc_mock
        )
        assert res["component_d_assessment"]["has_conflict"] is True
        assert res["component_d_assessment"]["final_label"] == "NEEDS FURTHER VERIFICATION"

def test_language_detection():
    eng_res = detect_language("This is a standard English news article reporting on finance.")
    assert eng_res["language"] == "English"
    assert eng_res["supported_for_ml"] is True
    
    hindi_res = detect_language("यह एक हिंदी समाचार लेख है जो नई दिल्ली से प्रकाशित हुआ है।")
    assert hindi_res["language"] == "Hindi"
    assert hindi_res["supported_for_ml"] is False

# =============================================================
# 3. OCR & IMAGE SYNCHRONIZATION TESTS
# =============================================================
def test_image_hash_generation():
    bytes1 = create_dummy_png_bytes("First Image")
    bytes2 = create_dummy_png_bytes("Second Image")
    hash1 = compute_image_hash(bytes1)
    hash2 = compute_image_hash(bytes2)
    
    assert hash1 != ""
    assert hash2 != ""
    assert hash1 != hash2, "Different images must produce different SHA-256 hashes"
    assert hash1 == compute_image_hash(bytes1), "Same image must produce identical SHA-256 hash"

def test_ocr_http_429_backoff_and_retry():
    """Tests that HTTP 429 triggers exponential backoff and returns informative guidance if quota remains busy."""
    bytes_img = create_dummy_png_bytes("Sample News")
    
    # Mock requests.post returning HTTP 429 with Retry-After header
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"Retry-After": "1"}
    
    with patch("src.ocr_engine._try_local_pytesseract", return_value=(None, "Unavailable")):
        with patch("requests.post", return_value=mock_resp_429) as mock_post:
            with patch("time.sleep", return_value=None):  # don't actually sleep in tests
                text, err, src = extract_text_from_image(bytes_img, api_key="test_key", max_retries=1)
                assert text is None
                assert "429" in err
                assert src is None
                assert mock_post.call_count >= 2

def test_ocr_timeout_handling():
    bytes_img = create_dummy_png_bytes("Sample News")
    with patch("src.ocr_engine._try_local_pytesseract", return_value=(None, "Unavailable")):
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
            with patch("time.sleep", return_value=None):
                text, err, src = extract_text_from_image(bytes_img, max_retries=1)
                assert text is None
                assert "timed out" in err.lower()

def test_ocr_local_pytesseract_fallback():
    bytes_img = create_dummy_png_bytes("Supreme Court Judgment")
    with patch("src.ocr_engine._try_local_pytesseract", return_value=("Supreme Court Judgment on River", None)):
        text, err, src = extract_text_from_image(bytes_img)
        assert text == "Supreme Court Judgment on River"
        assert err is None
        assert src == "Local Tesseract OCR"

def test_ocr_empty_and_invalid_image():
    out, err = preprocess_image_for_ocr(b"")
    assert out is None
    assert "no image data" in err.lower()

    # Oversized image
    huge = b"0" * (11 * 1024 * 1024)
    out, err = preprocess_image_for_ocr(huge)
    assert out is None
    assert "too large" in err.lower()

# =============================================================
# 4. URL VERIFICATION & SSRF TESTS
# =============================================================
def test_ssrf_blocks_localhost_and_internal_ips():
    is_safe, msg = is_safe_url("http://localhost:8501")
    assert not is_safe
    assert "localhost" in msg.lower() or "blocked" in msg.lower()

    is_safe, msg = is_safe_url("http://127.0.0.1:8000/api")
    assert not is_safe
    assert "blocked" in msg.lower() or "denied" in msg.lower()

    is_safe, msg = is_safe_url("http://169.254.169.254/latest/meta-data")
    assert not is_safe
    assert "blocked" in msg.lower()

    is_safe, msg = is_safe_url("ftp://example.com/archive.txt")
    assert not is_safe
    assert "protocol" in msg.lower()

def test_ssrf_allows_public_urls():
    is_safe, msg = is_safe_url("https://www.thehindu.com/news/national/")
    assert is_safe
    assert msg == ""

def test_scraper_handles_paywall_and_short_content():
    assert is_paywall_or_boilerplate("Subscribe now to continue reading this premium story.") is True
    assert is_paywall_or_boilerplate("The supreme court delivered a landmark verdict today.") is False

def test_scraper_handles_network_failure():
    with patch("src.scraper.is_safe_url", return_value=(True, "")):
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection failed")):
            res = scrape_article("https://nonexistent-news-domain-12345.com/story")
            assert res["success"] is False
            assert "unsuccessful" in res["error"].lower() or "failure" in res["error"].lower()

# =============================================================
# 5. FACT-CHECKING & CLAIM MATCHING TESTS
# =============================================================
def test_compute_semantic_relevance_calibration():
    # Exact match
    exact = compute_semantic_relevance(
        "Supreme Court orders Yamuna floodplain protection",
        "Supreme Court orders Yamuna floodplain protection"
    )
    assert exact == 1.0

    # Partial match
    partial = compute_semantic_relevance(
        "Yamuna river pollution penalties imposed by court",
        "Supreme Court calls for encroachment-free Yamuna floodplain, penalties on states over pollution"
    )
    assert 0.30 <= partial < 1.0, f"Expected realistic partial match, got {partial}"

    # Unrelated match
    unrelated = compute_semantic_relevance(
        "Mars rover discovers volcanic rock formation on red planet",
        "Supreme Court calls for encroachment-free Yamuna floodplain, penalties on states over pollution"
    )
    assert unrelated < 0.15, f"Expected near zero relevance for unrelated stories, got {unrelated}"

def test_classify_fact_check_rating():
    assert classify_fact_check_rating("False") == "False / Fabricated"
    assert classify_fact_check_rating("Pants on Fire!") == "False / Fabricated"
    assert classify_fact_check_rating("Misleading context") == "Misleading / Missing Context"
    assert classify_fact_check_rating("Partly False") == "Partly False / Mixture"
    assert classify_fact_check_rating("True") == "True / Verified"

def test_local_claim_database_debunk_retrieval():
    debunks = search_local_claim_database("COVID-19 vaccine contains 5G microchips")
    assert len(debunks) >= 1
    assert "False" in debunks[0]["rating"]
    assert "microchip" in debunks[0]["claim"].lower()

def test_query_fact_checking_pipeline_offline():
    with patch("src.fact_checker.search_live_news_corroboration", return_value=[]):
        res = query_fact_checking_pipeline("Unrecorded random event without coverage")
        assert res["has_debunk"] is False
        assert res["has_relevant_news"] is False
        assert res["external_status"] == "No sufficiently relevant matching articles found."

def test_rss_news_link_extraction_integrity():
    from src.fact_checker import search_live_news_corroboration
    mock_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
    <channel>
        <item>
            <title>Delhi air quality likely to turn Very Poor today - The Indian Express</title>
            <link>https://news.google.com/rss/articles/CBMiTEST12345</link>
            <pubDate>Mon, 16 Nov 2026 14:33:52 GMT</pubDate>
            <source url="https://indianexpress.com">The Indian Express</source>
        </item>
    </channel>
    </rss>"""
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_rss
    mock_resp.text = mock_rss.decode("utf-8")
    
    with patch("requests.get", return_value=mock_resp):
        results = search_live_news_corroboration("Delhi air quality likely to turn Very Poor today")
        assert len(results) >= 1
        item = results[0]
        assert item["url"].startswith("http")
        assert item["url"] != "#"
        assert item["source_url"] == "https://indianexpress.com"
        assert item["publisher"] == "The Indian Express"

