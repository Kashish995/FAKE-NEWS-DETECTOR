import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    BASE_DIR, MODEL_PATH, VECTORIZER_PATH, METRICS_JSON_PATH,
    CALIBRATION_PATH, CONFUSION_MATRIX_PATH, DB_FILE,
    NEWSAPI_KEY_ENV, FACTCHECK_API_KEY_ENV, OCR_API_KEY_ENV,
    VERDICT_RELIABLE, VERDICT_MISLEADING, VERDICT_VERIFY
)
from src.database import (
    init_db, save_prediction, save_feedback, get_history,
    get_analytics_summary, clear_logs, get_prediction_by_id
)
from src.classifier import load_ml_assets, analyze_four_components
from src.scraper import scrape_article
from src.ocr_engine import extract_text_from_image
from src.fact_checker import query_fact_checking_pipeline
from src.explainability import (
    explain_shap_linear, plot_shap_bar, get_highlighted_tokens_html
)
from src.pdf_report import generate_pdf_report
from src.ui_theme import apply_ui_theme, render_academic_disclaimer

# ---------- STREAMLIT PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="TruthLens – AI Fake News Detection & Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI Styling
apply_ui_theme()

# Initialize SQLite Database
init_db()

# Cache ML Model & Vectorizer
@st.cache_resource
def get_cached_models():
    return load_ml_assets()

try:
    model, vectorizer = get_cached_models()
except Exception as e:
    st.error(f"Failed to load machine learning models: {e}. Please ensure train_model.py has been run.")
    st.stop()

# ---------- SESSION RESTORATION & QUERY PARAMETERS ----------
# 1. User profile persistence
if "username" not in st.session_state:
    st.session_state.username = st.query_params.get("user", "guest")

# 2. Developer API keys persistence
if "newsapi_key" not in st.session_state:
    st.session_state.newsapi_key = NEWSAPI_KEY_ENV
if "factcheck_key" not in st.session_state:
    st.session_state.factcheck_key = FACTCHECK_API_KEY_ENV
if "ocr_key" not in st.session_state:
    st.session_state.ocr_key = OCR_API_KEY_ENV

# 3. Active analysis cache
if "active_analysis" not in st.session_state:
    st.session_state.active_analysis = None

# Navigation options
NAV_PAGES = [
    "🏠 Home",
    "🔍 Analyse News",
    "🌐 URL Verification",
    "📷 Image Analysis",
    "✅ Fact Checking",
    "📜 Analysis History",
    "📊 Analytics Dashboard",
    "ℹ️ About the Project"
]

# Restore page from query params if valid
query_page = st.query_params.get("page", "🔍 Analyse News")
default_nav_idx = NAV_PAGES.index(query_page) if query_page in NAV_PAGES else 1

# ---------- SIDEBAR CONTROLS ----------
with st.sidebar:
    st.markdown("## 🛡️ **TruthLens**")
    st.caption("AI-Powered Verification Core • B.Tech Capstone")
    st.markdown("---")
    
    selected_page = st.radio(
        "NAVIGATION",
        NAV_PAGES,
        index=default_nav_idx,
        key="nav_radio"
    )
    
    # Keep query param in sync with selected page
    if st.query_params.get("page") != selected_page:
        st.query_params["page"] = selected_page

    st.markdown("---")
    st.markdown("#### 👤 Session Profile")
    user_input = st.text_input("Active Username / ID", value=st.session_state.username, key="user_profile_input")
    if user_input.strip() and user_input.strip() != st.session_state.username:
        st.session_state.username = user_input.strip()
        st.query_params["user"] = user_input.strip()
        st.rerun()

    with st.expander("🔑 API Configuration (Optional)", expanded=False):
        st.caption("Configure keys. Built-in fallbacks remain active if omitted.")
        n_key = st.text_input("NewsAPI Key", value=st.session_state.newsapi_key, type="password", key="n_key_input")
        fc_key = st.text_input("Google Fact Check Key", value=st.session_state.factcheck_key, type="password", key="fc_key_input")
        ocr_k = st.text_input("OCR.space Key", value=st.session_state.ocr_key, type="password", key="ocr_k_input")
        if st.button("Save Keys", key="save_keys_btn"):
            st.session_state.newsapi_key = n_key.strip()
            st.session_state.factcheck_key = fc_key.strip()
            st.session_state.ocr_key = ocr_k.strip()
            st.success("API keys updated.")
            st.rerun()

    st.markdown("---")
    st.caption("Verification Engine: **Calibrated SVM (v2.1)**")

# Restore previous analysis from database if ID is in query params
query_analysis_id = st.query_params.get("id")
if query_analysis_id and st.session_state.active_analysis is None:
    try:
        loaded_record = get_prediction_by_id(int(query_analysis_id))
        if loaded_record:
            # Reconstruct analysis components
            raw_text = loaded_record["news_text"]
            # Clean URL domain tag if present
            clean_eval_text = raw_text
            eval_domain = ""
            if raw_text.startswith("[") and "]" in raw_text:
                parts = raw_text.split("]", 1)
                eval_domain = parts[0].replace("[", "").strip()
                clean_eval_text = parts[1].strip()
                
            fc_res = query_fact_checking_pipeline(clean_eval_text[:120], clean_eval_text)
            st.session_state.active_analysis = {
                "res": analyze_four_components(clean_eval_text, eval_domain, fc_res),
                "text": clean_eval_text,
                "input_mode": loaded_record.get("input_mode", "Text"),
                "log_id": loaded_record["id"]
            }
    except Exception:
        pass

# =========================================================================
# COMPONENT RENDERER: FOUR INDEPENDENT ANALYSIS COMPONENTS
# =========================================================================
def render_four_independent_components(analysis_data: dict, source_text: str, input_mode: str, domain_info: str = ""):
    res = analysis_data["res"] if "res" in analysis_data else analysis_data
    log_id = analysis_data.get("log_id")
    
    comp_a = res["component_a_ml"]
    comp_b = res["component_b_source"]
    comp_c = res["component_c_external"]
    comp_d = res["component_d_assessment"]
    
    # 1. Final Result Label & Color
    final_label = comp_d.get("final_label") or ("REAL NEWS" if comp_d["verdict"] == "Likely Reliable" else "FAKE NEWS" if comp_d["verdict"] == "Likely Misleading" else "UNABLE TO VERIFY")
    badge_color = comp_d.get("badge_color", "#22c55e" if final_label == "REAL NEWS" else "#ef4444" if final_label == "FAKE NEWS" else "#eab308")
    icon = "✅" if final_label == "REAL NEWS" else "⚠️" if final_label == "FAKE NEWS" else "🔍"
    
    # 2. Fake Probability
    fake_prob_pct = comp_d.get("fake_probability_pct", round(comp_a.get("probability_fake", 0.5) * 100, 1))
    
    # 3. Short Explanation (1-2 plain-English sentences)
    short_explanation = comp_d.get("explanation", "Verification completed.")
    
    # 4. News Source (Publisher name)
    publisher_name = comp_b.get("source_name", "Independent / Unlisted Source")
    if domain_info and publisher_name.startswith("Domain:"):
        publisher_name = domain_info
        
    # 5. Verification Note
    if comp_c.get("has_debunk"):
        verification_note = f"Prior debunk found in ClaimReview ({len(comp_c.get('verified_claims', []))} fact-check record(s) contradict this claim)."
    elif comp_c.get("has_relevant_news"):
        top_art = comp_c["relevant_news"][0]
        verification_note = f"Corroborated by independent coverage from {top_art['publisher']} ({top_art['relevance_score']}% relevance match)."
    else:
        verification_note = "No matching independent fact-checks or news reports found."
        
    # =============================================================
    # SIMPLIFIED RESULT CARD (SHOWN TO ORDINARY USERS)
    # =============================================================
    st.markdown(f"""
    <div style="background-color: rgba(15, 23, 42, 0.04); border: 2px solid {badge_color}; border-left: 8px solid {badge_color}; padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #64748b;">Verification Assessment</span>
            <span style="font-size: 1.05rem; font-weight: 700; color: {badge_color};">Fake Probability: <b>{fake_prob_pct}%</b></span>
        </div>
        <h2 style="color: {badge_color} !important; margin: 4px 0 12px 0; font-size: 2rem; border: none; padding: 0;">
            {icon} {final_label}
        </h2>
        <p style="margin: 0 0 14px 0; font-size: 1.05rem; color: inherit; line-height: 1.6;">
            {short_explanation}
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.9rem; padding-top: 10px; border-top: 1px solid rgba(148, 163, 184, 0.2);">
            <div><b>🏛️ News Source:</b> {publisher_name}</div>
            <div><b>🌐 Verification:</b> {verification_note}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =============================================================
    # OPTIONAL EXPANDABLE SECTION: VIEW DETAILED ANALYSIS
    # =============================================================
    with st.expander("🔍 View Detailed Analysis", expanded=False):
        if comp_d.get("has_conflict"):
            st.warning(f"⚠️ {comp_d.get('conflict_message', 'Conflicting evidence detected between stylistic model and external sources.')}")
            
        st.markdown("#### 🔬 Model Probabilities & Linguistic Features")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("P(Fake News)", f"{comp_a['probability_fake']*100:.1f}%")
        with col_m2:
            st.metric("P(Real News)", f"{comp_a['probability_real']*100:.1f}%")
        with col_m3:
            st.metric("Stylistic Signal", comp_a["prediction_label"])
            
        st.caption(f"Linguistic summary: {comp_a['summary']}")
        
        # Linear SHAP Feature Attribution
        shap_features = comp_a.get("shap_features", [])
        if shap_features:
            st.markdown("##### 📊 Top Feature Attributions (Linear Impact)")
            fig_shap = plot_shap_bar(shap_features)
            if fig_shap:
                st.pyplot(fig_shap)
                
        # Word-by-Word Token Highlights
        st.markdown("##### 🔤 Word-by-Word Explainable AI Highlights")
        highlighted_html = get_highlighted_tokens_html(source_text, model, vectorizer)
        st.markdown(
            f'<div style="background-color: rgba(148, 163, 184, 0.06); border: 1px solid rgba(148, 163, 184, 0.2); '
            f'padding: 14px; border-radius: 8px; line-height: 1.8; max-height: 280px; overflow-y: auto;">'
            f'{highlighted_html}</div>',
            unsafe_allow_html=True
        )
        
        # Fact-Checking & Live Coverage Details
        fcs = comp_c.get("verified_claims", [])
        rel_news = comp_c.get("relevant_news", [])
        
        if fcs:
            st.markdown(f"##### ⚠️ Verified Fact-Check Records ({len(fcs)} found)")
            for fc in fcs[:3]:
                st.markdown(f"• **[{fc.get('fact_checker', 'Fact Checker')}]**: *{fc.get('rating', 'Evaluated')}* - {fc.get('claim', '')}")
                if fc.get("review_url"):
                    st.markdown(f"  [Read Full Debunk]({fc['review_url']})")
                    
        if rel_news:
            st.markdown(f"##### 📰 Independent Corroborating News Coverage ({len(rel_news)} sources)")
            for art in rel_news:
                st.markdown(f"• **[{art.get('publisher', 'News')}]** [{art.get('title', 'Article')}]({art.get('url', '#')}) (Relevance: {art.get('relevance_score', 0)}%)")
                
        st.markdown("---")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.markdown("##### 📄 Export Verification Dossier")
            pdf_payload = {
                "log_id": log_id if log_id else "AUDIT",
                "input_mode": input_mode,
                "language": res.get("language_info", {}).get("language", "English"),
                "verdict": final_label,
                "confidence": comp_d["confidence"],
                "text": source_text,
                "clickbait_score": comp_b.get("clickbait_score", 0.0),
                "credibility_score": comp_b.get("credibility_score", 70.0),
                "probability": res.get("probability", [0.5, 0.5]),
                "shap_features": comp_a.get("shap_features", []),
                "fact_checks": fcs,
                "related_news": rel_news
            }
            try:
                pdf_bytes = generate_pdf_report(pdf_payload)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download PDF Verification Report",
                        data=pdf_bytes,
                        file_name=f"TruthLens_Report_{log_id if log_id else 'audit'}.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{log_id if log_id else 'tmp'}"
                    )
            except Exception as e:
                st.warning(f"PDF export temporarily unavailable: {e}")
                
        with col_act2:
            st.markdown("##### 💬 Verification Telemetry Feedback")
            if log_id:
                col_fb1, col_fb2 = st.columns(2)
                with col_fb1:
                    if st.button("👍 Correct Verdict", key=f"fb_ok_{log_id}"):
                        save_feedback(log_id, "Correct")
                        st.success("Telemetry logged: Correct")
                with col_fb2:
                    if st.button("👎 Incorrect Verdict", key=f"fb_bad_{log_id}"):
                        save_feedback(log_id, "Incorrect")
                        st.warning("Telemetry logged: Incorrect")
            else:
                st.caption("Telemetry feedback is active when prediction is stored.")

# =========================================================================
# PAGE 1: HOME
# =========================================================================
if selected_page == "🏠 Home":
    st.title("🛡️ TruthLens: AI-Powered Fake News Detection & Verification")
    st.markdown("### *A B.Tech Final-Year Capstone Project in Applied Machine Learning & NLP*")
    
    st.markdown("""
    **TruthLens** is an advanced verification system engineered to detect online misinformation, 
    evaluate editorial authenticity, and explain predictive reasoning through transparent Explainable AI (XAI).
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="tl-card">
            <h4>🔬 Multi-Signal ML</h4>
            <p>Benchmarked across <b>SVM, Logistic Regression, Naive Bayes, and Random Forest</b> with Platt Sigmoid Probability Calibration.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tl-card">
            <h4>🧠 Explainable AI (XAI)</h4>
            <p>Demystifying black-box predictions with <b>Linear SHAP</b> attributions and semantic token weight visualizers.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="tl-card">
            <h4>🌐 Multi-Source Fact Check</h4>
            <p>Corroborates claims against verified <b>ClaimReview</b> debunk databases and live mainstream news feeds.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📐 Four Independent Analysis Components")
    st.markdown("""
    ```mermaid
    graph LR
        In[Input Article] --> A[A. Machine Learning Classification]
        In --> B[B. Source Credibility Assessment]
        In --> C[C. External News Verification]
        A & B & C --> D[D. Final Evidence Assessment]
        D --> Out[Likely Reliable | Likely Misleading | Needs Verification]
    ```
    """)
    
    render_academic_disclaimer()

# =========================================================================
# PAGE 2: ANALYSE NEWS (TEXT CLAIM)
# =========================================================================
elif selected_page == "🔍 Analyse News":
    st.title("🔍 Text-Based News Claim Analysis")
    st.write("Input a news headline or complete article text to evaluate all four independent analysis components.")
    
    sample_options = [
        "Select a sample claim to test...",
        "Delhi HC directs MCD, CM to set up panel to monitor unsafe buildings during GRAP",
        "ISRO successfully launched its latest Earth Observation Satellite EOS-08 from Sriharikota.",
        "Breaking: Secret documents prove pharmaceutical companies hide cure for aging.",
        "COVID-19 vaccines contain 5G tracking microchips to control the population."
    ]
    selected_sample = st.selectbox("Quick Test Samples:", sample_options, key="sample_select")
    default_text = selected_sample if selected_sample != sample_options[0] else ""
    
    input_text = st.text_area(
        "Enter News Text or Claim:",
        value=default_text,
        height=180,
        placeholder="Paste full article text, headline, or claim here...",
        key="text_claim_area"
    )
    
    words = len(input_text.split())
    st.caption(f"Word Count: **{words}** words")
    
    if st.button("RUN TRUTHLENS DIAGNOSTICS", type="primary", key="btn_run_text"):
        if not input_text.strip():
            st.warning("Please enter claim text to analyze.")
        else:
            with st.spinner("Executing four independent analysis components..."):
                fc_result = query_fact_checking_pipeline(
                    headline=input_text[:120],
                    full_text=input_text,
                    google_key=st.session_state.factcheck_key,
                    newsapi_key=st.session_state.newsapi_key
                )
                res = analyze_four_components(
                    text=input_text,
                    domain_or_url="",
                    fact_check_result=fc_result
                )
                log_id = save_prediction(
                    news_text=input_text,
                    prediction=res["verdict"],
                    confidence=res["confidence"],
                    input_mode="Text",
                    clickbait_score=res["clickbait_score"],
                    credibility_score=res["credibility_score"],
                    username=st.session_state.username,
                    language=res["language_info"]["language"]
                )
                st.session_state.active_analysis = {
                    "res": res,
                    "text": input_text,
                    "input_mode": "Text",
                    "log_id": log_id
                }
                st.query_params["id"] = str(log_id)
                
    if st.session_state.active_analysis and st.session_state.active_analysis.get("input_mode") == "Text":
        render_four_independent_components(
            st.session_state.active_analysis,
            st.session_state.active_analysis["text"],
            "Text Claim"
        )
        
    render_academic_disclaimer()

# =========================================================================
# PAGE 3: URL VERIFICATION
# =========================================================================
elif selected_page == "🌐 URL Verification":
    st.title("🌐 News Article URL Verification")
    st.write("Paste a direct link to any live news article to extract content, evaluate domain reputation, and run verification.")
    
    url_input = st.text_input(
        "Paste Article URL:",
        placeholder="e.g. https://www.thehindu.com/incoming/.../article71683027.ece",
        key="url_input_box"
    )
    
    if st.button("FETCH & ANALYSE ARTICLE", type="primary", key="btn_run_url"):
        if not url_input.strip():
            st.warning("Please paste a valid article URL.")
        else:
            with st.spinner("Extracting article headline, body, and metadata using Trafilatura..."):
                scraped = scrape_article(url_input)
                
            # If extraction failed or paywall blocked content
            if not scraped["success"]:
                st.error("Article extraction unsuccessful. Please paste the article text manually.")
                st.info("The target web page is either protected by an aggressive subscription paywall, bot protection, or contains insufficient text.")
            else:
                extracted_text = scraped["full_text"]
                st.success(f"Article Identified: **{scraped['title']}** (Domain: `{scraped['domain']}`, {scraped['word_count']} words)")
                st.text_area("EXTRACTED ARTICLE PREVIEW", value=extracted_text, height=130, disabled=True, key="extracted_preview_area")
                
                with st.spinner("Executing four independent analysis components..."):
                    fc_result = query_fact_checking_pipeline(
                        headline=scraped["title"] if scraped["title"] else extracted_text[:120],
                        full_text=extracted_text,
                        google_key=st.session_state.factcheck_key,
                        newsapi_key=st.session_state.newsapi_key
                    )
                    res = analyze_four_components(
                        text=extracted_text,
                        domain_or_url=scraped["domain"],
                        fact_check_result=fc_result
                    )
                    log_id = save_prediction(
                        news_text=f"[{scraped['domain']}] {extracted_text}",
                        prediction=res["verdict"],
                        confidence=res["confidence"],
                        input_mode="URL",
                        clickbait_score=res["clickbait_score"],
                        credibility_score=res["credibility_score"],
                        username=st.session_state.username,
                        language=res["language_info"]["language"]
                    )
                    st.session_state.active_analysis = {
                        "res": res,
                        "text": extracted_text,
                        "input_mode": "URL",
                        "domain": scraped["domain"],
                        "log_id": log_id
                    }
                    st.query_params["id"] = str(log_id)
                    
    if st.session_state.active_analysis and st.session_state.active_analysis.get("input_mode") == "URL":
        render_four_independent_components(
            st.session_state.active_analysis,
            st.session_state.active_analysis["text"],
            "URL Scraper",
            domain_info=st.session_state.active_analysis.get("domain", "")
        )
        
    render_academic_disclaimer()

# =========================================================================
# PAGE 4: IMAGE ANALYSIS (OCR)
# =========================================================================
elif selected_page == "📷 Image Analysis":
    st.title("📷 Image-Based News Analysis (OCR)")
    st.write("Upload a news screenshot or headline graphic to extract text via Optical Character Recognition and assess credibility.")
    
    st.info("ℹ️ **Methodological Scope**: This module performs Optical Character Recognition (OCR) to extract text from screenshots. It classifies the extracted claim stylistically and cross-references it with news archives. It does not perform digital photo-forensics, EXIF tampering analysis, or deepfake detection.")
    
    uploaded_img = st.file_uploader("Upload News Screenshot (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"], key="img_uploader")
    
    if "ocr_extracted_text" not in st.session_state:
        st.session_state.ocr_extracted_text = ""
        
    if uploaded_img:
        st.image(uploaded_img, width=420, caption="Uploaded Document Preview")
        
        if st.button("EXTRACT TEXT FROM IMAGE", type="secondary", key="btn_run_ocr"):
            with st.spinner("Executing Optical Character Recognition (OCR)..."):
                extracted_text, ocr_err = extract_text_from_image(
                    uploaded_img.getvalue(),
                    api_key=st.session_state.ocr_key
                )
                
            if ocr_err:
                st.error(f"OCR Error: {ocr_err}")
            elif not extracted_text or len(extracted_text.split()) < 5:
                st.warning("Article extraction unsuccessful. Please paste the article text manually.")
            else:
                st.session_state.ocr_extracted_text = extracted_text
                st.success("Text extracted successfully! Please review and correct any OCR errors below before analysis.")
                
        if st.session_state.ocr_extracted_text:
            edited_text = st.text_area(
                "Review & Edit Extracted OCR Text:",
                value=st.session_state.ocr_extracted_text,
                height=140,
                key="ocr_edit_area"
            )
            
            if st.button("RUN TRUTHLENS DIAGNOSTICS", type="primary", key="btn_run_img_verify"):
                if not edited_text.strip():
                    st.warning("Please provide extracted or corrected text to analyze.")
                else:
                    with st.spinner("Evaluating extracted claim across all components..."):
                        fc_result = query_fact_checking_pipeline(
                            headline=edited_text[:120],
                            full_text=edited_text,
                            google_key=st.session_state.factcheck_key,
                            newsapi_key=st.session_state.newsapi_key
                        )
                        res = analyze_four_components(
                            text=edited_text,
                            domain_or_url="",
                            fact_check_result=fc_result
                        )
                        log_id = save_prediction(
                            news_text=f"[OCR] {edited_text}",
                            prediction=res["verdict"],
                            confidence=res["confidence"],
                            input_mode="Image OCR",
                            clickbait_score=res["clickbait_score"],
                            credibility_score=res["credibility_score"],
                            username=st.session_state.username,
                            language=res["language_info"]["language"]
                        )
                        st.session_state.active_analysis = {
                            "res": res,
                            "text": edited_text,
                            "input_mode": "Image OCR",
                            "log_id": log_id
                        }
                        st.query_params["id"] = str(log_id)
                        
    if st.session_state.active_analysis and st.session_state.active_analysis.get("input_mode") == "Image OCR":
        render_four_independent_components(
            st.session_state.active_analysis,
            st.session_state.active_analysis["text"],
            "Image OCR"
        )
        
    render_academic_disclaimer()

# =========================================================================
# PAGE 5: FACT CHECKING ENGINE
# =========================================================================
elif selected_page == "✅ Fact Checking":
    st.title("✅ Fact-Checking Search Engine")
    st.write("Directly search ClaimReview debunk databases and live journalistic corroboration feeds.")
    
    fc_query = st.text_input("Enter Claim or Topic to Fact-Check:", placeholder="e.g. COVID vaccine microchip or Delhi unsafe buildings", key="fc_query_box")
    
    if st.button("SEARCH FACT-CHECK ARCHIVES", type="primary", key="btn_run_fc"):
        if not fc_query.strip():
            st.warning("Please enter a claim to search.")
        else:
            with st.spinner("Querying ClaimReview databases and live feeds..."):
                fc_res = query_fact_checking_pipeline(
                    headline=fc_query,
                    full_text=fc_query,
                    google_key=st.session_state.factcheck_key,
                    newsapi_key=st.session_state.newsapi_key
                )
                
            claims = fc_res.get("verified_claims", [])
            news = fc_res.get("relevant_news", [])
            
            st.markdown(f"### Results for: *'{fc_query}'*")
            
            if claims:
                st.markdown(f"#### 🔍 Verified ClaimReview Debunks ({len(claims)})")
                for c in claims:
                    with st.container():
                        st.markdown(f"""
                        <div class="tl-card-misleading">
                            <span class="tl-badge tl-badge-red">{c.get('rating', 'Evaluated')}</span>
                            <h4 style="margin: 8px 0 4px 0;">Claim: {c.get('claim', '')}</h4>
                            <p style="margin: 0; font-size: 0.9rem;"><b>Claimant:</b> {c.get('claimant', 'Unknown')} | <b>Fact-Checker:</b> {c.get('fact_checker', 'IFCN Signatory')}</p>
                            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #64748b;">{c.get('review_title', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if c.get("review_url"):
                            st.markdown(f"[🔗 View Original Debunk Investigation]({c['review_url']})")
            else:
                st.info("No prior fact-check debunks recorded in ClaimReview databases for this query.")
                
            if news:
                st.markdown(f"#### 📰 Genuinely Relevant Multi-Source News ({len(news)})")
                for art in news:
                    st.markdown(f"• **[{art.get('publisher', 'News')}]** [{art.get('title', 'Headline')}]({art.get('url', '#')}) ({art.get('date', 'Recent')}) — *Relevance: {art.get('relevance_score')}%*")
            else:
                st.info("No sufficiently relevant matching articles found.")

    render_academic_disclaimer()

# =========================================================================
# PAGE 6: ANALYSIS HISTORY & AUDIT TRAIL
# =========================================================================
elif selected_page == "📜 Analysis History":
    st.title("📜 Prediction History & Audit Logs")
    st.write("Browse, filter, and inspect previously analyzed claims stored in the relational SQLite telemetry store.")
    
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_kw = st.text_input("Filter History by Keyword:", placeholder="Search claims or verdicts...", key="hist_search_box")
    with col_f2:
        user_filter = st.selectbox("User Scope:", ["All Users", f"Only {st.session_state.username}"], key="hist_user_scope")
    with col_f3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Logs", type="secondary", key="btn_clear_hist"):
            target = st.session_state.username if user_filter != "All Users" else None
            clear_logs(target)
            st.session_state.active_analysis = None
            st.query_params.clear()
            st.success("Selected history cleared.")
            st.rerun()
            
    target_user = st.session_state.username if user_filter != "All Users" else None
    df_history = get_history(target_user=target_user, search_query=search_kw, limit=100)
    
    if df_history.empty:
        st.info("No evaluation telemetry recorded yet matching criteria.")
    else:
        st.dataframe(df_history, use_container_width=True)
        csv_bytes = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export History as CSV",
            data=csv_bytes,
            file_name="truthlens_analysis_history.csv",
            mime="text/csv",
            key="dl_hist_csv"
        )

# =========================================================================
# PAGE 7: ANALYTICS DASHBOARD
# =========================================================================
elif selected_page == "📊 Analytics Dashboard":
    st.title("📊 Empirical Machine Learning Analytics Dashboard")
    st.write("Real evaluation metrics calculated strictly from held-out test datasets and live database telemetry.")
    
    if Path(METRICS_JSON_PATH).exists():
        with open(METRICS_JSON_PATH, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}
        
    if not Path(METRICS_JSON_PATH).exists() or not metrics_data.get("best_metrics"):
        st.info("ℹ️ Model evaluation metrics are not yet available. Please run `python train_model.py` to generate empirical evaluation metrics.")
    else:
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Model Test Accuracy", f"{best_metrics['accuracy']}%", delta="Held-out 20% test")
        with col_k2:
            st.metric("Test Precision", f"{best_metrics['precision']}%")
        with col_k3:
            st.metric("Test Recall", f"{best_metrics['recall']}%")
        with col_k4:
            st.metric("Test F1-Score", f"{best_metrics['f1_score']}%")
        
    st.markdown("---")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### 1. Multi-Algorithm Benchmarking Comparison")
        comp = metrics_data.get("comparison", [])
        if comp:
            df_comp = pd.DataFrame(comp)
            fig_comp, ax_comp = plt.subplots(figsize=(6, 3.4))
            fig_comp.patch.set_facecolor('none')
            ax_comp.set_facecolor('none')
            
            x = np.arange(len(df_comp))
            width = 0.35
            ax_comp.bar(x - width/2, df_comp["accuracy"], width, label="Accuracy", color="#3b82f6")
            ax_comp.bar(x + width/2, df_comp["f1_score"], width, label="F1-Score", color="#10b981")
            
            ax_comp.set_ylabel("Percentage (%)", color='#64748b', fontsize=8)
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(df_comp["classifier"], rotation=12, fontsize=8, color='#64748b')
            ax_comp.set_ylim(85, 100)
            ax_comp.spines['bottom'].set_color('#64748b')
            ax_comp.spines['left'].set_color('#64748b')
            ax_comp.spines['top'].set_color('none')
            ax_comp.spines['right'].set_color('none')
            ax_comp.tick_params(colors='#64748b', labelsize=8)
            ax_comp.legend(facecolor='none', edgecolor='none', labelcolor='#64748b', fontsize=8)
            st.pyplot(fig_comp)
            
    with col_v2:
        st.markdown("#### 2. Test Confusion Matrix (Linear SVM)")
        cm_data = metrics_data.get("confusion_matrix")
        if cm_data:
            cm = np.array(cm_data)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
            fig_cm.patch.set_facecolor('none')
            ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > 1000 else "#1e293b", fontweight="bold")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_xticklabels(["Pred Fake", "Pred Real"], color='#64748b', fontsize=8)
            ax_cm.set_yticks([0, 1])
            ax_cm.set_yticklabels(["Actual Fake", "Actual Real"], color='#64748b', fontsize=8)
            ax_cm.tick_params(colors='#64748b', labelsize=8)
            st.pyplot(fig_cm)
            
    st.markdown("---")
    
    st.markdown("#### 3. Real-Time Telemetry & Verdict Distribution")
    summary, _ = get_analytics_summary()
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Total Claims Analysed", summary["total"])
    with col_d2:
        st.metric("Reliable vs Misleading Ratio", f"{summary['reliable']} / {summary['misleading']}")
    with col_d3:
        st.metric("Ambiguous / Requiring Verification", summary["needs_verification"])
        
    if summary["total"] > 0:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        fig_pie.patch.set_facecolor('none')
        counts = [summary["reliable"], summary["misleading"], summary["needs_verification"]]
        labels = ["Reliable", "Misleading", "Needs Verification"]
        colors = ["#22c55e", "#ef4444", "#eab308"]
        valid_idx = [i for i, c in enumerate(counts) if c > 0]
        if valid_idx:
            ax_pie.pie(
                [counts[i] for i in valid_idx],
                labels=[labels[i] for i in valid_idx],
                colors=[colors[i] for i in valid_idx],
                autopct="%1.1f%%",
                textprops=dict(color="#64748b")
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

    render_academic_disclaimer()

# =========================================================================
# PAGE 8: ABOUT THE PROJECT
# =========================================================================
elif selected_page == "ℹ️ About the Project":
    st.title("ℹ️ About TruthLens: Capstone Documentation")
    
    st.markdown("""
    ### Project Overview
    **TruthLens** is a publication-grade B.Tech final-year major project developed to address 
    information pollution, online propaganda, and viral misinformation using rigorous Machine Learning and NLP.
    
    ### Four Independent Analysis Components
    1. **A. Machine Learning Classification**: Unbiased, calibrated linear SVM ($P(\\text{Real})$ and $P(\\text{Fake})$).
    2. **B. Source Credibility Assessment**: Evaluates domain authority and publisher reputation. Never determines truth alone.
    3. **C. External News Verification**: Real-time ClaimReview debunks and live multi-source news coverage with semantic relevance scoring.
    4. **D. Final Evidence Assessment**: 3-tier verdict synthesis with explicit conflict detection.
    """)
    
    render_academic_disclaimer()