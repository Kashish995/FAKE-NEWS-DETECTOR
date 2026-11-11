import io
import re
from datetime import datetime
from fpdf import FPDF
from typing import Dict, Any, List, Optional

def sanitize_for_pdf(text: Any) -> str:
    """
    Sanitizes Unicode strings to prevent FPDFUnicodeEncodingException when
    rendering with core Latin-1 fonts (Helvetica).
    Replaces smart quotes, em-dashes, bullets, and unsupported glyphs with ASCII/Latin-1 equivalents.
    """
    if text is None:
        return ""
    text = str(text)
    
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
        '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
        '\u2013': '-', '\u2014': '-', '\u2015': '-',
        '\u2026': '...', '\u2022': '*', '\u00a0': ' ',
        '\u2032': "'", '\u2033': '"', '\u00b4': "'", '`': "'",
        '–': '-', '—': '-', '‘': "'", '’': "'", '“': '"', '”': '"', '…': '...'
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
        
    # Replace non-latin-1 characters with readable placeholders
    return text.encode('latin-1', errors='replace').decode('latin-1')

class TruthLensPDF(FPDF):
    def header(self):
        # Header banner
        self.set_fill_color(15, 23, 42)  # Dark slate #0f172a
        self.rect(0, 0, 210, 24, 'F')
        
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "TruthLens - AI-Powered News Verification Report", ln=True, align="C")
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 4, "B.Tech Final-Year Capstone Project in Artificial Intelligence & NLP", ln=True, align="C")
        self.ln(8)
        
    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}} | TruthLens System Audit Telemetry | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

def generate_pdf_report(data: Dict[str, Any]) -> Optional[bytes]:
    """
    Generates a publication-grade PDF report from diagnostic analysis data.
    Gracefully sanitizes all text to prevent encoding crashes.
    """
    try:
        pdf = TruthLensPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Section: Executive Summary & Metadata
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "1. EXECUTIVE VERIFICATION SUMMARY", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        
        # Metadata Table
        col_w = 47
        log_id_str = sanitize_for_pdf(data.get('log_id', 'PREVIEW'))
        input_mode_str = sanitize_for_pdf(data.get('input_mode', 'Text'))
        lang_str = sanitize_for_pdf(data.get('language', 'English'))
        
        pdf.cell(col_w, 5, f"Audit ID: TL-{log_id_str}", border=0)
        pdf.cell(col_w, 5, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", border=0)
        pdf.cell(col_w, 5, f"Input Channel: {input_mode_str}", border=0)
        pdf.cell(col_w, 5, f"Language: {lang_str}", border=0, ln=True)
        pdf.ln(3)
        
        # Verdict Card Box
        verdict = data.get("verdict", "Needs Further Verification")
        if verdict == "Likely Reliable":
            box_r, box_g, box_b = 34, 197, 94   # Green
            bg_r, bg_g, bg_b = 240, 253, 244
        elif verdict == "Likely Misleading":
            box_r, box_g, box_b = 239, 68, 68   # Red
            bg_r, bg_g, bg_b = 254, 242, 242
        else:
            box_r, box_g, box_b = 234, 179, 8   # Amber
            bg_r, bg_g, bg_b = 254, 252, 232
            
        pdf.set_fill_color(bg_r, bg_g, bg_b)
        pdf.set_draw_color(box_r, box_g, box_b)
        pdf.set_line_width(0.6)
        pdf.rect(10, pdf.get_y(), 190, 22, 'DF')
        
        curr_y = pdf.get_y()
        pdf.set_xy(12, curr_y + 3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(box_r, box_g, box_b)
        pdf.cell(0, 7, f"SYSTEM VERDICT: {sanitize_for_pdf(verdict.upper())}", ln=True)
        
        pdf.set_xy(12, curr_y + 11)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        conf_pct = round(data.get("confidence", 0.5) * 100, 1)
        pdf.cell(0, 5, f"Calibrated Statistical Confidence: {conf_pct}% | Decision Category: {sanitize_for_pdf(verdict)}", ln=True)
        pdf.set_y(curr_y + 26)
        
        # Section: Evaluated Text Preview
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "2. EVALUATED CONTENT PREVIEW", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        raw_text = data.get("text", "")[:450].replace('\n', ' ')
        if len(data.get("text", "")) > 450:
            raw_text += "..."
        preview_text = sanitize_for_pdf(raw_text)
        
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 4.5, preview_text)
        pdf.ln(4)
        
        # Section: Multi-Signal Diagnostic Telemetry
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "3. MULTI-SIGNAL DIAGNOSTIC TELEMETRY", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        cb_score = data.get("clickbait_score", 0.0)
        cred_score = data.get("credibility_score", 70.0)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 5, "Calibrated SVM Score:", 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(47, 5, f"{conf_pct}%", 0)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 5, "Clickbait Index:", 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(47, 5, f"{int(cb_score)}/100", 0, ln=True)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 5, "Source Credibility:", 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(47, 5, f"{int(cred_score)}/100", 0)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 5, "Word Count:", 0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(47, 5, f"{len(data.get('text', '').split())} words", 0, ln=True)
        pdf.ln(4)
        
        # Section: Explainable AI (SHAP Weights)
        shap_features = data.get("shap_features", [])
        if shap_features:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 6, "4. EXPLAINABLE AI: TOP FEATURE ATTRIBUTIONS (SHAP)", ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
            
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 4, "Linear feature impacts (w_i * x_i) showing lexical impact toward Reliability (+) or Misleading (-):", ln=True)
            pdf.ln(2)
            
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(60, 5, "Token / N-gram", 1, 0, 'C')
            pdf.cell(40, 5, "SHAP Impact Score", 1, 0, 'C')
            pdf.cell(90, 5, "Attribution Influence", 1, 1, 'C')
            
            pdf.set_font("Helvetica", "", 8)
            for feat, val in shap_features[:8]:
                feat_clean = sanitize_for_pdf(feat)
                pdf.cell(60, 4.5, f" {feat_clean}", 1)
                pdf.cell(40, 4.5, f" {val:+.4f}", 1, 0, 'C')
                influence = "Pushes toward Reliable" if val > 0 else "Pushes toward Misleading"
                pdf.cell(90, 4.5, f" {influence}", 1, 1)
            pdf.ln(4)
            
        # Section: Fact-Checking & Corroboration
        fact_checks = data.get("fact_checks", [])
        related_news = data.get("related_news", [])
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "5. INDEPENDENT FACT-CHECKING & NEWS CORROBORATION", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        if fact_checks:
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(185, 28, 28)
            pdf.cell(0, 4.5, f"Verified ClaimReview Matches Found ({len(fact_checks)}):", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(71, 85, 105)
            for fc in fact_checks[:3]:
                fc_text = f" - [{fc.get('fact_checker', 'Fact Checker')}] Rating: {fc.get('rating', 'Evaluated')} | Claim: {fc.get('claim', '')[:90]}"
                pdf.multi_cell(0, 4, sanitize_for_pdf(fc_text))
        else:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 4.5, "No previously verified debunk entries recorded in ClaimReview databases for this query.", ln=True)
            
        if related_news:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 4.5, f"Relevant Corroboration from News Sources ({len(related_news)} matches):", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(71, 85, 105)
            for art in related_news[:3]:
                art_text = f" - [{art.get('publisher', 'News')}] {art.get('title', '')[:85]}"
                pdf.cell(0, 4, sanitize_for_pdf(art_text), ln=True)
        pdf.ln(4)
        
        # Section: Academic Disclaimer
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "ACADEMIC & METHODOLOGICAL DISCLAIMER", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(100, 116, 139)
        disclaimer_text = (
            "TruthLens uses statistical natural language processing (Calibrated SVM and TF-IDF feature representations) "
            "trained on established news corpuses. Machine learning predictions evaluate stylistic, linguistic, and structural "
            "conventions. They do not constitute an infallible claim to objective real-world truth. Critical user discernment "
            "and multi-source journalistic corroboration remain essential."
        )
        pdf.multi_cell(0, 3.5, sanitize_for_pdf(disclaimer_text))
        
        return bytes(pdf.output())
    except Exception as e:
        print(f"PDF Generation Exception: {e}")
        return None
