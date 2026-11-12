import streamlit as st

CUSTOM_CSS = """
<style>
/* Modern Slate & Indigo Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Main Container Padding */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1140px !important;
}

/* Card Panels */
.tl-card {
    background-color: rgba(30, 41, 59, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    transition: all 0.2s ease;
}

.tl-card-reliable {
    background-color: rgba(34, 197, 94, 0.06);
    border: 1.5px solid rgba(34, 197, 94, 0.3);
    border-left: 6px solid #22c55e;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
}

.tl-card-misleading {
    background-color: rgba(239, 68, 68, 0.06);
    border: 1.5px solid rgba(239, 68, 68, 0.3);
    border-left: 6px solid #ef4444;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
}

.tl-card-verify {
    background-color: rgba(234, 179, 8, 0.06);
    border: 1.5px solid rgba(234, 179, 8, 0.3);
    border-left: 6px solid #eab308;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
}

/* Badges */
.tl-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tl-badge-green {
    background-color: rgba(34, 197, 94, 0.15);
    color: #16a34a;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.tl-badge-red {
    background-color: rgba(239, 68, 68, 0.15);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.tl-badge-amber {
    background-color: rgba(234, 179, 8, 0.15);
    color: #ca8a04;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.03);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

/* High Contrast Headings */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}

/* Progress Bars */
.stProgress > div > div > div > div {
    border-radius: 6px;
}
</style>
"""

def apply_ui_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_verdict_banner(verdict: str, confidence: float, explanation: str):
    """Renders a prominent, high-contrast verdict banner."""
    conf_pct = round(confidence * 100, 1)
    
    if verdict == "Likely Reliable":
        card_class = "tl-card-reliable"
        badge_class = "tl-badge tl-badge-green"
        text_color = "#16a34a"
        icon = "✅"
    elif verdict == "Likely Misleading":
        card_class = "tl-card-misleading"
        badge_class = "tl-badge tl-badge-red"
        text_color = "#dc2626"
        icon = "⚠️"
    else:
        card_class = "tl-card-verify"
        badge_class = "tl-badge tl-badge-amber"
        text_color = "#ca8a04"
        icon = "🔍"
        
    html = f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span class="{badge_class}">TruthLens Verdict</span>
            <span style="font-size: 0.85rem; font-weight: 600; color: #64748b;">
                Calibrated Confidence: <b style="color: {text_color};">{conf_pct}%</b>
            </span>
        </div>
        <h2 style="color: {text_color} !important; margin: 0 0 8px 0; font-size: 1.75rem; border: none; padding: 0;">
            {icon} {verdict.upper()}
        </h2>
        <p style="margin: 0; font-size: 0.95rem; color: inherit; line-height: 1.55;">
            {explanation}
        </p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_academic_disclaimer():
    st.markdown("""
    <div style="background-color: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.2); 
                padding: 12px 16px; border-radius: 8px; font-size: 0.8rem; color: #64748b; margin-top: 2rem;">
        <b>⚖️ Academic & Methodological Disclaimer:</b> TruthLens evaluates stylistic, structural, and lexical properties 
        derived from machine learning training corpuses. Model predictions express calibrated statistical likelihoods 
        and do not constitute an infallible claim to objective real-world truth. Multi-source journalistic corroboration 
        and critical thinking remain indispensable.
    </div>
    """, unsafe_allow_html=True)
