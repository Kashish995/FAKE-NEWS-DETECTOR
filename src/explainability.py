import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def get_model_coefficients(model, vectorizer) -> Optional[np.ndarray]:
    """Extracts linear coefficients from SVM, Logistic Regression, or Calibrated classifiers."""
    try:
        if hasattr(model, "calibrated_classifiers_"):
            return np.mean([c.estimator.coef_[0] for c in model.calibrated_classifiers_], axis=0)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None
    except Exception:
        return None

def explain_shap_linear(text: str, model, vectorizer, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Computes linear SHAP attribution values (w_i * x_i) for active tokens.
    Positive values push toward Likely Reliable (Real);
    Negative values push toward Likely Misleading (Fake).
    """
    try:
        coefs = get_model_coefficients(model, vectorizer)
        if coefs is None:
            return []
            
        x = vectorizer.transform([text])
        non_zero_indices = x.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        
        shap_values = []
        for idx in non_zero_indices:
            feat_name = feature_names[idx]
            feat_val = x[0, idx]
            coef_val = coefs[idx]
            shap_impact = float(feat_val * coef_val)
            shap_values.append((feat_name, shap_impact))
            
        shap_values.sort(key=lambda item: item[1])
        return shap_values
    except Exception:
        return []

def plot_shap_bar(shap_values: List[Tuple[str, float]]):
    """Renders a clean horizontal SHAP feature impact bar chart."""
    if not shap_values:
        return None
        
    negatives = [x for x in shap_values if x[1] < -0.01][:5]
    positives = [x for x in shap_values if x[1] > 0.01][-5:]
    positives.reverse()
    
    selected = negatives + positives
    if not selected:
        return None
        
    selected.sort(key=lambda x: x[1])
    
    features = [x[0] for x in selected]
    values = [x[1] for x in selected]
    colors = ['#ef4444' if v < 0 else '#22c55e' for v in values]
    
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.barh(features, values, color=colors, edgecolor='#475569', height=0.55)
    
    ax.spines['bottom'].set_color('#64748b')
    ax.spines['left'].set_color('#64748b')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.axvline(0, color='#64748b', linestyle='--', linewidth=0.8)
    
    ax.set_xlabel("Attribution Weight (SHAP Impact)", color='#64748b', fontsize=8)
    ax.set_title("Linguistic Feature Impact (Explainable AI)", color='#475569', fontsize=9, fontweight='bold')
    
    return fig

def get_highlighted_tokens_html(text: str, model, vectorizer) -> str:
    """Returns HTML with colored spans for influential tokens."""
    try:
        coefs = get_model_coefficients(model, vectorizer)
        if coefs is None:
            return text
            
        vocab = vectorizer.vocabulary_
        analyzer = vectorizer.build_analyzer()
        words_in_text = analyzer(text)
        unique_words = set(words_in_text)
        
        word_scores = {}
        for word in unique_words:
            if word in vocab:
                idx = vocab[word]
                word_scores[word] = coefs[idx]
                
        tokens = re.split(r'(\W+)', text)
        highlighted = []
        
        for token in tokens:
            clean_tok = token.lower().strip()
            clean_tok = re.sub(r'[^\w\s]', '', clean_tok)
            score = word_scores.get(clean_tok)
            
            if score is not None and score < -0.05:
                # Misleading / Fake marker (red)
                highlighted.append(
                    f'<span style="background-color: rgba(239, 68, 68, 0.18); color: #ef4444; '
                    f'font-weight: 600; padding: 2px 4px; border-radius: 4px; '
                    f'border-bottom: 2px solid #ef4444;" title="Misleading weight: {score:.3f}">'
                    f'{token}</span>'
                )
            elif score is not None and score > 0.05:
                # Reliable / Real marker (green)
                highlighted.append(
                    f'<span style="background-color: rgba(34, 197, 94, 0.18); color: #22c55e; '
                    f'font-weight: 600; padding: 2px 4px; border-radius: 4px; '
                    f'border-bottom: 2px solid #22c55e;" title="Reliable weight: {score:.3f}">'
                    f'{token}</span>'
                )
            else:
                highlighted.append(token)
                
        return "".join(highlighted)
    except Exception:
        return text

# Alias for backward compatibility
get_highlighted_tokens = get_highlighted_tokens_html
