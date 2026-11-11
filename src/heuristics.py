import re
from typing import Tuple, List, Dict, Any

TRUSTED_MEDIA_DOMAINS = {
    "reuters.com": ("Reuters", 100),
    "apnews.com": ("Associated Press", 100),
    "bbc.com": ("BBC News", 95),
    "bbc.co.uk": ("BBC News", 95),
    "thehindu.com": ("The Hindu", 95),
    "indianexpress.com": ("The Indian Express", 95),
    "ndtv.com": ("NDTV", 90),
    "timesofindia.indiatimes.com": ("Times of India", 85),
    "hindustantimes.com": ("Hindustan Times", 85),
    "aninews.in": ("Asian News International (ANI)", 92),
    "ptinews.com": ("Press Trust of India (PTI)", 95),
    "theprint.in": ("ThePrint", 85),
    "scroll.in": ("Scroll.in", 85),
    "livemint.com": ("Mint", 90),
    "nytimes.com": ("The New York Times", 95),
    "washingtonpost.com": ("The Washington Post", 92),
    "theguardian.com": ("The Guardian", 92),
    "bloomberg.com": ("Bloomberg", 95),
    "aljazeera.com": ("Al Jazeera", 90),
    "dw.com": ("Deutsche Welle", 92),
    "france24.com": ("France 24", 90),
    "nature.com": ("Nature Journal", 100),
    "science.org": ("Science Magazine", 100),
}

DUBIOUS_MEDIA_DOMAINS = {
    "infowars.com": ("InfoWars", 15),
    "breitbart.com": ("Breitbart", 25),
    "naturalnews.com": ("Natural News", 15),
    "beforeitsnews.com": ("Before It's News", 10),
    "worldnewsdailyreport.com": ("World News Daily Report (Satire/Fake)", 5),
    "thegatewaypundit.com": ("The Gateway Pundit", 20),
    "rt.com": ("RT (State Sponsored)", 35),
    "sputnikglobe.com": ("Sputnik News", 35),
}

CLICKBAIT_TRIGGERS = [
    "shocking", "unbelievable", "you won't believe", "exposed", "miracle",
    "secret revealed", "disturbing", "jaw-dropping", "mind-blowing",
    "mystery solved", "tragedy strikes", "conspiracy", "destroys", "slams",
    "rips", "blasts", "bombshell", "what happened next will shock you"
]

def analyze_clickbait(text: str) -> Tuple[float, List[str]]:
    """Calculates clickbait/sensationalism index (0 to 100)."""
    if not text:
        return 0.0, []
        
    text_lower = text.lower()
    words = text.split()
    if not words:
        return 0.0, []
        
    matched_triggers = []
    for trigger in CLICKBAIT_TRIGGERS:
        if trigger in text_lower:
            matched_triggers.append(trigger)
            
    # Uppercase ratio in alphabetic tokens
    alpha_words = [w for w in words if w.isalpha() and len(w) > 1]
    caps_count = sum(1 for w in alpha_words if w.isupper())
    caps_ratio = (caps_count / len(alpha_words)) if alpha_words else 0.0
    
    # Exclamation & question mark frequency
    exclamations = text.count('!')
    questions = text.count('?')
    
    score = 0.0
    score += min(len(matched_triggers) * 15, 45)
    score += min(int(caps_ratio * 100), 25)
    score += min((exclamations + questions) * 8, 20)
    
    return min(float(score), 100.0), matched_triggers

def evaluate_source_credibility(domain_or_text: str) -> Tuple[int, str, str]:
    """
    Evaluates source credibility based on domain or publisher mentions.
    Returns: (score [0-100], source_name, classification)
    """
    cleaned = domain_or_text.lower().strip()
    
    # Direct domain matching
    for d, (name, score) in TRUSTED_MEDIA_DOMAINS.items():
        if d in cleaned:
            return score, name, "Verified Reputable Publisher"
            
    for d, (name, score) in DUBIOUS_MEDIA_DOMAINS.items():
        if d in cleaned:
            return score, name, "Known Questionable / High-Risk Domain"
            
    # Text mention matching
    for d, (name, score) in TRUSTED_MEDIA_DOMAINS.items():
        keyword = name.lower()
        if re.search(r'\b' + re.escape(keyword) + r'\b', cleaned):
            return score, name, "Verified Publisher Mention"
            
    for d, (name, score) in DUBIOUS_MEDIA_DOMAINS.items():
        keyword = name.lower()
        if re.search(r'\b' + re.escape(keyword) + r'\b', cleaned):
            return score, name, "Dubious Source Mention"
            
    return 70, "Independent / Unlisted Source", "Standard Unverified Domain"
