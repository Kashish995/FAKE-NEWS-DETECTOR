import re

# Reuters dateline and publisher leakage patterns
DATELINE_PATTERN = re.compile(r'^[A-Z][A-Za-z\.\s/]{2,40}\(Reuters\)\s*-\s*', re.UNICODE)
REUTERS_WORD_PATTERN = re.compile(r'\bReuters\b', re.IGNORECASE)

# Web boilerplate artifacts common in scraping/blogging
BOILERPLATE_PATTERNS = [
    re.compile(r'\bfeatured\s+image\b', re.IGNORECASE),
    re.compile(r'\bgetty\s+images\b', re.IGNORECASE),
    re.compile(r'\bread\s+more\b', re.IGNORECASE),
    re.compile(r'\bwatch\s+video\b', re.IGNORECASE),
    re.compile(r'\bfollow\s+us\s+on\s+twitter\b', re.IGNORECASE),
    re.compile(r'\blike\s+us\s+on\s+facebook\b', re.IGNORECASE),
    re.compile(r'\bsubscribe\s+now\b', re.IGNORECASE),
    re.compile(r'\bactive\s+subscription\b', re.IGNORECASE),
    re.compile(r'\bclick\s+here\b', re.IGNORECASE),
]

# Unicode character ranges for multilingual detection
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')
ODIA_REGEX = re.compile(r'[\u0B00-\u0B7F]')
LATIN_REGEX = re.compile(r'[a-zA-Z]')

def strip_source_leakage(text: str) -> str:
    """Removes known publisher signatures and dateline artifacts to prevent shortcut learning."""
    if not text:
        return ""
    text = str(text)
    # Strip Reuters datelines like 'WASHINGTON (Reuters) -'
    text = DATELINE_PATTERN.sub('', text, count=1)
    text = REUTERS_WORD_PATTERN.sub('', text)
    # Strip web boilerplate artifacts
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub('', text)
    return text.strip()

def clean_text_for_inference(text: str) -> str:
    """Cleans text for TF-IDF vectorization and feature extraction."""
    text = strip_source_leakage(text)
    # Collapse multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text: str) -> dict:
    """
    Identifies the language/script of the input text and verifies whether
    the machine learning model is validated for that language.
    
    Returns:
        dict: {
            'language': 'English' | 'Hindi' | 'Odia' | 'Mixed/Other',
            'supported_for_ml': bool,
            'notice': str
        }
    """
    if not text or len(text.strip()) == 0:
        return {
            'language': 'Unknown',
            'supported_for_ml': False,
            'notice': 'Empty input buffer.'
        }
    
    total_chars = len(text)
    devanagari_count = len(DEVANAGARI_REGEX.findall(text))
    odia_count = len(ODIA_REGEX.findall(text))
    latin_count = len(LATIN_REGEX.findall(text))
    
    if devanagari_count > total_chars * 0.15:
        return {
            'language': 'Hindi',
            'supported_for_ml': False,
            'notice': (
                'Hindi (Devanagari script) detected. TruthLens stylistic ML models were benchmarked '
                'on English news corpuses. To avoid biased predictions, stylistic ML is disabled for Hindi; '
                'fact-checking and semantic cross-referencing remain fully active.'
            )
        }
    elif odia_count > total_chars * 0.15:
        return {
            'language': 'Odia',
            'supported_for_ml': False,
            'notice': (
                'Odia script detected. TruthLens stylistic ML models were benchmarked '
                'on English news corpuses. Stylistic ML is disabled for Odia to preserve academic integrity; '
                'fact-checking and web corroboration remain active.'
            )
        }
    elif latin_count > 0:
        return {
            'language': 'English',
            'supported_for_ml': True,
            'notice': 'English language verified. Supported for full ML stylistic evaluation and XAI.'
        }
    else:
        return {
            'language': 'Mixed/Other',
            'supported_for_ml': False,
            'notice': 'Unrecognized or non-standard script detected. Requires external verification.'
        }

def extract_query_keywords(text: str, max_words: int = 6) -> str:
    """Extracts informative search keywords for fact-checking APIs."""
    first_sentence = text.split('.')[0] if '.' in text else text
    words = re.findall(r'\b[A-Za-z]{3,}\b', first_sentence)
    stop_words = {
        "the", "and", "that", "this", "with", "from", "have", "said",
        "they", "will", "would", "about", "there", "their", "which",
        "after", "breaking", "update", "latest", "reports", "today", "yesterday"
    }
    filtered = [w for w in words if w.lower() not in stop_words]
    selected = filtered[:max_words] if filtered else words[:max_words]
    return " ".join(selected)
