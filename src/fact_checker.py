import re
import json
import warnings
import urllib.parse
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Curated verified ClaimReview database of known viral misinformation
VERIFIED_CLAIM_DATABASE = [
    {
        "claim": "COVID-19 vaccines contain 5G tracking microchips",
        "keywords": ["covid", "vaccine", "5g", "microchip", "tracking"],
        "claimant": "Social Media Posts / WhatsApp",
        "fact_checker": "PolitiFact / Reuters Fact Check",
        "rating": "False / Fabricated",
        "review_url": "https://www.reuters.com/article/factcheck-coronavirus-5g-idUSL1N2P519H",
        "summary": "Medical regulators and global health agencies confirmed vaccines contain no microchips or electronic components."
    },
    {
        "claim": "Drinking salt water or lemon water cures coronavirus or cancer",
        "keywords": ["salt water", "lemon", "cure", "cancer", "coronavirus"],
        "claimant": "Viral Social Media Chain",
        "fact_checker": "WHO / FactCheck.org",
        "rating": "False",
        "review_url": "https://www.factcheck.org/2020/03/lemon-and-water-isnt-a-coronavirus-cure/",
        "summary": "There is no clinical or biochemical evidence that saline or lemon water neutralizes viral pathogens or cures malignancies."
    },
    {
        "claim": "UNESCO declared the Indian National Anthem the best anthem in the world",
        "keywords": ["unesco", "national anthem", "jana gana mana", "best anthem"],
        "claimant": "Viral WhatsApp Forward",
        "fact_checker": "India Today Fact Check / BoomLive",
        "rating": "False / Hoax",
        "review_url": "https://www.indiatoday.in/fact-check/story/unesco-best-national-anthem-hoax-280540-2015-08-14",
        "summary": "UNESCO explicitly confirmed it has never ranked, awarded, or rated national anthems of any country."
    },
    {
        "claim": "RBI to discontinue 500 rupee currency notes",
        "keywords": ["rbi", "500", "currency", "note", "discontinue", "ban"],
        "claimant": "Viral YouTube Videos",
        "fact_checker": "PIB Fact Check",
        "rating": "Fake News",
        "review_url": "https://twitter.com/PIBFactCheck",
        "summary": "The Reserve Bank of India has not issued any notification to demonetize or withdraw 500 denomination notes."
    },
    {
        "claim": "Secret documents prove pharmaceutical companies hide cure for aging",
        "keywords": ["secret documents", "cure for aging", "pharmaceutical", "hide cure"],
        "claimant": "Conspiracy Blogs",
        "fact_checker": "Snopes / Healthline",
        "rating": "Fabricated",
        "review_url": "https://www.snopes.com/",
        "summary": "No verified scientific or governmental records indicate a secret universal cure for cellular senescence exists."
    }
]

def extract_entities_and_keywords(text: str) -> List[str]:
    """
    Extracts high-information key phrases, acronyms, and proper nouns
    to construct focused search queries.
    """
    if not text:
        return []
    
    # 1. Capitalized multi-word phrases (e.g. 'Delhi High Court', 'Rekha Gupta')
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    
    # 2. Acronyms & uppercase abbreviations (e.g. 'HC', 'MCD', 'GRAP', 'ISRO', 'RBI', 'COVID')
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
    
    # 3. Informative lowercase nouns
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    stop_words = {
        "this", "that", "with", "from", "have", "said", "they", "will",
        "would", "about", "there", "their", "which", "after", "breaking",
        "update", "latest", "reports", "today", "yesterday", "news", "article",
        "incoming", "during", "directs", "orders", "panel"
    }
    meaningful_words = [w for w in words if w.lower() not in stop_words]
    
    combined = list(dict.fromkeys(acronyms + proper_nouns + meaningful_words[:6]))
    return combined

def compute_semantic_relevance(candidate_title: str, query_text: str) -> float:
    """
    Computes a semantic relevance score (0.0 to 1.0) between a candidate news title
    and the source article headline/content using token Jaccard and keyword overlap.
    """
    if not candidate_title or not query_text:
        return 0.0
        
    c_tokens = set(re.findall(r'\b[A-Za-z]{3,}\b', candidate_title.lower()))
    q_tokens = set(re.findall(r'\b[A-Za-z]{3,}\b', query_text.lower()))
    
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "after",
        "news", "live", "updates", "today", "yesterday", "says", "said"
    }
    c_meaningful = c_tokens - stop_words
    q_meaningful = q_tokens - stop_words
    
    if not c_meaningful or not q_meaningful:
        return 0.0
        
    overlap = c_meaningful.intersection(q_meaningful)
    
    # Check for crucial acronym overlap (e.g. GRAP, MCD, ISRO, RBI)
    q_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', query_text))
    c_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', candidate_title))
    acronym_hits = len(q_acronyms.intersection(c_acronyms))
    
    # Token Jaccard index
    jaccard = len(overlap) / len(c_meaningful.union(q_meaningful))
    overlap_ratio = len(overlap) / min(len(c_meaningful), len(q_meaningful))
    
    score = (overlap_ratio * 0.7) + (jaccard * 0.3)
    if acronym_hits > 0:
        score = min(score + 0.20, 1.0)
        
    return float(score)

def search_live_news_corroboration(
    headline: str,
    full_text: str = "",
    exclude_domain_or_publisher: str = ""
) -> List[Dict[str, Any]]:
    """
    Retrieves live multi-source news coverage and enforces strict semantic
    and entity-overlap filtering to reject unrelated search results.
    Explicitly filters out the target article's own publisher to prevent circular self-corroboration.
    """
    entities = extract_entities_and_keywords(headline)
    if not entities:
        entities = extract_entities_and_keywords(full_text[:300])
        
    if not entities:
        return []
        
    search_query = " ".join(entities[:5])
    
    try:
        encoded = urllib.parse.quote(search_query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, timeout=6)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.find_all('item')[:12]
        
        # Determine publisher keywords to exclude from self-corroboration
        norm_excl = ""
        if exclude_domain_or_publisher:
            clean_dom = exclude_domain_or_publisher.lower().replace("www.", "")
            base_name = clean_dom.split(".")[0].strip()
            norm_excl = re.sub(r'[^a-z0-9]', '', base_name)
        
        verified_matches = []
        for item in items:
            title_tag = item.find('title')
            link_tag = item.find('link')
            pub_date_tag = item.find('pubdate')
            source_tag = item.find('source')
            
            title_text = title_tag.text.strip() if title_tag else ""
            link_url = link_tag.text.strip() if link_tag else ""
            pub_date = pub_date_tag.text.strip() if pub_date_tag else "Recent"
            publisher = source_tag.text.strip() if source_tag else "News Media"
            
            # If publisher was in title suffix (e.g. 'Article Title - The Hindu')
            if ' - ' in title_text and (not publisher or publisher == "News Media"):
                parts = title_text.rsplit(' - ', 1)
                title_text = parts[0].strip()
                publisher = parts[1].strip()
                
            # Disqualify self-referential corroboration
            if norm_excl and len(norm_excl) >= 3:
                norm_pub = re.sub(r'[^a-z0-9]', '', publisher.lower())
                if norm_excl in norm_pub or norm_pub in norm_excl or (link_url and norm_excl in link_url.lower()):
                    continue
                
            # Compute semantic relevance score
            relevance = compute_semantic_relevance(title_text, f"{headline} {full_text[:200]}")
            
            # STRICT FILTER: Reject results with relevance below 25%
            if relevance < 0.25:
                continue
                
            verified_matches.append({
                "title": title_text,
                "publisher": publisher,
                "url": link_url,
                "date": pub_date,
                "relevance_score": int(relevance * 100),
                "evidence_type": "Supporting Coverage"
            })
            
        # Sort by relevance descending
        verified_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return verified_matches[:5]
        
    except Exception as e:
        print(f"External News Verification Exception: {e}")
        return []

def search_local_claim_database(query: str) -> List[Dict[str, Any]]:
    """Searches curated ClaimReview verified debunks by semantic keyword overlap."""
    if not query:
        return []
    q_lower = query.lower()
    matches = []
    for item in VERIFIED_CLAIM_DATABASE:
        hits = sum(1 for kw in item["keywords"] if kw in q_lower)
        # Require matching at least 2 distinct keywords or claim token overlap > 35%
        claim_rel = compute_semantic_relevance(item["claim"], query)
        if hits >= 2 or claim_rel >= 0.35:
            matches.append({
                "claim": item["claim"],
                "claimant": item["claimant"],
                "fact_checker": item["fact_checker"],
                "rating": item["rating"],
                "review_url": item["review_url"],
                "review_title": item["summary"],
                "evidence_type": "Contradicting / Debunk"
            })
    return matches

def search_google_fact_check_api(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Queries Google Fact Check Tools API (ClaimReview database)."""
    if not api_key:
        return []
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": query, "key": api_key, "pageSize": 5}
        response = requests.get(url, params=params, timeout=6)
        if response.status_code != 200:
            return []
        data = response.json()
        claims = data.get("claims", [])
        results = []
        for c in claims:
            for rev in c.get("claimReview", []):
                results.append({
                    "claim": c.get("text", query),
                    "claimant": c.get("claimant", "Unknown"),
                    "fact_checker": rev.get("publisher", {}).get("name", "Verified Fact Checker"),
                    "rating": rev.get("textualRating", "Evaluated"),
                    "review_url": rev.get("url", ""),
                    "review_title": rev.get("title", ""),
                    "evidence_type": "Contradicting / Debunk"
                })
        return results
    except Exception:
        return []

def query_fact_checking_pipeline(
    headline: str,
    full_text: str = "",
    google_key: str = "",
    newsapi_key: str = "",
    exclude_domain_or_publisher: str = ""
) -> Dict[str, Any]:
    """
    Executes the external news verification and fact-checking pipeline.
    Rejects unrelated search results and returns strictly relevant evidence.
    """
    # 1. Check for verified debunks
    search_claim = headline if headline else full_text[:120]
    api_claims = search_google_fact_check_api(search_claim, google_key)
    local_claims = search_local_claim_database(search_claim)
    all_debunks = api_claims + local_claims
    
    # 2. Retrieve strictly relevant live news coverage (excluding origin domain)
    relevant_news = search_live_news_corroboration(
        headline=headline,
        full_text=full_text,
        exclude_domain_or_publisher=exclude_domain_or_publisher
    )
    
    has_debunk = len(all_debunks) > 0
    has_relevant_news = len(relevant_news) > 0
    
    if has_debunk:
        status = "Prior Debunk Found in ClaimReview"
        summary = f"Identified {len(all_debunks)} verified fact-check record(s) evaluating this claim as false or misleading."
    elif has_relevant_news:
        status = "Corroborated by Verified News"
        top_n = relevant_news[0]
        summary = f"Corroborated by {len(relevant_news)} independent matching news report(s). Top match: {top_n['publisher']} ({top_n['relevance_score']}% relevance)."
    else:
        status = "No sufficiently relevant matching articles found."
        summary = "No sufficiently relevant matching articles found."
        
    return {
        "verified_claims": all_debunks,
        "relevant_news": relevant_news,
        "has_debunk": has_debunk,
        "has_relevant_news": has_relevant_news,
        "external_status": status,
        "external_summary": summary
    }
