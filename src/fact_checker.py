import re
import json
import warnings
import urllib.parse
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Local verified ClaimReview database of common viral misinformation and verified debunks
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
        "claim": "NASA satellite picture of India illuminated during Diwali",
        "keywords": ["nasa", "satellite", "diwali", "india lights", "illuminated"],
        "claimant": "Social Media Posts",
        "fact_checker": "Snopes",
        "rating": "False / Composite Image",
        "review_url": "https://www.snopes.com/fact-check/india-diwali-nasa-satellite/",
        "summary": "The image is an artistic composite created by NOAA researchers illustrating population growth over time, not a live satellite photo."
    }
]

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
            reviews = c.get("claimReview", [])
            for rev in reviews:
                results.append({
                    "claim": c.get("text", query),
                    "claimant": c.get("claimant", "Unknown"),
                    "fact_checker": rev.get("publisher", {}).get("name", "Verified Fact Checker"),
                    "rating": rev.get("textualRating", "Evaluated"),
                    "review_url": rev.get("url", ""),
                    "review_title": rev.get("title", ""),
                    "source_type": "Google Fact Check Tools API"
                })
        return results
    except Exception:
        return []

def search_local_claim_database(query: str) -> List[Dict[str, Any]]:
    """Searches curated ClaimReview verified debunks by keyword overlap."""
    q_lower = query.lower()
    matches = []
    for item in VERIFIED_CLAIM_DATABASE:
        keyword_hits = sum(1 for kw in item["keywords"] if kw in q_lower)
        if keyword_hits >= 2:
            matches.append({
                "claim": item["claim"],
                "claimant": item["claimant"],
                "fact_checker": item["fact_checker"],
                "rating": item["rating"],
                "review_url": item["review_url"],
                "review_title": item["summary"],
                "source_type": "TruthLens ClaimReview Corpus"
            })
    return matches

def search_live_news_corroboration(query: str) -> List[Dict[str, Any]]:
    """
    Retrieves live multi-source news coverage from Google News RSS feed
    without requiring third-party API keys or rate limits.
    """
    try:
        clean_q = re.sub(r'[^\w\s]', '', query).strip()
        encoded = urllib.parse.quote(clean_q[:80])
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.find_all('item')[:5]
        
        articles = []
        for item in items:
            title_tag = item.find('title')
            link_tag = item.find('link')
            pub_date = item.find('pubdate')
            source_tag = item.find('source')
            
            title_text = title_tag.text if title_tag else "Related Article"
            link_url = link_tag.text if link_tag else ""
            publisher = source_tag.text if source_tag else "Mainstream Media"
            
            articles.append({
                "title": title_text,
                "url": link_url,
                "publisher": publisher,
                "date": pub_date.text if pub_date else "Recent"
            })
        return articles
    except Exception:
        return []

def query_fact_checking_pipeline(query: str, google_key: str = "", newsapi_key: str = "") -> Dict[str, Any]:
    """
    Orchestrates the multi-layered fact-checking and news corroboration engine.
    """
    # 1. Check Google Fact Check API
    api_claims = search_google_fact_check_api(query, google_key)
    
    # 2. Check local ClaimReview verified database
    local_claims = search_local_claim_database(query)
    
    combined_claims = api_claims + local_claims
    
    # 3. Retrieve live multi-source corroboration
    related_news = search_live_news_corroboration(query)
    
    return {
        "query": query,
        "verified_claims": combined_claims,
        "related_news": related_news,
        "has_debunk": len(combined_claims) > 0,
        "news_coverage_count": len(related_news)
    }
