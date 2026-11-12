import re
import json
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Tuple, Dict, Any, Optional

PAYWALL_MARKERS = [
    "active subscription", "subscribe to continue", "read more with a subscription",
    "premium story", "login to read", "create an account to read", "subscriber-only",
    "you have reached your free article limit", "already a subscriber"
]

BOILERPLATE_PHRASES = [
    "all rights reserved", "terms of use", "privacy policy", "cookie policy",
    "advertisement", "sign up for our newsletter", "click here to subscribe",
    "follow us on", "download the app", "listen to this article"
]

def extract_slug_text(url: str) -> str:
    """Extracts a readable headline from URL slug as a fallback."""
    try:
        clean_url = url.strip().rstrip('/')
        path_segments = [s for s in clean_url.split('/') if s]
        if not path_segments:
            return "News Article"
        
        last_segment = path_segments[-1]
        last_segment = re.sub(r'\.[a-zA-Z]{2,5}$', '', last_segment)  # Strip .html, .ece, etc.
        last_segment = re.sub(r'-\d+$', '', last_segment)             # Strip numeric IDs
        words = last_segment.split('-')
        headline = " ".join([w for w in words if w.isalnum()]).strip()
        
        if (len(headline) < 10 or last_segment.isdigit()) and len(path_segments) > 2:
            second_last = path_segments[-2]
            second_last = re.sub(r'-\d+$', '', second_last)
            words = second_last.split('-')
            headline = " ".join([w for w in words if w.isalnum()]).strip()
            
        return headline.title() if headline else "News Article"
    except Exception:
        return "News Article"

def scrape_article(url: str) -> Dict[str, Any]:
    """
    Robust article extractor with paywall awareness, JSON-LD schema parsing,
    meta-description fallbacks, and boilerplate cleaning.
    """
    domain = urlparse(url).netloc.replace('www.', '')
    result = {
        "title": "",
        "body": "",
        "full_text": "",
        "domain": domain,
        "word_count": 0,
        "is_paywalled": False,
        "paywall_notice": None,
        "error": None
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            slug = extract_slug_text(url)
            result["title"] = slug
            result["body"] = f"Headline extracted from link: {slug}"
            result["full_text"] = f"{slug} - {result['body']}"
            result["word_count"] = len(result["full_text"].split())
            result["error"] = f"Web server returned HTTP {response.status_code}. Using headline fallback."
            return result
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title Extraction
        title = ""
        og_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'twitter:title'})
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()
            
        if not title:
            title = extract_slug_text(url)
            
        result["title"] = title
        
        # 2. Check JSON-LD for full article body or description
        json_ld_text = ""
        for script in soup.find_all('script', type='application/ld+json'):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get('@graph', [data])
                else:
                    items = []
                    
                for item in items:
                    if isinstance(item, dict) and item.get('@type') in ['NewsArticle', 'Article', 'ReportageNewsArticle']:
                        art_body = item.get('articleBody') or item.get('description')
                        if art_body and len(art_body) > len(json_ld_text):
                            json_ld_text = art_body.strip()
            except Exception:
                pass
                
        # 3. Target Article Body Containers
        article_elem = (
            soup.find('article') or 
            soup.find(attrs={'itemprop': 'articleBody'}) or
            soup.find('div', class_=re.compile(r'(article-body|story-content|story-body|article__body|entry-content)', re.I))
        )
        
        target_container = article_elem if article_elem else soup
        
        paragraphs = []
        raw_p_tags = target_container.find_all('p')
        
        paywall_detected = False
        for p in raw_p_tags:
            txt = p.get_text().strip()
            if len(txt) < 25:
                continue
                
            txt_lower = txt.lower()
            if any(pm in txt_lower for pm in PAYWALL_MARKERS):
                paywall_detected = True
                continue
                
            if any(bp in txt_lower for bp in BOILERPLATE_PHRASES):
                continue
                
            paragraphs.append(txt)
            
        body_text = "\n\n".join(paragraphs)
        
        # If body is short or paywall detected, augment with meta description or JSON-LD
        meta_desc = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'})
        desc_text = meta_desc.get('content', '').strip() if meta_desc else ""
        
        if (len(body_text.split()) < 40 or paywall_detected) and (desc_text or json_ld_text):
            augmented = []
            if desc_text:
                augmented.append(desc_text)
            if json_ld_text and json_ld_text != desc_text:
                augmented.append(json_ld_text)
            if body_text:
                augmented.append(body_text)
            body_text = "\n\n".join(augmented)
            
        if paywall_detected or len(body_text.split()) < 30:
            result["is_paywalled"] = True
            result["paywall_notice"] = (
                f"Publisher site ({domain}) appears subscription-gated or dynamically loaded. "
                "The extracted preview contains available summary/meta text. "
                "For highest precision, you may also paste full article text directly."
            )
            
        if not body_text:
            body_text = title
            
        result["body"] = body_text
        result["full_text"] = f"{title}\n\n{body_text}".strip()
        result["word_count"] = len(result["full_text"].split())
        return result
        
    except Exception as e:
        slug = extract_slug_text(url)
        result["title"] = slug
        result["body"] = f"Extracted from URL: {slug}"
        result["full_text"] = result["body"]
        result["word_count"] = len(result["full_text"].split())
        result["error"] = f"Extraction failed ({str(e)}). Using URL headline fallback."
        return result
