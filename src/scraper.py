import re
import json
import socket
import ipaddress
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, Tuple

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

def is_safe_url(url: str) -> Tuple[bool, str]:
    """
    Validates URL scheme and verifies destination to prevent Server-Side Request Forgery (SSRF).
    Blocks localhost, loopback (127.0.0.1, ::1), link-local, and private RFC-1918 IP addresses.
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL provided."
        
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in ('http', 'https'):
            return False, "Invalid URL protocol. Only HTTP and HTTPS are permitted."
            
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: Missing hostname."
            
        host_lower = hostname.lower()
        blocked_hosts = {
            'localhost', 'localhost.localdomain', '127.0.0.1', '::1',
            '0.0.0.0', '169.254.169.254', 'metadata.google.internal'
        }
        if host_lower in blocked_hosts or host_lower.endswith('.local') or host_lower.endswith('.internal'):
            return False, "Access Denied: Localhost and internal network destinations are blocked for security."
            
        # Check if the hostname is a literal IP address
        try:
            ip = ipaddress.ip_address(host_lower)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False, f"Access Denied: Internal or private IP address ({host_lower}) is blocked."
        except ValueError:
            # Not a raw IP, it's a domain name. Attempt DNS resolution to check resolved IP
            try:
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                    return False, f"Access Denied: Resolved to private IP address ({ip_str})."
            except socket.gaierror:
                # In sandbox or offline testing, DNS resolution may fail.
                # Allow standard public FQDN formats, but block non-standard hostnames
                if not re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+\.?$', hostname):
                    return False, f"Invalid or unresolvable hostname '{hostname}'."
                    
        return True, ""
    except Exception as e:
        return False, f"URL Validation Error: {e}"

PAYWALL_MARKERS = [
    "active subscription", "subscribe to continue", "read more with a subscription",
    "premium story", "login to read", "create an account to read", "subscriber-only",
    "you have reached your free article limit", "already a subscriber", "subscribed with another email",
    "account subscription benefits", "sign in to continue", "subscribe now", "become a member to read"
]

BOILERPLATE_PATTERNS = [
    re.compile(r'\ball\s+rights\s+reserved\b', re.IGNORECASE),
    re.compile(r'\bterms\s+of\s+use\b', re.IGNORECASE),
    re.compile(r'\bprivacy\s+policy\b', re.IGNORECASE),
    re.compile(r'\bcookie\s+policy\b', re.IGNORECASE),
    re.compile(r'\bsign\s+up\s+for\s+our\s+newsletter\b', re.IGNORECASE),
    re.compile(r'\bdownload\s+the\s+app\b', re.IGNORECASE),
    re.compile(r'\blisten\s+to\s+this\s+article\b', re.IGNORECASE),
]

MIN_BODY_WORD_COUNT = 60

def is_paywall_or_boilerplate(text: str) -> bool:
    """Checks if text predominantly matches paywall or subscription notices."""
    if not text:
        return True
    t_lower = text.lower()
    matches = sum(1 for marker in PAYWALL_MARKERS if marker in t_lower)
    return matches >= 1

def clean_extracted_body(text: str) -> str:
    """Cleans navigation, ads, and boilerplate phrases from body text."""
    if not text:
        return ""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_lines = []
    for line in lines:
        if is_paywall_or_boilerplate(line) and len(line.split()) < 30:
            continue
        cleaned = line
        for pat in BOILERPLATE_PATTERNS:
            cleaned = pat.sub('', cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > 20:
            cleaned_lines.append(cleaned)
    return "\n\n".join(cleaned_lines)

def extract_from_json_ld(html_content: str) -> Dict[str, str]:
    """Extracts headline and articleBody from JSON-LD schema scripts."""
    result = {"title": "", "body": "", "date": "", "publisher": ""}
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script', type='application/ld+json')
        for s in scripts:
            if not s.string:
                continue
            try:
                data = json.loads(s.string)
                items = data if isinstance(data, list) else data.get('@graph', [data]) if isinstance(data, dict) else []
                for item in items:
                    if isinstance(item, dict) and item.get('@type') in ['NewsArticle', 'Article', 'ReportageNewsArticle']:
                        title = item.get('headline') or item.get('name') or ""
                        body = item.get('articleBody') or item.get('description') or ""
                        date = item.get('datePublished') or item.get('dateModified') or ""
                        pub_data = item.get('publisher', {})
                        pub_name = pub_data.get('name', '') if isinstance(pub_data, dict) else ""
                        if len(body) > len(result["body"]):
                            result["title"] = title
                            result["body"] = body
                            result["date"] = date
                            result["publisher"] = pub_name
            except Exception:
                continue
    except Exception:
        pass
    return result

def scrape_article(url: str) -> Dict[str, Any]:
    """
    Robust article extraction pipeline:
    1. Validates URL protocol and SSRF security.
    2. Uses Trafilatura for primary extraction (removes ads, navigation, boilerplate).
    3. Uses JSON-LD schema and BeautifulSoup as secondary fallbacks.
    4. Validates length (minimum 60 words).
    5. Detects paywalls and access-denied responses.
    
    Returns structured dictionary with extraction status, title, body, word count, and domain.
    """
    domain = urlparse(url).netloc.replace('www.', '').lower()
    
    result = {
        "success": False,
        "title": "",
        "body": "",
        "full_text": "",
        "domain": domain,
        "publisher": domain,
        "word_count": 0,
        "publish_date": "",
        "error": None
    }
    
    # 0. SSRF Safety Verification
    is_safe, safety_err = is_safe_url(url)
    if not is_safe:
        result["error"] = safety_err
        return result
        
    extracted_title = ""
    extracted_body = ""
    extracted_date = ""
    extracted_pub = domain
    
    # 1. Primary Extraction using Trafilatura
    if TRAFILATURA_AVAILABLE:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                body = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                meta = trafilatura.extract_metadata(downloaded)
                if meta:
                    extracted_title = meta.title or ""
                    extracted_date = meta.date or ""
                    if meta.sitename:
                        extracted_pub = meta.sitename
                if body:
                    extracted_body = clean_extracted_body(body)
        except Exception:
            pass
            
    # 2. Secondary Extraction using requests and JSON-LD if Trafilatura was insufficient
    if len(extracted_body.split()) < MIN_BODY_WORD_COUNT:
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                json_data = extract_from_json_ld(resp.text)
                if len(json_data["body"].split()) > len(extracted_body.split()):
                    extracted_body = clean_extracted_body(json_data["body"])
                    if not extracted_title and json_data["title"]:
                        extracted_title = json_data["title"]
                    if not extracted_date and json_data["date"]:
                        extracted_date = json_data["date"]
                    if json_data["publisher"]:
                        extracted_pub = json_data["publisher"]
                        
                # If still empty, inspect HTML headings and paragraphs
                if not extracted_title or len(extracted_body.split()) < MIN_BODY_WORD_COUNT:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    if not extracted_title:
                        h1 = soup.find('h1') or soup.find('meta', property='og:title')
                        if h1:
                            extracted_title = h1.get_text().strip() if hasattr(h1, 'get_text') else h1.get('content', '').strip()
                    if len(extracted_body.split()) < MIN_BODY_WORD_COUNT:
                        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip().split()) > 8]
                        if paragraphs:
                            candidate_body = clean_extracted_body("\n\n".join(paragraphs))
                            if len(candidate_body.split()) > len(extracted_body.split()):
                                extracted_body = candidate_body
            elif resp.status_code in (401, 403):
                result["error"] = f"Access Denied (HTTP {resp.status_code}): The website blocked article extraction. Please copy and paste the text manually."
                return result
            elif resp.status_code == 404:
                result["error"] = "Article Not Found (HTTP 404): The specified URL does not exist or has been removed."
                return result
        except requests.exceptions.Timeout:
            result["error"] = "Connection Timeout: Target web server did not respond within 10 seconds."
            return result
        except Exception as e:
            result["error"] = f"Extraction Network Failure: {e}"
            return result
            
    # 3. Validation & Quality Checks
    words = extracted_body.split()
    word_count = len(words)
    
    # Check if the extracted content is predominantly a paywall notice
    if is_paywall_or_boilerplate(extracted_body) and word_count < 100:
        result["success"] = False
        result["error"] = "Article extraction unsuccessful: Web page is protected by a subscription paywall. Please paste the article text manually."
        return result
        
    if word_count < MIN_BODY_WORD_COUNT:
        result["success"] = False
        result["error"] = f"Insufficient article text extracted ({word_count} words; minimum {MIN_BODY_WORD_COUNT} required). The page may require JavaScript or user login."
        return result
        
    # Successful clean extraction
    result["success"] = True
    result["title"] = extracted_title
    result["body"] = extracted_body
    result["full_text"] = f"{extracted_title}\n\n{extracted_body}".strip() if extracted_title else extracted_body
    result["word_count"] = len(result["full_text"].split())
    result["publish_date"] = extracted_date
    result["publisher"] = extracted_pub
    result["error"] = None
    return result
