import io
import time
import hashlib
import requests
from typing import Tuple, Optional, List
from PIL import Image, ImageEnhance
from .config import OCR_API_KEY_ENV

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

# Backup public/demo keys to cycle if one is rate-limited
OCR_FALLBACK_KEYS = [
    "K88888888888957",  # Public registered demo key
    "helloworld"         # Default shared key
]

def compute_image_hash(image_bytes: bytes) -> str:
    """Generates a SHA-256 hex digest for image bytes to bind OCR state to the exact image."""
    if not image_bytes:
        return ""
    return hashlib.sha256(image_bytes).hexdigest()

def preprocess_image_for_ocr(image_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    """Validates image format, applies grayscale and contrast enhancement for optimal character recognition."""
    if not image_bytes:
        return None, "No image data received."
        
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return None, "File too large: Maximum supported image size is 10 MB."
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        allowed_formats = ("PNG", "JPEG", "JPG", "MPO", "WEBP", "BMP")
        if img.format and img.format.upper() not in allowed_formats:
            return None, f"Unsupported image format: {img.format}. Please upload PNG, JPG, or JPEG."
            
        # Convert RGBA, CMYK, or palette to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Grayscale conversion & contrast enhancement
        gray = img.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)
        
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        return buf.getvalue(), None
    except Exception as e:
        return None, f"Image Processing Error: Unable to read image file ({e})."

def _try_local_pytesseract(preprocessed_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Attempts local OCR using pytesseract if available and installed on system."""
    try:
        import pytesseract
        img = Image.open(io.BytesIO(preprocessed_bytes))
        text = pytesseract.image_to_string(img)
        if text and len(text.strip()) > 10:
            return text.strip(), None
        return None, "Local OCR produced empty output."
    except Exception as e:
        return None, f"Local OCR unavailable: {e}"

def extract_text_from_image(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    max_retries: int = 2
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts text from image bytes:
    1. Preprocesses image (size, format, contrast).
    2. Attempts local pytesseract OCR first (offline, zero rate limit).
    3. Falls back to OCR.space cloud API with exponential backoff and HTTP 429 retry handling.
    
    Returns:
        Tuple[extracted_text, error_message, ocr_source]
        ocr_source is either 'Local Tesseract OCR', 'OCR.space Cloud API', or None.
    """
    preprocessed_bytes, prep_err = preprocess_image_for_ocr(image_bytes)
    if prep_err:
        return None, prep_err, None
        
    # 1. Attempt Local OCR First
    local_text, local_err = _try_local_pytesseract(preprocessed_bytes)
    if local_text:
        return local_text, None, "Local Tesseract OCR"
        
    # 2. Prepare Cloud OCR Keys Pool
    keys_to_try: List[str] = []
    if api_key and api_key.strip():
        keys_to_try.append(api_key.strip())
    if OCR_API_KEY_ENV and OCR_API_KEY_ENV.strip() and OCR_API_KEY_ENV not in keys_to_try:
        keys_to_try.append(OCR_API_KEY_ENV.strip())
    for fb in OCR_FALLBACK_KEYS:
        if fb not in keys_to_try:
            keys_to_try.append(fb)
            
    last_error = None
    rate_limited = False

    for current_key in keys_to_try:
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "apikey": current_key,
                    "language": "eng",
                    "isOverlayRequired": False,
                    "detectOrientation": True,
                    "scale": True,
                    "OCREngine": 2
                }
                files = {
                    "file": ("image.png", preprocessed_bytes, "image/png")
                }
                
                response = requests.post(
                    "https://api.ocr.space/parse/image",
                    data=payload,
                    files=files,
                    timeout=15
                )
                
                # Handle Rate Limiting (HTTP 429)
                if response.status_code == 429:
                    rate_limited = True
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = 1.5 * (attempt + 1)
                    if retry_after and retry_after.isdigit():
                        sleep_time = min(float(retry_after), 5.0)
                        
                    if attempt < max_retries:
                        time.sleep(sleep_time)
                        continue
                    else:
                        last_error = "OCR Service is temporarily rate-limited (HTTP 429: Too Many Requests)."
                        break  # Try next key in pool
                        
                if response.status_code != 200:
                    last_error = f"OCR Service returned HTTP status {response.status_code}."
                    break
                    
                result = response.json()
                if result.get("OCRExitCode") == 1:
                    parsed = result.get("ParsedResults", [])
                    if parsed:
                        text = parsed[0].get("ParsedText", "").strip()
                        if text:
                            return text, None, "OCR.space Cloud API"
                        else:
                            return None, "No readable text detected in this image. Please ensure the screenshot has clear typography.", None
                            
                error_msg = result.get("ErrorMessage")
                if isinstance(error_msg, list) and len(error_msg) > 0:
                    last_error = str(error_msg[0])
                elif not error_msg:
                    last_error = "OCR Engine could not parse image text."
                else:
                    last_error = str(error_msg)
                break  # Non-429 error, no point retrying same request
                
            except requests.exceptions.Timeout:
                last_error = "OCR request timed out. Please check your network connection or try again."
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                break
            except Exception as e:
                last_error = f"OCR Connection Failure: {e}"
                break

    # If all options failed, construct a helpful and actionable error message
    if rate_limited:
        guidance = (
            "OCR Error: The public OCR service quota is currently busy (HTTP 429: Rate Limited). "
            "Suggestions:\n"
            "1. Wait a few seconds and click 'EXTRACT TEXT FROM IMAGE' again.\n"
            "2. Provide a free personal OCR.space key in the sidebar under 'API Configuration'.\n"
            "3. Or paste the article text directly into 'Review & Edit Extracted OCR Text' below."
        )
        return None, guidance, None
        
    final_msg = last_error or "Unable to extract text from image."
    return None, f"OCR Error: {final_msg}", None
