import io
import requests
from typing import Tuple, Optional
from PIL import Image, ImageEnhance
from .config import OCR_API_KEY_ENV

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

def preprocess_image_for_ocr(image_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    """Validates image format and applies grayscale and contrast enhancement."""
    if not image_bytes:
        return None, "No image data received."
        
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return None, "File too large: Maximum supported image size is 10 MB."
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ("PNG", "JPEG", "JPG", "MPO"):
            return None, f"Unsupported image format: {img.format}. Please upload PNG, JPG, or JPEG."
            
        # Convert RGBA or CMYK to RGB
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

def extract_text_from_image(image_bytes: bytes, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from image bytes with PIL preprocessing, local OCR fallback,
    and OCR.space cloud API.
    
    Returns:
        Tuple[extracted_text, error_message]
    """
    preprocessed_bytes, prep_err = preprocess_image_for_ocr(image_bytes)
    if prep_err:
        return None, prep_err
        
    # Check if local pytesseract is available
    try:
        import pytesseract
        img = Image.open(io.BytesIO(preprocessed_bytes))
        text = pytesseract.image_to_string(img)
        if text and len(text.strip()) > 10:
            return text.strip(), None
    except Exception:
        pass  # Fall through to cloud OCR
        
    # Cloud OCR via OCR.space
    key = api_key if (api_key and api_key.strip()) else OCR_API_KEY_ENV
    if not key:
        key = "helloworld"
        
    try:
        payload = {
            "apikey": key,
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
        if response.status_code != 200:
            return None, f"OCR Service returned HTTP status {response.status_code}."
            
        result = response.json()
        if result.get("OCRExitCode") == 1:
            parsed = result.get("ParsedResults", [])
            if parsed:
                text = parsed[0].get("ParsedText", "").strip()
                if text:
                    return text, None
                else:
                    return None, "No readable text detected in this image. Please ensure the screenshot has clear typography."
                    
        error_msg = result.get("ErrorMessage")
        if isinstance(error_msg, list) and len(error_msg) > 0:
            error_msg = error_msg[0]
        elif not error_msg:
            error_msg = "OCR Engine could not parse image text."
            
        return None, str(error_msg)
    except Exception as e:
        return None, f"OCR Connection Failure: {str(e)}"
