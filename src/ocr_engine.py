import requests
from typing import Tuple, Optional
from .config import OCR_API_KEY_ENV

def extract_text_from_image(image_bytes: bytes, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts text from image bytes using OCR.space REST API with local fallback checks.
    
    Returns:
        Tuple[extracted_text, error_message]
    """
    if not image_bytes:
        return None, "No image data received."
        
    # Check if local pytesseract is available
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
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
            "file": ("image.png", image_bytes, "image/png")
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
