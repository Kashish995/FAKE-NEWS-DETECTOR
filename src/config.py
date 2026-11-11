import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
DB_FILE = BASE_DIR / "news_logs.db"

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
CALIBRATION_PATH = MODEL_DIR / "calibration.pkl"
METRICS_JSON_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.npy"

# Verdict Tiers
VERDICT_RELIABLE = "Likely Reliable"
VERDICT_MISLEADING = "Likely Misleading"
VERDICT_VERIFY = "Needs Further Verification"

# Thresholds
MIN_WORDS_FOR_RELIABLE_ML = 50
CONFIDENCE_HIGH_THRESHOLD = 0.65
CONFIDENCE_LOW_THRESHOLD = 0.40

# Environment / Default API keys
NEWSAPI_KEY_ENV = os.getenv("NEWSAPI_KEY", "")
FACTCHECK_API_KEY_ENV = os.getenv("FACTCHECK_API_KEY", "")
OCR_API_KEY_ENV = os.getenv("OCR_API_KEY", "helloworld")
