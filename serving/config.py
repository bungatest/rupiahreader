import os
import secrets
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SERVING_DIR = Path(os.environ.get("SERVING_DIR", "C:/rupiah_serving" if os.name == "nt" and Path("C:/").exists() else str(BASE_DIR / "serving_storage")))
SERVING_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = SERVING_DIR / "models"
if not MODELS_DIR.exists() and (BASE_DIR / "model_coins" / "models").exists():
    MODELS_DIR = BASE_DIR / "model_coins" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = SERVING_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_DIR = SERVING_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = SERVING_DIR / "data"

REGISTRY_PATH = MODELS_DIR / "registry.json"
FIELD_LOG_PATH = LOGS_DIR / "field_eval.db"
FIELD_CSV_PATH = LOGS_DIR / "field_test.csv"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.jsonl"

FRONTEND_DIR = BASE_DIR / "rupiah-reader"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR

COIN_SERVER_URL = os.environ.get("COIN_SERVER_URL", "http://127.0.0.1:8000")

# Classes and metadata - Banknotes
CLASSES = ['idr_1000', 'idr_2000', 'idr_5000', 'idr_10000', 'idr_20000', 'idr_50000', 'idr_100000']

PRETTY_NOMINAL = {
    'idr_1000': 'Rp 1.000',
    'idr_2000': 'Rp 2.000',
    'idr_5000': 'Rp 5.000',
    'idr_10000': 'Rp 10.000',
    'idr_20000': 'Rp 20.000',
    'idr_50000': 'Rp 50.000',
    'idr_100000': 'Rp 100.000'
}

SPOKEN_NOMINAL = {
    'idr_1000': 'Seribu rupiah',
    'idr_2000': 'Dua ribu rupiah',
    'idr_5000': 'Lima ribu rupiah',
    'idr_10000': 'Sepuluh ribu rupiah',
    'idr_20000': 'Dua puluh ribu rupiah',
    'idr_50000': 'Lima puluh ribu rupiah',
    'idr_100000': 'Seratus ribu rupiah'
}

VALUE_MAPPING = {
    'idr_1000': 1000,
    'idr_2000': 2000,
    'idr_5000': 5000,
    'idr_10000': 10000,
    'idr_20000': 20000,
    'idr_50000': 50000,
    'idr_100000': 100000
}

# Classes and metadata - Coins
COIN_CLASSES = ['100', '200', '500', '1000', 'sisi_garuda']

COIN_PRETTY_NOMINAL = {
    '100': 'Rp 100',
    '200': 'Rp 200',
    '500': 'Rp 500',
    '1000': 'Rp 1.000',
    'sisi_garuda': 'Sisi Garuda / Pahlawan'
}

COIN_SPOKEN_NOMINAL = {
    '100': 'Seratus rupiah',
    '200': 'Dua ratus rupiah',
    '500': 'Lima ratus rupiah',
    '1000': 'Seribu rupiah',
    'sisi_garuda': 'Sisi gambar Garuda atau Pahlawan'
}

COIN_HERO_KNOWLEDGE = {
    '100': {'tokoh': 'Prof. Dr. Ir. Herman Johannes', 'nominal': 'Rp 100', 'tags': ['20_15_47', '20_16_23']},
    '200': {'tokoh': 'Dr. Tjipto Mangoenkoesoemo', 'nominal': 'Rp 200', 'tags': ['20_18_19', '20_19_22', '20_21_09']},
    '500': {'tokoh': 'Letjen TNI T.B. Simatupang', 'nominal': 'Rp 500', 'tags': ['20_22_07', '4daa1e38']},
    '1000': {'tokoh': 'Mr. I Gusti Ketut Pudja', 'nominal': 'Rp 1000', 'tags': ['20_23_50', '20_24_30', '20_25_40']}
}

# Fixed or generated access token for session
TOKEN_FILE = SERVING_DIR / "access_token.txt"
if TOKEN_FILE.exists():
    API_TOKEN = TOKEN_FILE.read_text(encoding="utf-8").strip()
else:
    API_TOKEN = secrets.token_hex(16)
    TOKEN_FILE.write_text(API_TOKEN, encoding="utf-8")

# Service limits
MAX_FRAME_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_FRAMES_PER_REQUEST = 8
RATE_LIMIT_PER_MINUTE = 120


# Compatibility aliases
SERVING_PORT = int(os.environ.get("SERVING_PORT", 8001))
SERVING_MODELS_DIR = MODELS_DIR
RUPIAH_READER_DIR = FRONTEND_DIR
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
CLASS_DISPLAY_NAMES = PRETTY_NOMINAL
SPOKEN_TEXT = SPOKEN_NOMINAL
DECISION_THRESHOLDS = {
    "consensus": 0.75,
    "mobilenet": 0.70,
    "fnn": 0.50,
    "cnn": 0.80,
}
