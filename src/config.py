from pathlib import Path
import torch

# --- Project Root ---
# The script is run from the project root
# Goes up two levels from src/config.py to multimodal_vector_db/
PROJECT_ROOT = Path(__file__).parent.parent 

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True, parents=True)
METADATA_DIR = DATA_DIR / "metadata"
METADATA_DIR.mkdir(exist_ok=True, parents=True)
IMAGE_SOURCE_DIR = PROJECT_ROOT / "images_to_process"
IMAGE_SOURCE_DIR.mkdir(exist_ok=True, parents=True)

# --- File Names ---
INDEX_FILE = INDEX_DIR / "vector_db.index"
# Using JSON for simplicity, consider SQLite for better scalability/querying
METADATA_FILE = METADATA_DIR / "vector_db_meta.json"

# --- Embedding Model ---
# Options: "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336", etc.
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"

# --- Faiss Indexing ---
# Choose metric: 'L2', 'IP' (Inner Product), COSINE.
FAISS_METRIC_TYPE = 'IP' # Or 'L2' Or 'COSINE'

# --- Search ---
DEFAULT_SEARCH_K = 5 # Default number of nearest neighbors to retrieve

# --- Ingestion ---
# How often to check the IMAGE_SOURCE_DIR (in seconds) if using polling
# Watchdog (event-based) is generally preferred if available
INGESTION_POLL_INTERVAL = 60
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# --- Device ---
# Automatically detect CUDA, fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Or force CPU: DEVICE = "cpu"