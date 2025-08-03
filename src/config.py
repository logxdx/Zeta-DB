from pathlib import Path
import torch

# --- Project Root ---
# The script is run from the project root
# Goes up two levels from src/config.py to zeta db/
PROJECT_ROOT = Path(__file__).parent.parent 

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True, parents=True)

# --- File Names ---
INDEX_FILE = INDEX_DIR / "zeta_db"

# --- Embedding Model ---
CLIP_MODEL_NAME = "jinaai/jina-clip-v2"
EMBEDDING_DIMENSION = 1024

# --- Search ---
DEFAULT_SEARCH_K = 10 # Default number of nearest neighbors to retrieve

# --- Ingestion ---
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# --- Device ---
# Automatically detect CUDA, fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Or force CPU: DEVICE = "cpu"