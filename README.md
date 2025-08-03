# Zeta DB: Multimodal Vector Database

Zeta DB is a Python-based multimodal vector database designed to store and search for images and text using their semantic embeddings. It leverages the power of CLIP models for generating embeddings and LanceDB for storage & similarity search.

## Features

- **Multimodal Embeddings:** Generates and stores embeddings for both images and text using CLIP models.
- **Efficient Similarity Search:** Utilizes LanceDB for fast and scalable nearest neighbor search on embedding vectors.
- **Persistent Storage:** LanceDB index is saved to disk.
- **Configurable:** Easily configure model names, storage paths, and search parameters via `src/config.py`.
- **Duplicate Prevention:** Basic mechanism to avoid re-indexing already processed image paths.
- **Normalization Option:** Supports normalization of vectors for cosine similarity searches.

## Project Structure

```
Zeta DB/
├── data/                     # Default directory for storing index
│   ├── index/                # Stores the LanceDB index file (vector_db.index)
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Project configuration
│   ├── embedding.py          # CLIP model loading and embedding generation
│   ├── index.py              # LanceDB index management
│   └── utils.py              # Utility functions (e.g., vector normalization)
├── zeta.py                   # Main entry point
├── .gitignore
├── README.md
└── requirements.txt          # Python dependencies
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/logxdx/Zeta-DB.git
    cd Zeta DB
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration (Optional):**
    Review and modify `src/config.py` if you need to change default paths, CLIP model, etc. The default configuration uses `jinaai/jina-clip-v2` with full_size embeddings (1024 dimensions). It will automatically use CUDA if available, otherwise CPU.

## Core Components

The `src` directory contains the core logic for Zeta DB, organized into the following modules:

- **`config.py`**: Centralized configuration for all system parameters, including model names, file paths, and embedding dimensions. This allows for easy customization and management of the database settings.

- **`embedding.py`**: Implements the `CLIPEmbedder` class, which is responsible for loading a pre-trained CLIP model and generating vector embeddings for both images and text. It's designed to be easily extensible to support other embedding models.

- **`index.py`**: Contains the `LanceDBIndex` class, which manages the `lancedb` vector index. This class handles the creation, storage, and retrieval of vectors, providing a simple interface for performing similarity searches.

- **`utils.py`**: A collection of utility functions used throughout the project. Currently, it includes a function for normalizing vectors, which is a crucial preprocessing step for certain distance metrics.

## Getting Started

To use Zeta DB, you'll need to have Python installed, along with the necessary dependencies. Once set up, you can run the `zeta.py` script to start the application with the gradio interface.

### Example Usage

```python
python zeta.py
```