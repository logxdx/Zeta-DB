# Zeta DB: Multimodal Vector Database

Zeta DB is a Python-based multimodal vector database designed to store and search for images and text using their semantic embeddings. It leverages the power of CLIP models for generating embeddings and Faiss for efficient similarity search.

## Features

*   **Multimodal Embeddings:** Generates and stores embeddings for both images and text using OpenAI's CLIP models.
*   **Efficient Similarity Search:** Utilizes Faiss for fast and scalable nearest neighbor search on embedding vectors.
*   **Persistent Storage:**
    *   Faiss index is saved to disk.
    *   Metadata (linking Faiss IDs to original data like file paths or text) is stored in a JSON file.
*   **Configurable:** Easily configure model names, storage paths, and search parameters via `src/config.py`.
*   **Duplicate Prevention:** Basic mechanism to avoid re-indexing already processed image paths.
*   **Thread-Safe Operations:** Metadata store operations are thread-safe.
*   **Normalization Option:** Supports normalization of vectors for cosine similarity searches.

## Project Structure

```
Zeta DB/
├── data/                     # Default directory for storing index and metadata
│   ├── index/                # Stores the Faiss index file (vector_db.index)
│   └── metadata/             # Stores the metadata JSON file (vector_db_meta.json)
├── images_to_process/        # Directory to place images for ingestion (monitored by ingestor)
├── scripts/                  # Utility scripts
│   ├── run_ingestor.py       # Script to run the ingestion process
│   └── query_cli.py          # Command-line interface for querying the database
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Project configuration
│   ├── embedding.py          # CLIP model loading and embedding generation
│   ├── index.py              # Faiss index management
│   ├── ingestion.py          # Logic for ingesting new data
│   ├── metadata_store.py     # Manages metadata associated with embeddings
│   └── utils.py              # Utility functions (e.g., vector normalization)
│   └── zeta.py               # Core class orchestrating components (likely)
├── tests/                    # Unit and integration tests
│   ├── data/                 # Test data
│   ├── test_embedding.py
│   └── test_indexer.py
├── .gitignore
├── README.md
└── requirements.txt          # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
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
    Review and modify `src/config.py` if you need to change default paths, CLIP model, Faiss metric, etc. The default configuration uses `openai/clip-vit-large-patch14-336` and `IP` (Inner Product) for Faiss metric. It will automatically use CUDA if available, otherwise CPU.

## Usage

### 1. Ingesting Data

The system is designed to ingest images placed in the `images_to_process/` directory.

*   **Run the Ingestor:**
    ```bash
    python scripts/run_ingestor.py
    ```
    This script will monitor the `images_to_process/` directory. When new images are added, it will:
    1.  Generate an embedding using the CLIP model ([`src/embedding.py`](./src/embedding.py)).
    2.  Add the embedding to the Faiss index ([`src/index.py`](./src/index.py)).
    3.  Store metadata (like the image path) in the metadata store ([`src/metadata_store.py`](./src/metadata_store.py)).
    The index and metadata will be saved to the paths defined in `src/config.py`.

### 2. Querying the Database

A command-line interface can be used to query the database.

*   **Run the Query CLI:**
    ```bash
    python scripts/query_cli.py
    ```
    This script will allow you to:
    *   **Query by text:** Enter a text description, and the system will find the most similar images/text in the database.
    *   **Query by image:** Provide an image path, and the system will find the most similar items.

    The CLI will:
    1.  Generate an embedding for your query (text or image).
    2.  Search the Faiss index for the nearest neighbors.
    3.  Retrieve metadata for the search results to display meaningful information (e.g., file paths of similar images).

## Core Components

*   **[`src/config.py`](./src/config.py):** Central configuration for paths, model names, and other parameters.
*   **[`src/embedding.py`](./src/embedding.py):** `CLIPEmbedder` class handles loading the specified CLIP model and processor, and provides methods to generate embeddings for images and text. It also dynamically determines the embedding dimension.
*   **[`src/index.py`](./src/index.py):** `FaissIndex` class manages the Faiss vector index. It supports loading/saving the index, adding new vectors, and performing similarity searches. It can be initialized with different metrics (L2, IP, COSINE).
*   **[`src/metadata_store.py`](./src/metadata_store.py):** `MetadataStore` class manages a JSON-based database to store metadata associated with each vector in the Faiss index. It handles ID assignment, duplicate path checking for images, and thread-safe operations.
*   **[`src/utils.py`](./src/utils.py):** Contains utility functions, such as `normalize_vectors` for preparing embeddings for cosine similarity.
*   **[`src/ingestion.py`](./src/ingestion.py):** (To be implemented) Will contain the logic for monitoring a source directory, processing new items, generating embeddings, and adding them to the index and metadata store.
*   **[`src/zeta.py`](./src/zeta.py):** (To be implemented) Likely the main class that orchestrates the embedder, indexer, and metadata store for high-level operations like adding items or searching.

## Dependencies

The main dependencies are:

*   `torch` & `torchvision`
*   `transformers` (for CLIP models)
*   `faiss-cpu` or `faiss-gpu` (for vector indexing)
*   `Pillow` (for image processing)
*   `numpy`

Ensure these are listed in your `requirements.txt` file.

## Future Enhancements

*   Implement `src/ingestion.py` with file system monitoring (e.g., using `watchdog`).
*   Develop `scripts/run_ingestor.py` to start and manage the ingestion process.
*   Develop `scripts/query_cli.py` for user interaction.
*   Implement the `src/zeta.py` facade class.
*   Add support for ingesting raw text snippets directly.
*   More robust duplicate detection (e.g., perceptual hashing for images, or content hashing for text).
*   Support for different/more advanced Faiss index types (e.g., `IndexIVFFlat`) for larger datasets.
*   Batch processing for ingestion and querying for improved performance.
*   API for programmatic access (e.g., using FastAPI or Flask).
*   More comprehensive unit and integration tests.
*   Option to use a more robust database (e.g., SQLite, PostgreSQL) for metadata storage.