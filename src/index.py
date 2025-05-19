"""
This class provides the core Faiss functionality. It handles loading/saving, initializing the correct index type based on the configuration, adding vectors (converting to `float32`), and searching. It also includes dimension checks and logging.
"""


import faiss
import numpy as np
import logging
from pathlib import Path
import time

# Import configuration
from src.config import FAISS_METRIC_TYPE, INDEX_FILE, DEFAULT_SEARCH_K

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissIndex:
    """
    Manages Faiss index operations including loading, saving, adding, and searching vectors.
    """
    def __init__(self, embedding_dim: int = 768, index_path: Path = INDEX_FILE, metric_type: str = FAISS_METRIC_TYPE):
        """
        Initializes the FaissIndex. Tries to load an existing index,
        otherwise prepares to initialize a new one upon first vector addition.

        Args:
            embedding_dim: The dimension of the vectors to be indexed.
            index_path: Path to the Faiss index file.
            metric_type: The distance metric ('L2' or 'IP' or 'COSINE').
        """
        self.index_path = Path(index_path)
        self.metric_type = metric_type.upper()
        self.embedding_dim = embedding_dim
        self.index = None

        if self.metric_type not in ['L2', 'IP', 'COSINE']:
            raise ValueError(f"Unsupported metric type: {self.metric_type}. Choose 'L2' or 'IP' or 'COSINE'.")

        self.load_index()

    def _initialize_index(self):
        """Initializes a new Faiss index based on the configuration."""
        if self.embedding_dim <= 0:
             raise ValueError("Embedding dimension must be positive to initialize index.")

        logger.info(f"Initializing new Faiss index: Dimension={self.embedding_dim}, Metric={self.metric_type}")
        try:
            if self.metric_type == 'L2':
                # Using IndexFlatL2 - simple baseline, exact search.
                # Consider more advanced indexes like IndexIVFFlat for larger datasets.
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(f"Successfully initialized IndexFlatL2 with dimension {self.embedding_dim}.")
            elif self.metric_type == 'IP' or self.metric_type == 'COSINE':
                # Using IndexFlatIP - simple baseline for inner product.
                # For COSINE similarity, vectors MUST be normalized (L2 norm = 1) BEFORE adding.
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if self.metric_type == 'COSINE':
                    logger.info(f"Successfully initialized IndexFlatIP for COSINE similarity with dimension {self.embedding_dim}. Ensure vectors are normalized before adding.")
                else: # self.metric_type == 'IP'
                    logger.info(f"Successfully initialized IndexFlatIP for Inner Product similarity with dimension {self.embedding_dim}.")
            # Note: The __init__ method already validates metric_type, so no else needed here for unknown types.
        except Exception as e:
            logger.error(f"Failed to initialize Faiss index: {e}", exc_info=True)
            self.index = None # Ensure index is None if init fails
            raise # Re-raise the exception

    def load_index(self):
        """Loads the Faiss index from the specified path."""
        if self.index_path.exists():
            logger.info(f"Attempting to load index from {self.index_path}...")
            try:
                start_time = time.time()
                self.index = faiss.read_index(str(self.index_path))
                end_time = time.time()
                logger.info(f"Successfully loaded index with {self.index.ntotal} vectors in {end_time - start_time:.2f} seconds.")

                # Validate dimension
                if self.index.d != self.embedding_dim:
                    logger.error(f"Loaded index dimension ({self.index.d}) does not match expected dimension ({self.embedding_dim}).")
                    raise ValueError(f"Index dimension mismatch: loaded={self.index.d}, expected={self.embedding_dim}")

            except Exception as e:
                logger.error(f"Failed to load index from {self.index_path}: {e}", exc_info=True)
                self.index = None # Ensure index is None if loading fails
                raise
        else:
            logger.info(f"Index file not found at {self.index_path}. Index will be initialized on first add.")
            self.index = None # Explicitly set to None

    def save_index(self):
        """Saves the current Faiss index to the specified path."""
        if self.index is None:
            logger.warning("Save aborted: Index is not initialized or loaded.")
            return

        if self.index.ntotal == 0:
            logger.warning("Save aborted: Index is empty.")
            return # Uncomment to prevent saving empty indexes

        logger.info(f"Saving index with {self.index.ntotal} vectors to {self.index_path}...")
        # Ensure the directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            start_time = time.time()
            faiss.write_index(self.index, str(self.index_path))
            end_time = time.time()
            logger.info(f"Index saved successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to save index to {self.index_path}: {e}", exc_info=True)
            raise # Re-raise the exception

    def add_vectors(self, vectors: np.ndarray):
        """
        Adds one or more vectors to the index. Initializes the index if it's the first addition.

        Args:
            vectors: A numpy array of vectors (1D or 2D). Vectors should be float32.
                     For 'IP' metric used for Cosine Similarity, vectors should be normalized *before* calling this method.
        """
        if vectors is None or vectors.size == 0:
            logger.warning("Add vectors skipped: input is None or empty.")
            return

        # Ensure vectors are 2D and float32, as required by Faiss
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        if vectors.dtype != np.float32:
            logger.warning(f"Input vectors dtype is {vectors.dtype}, converting to float32.")
            vectors = vectors.astype(np.float32)

        # Check dimension consistency
        if vectors.shape[1] != self.embedding_dim:
            logger.error(f"Vector dimension mismatch: expected {self.embedding_dim}, got {vectors.shape[1]}. Skipping add.")
            return

        # Initialize index if it doesn't exist yet (first add)
        if self.index is None:
            self._initialize_index()
            if self.index is None: # Check if initialization failed
                 logger.error("Cannot add vectors: Index initialization failed.")
                 return # Or raise an error

        try:
            self.index.add(vectors)
            logger.debug(f"Added {vectors.shape[0]} vectors. Index total: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to add vectors to Faiss index: {e}", exc_info=True)

    def search(self, query_vectors: np.ndarray, k: int = DEFAULT_SEARCH_K) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Searches the index for the k nearest neighbors of the query vector(s).

        Args:
            query_vectors: A numpy array of query vectors (1D or 2D). Should be float32.
                           For 'IP' metric used for Cosine Similarity, query should be normalized *before* calling.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple containing:
            - distances (np.ndarray): The distances/similarities of the neighbors. Shape (num_queries, k).
                                      For L2, lower is better. For IP, higher is better.
            - indices (np.ndarray): The Faiss indices of the neighbors. Shape (num_queries, k).
                                    Contains -1 for slots where fewer than k neighbors were found.
            Returns (None, None) if the index is empty or not ready.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search skipped: Index is not initialized or is empty.")
            return None, None

        if query_vectors is None or query_vectors.size == 0:
            logger.warning("Search skipped: query_vectors is None or empty.")
            return None, None

        # Ensure query is 2D and float32
        if query_vectors.ndim == 1:
            query_vectors = np.expand_dims(query_vectors, axis=0)
        if query_vectors.dtype != np.float32:
            logger.warning(f"Query vectors dtype is {query_vectors.dtype}, converting to float32.")
            query_vectors = query_vectors.astype(np.float32)

        # Check dimension consistency
        if query_vectors.shape[1] != self.embedding_dim:
            logger.error(f"Query vector dimension mismatch: expected {self.embedding_dim}, got {query_vectors.shape[1]}. Aborting search.")
            return None, None # Or raise ValueError

        num_queries = query_vectors.shape[0]
        logger.debug(f"Searching for {k} nearest neighbors for {num_queries} query vector(s)...")

        try:
            distances, indices = self.index.search(query_vectors, k)
            logger.debug(f"Search completed. Found indices shape: {indices.shape}, distances shape: {distances.shape}")
            return distances, indices
        except Exception as e:
            logger.error(f"Faiss search failed: {e}", exc_info=True)
            return None, None # Or raise

    @property
    def count(self) -> int:
        """Returns the total number of vectors currently in the index."""
        return self.index.ntotal if self.index else 0

    @property
    def dimension(self) -> int:
        """Returns the dimension of the vectors in the index."""
        if self.index is not None:
            return self.index.d
        return self.embedding_dim # Fallback to configured dim if index not yet ready
