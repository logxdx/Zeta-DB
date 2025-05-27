import json
import logging
from pathlib import Path
import threading
from typing import Dict, Any, Optional, List

# Import configuration
from . import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetadataStore:
    """
    Manages the mapping between Faiss index IDs and item metadata (e.g., file paths, text).
    Uses a JSON file for persistence.
    """
    def __init__(self, metadata_path: Path = config.METADATA_FILE):
        """
        Initializes the MetadataStore, loading existing data if available.

        Args:
            metadata_path: Path to the metadata JSON file.
        """
        self.metadata_path = Path(metadata_path)
        # Use a lock for thread safety when accessing/modifying metadata
        self._lock = threading.Lock()
        # In-memory store: {faiss_id (int): {"type": "image/text", "path": "...", "text": "...", ...}}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        # Keep track of the next ID to assign
        self._next_id: int = 0
        # For quick lookups to prevent duplicates (e.g., set of image paths)
        self._existing_paths: set[str] = set()
        # Consider adding a set for existing text if duplicate text prevention is needed

        self.load_metadata()

    def load_metadata(self):
        """Loads metadata from the JSON file."""
        with self._lock:
            if self.metadata_path.exists():
                logger.info(f"Attempting to load metadata from {self.metadata_path}...")
                try:
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        # Load raw data which might have string keys
                        raw_data: Dict[str, Dict[str, Any]] = json.load(f)

                        # Convert string keys back to integers and find max ID
                        max_id = -1
                        temp_metadata = {}
                        temp_paths = set()
                        for str_id, meta_item in raw_data.items():
                            try:
                                current_id = int(str_id)
                                temp_metadata[current_id] = meta_item
                                if meta_item.get("type") == "image" and "path" in meta_item:
                                    temp_paths.add(meta_item["path"])
                                # Add similar check for text if needed
                                if current_id > max_id:
                                    max_id = current_id
                            except ValueError:
                                logger.warning(f"Skipping invalid key '{str_id}' in metadata file.")

                        self._metadata = temp_metadata
                        self._existing_paths = temp_paths
                        self._next_id = max_id + 1
                        logger.info(f"Successfully loaded {len(self._metadata)} metadata entries. Next ID: {self._next_id}")

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from {self.metadata_path}. Starting with empty metadata.", exc_info=True)
                    self._metadata = {}
                    self._next_id = 0
                    self._existing_paths = set()
                except Exception as e:
                    logger.error(f"Failed to load metadata from {self.metadata_path}: {e}", exc_info=True)
                    # Decide whether to start fresh or raise
                    self._metadata = {}
                    self._next_id = 0
                    self._existing_paths = set()
            else:
                logger.info(f"Metadata file not found at {self.metadata_path}. Starting fresh.")
                self._metadata = {}
                self._next_id = 0
                self._existing_paths = set()

    def save_metadata(self):
        """Saves the current metadata to the JSON file."""
        with self._lock:
            logger.info(f"Saving {len(self._metadata)} metadata entries to {self.metadata_path}...")
            # Ensure the directory exists
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Convert integer keys to strings for JSON compatibility
                data_to_save = {str(k): v for k, v in self._metadata.items()}
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=4)
                logger.info("Metadata saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save metadata to {self.metadata_path}: {e}", exc_info=True)
                # Decide if we should raise the error

    def add_metadata(self, metadata_item: Dict[str, Any]) -> Optional[int]:
        """
        Adds a new metadata item and assigns it the next available ID.

        Args:
            metadata_item: A dictionary containing the metadata (e.g., {"type": "image", "path": "..."}).

        Returns:
            The assigned Faiss ID (int) for the new item, or None if addition failed (e.g., duplicate).
        """
        if not isinstance(metadata_item, dict) or "type" not in metadata_item:
             logger.error(f"Invalid metadata item format: {metadata_item}")
             return None

        # --- Duplicate Check ---
        # Check for duplicate image paths
        item_type = metadata_item.get("type")
        item_path = metadata_item.get("path")
        if item_type == "image" and item_path:
            if item_path in self._existing_paths:
                logger.warning(f"Duplicate image path detected: '{item_path}'. Skipping add.")
                # Optionally, find and return the existing ID
                # for existing_id, existing_meta in self._metadata.items():
                #     if existing_meta.get("path") == item_path:
                #         return existing_id
                return None
        # Add similar check for text content if needed

        # --- Add Item ---
        with self._lock:
            current_id = self._next_id
            self._metadata[current_id] = metadata_item
            # Update quick lookup set if it's an image path
            if item_type == "image" and item_path:
                self._existing_paths.add(item_path)

            self._next_id += 1
            logger.debug(f"Added metadata for ID {current_id}: {metadata_item}")
            return current_id

    def get_metadata(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a given Faiss ID.

        Args:
            faiss_id: The Faiss index ID.

        Returns:
            The metadata dictionary, or None if the ID is not found.
        """
        with self._lock:
            return self._metadata.get(faiss_id)

    def get_metadata_batch(self, faiss_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
        """
        Retrieves metadata for a batch of Faiss IDs.

        Args:
            faiss_ids: A list of Faiss index IDs.

        Returns:
            A dictionary mapping requested Faiss IDs to their metadata (or None if not found).
        """
        results = {}
        with self._lock:
            for faiss_id in faiss_ids:
                results[faiss_id] = self._metadata.get(faiss_id)
        return results

    def get_next_id(self) -> int:
        """Returns the next ID that will be assigned."""
        # No lock needed for reading a simple integer if atomicity isn't critical across operations
        # If strict atomicity with add is needed, use the lock here too.
        return self._next_id

    def path_exists(self, file_path: str) -> bool:
        """Checks if a given file path already exists in the metadata."""
        # No lock needed for reading the set if updates are atomic via add_metadata's lock
        return file_path in self._existing_paths

    @property
    def count(self) -> int:
        """Returns the total number of metadata entries."""
        with self._lock:
            return len(self._metadata)

# Example usage (usually instantiated elsewhere)
# if __name__ == '__main__':
#     store = MetadataStore()
#     print(f"Initial count: {store.count}, Next ID: {store.get_next_id()}")
#
#     # Add some data
#     id1 = store.add_metadata({"type": "image", "path": "/path/to/image1.jpg", "source": "camera1"})
#     id2 = store.add_metadata({"type": "text", "text": "A description of image 1."})
#     id3 = store.add_metadata({"type": "image", "path": "/path/to/image2.png"})
#     # Try adding duplicate path
#     id4 = store.add_metadata({"type": "image", "path": "/path/to/image1.jpg"})
#
#     print(f"Added IDs: {id1}, {id2}, {id3}, {id4}") # id4 should be None
#     print(f"Metadata for ID {id1}: {store.get_metadata(id1)}")
#     print(f"Metadata for ID 99 (non-existent): {store.get_metadata(99)}")
#     print(f"Path exists '/path/to/image1.jpg': {store.path_exists('/path/to/image1.jpg')}")
#     print(f"Path exists '/path/to/other.jpg': {store.path_exists('/path/to/other.jpg')}")
#     print(f"Current count: {store.count}, Next ID: {store.get_next_id()}")
#
#     # Save the metadata
#     store.save_metadata()
#
#     # Load again to test persistence
#     store2 = MetadataStore()
#     print(f"\nLoaded store count: {store2.count}, Next ID: {store2.get_next_id()}")
#     print(f"Metadata for ID {id2} from loaded store: {store2.get_metadata(id2)}")
#     print(f"Path exists '/path/to/image1.jpg' in loaded store: {store2.path_exists('/path/to/image1.jpg')}")

"""
Key features:

1.  **Persistence:** Loads from and saves to the JSON file specified in `config.py`.
2.  **In-Memory Cache:** Keeps the metadata in a dictionary (`_metadata`) for fast lookups.
3.  **ID Management:** Automatically assigns sequential integer IDs (`_next_id`).
4.  **Duplicate Prevention:** Includes a basic check (`path_exists` and logic in `add_metadata`) to avoid adding images with the same path multiple times. You could extend this for text content if needed.
5.  **Thread Safety:** Uses a `threading.Lock` to protect access to the shared metadata dictionary and path set, making it safer if used in multi-threaded contexts (like concurrent ingestion and querying).
6.  **Batch Retrieval:** Added `get_metadata_batch` for efficiency when looking up multiple results from a search.
7.  **Error Handling:** Includes basic error handling for file I/O and JSON parsing.

Now we have the core components: configuration, embedding, indexing, and metadata storage. The next logical step is `ingestion.py` to tie these together for processing new images.
"""