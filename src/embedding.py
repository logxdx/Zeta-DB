from transformers import AutoModel
from PIL import Image, UnidentifiedImageError
import numpy as np
import logging

# Import configuration and utility functions
from .config import CLIP_MODEL_NAME, DEVICE, EMBEDDING_DIMENSION
from .utils import normalize_vectors

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Handles loading the CLIP model and generating embeddings."""

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: str = DEVICE,
        embedding_dim: int = EMBEDDING_DIMENSION,
    ):
        """
        Initializes the CLIP model and processor.

        Args:
            metric_type: The type of distance metric to use for FAISS indexing ('l2', 'DOT', or 'COSINE').
            embedding_dim: The dimension of the embeddings to be generated (64 to 1024).
        """
        self.device = device
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        logger.info(
            f"Initializing CLIPEmbedder with model: {self.model_name} on device: {self.device}"
        )
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
            logger.info("CLIP model loaded successfully.")
            # Get embedding dimension dynamically
        except Exception as e:
            logger.error(
                f"Failed to load CLIP model '{self.model_name}': {e}", exc_info=True
            )
            raise

    def embed_image(
        self, image_path: str, normalize: bool = False
    ) -> np.ndarray | None:
        """
        Generates an embedding for a given image file.

        Args:
            image_path: Path to the image file.

        Returns:
            A numpy array representing the image embedding, or None if an error occurs.
        """
        if not image_path:
            logger.warning("Attempted to embed image from an empty path.")
            return None
        logger.debug(f"Attempting to embed image: {image_path}")
        try:
            img = Image.open(image_path)
            image = img.convert("RGB")  # Ensure image is RGB
            image_embeddings = self.model.encode_image(
                image, truncate_dim=self.embedding_dim
            )
            if normalize:
                image_embeddings = normalize_vectors(image_embeddings)
            # Ensure the output is consistently 1D for single images after potential normalization
            logger.info(f"Successfully embedded image: {image_path}")
            return (
                image_embeddings.squeeze()
                if image_embeddings.ndim > 1
                else image_embeddings
            )
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except UnidentifiedImageError:
            logger.error(
                f"Cannot identify image file (possibly corrupt or unsupported format): {image_path}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to embed image '{image_path}': {e}", exc_info=True)
            return None

    def embed_text(self, text: str, normalize: bool = False) -> np.ndarray | None:
        """
        Generates an embedding for a given text string.

        Args:
            text: The text string to embed.

        Returns:
            A numpy array representing the text embedding, or None if an error occurs.
        """
        if not text:
            logger.warning("Attempted to embed empty text.")
            return None
        logger.debug(f"Attempting to embed text: {text[:50]}...")
        try:
            text_embeddings = self.model.encode_text(
                text, truncate_dim=self.embedding_dim
            )
            if normalize:
                text_embeddings = normalize_vectors(text_embeddings)
            # Ensure the output is consistently 1D for single text inputs
            logger.info(f"Successfully embedded text")
            return (
                text_embeddings.squeeze()
                if text_embeddings.ndim > 1
                else text_embeddings
            )
        except Exception as e:
            logger.error(f"Failed to embed text '{text[:50]}...': {e}", exc_info=True)
            return None
