import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import logging

# Import configuration and utility functions
from src.config import CLIP_MODEL_NAME, DEVICE, FAISS_METRIC_TYPE
from src.utils import normalize_vectors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPEmbedder:
    """Handles loading the CLIP model and generating embeddings."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE, metric_type: str = FAISS_METRIC_TYPE):
        """
        Initializes the CLIP model and processor.

        Args:
            model_name: The name of the pre-trained CLIP model to load.
            device: The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model_name = model_name
        self.normalize = metric_type == 'COSINE'
        logger.info(f"Initializing CLIPEmbedder with model: {self.model_name} on device: {self.device}")
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval() # Set model to evaluation mode
            logger.info("CLIP model and processor loaded successfully.")
            # Get embedding dimension dynamically
            self.embedding_dim = self._get_embedding_dim()
            logger.info(f"Detected embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model '{self.model_name}': {e}", exc_info=True)
            raise

    def _get_embedding_dim(self) -> int:
        """Helper to determine the output embedding dimension."""
        # Embed a dummy text to find the output dimension
        dummy_text = "dimension check"
        try:
            embedding = self.embed_text(dummy_text) # Don't normalize for dim check
            if embedding is not None:
                return embedding.shape[-1]
            else:
                # Fallback based on common model knowledge if embedding failed
                logger.warning("Embedding dummy text failed, attempting fallback dimension check.")
                if "base" in self.model_name.lower(): return 512
                if "large" in self.model_name.lower(): return 768
                raise ValueError("Could not determine embedding dimension.")
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {e}", exc_info=True)
            # Provide common defaults or raise error
            if "base" in self.model_name.lower():
                logger.warning("Falling back to dimension 512 based on model name \'base\'.")
                return 512
            if "large" in self.model_name.lower():
                logger.warning("Falling back to dimension 768 based on model name \'large\'.")
                return 768
            raise ValueError(f"Could not determine embedding dimension for model {self.model_name}.") from e

    def embed_image(self, image_path: str) -> np.ndarray | None:
        """
        Generates an embedding for a given image file.

        Args:
            image_path: Path to the image file.

        Returns:
            A numpy array representing the image embedding, or None if an error occurs.
        """
        try:
            image = Image.open(image_path).convert("RGB") # Ensure image is RGB
            tokens = self.processor(text=None, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**tokens)
            vec = image_features.detach().cpu().numpy()

            if self.normalize:
                vec = normalize_vectors(vec)

            # Ensure the output is consistently 1D for single images after potential normalization
            return vec.squeeze() if vec.ndim > 1 else vec

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to embed image '{image_path}': {e}", exc_info=True)
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
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
        try:
            tokens = self.processor(text=text, images=None, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**tokens)
            vec = text_features.detach().cpu().numpy()

            if self.normalize:
                vec = normalize_vectors(vec)

            # Ensure the output is consistently 1D for single text inputs
            return vec.squeeze() if vec.ndim > 1 else vec

        except Exception as e:
            logger.error(f"Failed to embed text '{text[:50]}...': {e}", exc_info=True)
            return None
