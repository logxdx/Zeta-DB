import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors to unit L2 norm (makes them length 1).

    Args:
        vectors: A numpy array of vectors (1D or 2D).

    Returns:
        The normalized vectors. Returns the original vector if the norm is zero.
    """

    if vectors.ndim == 1: # Single vector
        norm = np.linalg.norm(vectors)
        if norm == 0:
           logger.warning("Cannot normalize a zero vector.")
           return vectors
        return (vectors.T / norm).T
    elif vectors.ndim == 2: # Batch of vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Prevent division by zero for zero vectors in the batch
        zero_norms_mask = (norms == 0)
        if np.any(zero_norms_mask):
            logger.warning(f"Found {np.sum(zero_norms_mask)} zero vectors in batch during normalization. They will remain zero vectors.")
            # Avoid division by zero, keep zero vectors as they are.
            norms[zero_norms_mask] = 1e-10 # Replace norm with small value to avoid NaN, result will be ~0 vector
            # Or handle differently, e.g., return original vectors for those rows?
            # For now, they become near-zero vectors after division.

        return vectors / norms
    else:
        raise ValueError(f"Unsupported vector dimensions: {vectors.ndim}. Expected 1 or 2.")
