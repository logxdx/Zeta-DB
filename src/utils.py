import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_vectors(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes vector to unit L2 norm (makes them length 1).

    Args:
        vector: A numpy array of vector (1D or 2D).

    Returns:
        The normalized vector. Returns the original vector if the norm is zero.
    """

    try:
        if vector.ndim == 1: # Single vector
            norm = np.linalg.norm(vector)
            if norm == 0:
                logger.warning("Cannot normalize a zero vector.")
                return vector
            return (vector.T / norm).T
        else:
            norms = np.linalg.norm(vector, axis=-1, keepdims=True)
            # Prevent division by zero for zero vector in the batch
            zero_norms_mask = (norms == 0)
            if np.any(zero_norms_mask):
                logger.warning(f"Found {np.sum(zero_norms_mask)} zero vector in batch during normalization. They will remain zero vector.")
                # Avoid division by zero, keep zero vector as they are.
                norms[zero_norms_mask] = 1e-10 # Replace norm with small value to avoid NaN, result will be ~0 vector
                # Or handle differently, e.g., return original vector for those rows?
                # For now, they become near-zero vector after division.

            return vector / norms
    except Exception as e:
        raise ValueError(f"Error normalizing vector: {e}") from e
