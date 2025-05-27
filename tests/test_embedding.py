# tests for embedding.py

import torch
from ..src.embedding import embedding

if __name__ == "__main__":
    embedder = CLIPEmbedder()
    print(f"Embedder initialized. Dimension: {embedder.embedding_dim}")
    text_vec = embedder.embed_text("An astronaut in a jungle watching the moon.")
    img_vec = embedder.embed_image("./data/test.jpg")
    if text_vec is not None:
        print("Text embedding shape:", text_vec.shape)
    if img_vec is not None:
        print("Image embedding shape:", img_vec.shape)

    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(text_vec), torch.tensor(img_vec), dim=0
    )
    print(f"Cosine similarity: {similarity.item():.4f}")
    
    text_vec_1 = embedder.embed_text("An astronaut on a planet.")
    text_vec_2 = embedder.embed_text("An astronaut on a moon.")
    if text_vec_1 is not None:
        print("Text embedding shape:", text_vec_1.shape)

    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(text_vec_2), torch.tensor(text_vec_1), dim=0
    )
    print(f"Cosine similarity: {similarity.item():.4f}")
