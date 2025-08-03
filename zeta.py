import gradio as gr
import logging
import os
import uuid
from pathlib import Path
from typing import List, Union

from src.config import SUPPORTED_IMAGE_EXTENSIONS, DEFAULT_SEARCH_K
from src.embedding import CLIPEmbedder
from src.index import LanceDBIndex, VectorObject

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Global Components ---
try:
    embedder = CLIPEmbedder()
    indexer = LanceDBIndex()
    logger.info("Successfully initialized embedder and indexer.")
except Exception as e:
    logger.error(f"Fatal error during initialization: {e}", exc_info=True)
    # Gracefully exit if core components fail
    embedder = None
    indexer = None


def index_folder(table_name: str, folder_path: str) -> str:
    """
    Scans a folder for supported images, generates embeddings, and adds them to the index.

    Args:
        folder_path: The absolute path to the folder to index.

    Returns:
        A status message indicating the result of the indexing operation.
    """
    if not embedder or not indexer:
        return "Error: Core components not initialized. Please check logs."
    if not folder_path or not os.path.isdir(folder_path):
        logger.warning(f"Invalid folder path provided: {folder_path}")
        return "Error: Please provide a valid folder path."

    logger.info(f"Starting to index folder: {folder_path}")
    logger.info(f"Using Table: {table_name}")

    path_obj = Path(folder_path)
    image_paths = [
        p for p in path_obj.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_paths:
        logger.info("No supported image files found in the folder.")
        return "No new images to index."

    docs_to_add = []
    for image_path in image_paths:
        try:
            embedding = embedder.embed_image(str(image_path))
            if embedding is not None:

                doc_id = str(uuid.uuid4())
                item = VectorObject(
                    id=doc_id,
                    vector=embedding.tolist(),
                    type="image",
                    path=str(image_path),
                ).model_dump(mode="json")
                docs_to_add.append(item)

        except Exception as e:
            logger.error(
                f"Failed to process and embed image {image_path}: {e}", exc_info=True
            )

    if not docs_to_add:
        return "Could not process any images for indexing."

    try:
        indexer.insert_documents(docs_to_add, table_name=table_name)
        status_message = (
            f"Successfully indexed {len(docs_to_add)} images from {folder_path}."
        )
        logger.info(status_message)
        return status_message
    except Exception as e:
        logger.error(f"Failed to insert documents into index: {e}", exc_info=True)
        return "Error: Failed to save new index entries."


def search_index(
    query: Union[str, Path],
    top_k: int = DEFAULT_SEARCH_K,
    table_name: str = "embeddings",
) -> List[tuple[str, str]]:
    """
    Searches the index using a text query or an image file.

    Args:
        query: The text string or image path for the search.
        top_k: The number of top results to retrieve.

    Returns:
        A list of tuples, where each tuple contains the file path and a score/distance.
    """
    if not embedder or not indexer:
        logger.error("Search cannot proceed: core components not initialized.")
        return []

    embedding = None
    if isinstance(query, str) and os.path.isfile(query):
        logger.info(f"Performing image search for: {query}")
        embedding = embedder.embed_image(query)
    elif isinstance(query, str):
        logger.info(f"Performing text search for: '{query}'")
        embedding = embedder.embed_text(query)

    if embedding is None:
        logger.warning("Could not generate an embedding for the query.")
        return []

    try:
        results = indexer.search(
            query_vector=embedding.tolist(), top_k=top_k, table_name=table_name
        )
        # The result object from to_pydantic is a list of VectorObject models
        processed_results = [res["path"] for res in results]
        logger.info(f"Search found {len(processed_results)} results.")
        return processed_results
    except Exception as e:
        logger.error(f"An error occurred during search: {e}", exc_info=True)
        return []


def create_gradio_interface():
    """
    Creates and launches the Gradio web interface for the search application.
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Zeta DB Visual Search") as interface:
        gr.Markdown("<h1>Zeta DB: Visual & Text Search</h1>")

        with gr.Tab("Search"):
            with gr.Row():
                with gr.Column(scale=3):
                    results_gallery = gr.Gallery(
                        label="Search Results", show_label=True, elem_id="gallery"
                    )

                with gr.Column(scale=2):
                    text_query = gr.Textbox(
                        label="Text Query",
                        placeholder="e.g., 'a red car on a sunny day'",
                    )
                    image_query = gr.Image(type="filepath", label="Image Query")
                    table_name = gr.Dropdown(
                        choices=indexer.get_tables(),
                        label="Table Name",
                        value=indexer.get_tables()[0],
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=DEFAULT_SEARCH_K,
                        step=1,
                        label="Number of Results (Top-K)",
                    )
                    search_button = gr.Button("Search", variant="primary")

            # Handlers for search actions
            text_query.submit(
                fn=search_index,
                inputs=[text_query, top_k_slider, table_name],
                outputs=results_gallery,
            )
            search_button.click(
                fn=search_index,
                inputs=[text_query, top_k_slider, table_name],
                outputs=results_gallery,
            )
            image_query.upload(
                fn=search_index,
                inputs=[image_query, top_k_slider, table_name],
                outputs=results_gallery,
            )

        with gr.Tab("Index Management"):
            with gr.Row():
                table_name = gr.Textbox(label="Table Name")
                folder_input = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter the full path to your image folder",
                )
                index_button = gr.Button("Index Folder", variant="primary")
            index_status = gr.Label(label="Indexing Status")
            index_button.click(
                fn=index_folder, inputs=[table_name, folder_input], outputs=index_status
            )

    logger.info("Launching Gradio interface...")
    interface.launch()


if __name__ == "__main__":
    if embedder and indexer:
        create_gradio_interface()
    else:
        logger.critical(
            "Application cannot start because core components failed to initialize."
        )
