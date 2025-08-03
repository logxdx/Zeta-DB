"""
This class provides the core Vector BD functionality. It handles loading/saving, initializing the correct index type based on the configuration, adding vectors, and searching.
"""

from datetime import timedelta
from pathlib import Path
from typing import Optional, Union, Literal, List, Dict

import lancedb
from pydantic import BaseModel
import pyarrow as pa

from .config import INDEX_FILE, DEFAULT_SEARCH_K, EMBEDDING_DIMENSION


# class VectorObject(BaseModel):
class VectorObject(BaseModel):
    """
    Schema for vectors stored in LanceDB.
    """

    id: str
    vector: List
    type: Literal["image", "text"] = "text"
    path: Optional[Path] = None


class LanceDBIndex:
    """
    LanceDBIndex wraps lancedb for storing and querying vector embeddings
    along with metadata for images and text documents.
    """

    def __init__(
        self,
        index_path: Union[str, Path] = INDEX_FILE,
        table_name: str = "embeddings",
    ):
        self.db = lancedb.connect(str(index_path), read_consistency_interval=timedelta(seconds=1))
        self.table_name = table_name
        self.table_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
                pa.field("type", pa.string()),
                pa.field("path", pa.string()),
            ]
        )

    def insert_documents(self, docs: List[Dict], table_name: str = "embeddings") -> None:
        """
        Insert a list of VectorObject documents into the index.
        """
        for name in self.db.table_names():
            if name == table_name:
                self.table = self.db.open_table(name)
                break
        else:
            self.table = self.db.create_table(table_name, exist_ok=True, schema=self.table_schema)
        self.table.add(docs)

    def get_tables(self) -> List[str]:
        return self.db.table_names()
    
    def drop_table(self, table_name: str) -> None:
        try:
            self.db.drop_table(table_name)
        except Exception as e:
            print(f"Error dropping table {table_name}: {e}")

    def search(
        self,
        query_vector: List[float],
        table_name: str = "embeddings",
        top_k: int = DEFAULT_SEARCH_K,
    ) -> List[Dict]:
        """
        Search the index for nearest neighbors to the query vector.

        Args:
            query_vector: query embedding
            top_k: number of nearest neighbors to return
        Returns:
            list of VectorObject results with distances
        """
        for name in self.db.table_names():
            if name == table_name:
                self.table = self.db.open_table(name)
                break
        else:
            print(f"Table {table_name} not found in database.")
        results = (
            self.table.search(query=query_vector).limit(top_k).to_list()
        )
        return results


# Example Usage:
# index = LanceDBIndex("./my_lancedb", table_name="docs")
# index.create_index(vector_dim=512, metric="IP")
# docs = [VectorObject(id="1", vector=[...], document=Document(id="1", type="text", content="Hello", timestamp=datetime.utcnow()))]
# index.insert_documents(docs)
# results = index.search(query_vector=[...], top_k=5)
# index.delete_by_id("1")
# index.delete_index()
