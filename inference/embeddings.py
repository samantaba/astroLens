"""
Embedding Store using FAISS for similarity search.

Stores 768-dim embeddings and enables fast nearest-neighbor search.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    FAISS-based embedding store for similarity search.
    
    Stores image embeddings (768-dim) and enables:
    - Add new embeddings
    - Search for similar embeddings
    - Save/load index to disk
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_path: str = str(FAISS_INDEX_PATH),
    ):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index = None
        self.id_map = {}  # Maps FAISS internal index to image_id
        self._next_idx = 0

        # Try to load existing index
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create new one."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                # Load id_map (stored alongside index)
                map_path = self.index_path.with_suffix(".map.npy")
                if map_path.exists():
                    self.id_map = dict(enumerate(np.load(map_path)))
                    self._next_idx = len(self.id_map)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self._create_index()
        else:
            self._create_index()

    def _create_index(self):
        """Create a new FAISS index."""
        import faiss
        
        # Use IVFFlat for scalability, or Flat for simplicity
        # For small datasets (<10K), Flat is fine
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after normalization)
        self.id_map = {}
        self._next_idx = 0
        logger.info("Created new FAISS index")

    def add(self, image_id: int, embedding: np.ndarray) -> int:
        """
        Add an embedding to the index.
        
        Args:
            image_id: Database image ID
            embedding: 768-dim numpy array
        
        Returns:
            Internal FAISS index
        """
        # Check if image_id already exists - skip if so (no duplicates)
        if image_id in self.id_map.values():
            # Already indexed, skip to avoid duplicates
            return -1
        
        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32).reshape(1, -1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Add to index
        self.index.add(embedding)
        
        # Map FAISS index to image_id
        self.id_map[self._next_idx] = image_id
        idx = self._next_idx
        self._next_idx += 1

        # Save periodically
        if self._next_idx % 100 == 0:
            self.save()

        return idx

    def search(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar embeddings.
        
        Args:
            embedding: Query embedding (768-dim)
            k: Number of results
        
        Returns:
            (image_ids, similarities)
        """
        if self.index.ntotal == 0:
            return [], []

        # Normalize query
        embedding = embedding.astype(np.float32).reshape(1, -1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Search
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(embedding, k)

        # Map back to image_ids
        image_ids = [self.id_map.get(int(idx), -1) for idx in indices[0]]
        sims = [float(s) for s in similarities[0]]

        return image_ids, sims

    def remove(self, image_id: int):
        """
        Remove an embedding by image_id.
        
        Note: FAISS doesn't support efficient removal. For now, we just
        mark as removed and rebuild index periodically.
        """
        # Find internal index
        for idx, img_id in self.id_map.items():
            if img_id == image_id:
                del self.id_map[idx]
                break
        # TODO: Rebuild index to actually remove

    def save(self):
        """Save index to disk."""
        import faiss
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        # Save id_map
        map_path = self.index_path.with_suffix(".map.npy")
        np.save(map_path, list(self.id_map.values()))
        
        logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")

    def count(self) -> int:
        """Get number of stored embeddings."""
        return self.index.ntotal if self.index else 0

