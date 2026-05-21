from typing import List

import numpy as np
from fastembed import TextEmbedding
import logging


class FastEmbedVerifier:
    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.85
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Semantic model: {model_name}")
        self.model = TextEmbedding(model_name)
        self.threshold = threshold

    def embed_unique_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        try:
            embeddings = list(self.model.embed(texts, batch_size=batch_size))
            embeddings = np.array(embeddings, dtype=np.float32)
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12  
            return embeddings / norms
        except Exception as e:
            self.logger.error(f"Error in batch embedding generation: {e}")
            raise e
