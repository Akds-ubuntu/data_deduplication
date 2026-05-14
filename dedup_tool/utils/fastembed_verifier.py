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

    def verify(self, text1: str, text2: str) -> bool:
        try:
            embeddings = list(self.model.embed([text1, text2]))
            emb1, emb2 = embeddings[0], embeddings[1]

            cosine_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            return cosine_sim >= self.threshold
        except Exception as e:
            self.logger.error(f"Error in semantic verification: {e}")
            return False
