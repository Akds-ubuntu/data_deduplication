from collections import defaultdict
import logging
from typing import Dict, List, Any
import numpy as np
from fastembed import TextEmbedding
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from dedup_tool.core.strategy import DedupStrategy
from dedup_tool.core.strategyregistry import StrategyRegistry


@StrategyRegistry.register("semantic")
class SemanticDedup(DedupStrategy):
    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.85
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

        self.model = TextEmbedding(model_name=self.model_name)
        self.all_embeddings: np.ndarray = np.array([])

    @classmethod
    def from_config(cls, config):

        model_name = config.model_name
        return cls(model_name=model_name, threshold=config.threshold)

    def deduplicate(self, texts: List[str]) -> Dict[str, Any]:
        if not texts:
            return {"clusters": {}, "metadata": {}}

        self.logger.info(
            f"Generating embeddings for {len(texts)} texts using FastEmbed..."
        )

        self.all_embeddings = np.array(list(self.model.embed(texts)))

        clusters = self._find_clusters(len(texts))

        return {
            "clusters": clusters,
            "metadata": {
                "num_texts": len(texts),
                "num_clusters": len(clusters),
                "model": self.model_name,
                "threshold": self.threshold,
                "engine": "ONNX/FastEmbed",
                "ngram_size": "N/A (Semantic)",
            },
        }

    def compute_similarity(self, idx1: int, idx2: int) -> float:
        """
        Считает косинусное сходство между двумя документами по их индексам.
        """
        if self.all_embeddings is None or self.all_embeddings.size == 0:
            raise ValueError(
                "Сначала нужно вызвать метод deduplicate(), чтобы создать эмбеддинги."
            )

        emb1 = self.all_embeddings[idx1].reshape(1, -1)
        emb2 = self.all_embeddings[idx2].reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0][0])

    def _find_clusters(self, num_docs: int) -> Dict[int, List[int]]:
        eps = 1 - self.threshold
        clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine")
        labels = clustering.fit_predict(self.all_embeddings)

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                clusters[label].append(idx)
        return dict(clusters)
