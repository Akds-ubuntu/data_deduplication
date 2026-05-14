from collections import defaultdict
import logging
import time
from typing import Dict, List, Any
import numpy as np
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from tqdm import tqdm
from dedup_tool.core.strategy import DedupStrategy
from dedup_tool.core.strategyregistry import StrategyRegistry
from dedup_tool.utils.union_find import UnionFind


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
        self.index = None

    @classmethod
    def from_config(cls, config):

        model_name = config.model_name
        return cls(model_name=model_name, threshold=config.threshold)

    def deduplicate(self, texts: List[str], batch_size: int = 32) -> Dict[str, Any]:
        if not texts:
            return {"clusters": {}, "metadata": {}}

        self.logger.info(f"Starting embedding generation for {len(texts)} texts...")

        all_embs = []

       
        for i in tqdm(
            range(0, len(texts), batch_size), desc="Generating Embeddings"
        ):
            batch = texts[i : i + batch_size]
            batch_embs = list(self.model.embed(batch))
            all_embs.extend(batch_embs)

        self.logger.info("Normalizing embeddings...")
        self.all_embeddings = np.array(all_embs, dtype=np.float32)

        faiss.normalize_L2(self.all_embeddings)

        self.logger.info(f"Building FAISS Index (Size: {self.all_embeddings.shape})...")

        dim = self.all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.all_embeddings)

        start_index = time.time()
        clusters = self._find_clusters(len(texts))

        self.logger.info(
            f"Indexing & Clustering finished in {time.time() - start_index:.2f}s"
        )

        return {
            "clusters": clusters,
            "metadata": {
                "num_texts": len(texts),
                "num_clusters": len(clusters),
                "model": self.model_name,
                "threshold": self.threshold,
                "engine": "FAISS/FastEmbed",
            },
        }

    def compute_similarity(self, idx1: int, idx2: int) -> float:
        if self.all_embeddings is None or self.all_embeddings.size == 0:
            raise ValueError(
                "Сначала нужно вызвать метод deduplicate(), чтобы создать эмбеддинги."
            )

        emb1 = self.all_embeddings[idx1].reshape(1, -1)
        emb2 = self.all_embeddings[idx2].reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0][0])

    def _find_clusters(self, num_docs: int) -> Dict[int, List[int]]:
        if self.index is None:
            raise ValueError("Индекс не построен. Сначала вызовите deduplicate().")

        radius = self.threshold
        limits, distances, indices = self.index.range_search(
            self.all_embeddings, radius
        )

        uf = UnionFind(num_docs)

        for i in range(num_docs):
            start, end = limits[i], limits[i + 1]
            neighbors = indices[start:end]

            for neighbor in neighbors:
                if i != neighbor:
                    uf.union(i, neighbor)

        clusters = defaultdict(list)
        for i in range(num_docs):
            root = uf.find(i)
            clusters[root].append(i)

        return dict(clusters)
