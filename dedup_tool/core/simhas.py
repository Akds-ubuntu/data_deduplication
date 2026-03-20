import logging
import re
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

from dedup_tool.core.strategy import DedupStrategy
from dedup_tool.core.strategyregistry import StrategyRegistry
from dedup_tool.utils.sha_hash import sha1_hash
from dedup_tool.utils.union_find import UnionFind

NON_ALPHA = re.compile(r"\W+")


@StrategyRegistry.register("simhash")
class SimHashDedup(DedupStrategy):
    """SimHash-based deduplication strategy."""

    def __init__(self, ngram_size: int = 3, threshold: int = 3):
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.all_signatures: List[np.uint64] = []

        self._shifts = np.arange(64, dtype=np.uint64)

    @classmethod
    def from_config(cls, config):
        return cls(ngram_size=config.ngram_size, threshold=config.threshold)

    def deduplicate(self, texts: List[str]) -> Dict[str, Any]:
        self.logger.debug(f"Starting SimHash deduplication for {len(texts)} texts")
        self.all_signatures = self._generate_signatures(texts)
        clusters = self._find_clusters(len(texts))
        self.logger.debug(f"Found {len(clusters)} clusters")
        return {
            "clusters": clusters,
            "signatures": self.all_signatures,
            "metadata": {
                "ngram_size": self.ngram_size,
                "threshold_bits": self.threshold,
                "num_texts": len(texts),
                "num_clusters": len(clusters),
                "algorithm": "SimHash",
            },
        }

    def _generate_signatures(self, texts: List[str]) -> List[np.uint64]:
        signatures = []
        for text in texts:
            if len(text) < self.ngram_size:
                tokens = {text} if text else set()
            else:
                tokens = {
                    text[i : i + self.ngram_size]
                    for i in range(len(text) - self.ngram_size + 1)
                }

            signatures.append(self._compute_simhash(tokens))

        return signatures

    def _compute_simhash(self, tokens: set) -> np.uint64:
        """Векторизованное вычисление SimHash."""
        if not tokens:
            return np.uint64(0)

        hashvalues = np.array(
            [sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64
        )

        if hashvalues.size == 0:
            return np.uint64(0)

        bits = (hashvalues[:, None] >> self._shifts) & np.uint64(1)

        weights = np.where(bits == 1, 1, -1)

        v = np.sum(weights, axis=0)

        final_bits = np.where(v > 0, np.uint64(1), np.uint64(0))

        ans = np.bitwise_or.reduce(final_bits << self._shifts)

        return ans

    def _find_clusters(self, num_docs: int) -> Dict[int, List[int]]:
        uf = UnionFind(num_docs)
        if self.threshold > 3:
            self.logger.warning(
                "Threshold > 3. Block indexing is optimized for threshold <= 3."
            )

        index = defaultdict(list)
        for doc_id in range(num_docs):
            sig = self.all_signatures[doc_id]

            blocks = [
                (sig >> np.uint64(0)) & np.uint64(0xFFFF),
                (sig >> np.uint64(16)) & np.uint64(0xFFFF),
                (sig >> np.uint64(32)) & np.uint64(0xFFFF),
                (sig >> np.uint64(48)) & np.uint64(0xFFFF),
            ]

            candidates = set()
            for i, block_val in enumerate(blocks):
                key = (i, block_val)
                for candidate_id in index[key]:
                    candidates.add(candidate_id)
                index[key].append(doc_id)

            for candidate_id in candidates:
                if uf.find(doc_id) != uf.find(candidate_id):
                    dist = self._hamming_distance(
                        sig, self.all_signatures[candidate_id]
                    )
                    if dist <= self.threshold:
                        uf.union(doc_id, candidate_id)

        clusters = defaultdict(list)
        for doc_id in range(num_docs):
            root = uf.find(doc_id)
            clusters[root].append(doc_id)

        return dict(clusters)

    def compute_similarity(self, idx1: int, idx2: int) -> float:
        if not self.all_signatures:
            raise ValueError("Must call deduplicate() first")
        h1 = self.all_signatures[idx1]
        h2 = self.all_signatures[idx2]
        dist = self._hamming_distance(h1, h2)
        similarity = 1.0 - (2.0 * dist / 64.0)
        return max(0.0, similarity)

    @staticmethod
    def _hamming_distance(h1: np.uint64, h2: np.uint64) -> int:
        return int(h1 ^ h2).bit_count()
