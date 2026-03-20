import logging
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

from dedup_tool.utils.fastembed_verifier import FastEmbedVerifier
from dedup_tool.index.lsh import LSHIndex

from dedup_tool.core.strategyregistry import StrategyRegistry
from dedup_tool.utils import UnionFind, embed_func
from dedup_tool.utils.embeded import MERSENNE_PRIME

from .strategy import DedupStrategy


@StrategyRegistry.register("minhash")
class MinHashDedup(DedupStrategy):
    """MinHash-based deduplication using LSH."""

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        bands: int = 16,
        seed: int = 42,
        threshold: float = 0.6,
        threshold_semantic: float = 0.8,
        verify_ast: bool = False,
        use_semantic: bool = False,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        """Initialize MinHash deduplicator.

        Args:
            num_perm: Number of permutations
            ngram_size: Size of n-grams
            bands: Number of LSH bands
            seed: Random seed
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.bands = bands
        self.seed = seed
        self.threshold = threshold
        self.verify_ast = verify_ast
        self.use_semantic = use_semantic
        self.logger = logging.getLogger(__name__)

        self.rows_per_band = num_perm // bands
        self.permutations = None
        self.hashranges = None
        self.all_signatures = None
        self._prepare_permutations()
        self.verifier = None
        if self.use_semantic:
            self.logger.info(
                f"Loading semantic verifier: {model_name}, threshold: {threshold_semantic}"
            )
            self.verifier = FastEmbedVerifier(
                model_name=model_name, threshold=threshold_semantic
            )

        self._prepare_permutations()

    @classmethod
    def from_config(cls, config):
        return cls(
            num_perm=config.num_perm,
            ngram_size=config.ngram_size,
            bands=config.bands,
            threshold=config.threshold,
            verify_ast=config.verify_ast,
            use_semantic=getattr(config, "use_semantic_lsh", False),
            model_name=getattr(config, "model_name", "BAAI/bge-small-en-v1.5"),
            threshold_semantic=getattr(config, "threshold_semantic_lsh", 0.8),
        )

    def _prepare_permutations(self):
        """Prepare permutation parameters for MinHash."""
        self.logger.info(
            f"Params: num_perm={self.num_perm}, bands={self.bands}, threshold={self.threshold}, ngram_size={self.ngram_size}"
        )
        rng = np.random.default_rng(self.seed)
        self.permutations = np.array(
            [
                rng.integers(
                    1, MERSENNE_PRIME, size=self.num_perm, dtype=np.uint64
                ),  # a
                rng.integers(
                    0, MERSENNE_PRIME, size=self.num_perm, dtype=np.uint64
                ),  # b
            ]
        )
        rows_per_band = self.num_perm // self.bands
        rem = self.num_perm % self.bands
        self.hashranges = []
        start = 0
        for i in range(self.bands):
            end = start + rows_per_band + (1 if i < rem else 0)
            self.hashranges.append((start, end))
            start = end
        self.logger.debug(
            f"Prepared {self.bands} bands, {self.rows_per_band} rows each"
        )

    def deduplicate(self, texts: List[str]) -> Dict[str, Any]:
        """Deduplicate texts using MinHash + LSH.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with clusters, signatures, and metadata
        """
        self.logger.debug(f"Starting deduplication for {len(texts)} texts")

        self.all_signatures = self._generate_signatures(texts)

        lsh_index = self._build_hash_tables()

        clusters = self._find_clusters(
            texts,
            len(texts),
            lsh_index.hash_tables,
            self.threshold,
            verifier=self.verifier,
        )

        self.logger.debug(f"Found {len(clusters)} clusters")

        return {
            "clusters": clusters,
            "signatures": self.all_signatures,
            "metadata": {
                "num_perm": self.num_perm,
                "ngram_size": self.ngram_size,
                "bands": self.bands,
                "num_texts": len(texts),
                "num_clusters": len(clusters),
            },
        }

    def _generate_signatures(self, texts: List[str]) -> List[Dict]:
        """Generate MinHash signatures for all texts."""
        all_signatures = []
        # with tqdm(
        #     total=len(texts), desc="Generating MinHash signatures", disable=False
        # ) as pbar:
        for i, text in enumerate(texts):
            result = embed_func(
                content=text,
                idx=i,
                num_perm=self.num_perm,
                ngram_size=self.ngram_size,
                hashranges=self.hashranges,
                permutations=self.permutations,
            )
            all_signatures.append(result)

        self.logger.debug("MinHash signature generation completed")
        return all_signatures

    def _build_hash_tables(self) -> List[Dict]:
        """Build LSH hash tables from signatures."""
        num_bands = len(self.all_signatures[0]["__signatures__"])
        hash_tables = [defaultdict(list) for _ in range(num_bands)]

        for doc in self.all_signatures:
            doc_id = doc["__id__"]
            for band_idx, band_bytes in enumerate(doc["__signatures__"]):
                hash_tables[band_idx][band_bytes].append(doc_id)

        self.logger.debug(f"Built {len(hash_tables)} hash tables")
        # return hash_tables
        lsh_index = LSHIndex(num_bands=num_bands)
        for doc in self.all_signatures:
            lsh_index.add_signature(
                doc_id=doc["__id__"], signature=doc["__signatures__"]
            )
        stats = lsh_index.get_bucket_stats()
        self.logger.debug(
            f"LSH Index built: {stats['doc_count']} docs in {stats['num_bands']} bands. "
            f"Avg bucket size: {stats['avg_bucket_size']:.2f}, "
            f"Max bucket size: {stats['max_bucket_size']}"
        )

        return lsh_index

    def _find_clusters(
        self,
        texts: List[str],
        num_docs: int,
        hash_tables: List[Dict],
        threshold: float,
        verifier=None,
    ) -> Dict[int, List[int]]:
        """Find clusters using Union-Find on hash table buckets."""
        uf = UnionFind(num_docs)
        checked_pairs = set()

        for table in hash_tables:
            for bucket in table.values():
                if len(bucket) <= 1:
                    continue
                first = bucket[0]
                for doc_id in bucket[1:]:
                    pair = tuple(sorted((first, doc_id)))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    if self.compute_similarity(first, doc_id) >= threshold:
                        is_duplicate = True
                        if verifier is not None:
                            is_duplicate = verifier.verify(texts[first], texts[doc_id])
                        if is_duplicate:
                            uf.union(first, doc_id)

        clusters = defaultdict(list)
        for doc_id in range(num_docs):
            root = uf.find(doc_id)
            clusters[root].append(doc_id)

        return dict(clusters)

    def compute_similarity(self, idx1: int, idx2: int) -> float:
        """Compute Jaccard similarity between two documents.

        This is an estimate based on matching hash bands.
        """
        sig1 = self.all_signatures[idx1]["__raw_hashes__"]
        sig2 = self.all_signatures[idx2]["__raw_hashes__"]

        return np.mean(sig1 == sig2)
