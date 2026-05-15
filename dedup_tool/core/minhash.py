import itertools
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm 

from dedup_tool.utils.batch_reader import process_document
from dedup_tool.utils.fastembed_verifier import FastEmbedVerifier
from dedup_tool.index.lsh import LSHIndex

from dedup_tool.core.strategyregistry import StrategyRegistry
from dedup_tool.utils import UnionFind, embed_func
from dedup_tool.utils.embeded import MERSENNE_PRIME

from .strategy import DedupStrategy


@StrategyRegistry.register("minhash")
class MinHashDedup(DedupStrategy):

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
        mode: str = "text",
    ):

        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.bands = bands
        self.seed = seed
        self.threshold = threshold
        self.verify_ast = verify_ast
        self.use_semantic = use_semantic
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False

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
            mode=getattr(config, "mode", "text"),
        )

    def _prepare_permutations(self):
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
        self.logger.debug(f"Starting deduplication for {len(texts)} texts")

        self.all_signatures = self._generate_signatures(texts)

        self.hash_matrix = np.array(
            [doc["__raw_hashes__"] for doc in self.all_signatures],
            dtype=np.uint64,
        )

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
        tasks = [
            (
                i,
                text,
                self.num_perm,
                self.ngram_size,
                self.hashranges,
                self.permutations,
                self.mode,
            )
            for i, text in enumerate(texts)
        ]

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(process_document, tasks, chunksize=256),
                total=len(tasks),
                desc="Generating MinHash signatures",
                unit="docs"
            ))

        return results


    def _build_hash_tables(self) -> List[Dict]:
        num_bands = len(self.all_signatures[0]["__signatures__"])
        hash_tables = [defaultdict(list) for _ in range(num_bands)]

        for doc in self.all_signatures:
            doc_id = doc["__id__"]
            for band_idx, band_bytes in enumerate(doc["__signatures__"]):
                hash_tables[band_idx][band_bytes].append(doc_id)

        self.logger.debug(f"Built {len(hash_tables)} hash tables")
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
        texts,
        num_docs,
        hash_tables,
        threshold,
        verifier=None,
    ):
        uf = UnionFind(num_docs)
        pairs = []

        for table in hash_tables:
            for bucket in table.values():
                if len(bucket) < 2:
                    continue

                for pair in itertools.combinations(bucket, 2):
                    pairs.append(pair)

        if not pairs:
            return {}

        pairs = np.unique(np.array(pairs, dtype=np.int32), axis=0)

        left = self.hash_matrix[pairs[:, 0]]
        right = self.hash_matrix[pairs[:, 1]]

        sims = np.count_nonzero(left == right, axis=1) / self.num_perm

        valid_pairs = pairs[sims >= threshold]

        for i, j in tqdm(valid_pairs, desc="Clustering", leave=False):
            if verifier is None or verifier.verify(texts[i], texts[j]):
                uf.union(i, j)

        clusters = defaultdict(list)
        for i in range(num_docs):
            clusters[uf.find(i)].append(i)

        return dict(clusters)

    def compute_similarity(self, idx1: int, idx2: int) -> float:
        sig1 = self.all_signatures[idx1]["__raw_hashes__"]
        sig2 = self.all_signatures[idx2]["__raw_hashes__"]

        return np.mean(sig1 == sig2)
