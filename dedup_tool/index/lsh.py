from typing import Dict, List, Set
from collections import defaultdict


class LSHIndex:
    """Locality Sensitive Hashing index for fast similarity queries."""

    def __init__(self, num_bands: int):
        """Initialize LSH index.

        Args:
            num_bands: Number of bands in the LSH structure
        """
        self.num_bands = num_bands
        self.hash_tables: List[Dict] = [defaultdict(list) for _ in range(num_bands)]
        self.doc_count = 0

    def add_signature(self, doc_id: int, signature: List[bytes]):
        """Add a document signature to the index.

        Args:
            doc_id: Document identifier
            signature: List of hash values (one per band)
        """
        for band_idx, band_hash in enumerate(signature):
            self.hash_tables[band_idx][band_hash].append(doc_id)
        self.doc_count += 1

    def query(self, signature: List[bytes]) -> Set[int]:
        """Find all documents with matching hash bands.

        Args:
            signature: Query signature

        Returns:
            Set of document IDs that match at least one band
        """
        candidates = set()
        for band_idx, band_hash in enumerate(signature):
            if band_hash in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_hash])
        return candidates

    def get_bucket_stats(self) -> Dict:
        """Get statistics about hash table distribution."""
        stats = {
            "num_bands": self.num_bands,
            "doc_count": self.doc_count,
            "buckets_per_band": [],
            "avg_bucket_size": 0,
            "max_bucket_size": 0,
        }

        total_buckets = 0
        total_docs = 0
        max_size = 0

        for table in self.hash_tables:
            num_buckets = len(table)
            stats["buckets_per_band"].append(num_buckets)
            total_buckets += num_buckets

            for bucket in table.values():
                bucket_size = len(bucket)
                total_docs += bucket_size
                max_size = max(max_size, bucket_size)

        if total_buckets > 0:
            stats["avg_bucket_size"] = total_docs / total_buckets
        stats["max_bucket_size"] = max_size

        return stats
