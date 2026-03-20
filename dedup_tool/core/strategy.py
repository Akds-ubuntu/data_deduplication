from abc import ABC, abstractmethod
from typing import Dict, List, Any

from dedup_tool.config.settings import DedupConfig


class DedupStrategy(ABC):
    """Base class for deduplication strategies."""

    @abstractmethod
    def deduplicate(self, texts: List[str]) -> Dict[str, Any]:
        """Deduplicate a list of texts."""
        pass

    @abstractmethod
    def compute_similarity(self, idx1: int, idx2: int) -> float:
        """Compute similarity between two documents."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: DedupConfig) -> "DedupStrategy":
        pass
