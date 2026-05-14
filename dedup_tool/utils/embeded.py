import logging
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from dedup_tool.utils import sha1_hash
from dedup_tool.utils.tokenizers import code_shingles, text_shingles

MAX_HASH = np.uint64(2**64 - 1)
MERSENNE_PRIME = np.uint64(2**61 - 1)


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    mode: str = "text",
) -> Dict[str, Any]:
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)

    if mode == "text":
        tokens = text_shingles(content, ngram_size)
    elif mode == "code":
        tokens = code_shingles(content, ngram_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hashvalues: np.ndarray = np.array(
        [sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)

    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx, "__raw_hashes__": hashvalues}
