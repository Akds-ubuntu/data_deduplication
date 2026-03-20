import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from dedup_tool.utils import sha1_hash

MAX_HASH = np.uint64(2**64 - 1)
MERSENNE_PRIME = np.uint64(2**61 - 1)
NON_ALPHA = re.compile(r"\W+")


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    logging.debug(f"Маска получилась {masks}")
    # tokens: Set[str] = {
    #     " ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)
    # }
    tokens = {content[i : i + ngram_size] for i in range(len(content) - ngram_size + 1)}
    logging.debug(f"Документ {idx}: {len(tokens)} нграммы: {tokens}")

    hashvalues: np.ndarray = np.array(
        [sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    logging.debug(f"Документ {idx}: Хэш-значения нграмм: {hashvalues}")
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME,
        MAX_HASH,
    )
    logging.debug(f"Документ {idx}: Перемешанные хэш-значения: {permuted_hashvalues}")
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)

    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    logging.debug(f"Документ {idx}: Подписи для LSH: {Hs}")
    return {"__signatures__": Hs, "__id__": idx, "__raw_hashes__": hashvalues}
