import numpy as np
import xxhash


def sha1_hash(data: bytes) -> np.uint64:
    # hash_bytes = hashlib.sha1(data).digest()[:8]
    # return np.uint64(int.from_bytes(hash_bytes, byteorder="little"))
    return np.uint64(xxhash.xxh64_intdigest(data))
