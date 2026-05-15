import orjson

from dedup_tool.utils.embeded import embed_func


def read_jsonl_in_batches(file_path, text_field, batch_size=10000):
    batch = []

    with open(file_path, "rb") as f:
        for line_idx, line in enumerate(f):
            try:
                obj = orjson.loads(line)

                if text_field not in obj:
                    continue

                batch.append(
                    (
                        line_idx,
                        obj,
                        str(obj[text_field]),
                    )
                )

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except Exception:
                continue

    if batch:
        yield batch


def process_document(args):
    idx, text, num_perm, ngram_size, hashranges, permutations, mode = args

    return embed_func(
        content=text,
        idx=idx,
        num_perm=num_perm,
        ngram_size=ngram_size,
        hashranges=hashranges,
        permutations=permutations,
        mode=mode,
    )
