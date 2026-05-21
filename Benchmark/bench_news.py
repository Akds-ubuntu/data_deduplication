import time
import numpy as np
from datasets import load_dataset
from sklearn.metrics import adjusted_rand_score, v_measure_score
from collections import defaultdict

from dedup_tool.cli import get_strategy
from dedup_tool.config.settings import DedupConfig


def run_news_benchmark(algorithm: str, config_kwargs: dict, sample_size: int = 1000):
    print(f"--- {algorithm.upper()} Benchmark {sample_size} docs ---")

    ds = load_dataset("chenghao/NEWS-COPY-eval", split="test", streaming=True)
    texts, true_labels = [], []
    for i, item in enumerate(ds):
        if i >= sample_size:
            break
        texts.append(item["article"])
        true_labels.append(item["cluster"])
    n = len(texts)
    print(f"Loaded {n} docs")

    config = DedupConfig(algorithm=algorithm, **config_kwargs)
    strategy = get_strategy(config)
   
    start = time.time()
    result = strategy.deduplicate(texts)
    elapsed = time.time() - start

    clusters = result["clusters"]

    predicted_labels = [-1] * n  
    for root_id, members in clusters.items():
        for idx in members:
            predicted_labels[idx] = root_id

    ari = adjusted_rand_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    print(f"Done in {elapsed:.2f}s")
    print("\n--- Results ---")
    print(f"ARI:        {ari:.4f}")
    print(f"V-measure:  {v_measure:.4f}")
    print(f"Speed:      {n / elapsed:.2f} docs/sec")

    cluster_sizes = [len(v) for v in clusters.values()]
    if cluster_sizes:
        print(f"Largest cluster: {max(cluster_sizes)}")
        print(f"Singletons:      {cluster_sizes.count(1)}")
    print()  

    return ari, v_measure, elapsed

if __name__ == "__main__":
    run_news_benchmark(
        algorithm="minhash",
        config_kwargs={
            "num_perm": 512,
            "ngram_size": 2,
            "bands": 128,
            "threshold": 0.34,
            "use_semantic_lsh": False,
            "mode": "text",
        },
        sample_size=14211
    )

    # run_news_benchmark(
    #     algorithm="simhash",
    #     config_kwargs={
    #         "ngram_size": 3,
    #         "threshold": 12,
    #         "num_blocks": 13,
    #         "jaccard_threshold": 0.65,
    #     },
    #     sample_size=14211
    # )

    # run_news_benchmark(
    #     algorithm="semantic",
    #     config_kwargs={
    #         "model_name": "BAAI/bge-small-en-v1.5",
    #         "threshold": 0.85,
    #     },
    #     sample_size=14211
    # )