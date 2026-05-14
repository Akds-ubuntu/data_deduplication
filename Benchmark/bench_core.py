import time
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
from dedup_tool.cli import get_strategy
from dedup_tool.config.settings import DedupConfig
from dedup_tool.utils.union_find import UnionFind


def classify_prediction(gt_dups: set, pred_dups: set) -> str:
    if not gt_dups:
        return "TN" if not pred_dups else "FP"
    else:
        if not pred_dups:
            return "FN"
        return "TP" if gt_dups == pred_dups else "FP"


def evaluate_metrics(true_dup_sets, pred_dup_sets):
    n = len(true_dup_sets)
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    exact_match = 0
    for i in range(n):
        gt = true_dup_sets[i]
        pred = pred_dup_sets[i]
        cls = classify_prediction(gt, pred)
        counts[cls] += 1
        if gt == pred:
            exact_match += 1

    TP, TN, FP, FN = counts["TP"], counts["TN"], counts["FP"], counts["FN"]

    precision_dup = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_dup    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    non_dup_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    non_dup_recall    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    macro_f1 = (precision_dup + non_dup_precision) / 2
    accuracy = exact_match / n

    return {
        "precision_dup": precision_dup,
        "recall_dup": recall_dup,
        "non_dup_precision": non_dup_precision,
        "non_dup_recall": non_dup_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "counts": counts,
        "n": n,
    }


def run_benchmark(
    algorithm: str,
    config_kwargs: dict,
    sample_size: int = 20000
):

    print(f"--- {algorithm.upper()} Benchmark on pinecone/core-2020-05-10-deduplication ({sample_size} docs) ---")

    ds = load_dataset(
        "pinecone/core-2020-05-10-deduplication", split="train", streaming=True
    )
    texts = []
    core_ids = []
    labelled_duplicates = []

    for i, item in enumerate(ds):
        if i >= sample_size:
            break
        abstract = item.get("original_abstract", "")
        title = item.get("original_title", "")
        text = f"{title}. {abstract}" if title else abstract
        texts.append(text)
        core_ids.append(item["core_id"])
        labelled_duplicates.append(item.get("labelled_duplicates", []))

    n = len(texts)
    print(f"Loaded {n} documents")

    core_id_to_idx = {cid: idx for idx, cid in enumerate(core_ids)}
    uf_true = UnionFind(n)
    for i, dup_ids in enumerate(labelled_duplicates):
        for dup_id in dup_ids:
            if dup_id in core_id_to_idx:
                uf_true.union(i, core_id_to_idx[dup_id])

    true_clusters_idx = defaultdict(list)
    for i in range(n):
        root = uf_true.find(i)
        true_clusters_idx[root].append(i)

    true_dup_sets = {}
    for indices in true_clusters_idx.values():
        cluster_core_ids = {core_ids[i] for i in indices}
        for i in indices:
            true_dup_sets[i] = cluster_core_ids - {core_ids[i]}

    config = DedupConfig(algorithm=algorithm, **config_kwargs)
    strategy = get_strategy(config)

    start = time.time()
    result = strategy.deduplicate(texts)
    elapsed = time.time() - start

    pred_clusters = result["clusters"]

    pred_dup_sets = {i: set() for i in range(n)}
    for indices in pred_clusters.values():
        cluster_core_ids = {core_ids[i] for i in indices}
        for i in indices:
            pred_dup_sets[i] = cluster_core_ids - {core_ids[i]}

    metrics = evaluate_metrics(true_dup_sets, pred_dup_sets)

    print(f"Done in {elapsed:.2f}s ({n / elapsed:.2f} docs/sec)")
    print("\n--- Document-level Metrics (text-dedup style) ---")
    print(f"Precision (Duplicates):     {metrics['precision_dup']:.4f}")
    print(f"Recall (Duplicates):        {metrics['recall_dup']:.4f}")
    print(f"Precision (Non Duplicates): {metrics['non_dup_precision']:.4f}")
    print(f"Recall (Non Duplicates):    {metrics['non_dup_recall']:.4f}")
    print(f"Macro F1 score:             {metrics['macro_f1']:.4f}")
    print(f"Accuracy:                   {metrics['accuracy']:.4f}")
    c = metrics['counts']
    print(f"Class distribution: TP={c['TP']}, TN={c['TN']}, FP={c['FP']}, FN={c['FN']}")
    print()   
if __name__ == "__main__":
    run_benchmark(
        algorithm="minhash",
        config_kwargs={
            "num_perm": 512,
            "ngram_size": 2,
            "bands": 128,
            "threshold": 0.6,
            "use_semantic_lsh": False,
            "mode": "text",
        },
        sample_size=100000  
    )

    # SimHash
    # run_benchmark(
    #     algorithm="simhash",
    #     config_kwargs={
    #         "ngram_size": 3,
    #         "threshold": 12,
    #         "num_blocks": 13,
    #         "jaccard_threshold": 0.6,
    #     },
    #     sample_size=100000
    # )

    # # Semantic
    # run_benchmark(
    #     algorithm="semantic",
    #     config_kwargs={
    #         "model_name": "BAAI/bge-small-en-v1.5",
    #         "threshold": 0.85,
    #     },
    #     sample_size=100000
    # )