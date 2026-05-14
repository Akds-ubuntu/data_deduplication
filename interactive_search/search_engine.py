import time
import numpy as np
import faiss
from typing import Dict, Any

from dedup_tool.core.minhash import MinHashDedup
from dedup_tool.core.semantic import SemanticDedup
from interactive_search.data_loader import load_data
from interactive_search.utils import get_multiline_input


class InteractiveSearchEngine:
    def __init__(self, dataset_path: str, limit: int = 1000):
        self.texts = load_data(dataset_path, limit)
        if not self.texts:
            raise ValueError("Данные не загружены! Проверьте путь.")

        print(
            f"[Система] Загружено {len(self.texts)} документов. Начинаем индексацию..."
        )

        start = time.time()
        self.mh_dedup = MinHashDedup(num_perm=128, threshold=0.64, bands=16)
        self.mh_dedup.deduplicate(self.texts)
        self.lsh_index = self.mh_dedup._build_hash_tables()
        self.mh_index_time = time.time() - start
        print(f"[MinHash] Индекс построен за {self.mh_index_time:.4f} сек.")

        start = time.time()
        self.sem_dedup = SemanticDedup(threshold=0.6)
        self.sem_dedup.deduplicate(self.texts)
        self.sem_index_time = time.time() - start
        print(f"[FAISS] Индекс построен за {self.sem_index_time:.4f} сек.")
        print("-" * 60)

    def search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        start_mh = time.time()
        query_sig_data = self.mh_dedup._generate_signatures([query_text])[0]
        query_sig = query_sig_data["__signatures__"]
        query_raw = query_sig_data["__raw_hashes__"]

        candidates = set()
        for i, band_hash in enumerate(query_sig):
            candidates.update(self.lsh_index.hash_tables[i].get(band_hash, []))

        mh_results = []
        for cand_idx in candidates:
            cand_raw = self.mh_dedup.all_signatures[cand_idx]["__raw_hashes__"]
            jaccard = np.mean(query_raw == cand_raw)
            mh_results.append((cand_idx, jaccard))

        mh_results.sort(key=lambda x: x[1], reverse=True)
        mh_top = mh_results[:top_k]
        mh_time = time.time() - start_mh

        start_sem = time.time()
        query_vector = np.array(
            list(self.sem_dedup.model.embed([query_text])), dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        D, I = self.sem_dedup.index.search(query_vector, k=top_k)
        sem_time = time.time() - start_sem

        sem_top = [(I[0][i], D[0][i]) for i in range(len(I[0])) if I[0][i] != -1]

        return {
            "mh_time": mh_time,
            "mh_candidates_count": len(candidates),
            "mh_top": mh_top,
            "sem_time": sem_time,
            "sem_top": sem_top,
        }

    def print_results(self, results: dict):
        def truncate(text, max_len=80):
            clean_text = text.replace("\n", " ").replace("\r", "").strip()
            return (
                clean_text
                if len(clean_text) <= max_len
                else clean_text[:max_len] + "..."
            )

        print("\n" + "=" * 80)
        print(f"{'СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПОИСКА':^80}")
        print("=" * 80)

        print(
            f"[MinHash LSH] Время поиска: {results['mh_time']:.5f} сек | Кандидатов в бакетах: {results['mh_candidates_count']}"
        )
        if not results["mh_top"]:
            print("   -> Совпадений не найдено (ни один бакет не совпал).")
        else:
            for rank, (idx, score) in enumerate(results["mh_top"], 1):
                preview = truncate(self.texts[idx])
                print(f"   {rank}. [ID: {idx:>4}] [Jaccard: {score:.3f}] {preview}")

        print("-" * 80)

        print(
            f"[FAISS Semantic] Время поиска: {results['sem_time']:.5f} сек | Векторный поиск"
        )
        if not results["sem_top"]:
            print("   -> Совпадений не найдено.")
        else:
            for rank, (idx, distance) in enumerate(results["sem_top"], 1):
                preview = truncate(self.texts[idx])
                print(f"   {rank}. [ID: {idx:>4}] [Distance: {distance:.3f}] {preview}")
        print("=" * 80 + "\n")
