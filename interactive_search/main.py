import sys
from interactive_search.search_engine import InteractiveSearchEngine
from interactive_search.utils import get_multiline_input

if __name__ == "__main__":
    dataset_path = "datasets/BigCloneBench_dataset.jsonl"  

    try:
        engine = InteractiveSearchEngine(dataset_path, limit=2000)

        while True:
            query = get_multiline_input()

            if query == "EXIT":
                print("Выход из программы. До свидания!")
                break

            if not query.strip():
                print(" Запрос пустой, попробуйте еще раз.\n")
                continue

            results = engine.search(query, top_k=5)
            engine.print_results(results)

    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
