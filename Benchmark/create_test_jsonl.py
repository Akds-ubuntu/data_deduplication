import json
from typing import Dict
import pandas as pd


def codex_test():
    classter_code: Dict[int, str] = {}
    with open("BigCloneBench dataset.jsonl", "r") as infile:
        for line in infile:
            data = json.loads(line)
            classter_code[data["idx"]] = data["func"]

    with (
        open("CodeXGLUE Test.txt", "r") as infile,
        open("CodeXGLUE Test.jsonl", "w") as outfile,
    ):
        for line in infile:
            id1, id2, label = line.split()
            if id1 in classter_code and id2 in classter_code:
                pair = {
                    "code1": classter_code[id1],
                    "code2": classter_code[id2],
                    "label": int(label),
                }
                outfile.write(json.dumps(pair) + "\n")


def quora_pairs():
    df = pd.read_csv("/Users/alexrgz/Desktop/questions.csv")

    df = df[["question1", "question2", "is_duplicate"]].copy()
    df.columns = ["code1", "code2", "label"]

    df = df.dropna()

    df.to_json("quora_pairs.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"Готово! Файл сохранен как quora_pairs.jsonl. Обработано {len(df)} пар.")


if __name__ == "__main__":
    quora_pairs()
