import json
from typing import List


def load_data(path: str, limit: int = 1000) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                data = json.loads(line)
                content = data.get("function", data.get("func", ""))
                if content:
                    texts.append(content)
            except Exception:
                continue
    return texts
