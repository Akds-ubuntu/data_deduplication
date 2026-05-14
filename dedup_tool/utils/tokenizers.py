import re

NON_ALPHA = re.compile(r"\W+")
MULTISPACE = re.compile(r"\s+")


MULTISPACE = re.compile(r"\s+")
NON_ALPHANUM_SPACE = re.compile(r"[^a-zа-яё0-9 ]+")


def text_char_shingles(content: str, ngram_size: int) -> set:
    text = content.lower().strip()
    text = text.replace("'", " ").replace("-", " ")
    text = MULTISPACE.sub(" ", text)
    text = NON_ALPHANUM_SPACE.sub("", text)
    if not text:
        return set()
    if len(text) < ngram_size:
        return {text}
    return {text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)}


def text_shingles(content: str, ngram_size: int):
    content = content.lower()

    words = NON_ALPHA.split(content)
    words = [w for w in words if w]

    if len(words) < ngram_size:
        return {" ".join(words)}

    return {
        " ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
    }


def code_shingles(content: str, ngram_size: int):
    content = content.strip()

    content = MULTISPACE.sub(" ", content)

    if len(content) < ngram_size:
        return {content}

    return {content[i : i + ngram_size] for i in range(len(content) - ngram_size + 1)}
