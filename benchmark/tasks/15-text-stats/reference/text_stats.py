"""文本统计。"""

import string

_STRIP_CHARS = string.punctuation + "，。！？；：“”‘’（）【】"


def word_frequencies(text):
    counts = {}
    for raw in text.split():
        word = raw.lower().strip(_STRIP_CHARS)
        if not word:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts
