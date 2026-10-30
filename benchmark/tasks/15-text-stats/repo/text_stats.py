"""文本统计。"""


def word_frequencies(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts
