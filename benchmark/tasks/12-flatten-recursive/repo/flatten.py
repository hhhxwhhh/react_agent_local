"""展平嵌套序列。"""


def flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
