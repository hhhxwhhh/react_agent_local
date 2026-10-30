"""展平嵌套序列。"""


def flatten(items):
    result = []
    for item in items:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
