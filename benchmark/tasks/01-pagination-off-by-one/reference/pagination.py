"""分页工具。"""


def paginate(items, page, per_page):
    """返回第 page 页（从 1 开始）的数据。"""
    if page < 1:
        raise ValueError("page 必须 >= 1")
    if per_page < 1:
        raise ValueError("per_page 必须 >= 1")
    start = (page - 1) * per_page
    return list(items[start:start + per_page])
