"""分页工具。"""


def paginate(items, page, per_page):
    """返回第 page 页（从 1 开始）的数据。"""
    start = (page - 1) * per_page
    end = start + per_page - 1
    return items[start:end]
