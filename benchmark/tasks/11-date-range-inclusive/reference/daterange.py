"""日期范围生成器。"""

from datetime import timedelta


def date_range(start, end):
    if start > end:
        raise ValueError("start 不能晚于 end")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
