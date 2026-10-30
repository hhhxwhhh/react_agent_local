"""日期范围生成器。"""

from datetime import timedelta


def date_range(start, end):
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)
