"""递归 Fibonacci。"""

from functools import lru_cache


@lru_cache(maxsize=None)
def fib(n):
    if n < 0:
        raise ValueError("n 必须 >= 0")
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
