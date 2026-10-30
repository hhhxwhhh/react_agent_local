"""重试工具。"""

import time


def retry(func, attempts=3, base_delay=0.0, sleep=time.sleep, exceptions=(Exception,)):
    if attempts < 1:
        raise ValueError("attempts 必须 >= 1")
    if base_delay < 0:
        raise ValueError("base_delay 不能为负")
    last = None
    for index in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
            if index == attempts - 1:
                raise
            sleep(base_delay * (2 ** index))
    raise last
