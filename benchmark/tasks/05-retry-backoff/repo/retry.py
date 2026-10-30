"""重试工具。"""

import time


def retry(func, attempts=3, base_delay=0.0, sleep=time.sleep, exceptions=(Exception,)):
    last = None
    for index in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
            if index == attempts - 1:
                raise
            sleep(base_delay)
    raise last
