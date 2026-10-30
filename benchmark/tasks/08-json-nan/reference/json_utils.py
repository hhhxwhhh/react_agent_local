"""JSON 序列化辅助。"""

import json
import math


def _check(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("不支持 NaN 或 Infinity")
    if isinstance(value, dict):
        for key, item in value.items():
            _check(key)
            _check(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check(item)


def dumps_safe(value):
    _check(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
