"""JSON 序列化辅助。"""

import json


def dumps_safe(value):
    return json.dumps(value, ensure_ascii=False)
