"""极简事件总线。

任务：实现 subscribe / unsubscribe / emit / clear。
"""


def subscribe(event, handler, priority=0):
    """订阅事件。priority 越大越先执行。"""
    raise NotImplementedError


def unsubscribe(event, handler):
    """取消订阅；成功返回 True，未找到返回 False。"""
    raise NotImplementedError


def emit(event, payload=None):
    """按优先级调用处理函数，返回成功处理函数的返回值列表。"""
    raise NotImplementedError


def clear(event=None):
    """清空全部订阅，或只清空指定事件。"""
    raise NotImplementedError
