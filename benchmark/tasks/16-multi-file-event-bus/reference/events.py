"""极简事件总线：按优先级、稳定顺序、异常隔离地分发事件。"""

_subscribers = {}


def subscribe(event, handler, priority=0):
    _subscribers.setdefault(event, []).append((priority, handler))


def unsubscribe(event, handler):
    for index, (_, existing) in enumerate(_subscribers.get(event, [])):
        if existing is handler or existing == handler:
            _subscribers[event].pop(index)
            return True
    return False


def emit(event, payload=None):
    handlers = sorted(_subscribers.get(event, []), key=lambda item: -item[0])
    results = []
    for _, handler in handlers:
        try:
            results.append(handler(payload))
        except Exception:
            continue
    return results


def clear(event=None):
    if event is None:
        _subscribers.clear()
    else:
        _subscribers.pop(event, None)
