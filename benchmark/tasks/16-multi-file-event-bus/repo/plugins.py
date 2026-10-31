"""默认插件：把审计处理器注册到订单事件。"""

from events import subscribe


def audit(payload=None):
    return "audit:order.created"


def register_defaults():
    subscribe("order.created", audit, priority=-10)
    return 1
