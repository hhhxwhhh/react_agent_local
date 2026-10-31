"""价格格式化：输入整数分，输出美元文本。"""


def format_price(cents):
    return f"${cents / 100:.2f}"
