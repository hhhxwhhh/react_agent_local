"""价格格式化：输入整数分，输出美元文本。"""


def format_price(cents):
    return "$" + str(cents / 100)
