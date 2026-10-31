"""购物车：价格以整数分存储。"""


def add_item(cart, name, price, qty=1):
    if qty < 1:
        raise ValueError("qty")
    cart.append({"name": name, "price": price, "qty": qty})


def total(cart):
    return sum(item["price"] for item in cart)


def apply_discount(cart, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent")
    factor = 1 - percent / 100
    for item in cart:
        item["price"] = int(item["price"] * factor)
