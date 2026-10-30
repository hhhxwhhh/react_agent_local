"""用户创建。"""

import re


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_user(data):
    for field in ("username", "email", "age"):
        if field not in data:
            raise ValueError(f"缺少字段: {field}")
    username = data["username"]
    email = data["email"]
    age = data["age"]
    if not isinstance(username, str) or not _USERNAME_RE.fullmatch(username):
        raise ValueError("username 必须是 3-20 位字母、数字或下划线")
    if not isinstance(email, str) or not _EMAIL_RE.fullmatch(email):
        raise ValueError("email 格式不合法")
    if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 150:
        raise ValueError("age 必须是 0-150 的整数")
    return {"username": username, "email": email, "age": age}
