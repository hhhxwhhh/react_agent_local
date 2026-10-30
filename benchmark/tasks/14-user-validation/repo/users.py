"""用户创建。"""


def create_user(data):
    return {
        "username": data["username"],
        "email": data["email"],
        "age": data["age"],
    }
