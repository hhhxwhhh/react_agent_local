"""用户查询。"""


def find_user(conn, username):
    cursor = conn.execute(
        "SELECT id, username FROM users WHERE username = ?", (username,)
    )
    return cursor.fetchall()
