from werkzeug.security import check_password_hash, generate_password_hash
from .db import get_db


def register(username, password, data_base):
    error = None

    if not username:
        error = 'Username is required.'
    elif not password:
        error = 'Password is required.'

    if error is None:
        data_base.execute(
            "INSERT INTO user (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        data_base.commit()
    return error


def login(username, password, data_base):
    user = data_base.execute(
        'SELECT * FROM user WHERE username = ?', (username,)
    ).fetchone()
    error = None

    if not check_password_hash(user['password'], password):
        error = 'Incorrect password.'
    return error
