import os

from flask import Flask, request
from .gdoc import authorize_and_return_worksheet, update_ws_stats
from ast import literal_eval
from . import db
from . import auth


def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'grader.sqlite'),
    )

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/grade', methods=['POST', 'GET'])
    def grader():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            # TODO: надо принимать функцию, чтобы ее прогнать по тестам (безопасно) и получить stats, но пока так
            stats = literal_eval(request.form['stats'])

            data_base = db.get_db()
            users = [username[0] for username in data_base.execute('SELECT username FROM user')]
            if username not in users:
                error = auth.register(username, password)
            else:
                error = auth.login(username, password)

            # TODO: как-то вывести ошибку пользователю в ноутбук
            if error:
                return error

            worksheet = authorize_and_return_worksheet()
            update_ws_stats(worksheet, username, stats)
        return "Grading grades"

    db.init_app(app)

    return app
