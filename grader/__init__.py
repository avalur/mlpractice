from flask import Flask, request
from .gdoc import authorize_and_return_worksheet, update_ws_stats
from ast import literal_eval


def create_app():
    app = Flask(__name__)

    @app.route('/grade', methods=['POST', 'GET'])
    def grader():
        if request.method == 'POST':
            name = request.form['name']
            # TODO: надо принимать функцию, чтобы ее прогнать по тестам (безопасно) и получить stats, но пока так
            stats = literal_eval(request.form['stats'])

            worksheet = authorize_and_return_worksheet()
            update_ws_stats(worksheet, name, stats)
        return "Grading grades"

    return app
