from flask import Flask, request
from .gdoc import authorize_and_return_worksheet, update_ws_stats


def create_app():
    app = Flask(__name__)

    @app.route('/', methods=['POST', 'GET'])
    def grader():
        if request.method == 'POST':
            name = request.form['name']
            # TODO: надо как-то принимать функцию, чтобы ее прогнать по тестам и получить stats
            # TODO: так что пока stats будет пустым
            stats = {
                "linear_classifier": {
                    'softmax': 0,
                    'cross_entropy_loss': 0,
                    'softmax_with_cross_entropy': 0,
                    'l2_regularization': 0,
                    'linear_softmax': 0,
                },
                'rnn_torch': {
                    'make_token_to_id': 0,
                    'make_tokens': 0,
                },
            }  # somehow grade solutions instead of this empty dict
            worksheet = authorize_and_return_worksheet()
            update_ws_stats(worksheet, name, stats)
        return "Hello everyone"

    return app
