import random
import string

try:
    from mlpractice_solutions.mlpractice_solutions\
        .rnn_torch_solution import make_token_to_id
except ImportError:
    make_token_to_id = None


def test_all(make_token_to_id=make_token_to_id):
    test_interface(make_token_to_id)
    test_len(make_token_to_id)
    test_random(make_token_to_id, 100)
    print('All tests passed!')


def test_interface(make_token_to_id=make_token_to_id):
    tokens = ['a', 'b', 'c', 'd']
    token2id = make_token_to_id(tokens)

    assert isinstance(token2id, dict), \
        'make_token_to_id must return a dict of str'
    for key, value in token2id.items():
        assert isinstance(key, str), \
            'make_token_to_id must return a dict with str keys'
        assert isinstance(value, int), \
            'make_token_to_id must return a dict with int values'


def test_len(make_token_to_id=make_token_to_id):
    tokens = ['Make', 'ML', 'great']
    token2id = make_token_to_id(tokens)

    assert len(token2id) == len(tokens), \
        'Returned dict must be the same length as the tokens list'


def test_random(make_token_to_id=make_token_to_id, iterations=1):
    random.seed(42)

    for _ in range(iterations):
        tokens = list(set(random.choices(string.ascii_letters, k=20)))
        token2id = make_token_to_id(tokens)

        for i, token in enumerate(tokens):
            assert token2id[tokens[i]] == i, \
                "Token identifier must be it's position in tokens list"
