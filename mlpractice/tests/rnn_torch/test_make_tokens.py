import random
import string

from mlpractice.utils import ExceptionInterception

try:
    from mlpractice_solutions.mlpractice_solutions\
        .rnn_torch_solution import make_tokens
except ImportError:
    make_tokens = None


def test_all(make_tokens=make_tokens):
    test_interface(make_tokens)
    test_simple(make_tokens)
    test_random(make_tokens, 100)
    print('All tests passed!')


def test_interface(make_tokens=make_tokens):
    with ExceptionInterception():
        input_text = "Sample text"
        tokens = make_tokens(input_text)

        assert isinstance(tokens, list), \
            "make_tokens must return a list of str"
        for token in tokens:
            assert isinstance(token, str), \
                "make_tokens must return a list of str"
            assert len(token) == 1, \
                "All tokens must be of length 1"


def test_simple(make_tokens=make_tokens):
    with ExceptionInterception():
        input_text = "Make ML practice great!"
        expected = {
            'M', 'a', 'k', 'e', ' ', 'L', 'p', 'r', 'c', 't', 'i', 'g', '!',
        }

        tokens = make_tokens(input_text)
        assert set(tokens) == expected, \
            "Have you heard of the set datatype in Python?"


def test_random(make_tokens=make_tokens, iterations=1):
    with ExceptionInterception():
        random.seed(42)

        for _ in range(iterations):
            input_text = ''.join(random.choices(string.ascii_letters, k=20))
            expected = set(input_text)

            tokens = make_tokens(input_text)
            assert isinstance(tokens, list), \
                "make_tokens must return a list of str"
            for token in tokens:
                assert isinstance(token, str), \
                    "make_tokens must return a list of str"
                assert len(token) == 1, \
                    "All tokens must be of length 1"

            assert set(tokens) == expected, \
                "Have you heard of the set datatype in Python?"
