from mlpractice_solutions.mlpractice_solutions.rnn_torch_solution \
    import make_tokens


def test_simple():
    input_text = "Make ML practice great!"
    expected = set(input_text)

    assert set(make_tokens(input_text)) == expected, \
        "Have you heard of the set datatype in Python?"
