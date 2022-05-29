import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from IPython.display import clear_output


def make_tokens(input_text):
    r"""Makes a list of all unique characters in the `input_text`.

    Parameters
    ----------
    input_text : str
        Input text for RNN training. Should be a simple plain text file.

    Returns
    -------
    tokens : list of str
        List with all unique tokens.
    """
    # Your final implementation shouldn't have any loops
    # TODO: implement make_tokens
    raise NotImplementedError('Not implemented!')


def make_token_to_id(tokens):
    r"""Creates a mapping between tokens and its int identifiers.

    Parameters
    ----------
    tokens : list of str
        List with all unique tokens.

    Returns
    -------
    token_to_id : dict of str
        Tokens to its identifier (index in tokens list).
    """
    # TODO: implement make_token_to_id
    raise NotImplementedError('Not implemented!')


class CharRNNCell(nn.Module):
    r"""Vanilla RNN cell with tanh non-linearity.

    Parameters
    ----------
    num_tokens : int
        Size of the token dictionary.
    embedding_size : int
        Size of the token embedding vector.
    rnn_num_units : int
        A number of features in the hidden state vector.

    Attributes
    ----------
    num_units : int
        A number of features in the hidden state vector.
    embedding : nn.Embedding
        An embedding layer that converts character id to a vector.
    rnn_update : nn.Linear
        A linear layer that creates a new hidden state vector.
    rnn_to_logits : nn.Linear
        An output layer that predicts probabilities of next phoneme.
    """
    def __init__(self, num_tokens, embedding_size=16, rnn_num_units=64):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units

        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(
            embedding_size + rnn_num_units,
            rnn_num_units,
        )
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, h_prev):
        r"""Compute h_next(x, h_prev) and log(P(x_next | h_next)).
        We'll call it repeatedly to produce the whole sequence.

        Parameters
        ----------
        x : torch.LongTensor, shape(batch_size)
            Batch of character ids.
        h_prev : torch.FloatTensor, shape(batch_size, num_units)
            Previous rnn hidden states.

        Returns
        -------
        h_next : torch.FloatTensor, shape(batch_size, num_units)
            Next rnn hidden states.
        x_next_proba : torch.FloatTensor, shape(batch_size, num_tokens)
            Predicted probabilities for the next token.
        """
        # get vector embedding of x
        x_emb = self.embedding(x)

        # TODO: compute next hidden state using self.rnn_update
        # hint: use torch.cat(..., dim=...) for concatenation
        raise NotImplementedError('Not implemented!')
        # h_next = ...

        h_next = torch.tanh(h_next)

        # TODO: compute logits for next character probs
        raise NotImplementedError('Not implemented!')
        # logits = ...

        return h_next, F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        r"""Returns rnn state before it processes first input (aka h_0)."""
        return torch.zeros(batch_size, self.num_units)


def rnn_loop(char_rnn, batch_ix):
    r"""Computes log P(next_character) for all time-steps in lines_ix."""
    batch_size, max_length = batch_ix.size()
    hid_state = char_rnn.initial_state(batch_size)
    log_probs = []

    for x_t in batch_ix.transpose(0, 1):
        hid_state, log_p_next = char_rnn(x_t, hid_state)
        log_probs.append(log_p_next)

    return torch.stack(log_probs, dim=1)


def train_rnn(encoded_lines, model, optimizer, iterations=1000):
    r"""Trains RNN on a given text.

    Parameters
    ----------
    encoded_lines : np.ndarray, shape(n_samples, MAX_LENGTH)
        Lines of input text converted to a matrix.
    model : torch.nn.Module
        A model to train.
    optimizer : torch.optim.Optimizer
        Optimizer that will be used to train a model.
    iterations : int, optional
        Number of optimization steps that the model will make.

    Returns
    -------
    training_history : list of float
        Training history consisting of mean-loss-per-iteration records.
    """
    training_history = []

    for i in range(iterations):
        batch_indices = np.random.choice(len(encoded_lines), 32, replace=False)

        batch_ix = encoded_lines[batch_indices]
        batch_ix = torch.tensor(batch_ix, dtype=torch.int64)

        # TODO: implement train loop
        raise NotImplementedError('Not implemented!')

        log_p_seq = rnn_loop(char_rnn, batch_ix)

        # TODO: compute loss

        # loss = ...

        # TODO: train with backprop

        training_history.append(loss.item())
        if (i + 1) % 100 == 0:
            clear_output(True)
            plt.plot(training_history, label='loss')
            plt.legend()
            plt.show()

    return training_history

