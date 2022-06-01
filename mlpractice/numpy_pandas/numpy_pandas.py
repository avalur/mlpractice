

def construct_array(X, row_indices, col_indices):
    r"""Constructs slice of given matrix by indices
    row_indices and col_indices:
    [X[row_indices[0], col_indices[0]], ... ,
            X[row_indices[N-1], col_indices[N-1]]]

    Parameters
    ----------
    X : np.ndarray, dim = 2
        Input matrix.
    row_indices : list of int
    col_indices : list of int

    Returns
    -------
    result : np.ndarray
        Matrix slice
    """
    raise NotImplementedError('Not implemented!')


def construct_matrix(first_array, second_array):
    r"""Constructs matrix from pair of arrays

    Parameters
    ----------
    first_array : np.ndarray
    second_array : np.ndarray

    Returns
    -------
    result : np.ndarray
        Constructed matrix
    """
    raise NotImplementedError('Not implemented!')


def nonzero_product(matrix):
    r"""Computes product of nonzero diagonal elements of matrix
    If all diagonal elements are zeros, then returns None

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    result : Optional[float]
        Product value or None
    """
    raise NotImplementedError('Not implemented!')


def max_element_spec(x):
    r"""Returns max element in front of which is zero for input array.
    If appropriate elements are absent, then returns None

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    result : Optional[float]
        Max element value or None
    """
    raise NotImplementedError('Not implemented!')


def nearest_value(matrix, value) :
    r"""Finds nearest value in matrix.
    If matrix is empty returns None

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    result : Optional[float]
        Nearest value in matrix or None
    """
    raise NotImplementedError('Not implemented!')


def get_unique_rows(matrix):
    r"""Computes unique rows of matrix

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    result : np.ndarray
        Matrix of unique rows
    """
    raise NotImplementedError('Not implemented!')


def replace_nans(matrix) :
    r"""Replaces all nans in matrix with average of other values.
    If all values are nans, then returns zero matrix of the same size

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    result : np.ndarray
        Matrix after replacing
    """
    raise NotImplementedError('Not implemented!')


def scale(matrix):
    r"""Scales each column of matrix,
    namely, subtracts its mean from the column and
    divides the column by the standard deviation.

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    result : np.ndarray
        Matrix after scaling
    """
    raise NotImplementedError('Not implemented!')


def lin_alg_function(x):
    r"""Returns tuple of different matrix's properties:
    determinant, track, smallest and largest elements,
    Frobenius norm, eigenvalues and inverse matrix

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    result : tuple
        Tuple of different matrix's properties
    """
    raise NotImplementedError('Not implemented!')
