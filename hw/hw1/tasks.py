import numpy as np


# 2 points
def euclidean_distance(X, Y) -> np.ndarray:
    """
    Compute element wise euclidean distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the Euclidean distance between the corresponding pair of vectors from the arrays X and Y
    """
    new_Y = Y[:, np.newaxis]
    return np.transpose(np.sqrt(np.sum(np.pow(new_Y - X, 2), axis=2)))

# 2 points
def cosine_distance(X, Y) -> np.ndarray:
    """
    Compute element wise cosine distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the cosine distance between the corresponding pair of vectors from the arrays X and Y
    """
    new_Y = Y[:, np.newaxis]
    dot_prod = np.transpose(np.sum(np.multiply(X, new_Y), axis=2)) / np.sqrt(np.sum(np.pow(X, 2), axis=1).reshape(-1, 1)) / np.sqrt(np.sum(np.pow(Y, 2), axis=1))
    return (1 - dot_prod)


# 1 point
def manhattan_distance(X, Y) -> np.ndarray:
    """
    Compute element wise manhattan distance.

    Parameters
    ----------
    X: np.ndarray of size M * K
    Y: np.ndarray of size N * K

    Returns
    -------
    np.ndarray of size M * N
        Each element of which is the manhattan distance between the corresponding pair of vectors from the arrays X and Y
    """
    raise NotImplementedError()