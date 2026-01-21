import dis
import numpy as np
from numpy.testing import assert_allclose
from broadcasting_exercises.grading import test

rng = np.random.default_rng()


def assert_no_python_loops(fn):
    """
    Heuristic: flags common Python-level loops/comprehensions by checking bytecode.
    Not perfect, but good enough to discourage "just loop".
    """
    bytecode = dis.Bytecode(fn)
    loop_ops = {"FOR_ITER", "SETUP_LOOP"}
    if any(instr.opname in loop_ops for instr in bytecode):
        raise AssertionError(
            "Python loop detected (FOR_ITER/SETUP_LOOP). Use vectorized NumPy broadcasting."
        )


def rand_shape(min_ndim=1, max_ndim=3, min_size=1, max_size=7):
    ndim = rng.integers(min_ndim, max_ndim + 1)
    return tuple(int(rng.integers(min_size, max_size + 1)) for _ in range(ndim))


def _test_ex1(fn):
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(20, 100))
        D = int(rng.integers(1, 20))
        X = rng.normal(size=(N, D))

        got = fn(X)
        assert got.shape == (N, D)

        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        # avoid divide-by-zero in pathological tiny samples
        sd = np.where(sd == 0, 1.0, sd)
        expected = (X - mu) / sd
        assert_allclose(got, expected)


def _test_ex2(fn):
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(1, 40))
        M = int(rng.integers(1, 35))
        a = rng.normal(size=(N,))
        b = rng.normal(size=(M,))

        got = fn(a, b)
        assert got.shape == (N, M)

        expected = a[:, None] + b[None, :]
        assert_allclose(got, expected)


def ref_row_gather(X, idx):
    N = X.shape[0]
    return X[np.arange(N), idx]


def _test_ex3(fn):
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(1, 50))
        D = int(rng.integers(1, 30))
        X = rng.normal(size=(N, D))
        idx = rng.integers(0, D, size=(N,))

        got = fn(X, idx)
        expected = ref_row_gather(X, idx)

        assert got.shape == expected.shape
        assert_allclose(got, expected)


def ref_row_scatter_add(X, idx, values):
    """
    Reference implementation for row_scatter_add.

    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (N, D).
    idx : np.ndarray
        Index array of shape (N, K) with values in [0, D).
    values : np.ndarray
        Values to add, shape (N, K).

    Returns
    -------
    np.ndarray
        Copy of X with values added at specified indices, shape (N, D).
    """
    out = X.copy()
    N = X.shape[0]
    rows = np.arange(N)[:, None]
    np.add.at(out, (rows, idx), values)
    return out


def _test_ex4(fn):
    """
    Test function for Exercise 4: Row-wise Scatter Add.

    Validates that the provided function correctly performs scatter-add
    operations using broadcasted advanced indexing, without Python-level loops.

    Parameters
    ----------
    fn : callable
        Function to test with signature fn(X, idx, values) -> np.ndarray
    """
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(1, 40))
        D = int(rng.integers(5, 25))
        K = int(rng.integers(1, 20))

        # encourage repeats by making D small sometimes
        if rng.random() < 0.3:
            D = int(rng.integers(1, 6))

        X = rng.normal(size=(N, D))
        idx = rng.integers(0, D, size=(N, K))
        values = rng.normal(size=(N, K))

        got = fn(X, idx, values)
        expected = ref_row_scatter_add(X, idx, values)

        assert got.shape == (N, D)
        assert_allclose(got, expected)


def _ref_one_hot(idx):
    """
    Reference implementation for one-hot encoding.

    Parameters
    ----------
    idx : np.ndarray
        Integer array of class labels, shape (N,).

    Returns
    -------
    np.ndarray
        One-hot encoded matrix of shape (N, K) where K = max(idx) + 1.
    """
    N = idx.shape[0]
    K = idx.max() + 1
    out = np.zeros((N, K), dtype=int)
    out[np.arange(N), idx] = 1
    return out


def _test_ex5(fn):
    """
    Test function for Exercise 5: One-Hot Encoding.

    Validates that the provided function correctly creates a one-hot encoded
    matrix using broadcasting, without Python-level loops.

    Parameters
    ----------
    fn : callable
        Function to test with signature fn(idx) -> np.ndarray
    """
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(1, 50))
        K = int(rng.integers(2, 20))
        idx = rng.integers(0, K, size=(N,))

        got = fn(idx)
        expected = _ref_one_hot(idx)

        assert got.shape == expected.shape, (
            f"Expected shape {expected.shape}, got {got.shape}"
        )
        assert_allclose(got, expected)


def _ref_pairwise_sq_dist(X, Y):
    """
    Reference implementation for pairwise squared Euclidean distances.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D).
    Y : np.ndarray
        Array of shape (M, D).

    Returns
    -------
    np.ndarray
        Pairwise squared distances of shape (N, M).
    """
    N = X.shape[0]
    M = Y.shape[0]
    dist2 = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist2[i, j] = np.sum((X[i] - Y[j]) ** 2)
    return dist2


def _test_ex6(fn):
    """
    Test function for Exercise 6: Pairwise Squared Distances.

    Validates that the provided function correctly computes pairwise squared
    Euclidean distances using broadcasting, without Python-level loops.

    Parameters
    ----------
    fn : callable
        Function to test with signature fn(X, Y) -> np.ndarray
    """
    assert_no_python_loops(fn)

    for _ in range(10):
        N = int(rng.integers(1, 30))
        M = int(rng.integers(1, 30))
        D = int(rng.integers(1, 20))

        X = rng.normal(size=(N, D))
        Y = rng.normal(size=(M, D))

        got = fn(X, Y)
        expected = _ref_pairwise_sq_dist(X, Y)

        assert got.shape == (N, M), f"Expected shape {(N, M)}, got {got.shape}"
        assert_allclose(got, expected)


def _ref_gather_last_axis(X, choice):
    """
    Reference implementation for batched gather along last axis.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (B, T, K).
    choice : np.ndarray
        Integer array of shape (B, T) with values in [0, K).

    Returns
    -------
    np.ndarray
        Gathered values of shape (B, T).
    """
    B, T, K = X.shape
    out = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            out[b, t] = X[b, t, choice[b, t]]
    return out


def _test_ex7(fn):
    """
    Test function for Exercise 7: Batched Gather Along Last Axis.

    Validates that the provided function correctly gathers values from the last
    axis using broadcasted indexing, without Python-level loops.

    Parameters
    ----------
    fn : callable
        Function to test with signature fn(X, choice) -> np.ndarray
    """
    assert_no_python_loops(fn)

    for _ in range(10):
        B = int(rng.integers(1, 20))
        T = int(rng.integers(1, 30))
        K = int(rng.integers(2, 25))

        X = rng.normal(size=(B, T, K))
        choice = rng.integers(0, K, size=(B, T))

        got = fn(X, choice)
        expected = _ref_gather_last_axis(X, choice)

        assert got.shape == (B, T), f"Expected shape {(B, T)}, got {got.shape}"
        assert_allclose(got, expected)


def _ref_softmax(logits, axis=-1):
    """
    Reference implementation for stable softmax.

    Parameters
    ----------
    logits : np.ndarray
        Input array of logits.
    axis : int
        Axis along which to compute softmax.

    Returns
    -------
    np.ndarray
        Softmax probabilities.
    """
    max_val = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_val
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def _test_ex8(fn):
    """
    Test function for Exercise 8: Stable Softmax.

    Validates that the provided function correctly computes numerically stable
    softmax using broadcasting, without Python-level loops.

    Parameters
    ----------
    fn : callable
        Function to test with signature fn(logits, axis=-1) -> np.ndarray
    """
    assert_no_python_loops(fn)

    # Test with various shapes and axes
    for _ in range(10):
        ndim = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(2, 10)) for _ in range(ndim))
        axis = int(rng.integers(-ndim, ndim))

        logits = rng.normal(size=shape)

        got = fn(logits, axis=axis)
        expected = _ref_softmax(logits, axis=axis)

        assert got.shape == expected.shape, (
            f"Expected shape {expected.shape}, got {got.shape}"
        )
        assert_allclose(got, expected)

        # Check that softmax sums to 1 along the axis
        sums = np.sum(got, axis=axis)
        assert_allclose(sums, np.ones_like(sums))

    # Test numerical stability with large values
    for _ in range(5):
        shape = (10, 20)
        logits = rng.normal(size=shape) * 100 + 500  # Large values
        got = fn(logits, axis=-1)
        expected = _ref_softmax(logits, axis=-1)

        assert np.all(np.isfinite(got)), "Softmax output contains non-finite values"
        assert_allclose(got, expected)


test_ex1 = test(_test_ex1)
test_ex2 = test(_test_ex2)
test_ex3 = test(_test_ex3)
test_ex4 = test(_test_ex4)
test_ex5 = test(_test_ex5)
test_ex6 = test(_test_ex6)
test_ex7 = test(_test_ex7)
test_ex8 = test(_test_ex8)

__all__ = [
    "test_ex1",
    "test_ex2",
    "test_ex3",
    "test_ex4",
    "test_ex5",
    "test_ex6",
    "test_ex7",
    "test_ex8",
]
