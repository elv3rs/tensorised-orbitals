import numpy as np


def as_cores(mps):
    return [np.asarray(c) for c in mps]


def contract_mps_mpo_mps(mps_a, mpo, mps_b, dtype=np.float64):
    """Computes <mps_a|mpo|mps_b>. Assumes mps as tt with cores
    (x_l, p, x_r), and mpo with shape (x_l, p1, p2, x_r).

    Contraction order adapted from:
        Stoudenmire, E., & White, S. (2010). Minimally entangled typical
        thermal state algorithms. New Journal of Physics, 12(5), 055026. (eq. 22)
    """
    mps_a, mpo, mps_b = as_cores(mps_a), as_cores(mpo), as_cores(mps_b)
    L = np.ones((1, 1, 1), dtype=dtype)
    for A, W, B in zip(mps_a, mpo, mps_b):
        A = np.asarray(A.conj(), dtype=dtype)
        W = np.asarray(W, dtype=dtype)
        B = np.asarray(B, dtype=dtype)

        L = np.tensordot(L, A, [[0], [0]])
        L = np.tensordot(L, W, [[0, 2], [0, 1]])
        L = np.tensordot(L, B, [[0, 2], [0, 1]])

    assert np.prod(L.shape) == 1
    return np.squeeze(L)


def contract_mps_mps_mps(mps_a, mps_b, mps_c, dtype=np.float64):
    """Computes <mps_a|mps_b|mps_c>."""
    mps_a, mps_b, mps_c = as_cores(mps_a), as_cores(mps_b), as_cores(mps_c)
    L_new = np.ones((1, 1, 1), dtype=dtype)
    for A, B, C in zip(mps_a, mps_b, mps_c):
        A = np.asarray(A.conj(), dtype=dtype)
        B = np.asarray(B, dtype=dtype)
        C = np.asarray(C, dtype=dtype)
        L = L_new
        L_new = None
        for p in range(A.shape[1]):
            L_p = np.tensordot(C[:, p, :], L, [[0], [2]])
            L_p = np.tensordot(B[:, p, :], L_p, [[0], [2]])
            L_p = np.tensordot(A[:, p, :], L_p, [[0], [2]])
            if L_new is None:
                L_new = L_p
            else:
                L_new += L_p
    assert np.prod(L_new.shape) == 1
    return np.squeeze(L_new)


def contract_mps_mps(mps_a, mps_b, dtype=np.float64):
    """Computes <mps_a|mps_b>"""
    mps_a, mps_b = as_cores(mps_a), as_cores(mps_b)
    L = np.ones((1, 1), dtype=dtype)

    for A, B in zip(mps_a, mps_b):
        A = np.asarray(A, dtype=dtype)
        B = np.asarray(B, dtype=dtype)
        L = np.tensordot(L, B, axes=[[0], [0]])
        L = np.tensordot(L, A, axes=[[0, 1], [0, 1]])

    assert np.prod(L.shape) == 1
    return np.squeeze(L)
