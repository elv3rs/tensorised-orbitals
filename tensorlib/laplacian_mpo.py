# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Merlin Elvers
# Portions copyright (C) 2009-2012 Ivan Oseledets, Sergey Dolgov,
# Vladimir Kazeev, Thomas Mach, Olga Lebedeva, Dmitry Savostyanov,
# Pavel Zhlobich, and Le Song.
#
# Derived from TT-Toolbox's exp/tt_qlaplace_dd.m. See
# LICENSE for the pinned source and upstream MIT license.

import numpy as np


def build_laplacian_mpo(nBits):
    """Returns the QTT Laplacian MPO for grouped ordering (x1 x2,... y1,y2...).
    Assumes Dirichlet-Dirichlet BC, max bond rank 4.
    Grid spacing not baked in: laplace = - this * h

    Ported from the reference MATLAB implementation (tt_qlaplace_dd.m by Vladimir Kazeev 2010-09-03).

    Paper:
        Vladimir A. Kazeev and Boris N. Khoromskij,
        "On explicit QTT representation of Laplace operator and its inverse",
        Preprint No. 75, 2010, Max-Planck Institute for Mathematics in the Sciences.
        http://www.mis.mpg.de/publications/preprints/2010/prepr2010-75.html

    Matlab reference (accessed 2026-04-23):
        https://github.com/oseledets/TT-Toolbox/blob/master/exp/tt_qlaplace_dd.m

    Parameters
    ----------
    nBits : integer array specifying nBit per dimension

    Returns
    -------
    cores : list of sum(nBits) MPO cores with shapes (x_left, 2, 2, x_right).
    """
    nBits = np.asarray(nBits, dtype=int)

    max_dimension = len(nBits)
    assert np.all(nBits > 1)
    assert max_dimension > 0

    # 2x2 blocks
    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    JT = np.array([[0.0, 0.0], [1.0, 0.0]])
    D = 2 * I - J - JT

    cores = []

    # 1d, Corollary 3.2
    if max_dimension == 1:
        for bit in range(nBits[0]):
            if bit == 0:
                core = np.zeros((1, 2, 2, 3))
                core[0, :, :, 0] = D
                core[0, :, :, 1] = -J
                core[0, :, :, 2] = -JT
            elif bit == nBits[0] - 1:
                core = np.zeros((3, 2, 2, 1))
                core[0, :, :, 0] = I
                core[1, :, :, 0] = JT
                core[2, :, :, 0] = J
            else:
                core = np.zeros((3, 2, 2, 3))
                core[0, :, :, 0] = I
                core[1, :, :, 1] = J
                core[2, :, :, 2] = JT
                core[1, :, :, 0] = JT
                core[2, :, :, 0] = J
            cores.append(core)
        return cores

    # n-d, Corollary 5.3
    for dimension in range(max_dimension):
        for bit in range(nBits[dimension]):
            if bit == 0:
                if dimension == 0:
                    core = np.zeros((1, 2, 2, 4))
                    core[0, :, :, 0] = D
                    core[0, :, :, 1] = -J
                    core[0, :, :, 2] = -JT
                    core[0, :, :, 3] = I
                elif dimension == max_dimension - 1:
                    core = np.zeros((2, 2, 2, 3))
                    core[0, :, :, 0] = D
                    core[0, :, :, 1] = -J
                    core[0, :, :, 2] = -JT
                    core[1, :, :, 0] = I
                else:
                    core = np.zeros((2, 2, 2, 4))
                    core[0, :, :, 0] = D
                    core[0, :, :, 1] = -J
                    core[0, :, :, 2] = -JT
                    core[0, :, :, 3] = I
                    core[1, :, :, 0] = I
            elif bit == nBits[dimension] - 1:
                if dimension == max_dimension - 1:
                    core = np.zeros((3, 2, 2, 1))
                    core[0, :, :, 0] = I
                    core[1, :, :, 0] = JT
                    core[2, :, :, 0] = J
                else:
                    core = np.zeros((4, 2, 2, 2))
                    core[3, :, :, 0] = I
                    core[0, :, :, 1] = I
                    core[1, :, :, 1] = JT
                    core[2, :, :, 1] = J
            else:
                if dimension == max_dimension - 1:
                    core = np.zeros((3, 2, 2, 3))
                    core[0, :, :, 0] = I
                    core[1, :, :, 1] = J
                    core[2, :, :, 2] = JT
                    core[1, :, :, 0] = JT
                    core[2, :, :, 0] = J
                else:
                    core = np.zeros((4, 2, 2, 4))
                    core[0, :, :, 0] = I
                    core[1, :, :, 1] = J
                    core[2, :, :, 2] = JT
                    core[1, :, :, 0] = JT
                    core[2, :, :, 0] = J
                    core[3, :, :, 3] = I
            cores.append(core)

    return cores
