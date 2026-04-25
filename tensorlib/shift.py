# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Merlin Elvers
# Portions copyright (c) 2024 tensor4all collaboration.
#
# The carry-based shift construction is derived from tensor4all-rs's
# tensor4all-quanticstransform shift operator, itself a port of Quantics.jl.
# See LICENSE for pinned sources and license information.

import numpy as np
import xfacpy
from tensorlib.contraction import as_cores


def compress_mps_svd(cores, reltol=1e-12, max_bond_dim=0):
    tt = xfacpy.TensorTrain([np.asarray(c, dtype=float) for c in as_cores(cores)])
    tt.compressSVD(reltol, max_bond_dim)
    return as_cores(tt)


def shift_3d(mps, nBit, dr, L=40):
    h = L / 2**nBit

    chi_orig = max((c.shape[2] for c in as_cores(mps)[:-1]))

    for dim, delta in enumerate(dr):
        shift_i = int(round(delta / h))
        if not shift_i:
            continue
        mps = fast_shift(
            mps, nBit, dim, shift_i, n_dims=3, periodic=False, compress=False
        )

    # Compress once
    return compress_mps_svd(mps, reltol=1e-12, max_bond_dim=chi_orig)


def fast_shift(mps, nBit, dim, shift_int, n_dims=3, periodic=False, compress=False):
    transposed = not periodic and shift_int < 0

    mps_cores = as_cores(mps)
    N = 1 << nBit
    shift_mod = (-shift_int if transposed else shift_int) % N
    if shift_mod == 0:
        return [c.copy() for c in mps_cores]

    shift_bits = [(shift_mod >> k) & 1 for k in range(nBit)]
    new_cores = [c.copy() for c in mps_cores]
    dim_start = dim * nBit

    for local_k in range(nBit):
        site = dim_start + local_k
        A = mps_cores[site]
        chi_l, d, chi_r = A.shape
        add_bit = shift_bits[local_k]
        is_first = local_k == 0
        is_last = local_k == nBit - 1

        wl = 1 if is_first else 2
        wr = 1 if is_last else 2

        c_in_max = 1 if is_first else 2
        c_in_base = 0

        new_core = np.zeros((wl, chi_l, 2, wr, chi_r))

        for c_in_idx in range(c_in_max):
            c_in = c_in_base if is_first else c_in_idx
            for i in range(2):
                total = i + add_bit + c_in
                j = total & 1
                c_out = (total >> 1) & 1

                # Transposed MPO swaps the physical in/out legs.
                dst, src = (i, j) if transposed else (j, i)

                if is_last:
                    if periodic or c_out == 0:
                        new_core[c_in_idx, :, dst, 0, :] += A[:, src, :]
                else:
                    new_core[c_in_idx, :, dst, c_out, :] += A[:, src, :]

        new_cores[site] = np.ascontiguousarray(
            new_core.reshape(wl * chi_l, 2, wr * chi_r)
        )

    if compress is False:
        return new_cores
    if compress is True:
        chi_orig = max((c.shape[2] for c in mps_cores[:-1]), default=0)
        return compress_mps_svd(new_cores, reltol=1e-12, max_bond_dim=chi_orig)
    return compress_mps_svd(new_cores, reltol=float(compress))
