import xfacpy
from time import perf_counter
import pathlib
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

from tensorlib.contraction import contract_mps_mpo_mps, contract_mps_mps_mps
from tensorlib.laplacian_mpo import build_laplacian_mpo

side_length = 40
n_sample = np.array([8, 12, 16, 20])
x_sample = np.array([10, 20, 30, 40, 50, 60, 70])
x_overrank = 256  # int or False

cachefile = pathlib.Path(f"figures/cache_{x_overrank}.npy")
figurepath = pathlib.Path(f"figures/figure_{x_overrank}.png")
allow_cache = True
overrank_cache = {}


def main():
    data = get_data()
    make_plot(data)
    print_data(data)


def inv_r(r):
    eps = 1e-10
    return 1 / (np.linalg.norm(r) + eps)


def h1s(r):
    return np.exp(-np.linalg.norm(r)) / np.sqrt(np.pi)


def get_error(nBit, bondDim):
    psi = get_qtt(h1s, nBit, bondDim)
    coulomb = get_qtt(inv_r, nBit, bondDim)
    laplace = build_laplacian_mpo([nBit, nBit, nBit])

    K = contract_mps_mpo_mps(psi, laplace, psi) * 0.5 * coulomb.grid.deltaX
    P = -contract_mps_mps_mps(psi, coulomb, psi) * coulomb.grid.deltaVolume
    return np.abs(np.array([K - 0.5, P + 1.0, K + P + 0.5]))


def get_data():
    if cachefile.exists() and allow_cache:
        print("Results restored from", cachefile)
        return np.load(cachefile)
    print("Results not cached, recomputing.")

    res = np.zeros((n_sample.size, x_sample.size, 3))
    for i, n in enumerate(n_sample):
        for j, x in reversed(list(enumerate(x_sample))):
            res[i, j, :] = get_error(n, x)
    np.save(cachefile, res)
    print("Saved results to", cachefile)
    return res


def do_tci(f, nBit, bondDim):
    h = side_length / 2**nBit
    a = -(side_length + h) / 2
    b = a + side_length

    qgrid = xfacpy.QuanticsGrid(a=a, b=b, dim=3, nBit=nBit, grouped=True)
    params = xfacpy.TensorCI2Param()
    params.bondDim = bondDim
    params.reltol = 1e-12
    params.pivot1 = qgrid.coord_to_id([0.5, 0.5, 0.5])
    ci = xfacpy.QTensorCI(f=f, qgrid=qgrid, args=params)

    iteration = 0
    mDim = 0
    aDim = 0
    t0 = perf_counter()
    while True:
        ci.iterate()

        # Debug print statements:
        iteration += 1
        lastMdim = mDim
        lastAdim = aDim

        bond_dims = np.array([core.shape[0] for core in ci.tt.core[1:-1]])
        mDim = max(bond_dims)
        aDim = np.mean(bond_dims)
        pivotErr = ci.pivotError[-1]
        elapsed = int(perf_counter() - t0)
        if iteration > 1:
            print("\033[F\033[K", end="")
        print(
            f"Iteration {iteration}: rank m{mDim} a{int(round(aDim))}, pivotErr {pivotErr:.1e}, elapsed {elapsed}s"
        )

        if lastMdim < mDim or lastAdim < aDim:
            continue

        if ci.isDone():
            break

    return ci.get_qtt()


def get_qtt(f, nBit, bondDim):
    if not x_overrank:
        return do_tci(f, nBit, bondDim)

    key = f"{f.__name__}{nBit}"
    if key not in overrank_cache:
        overrank_cache[key] = [do_tci(f, nBit, x_overrank), x_overrank]

    qtt, lastBond = overrank_cache[key]
    # Truncation only ever removes rank, so x_sample has to descend.
    assert bondDim <= lastBond
    qtt.tt.compressSVD(0.0, bondDim)
    overrank_cache[key][1] = bondDim

    return qtt


def print_data(data):
    for i, n in enumerate(n_sample):
        print(f"\nn = {n}\nx  |K-K^th| |P-P^th| |E-E^th|")
        for j, x in enumerate(x_sample):
            dK, dP, dE = data[i, j, :]
            print(f"{x} {dK:.2e} {dP:.2e} {dE:.2e}")


def make_plot(data):
    ylabels = ["$|K-K^{th}|$ [Ha]", "$|P-P^{th}|$ [Ha]", "$|E-E^{th}|$ [Ha]"]
    colors = ["tab:blue", "tab:orange", "tab:green", "black"]

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[1].set_title(
        f"H1s, grouped ordering, qtt on $[-{side_length/2},{side_length/2}]^3$, overrank={x_overrank}"
    )
    plt.subplots_adjust(wspace=0.35)

    for i, n in enumerate(n_sample):
        for j in range(3):
            ax[j].semilogy(
                x_sample, data[i, :, j], marker="o", color=colors[i], label=f"n={n}"
            )
            ax[j].set_xlabel(r"$\chi$")
            ax[j].set_ylabel(ylabels[j])
            ax[j].yaxis.set_minor_locator(ticker.NullLocator())
            ax[j].set_ylim(3e-9, 3e0)
            ax[j].set_xlim(5, 65)
    ax[0].legend(loc="upper left", ncol=2, frameon=False)

    plt.savefig(figurepath)
    print("Saved plot to", figurepath)


if __name__ == "__main__":
    main()
