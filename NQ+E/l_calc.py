import numpy as np
import time


def l_calc(sites, j_ops):
    if sites < 2 or sites > 7:
        print("Sites must be between 2 and 7.")
        return None

    try:
        h = np.load(f"../NQ/H_{sites}Q.npy")
    except FileNotFoundError:
        print("Hamiltonian file not found.")
        return None

    d = h.shape[0]
    i = np.eye(d, dtype=complex)
    l_sup_op = -1j * (np.kron(i, h) - np.kron(h.T, i))

    # Create Lindblad operators embedded per site
    l_ops = []
    for j_op in j_ops:
        for site in range(sites):
            i1 = np.eye(2 ** site, dtype=complex)
            i2 = np.eye(2 ** (sites - site - 1), dtype=complex)
            l_embedded = np.kron(np.kron(i1, j_op), i2)
            l_ops.append(l_embedded)

    for l_op in l_ops:
        l_sup_op += np.kron(l_op.conj(), l_op)
        l_sup_op -= 0.5 * (np.kron(i, l_op.conj().T @ l_op) +
                           np.kron((l_op.conj().T @ l_op).T, i))

    print(l_sup_op)
    np.save(f"L_{sites}Q_Z.npy", l_sup_op)
    return


if __name__ == "__main__":
    n_sites = 4

    print(f"Calculating the Lindblad Super Operator for {n_sites} sites...\n")

    X = np.array([
        [0 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j]
    ])

    Y = np.array([
        [0 + 0j, 0 - 1j],
        [0 + 1j, 0 + 0j]
    ])

    Z = np.array([
        [1 + 0j, 0 + 0j],
        [0 + 0j, -1 + 0j]
    ])

    for i in range(2, 7):

        start = time.time()
        l_calc(sites=i, j_ops=[Z])
        end = time.time()

        print(f"\nTime taken: {end - start:.4f} seconds")
