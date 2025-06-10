import numpy as np
import time


def h_calc(sites):

    if sites < 2:
        print("Too few sites, they should be 2 or more")
        return -1

    if sites > 14:
        print("Too much memory involved, choose less then 14 sites")
        return -1

    # Define Pauli Matrices

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

    # Calculate the Hamiltonian

    h = np.zeros((2**sites, 2**sites), dtype=complex)

    for site in range(0, sites - 1):
        i1 = np.eye(2 ** site, dtype=complex)
        i2 = np.eye(2 ** (sites - site - 2), dtype=complex)
        h += (np.kron(np.kron(np.kron(i1, X), X), i2) + np.kron(np.kron(np.kron(i1, Y), Y), i2)
              + np.kron(np.kron(np.kron(i1, Z), Z), i2))
        print(f"Completed at {int(((site + 1) * 100) / (sites - 1))}%")

    print("\nHamiltonian Matrix:")
    print(h)

    np.save(f"H_{n_sites}Q.npy", h)
    print("\nFile saved!")

    return


if __name__ == "__main__":
    n_sites = 2

    print(f"Calculating the Hamiltonian for {n_sites} sites...\n")

    start = time.time()
    hamiltonian = h_calc(sites=n_sites)
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
