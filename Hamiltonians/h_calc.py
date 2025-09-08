from scipy.sparse import csr_matrix, identity, kron, save_npz

import time


def h_calc(sites, delta):

    if sites < 2:
        print("Too few sites, they should be 2 or more")
        return -1

    if sites > 22:
        print("Too much memory involved, choose less then 22 sites")
        return -1

    # Define Pauli Matrices

    X = csr_matrix([
        [0 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j]
    ], dtype=complex)

    Y = csr_matrix([
        [0 + 0j, 0 - 1j],
        [0 + 1j, 0 + 0j]
    ], dtype=complex)

    Z = csr_matrix([
        [1 + 0j, 0 + 0j],
        [0 + 0j, -1 + 0j]
    ], dtype=complex)

    # Calculate the Hamiltonian

    h = csr_matrix((2**sites, 2**sites), dtype=complex)

    for site in range(0, sites - 1):
        i1 = identity(2**site, format='csr', dtype=complex)
        i2 = identity(2 ** (sites - site - 2), format='csr', dtype=complex)
        h += (kron(kron(kron(i1, X), X), i2) + kron(kron(kron(i1, Y), Y), i2)
              + delta * kron(kron(kron(i1, Z), Z), i2))
        print(f"\rCompleted at {int(((site + 1) * 100) / (sites - 1))}%", end='')

    print("\nHamiltonian Matrix:")
    print(h)

    save_npz(f"Delta_{int(delta*10)}/H_{sites}Q", h)
    print("\nFile saved!")

    return


if __name__ == "__main__":

    print(f"Calculating the Hamiltonian...\n")

    start = time.time()
    h_calc(sites=2, delta=0.5)
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
