from scipy.sparse import csr_matrix, identity, kron, save_npz, load_npz
import time


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

HIGH = X + 1j * Y
LOW = X - 1j * Y


def l_calc(sites, gamma, delta, noise):

    if sites < 2 or sites > 17:
        print("Sites must be between 2 and 17.")
        return 1

    h = load_npz(f"../Hamiltonians/Delta_{int(delta*10)}/H_{sites}Q.npz")

    i = identity(2**sites, format='csr', dtype=complex)
    l_sup_op = -1j * (kron(i, h) - kron(h.T, i))

    not_unitary = csr_matrix((2**(2 * sites), 2**(2 * sites)), dtype=complex)

    # Create Lindblad operators embedded per site
    if noise == "X":
        j_ops = [X]
        mode = 0
    elif noise == "Y":
        j_ops = [Y]
        mode = 0
    elif noise == "Z":
        j_ops = [Z]
        mode = 0
    elif noise == "XXZ":
        j_ops = [X, Y, Z]
        mode = 1
    elif noise == "XXZ+Y":
        j_ops = None
        mode = 2
    elif noise == "X_Y_Z":
        j_ops = [X, Y, Z]
        mode = 3
    elif noise == "HL":
        j_ops = [HIGH, LOW]
        mode = 4
    else:
        print("Insert valid noise")
        return

    l_ops = []
    if mode == 0 or mode == 2:
        if mode == 2:
            j_ops = [Y]
        for j_op in j_ops:
            for site in range(sites):
                i1 = identity(2 ** site, format='csr', dtype=complex)
                i2 = identity(2 ** (sites - site - 1), format='csr', dtype=complex)
                l_embedded = kron(kron(i1, j_op), i2)
                l_ops.append(l_embedded)

    if mode == 1 or mode == 2:
        if mode == 2:
            j_ops = [X, Y, Z]
        for site in range(0, sites - 1):
            i1 = identity(2 ** site, format="csr", dtype=complex)
            i2 = identity(2 ** (sites - site - 2), format="csr", dtype=complex)

            # start with a clean CSR matrix
            l_embedded = csr_matrix((2 ** sites, 2 ** sites), dtype=complex)

            # XX + YY + delta*ZZ
            l_embedded += kron(kron(kron(i1, j_ops[0]), j_ops[0]), i2).tocsr()  # X X
            l_embedded += kron(kron(kron(i1, j_ops[1]), j_ops[1]), i2).tocsr()  # Y Y
            l_embedded += delta * kron(kron(kron(i1, j_ops[2]), j_ops[2]), i2).tocsr()  # δ Z Z

            # ensure CSR format before appending
            l_ops.append(l_embedded.tocsr())

    if mode == 3:
        for j_op in j_ops:
            l_op = csr_matrix((2**sites, 2**sites), dtype=complex)
            for site in range(sites):
                i1 = identity(2 ** site, format='csr', dtype=complex)
                i2 = identity(2 ** (sites - site - 1), format='csr', dtype=complex)
                l_op += kron(kron(i1, j_op), i2)
            l_ops.append(l_op)

    if mode == 4:
        for site in range(0, sites - 1):
            i1 = identity(2 ** site, format="csr", dtype=complex)
            i2 = identity(2 ** (sites - site - 2), format="csr", dtype=complex)

            # start with a clean CSR matrix
            l_embedded = csr_matrix((2 ** sites, 2 ** sites), dtype=complex)

            # XX + YY + delta*ZZ
            l_embedded += kron(kron(kron(i1, j_ops[0]), j_ops[0]), i2).tocsr()  # High High
            l_embedded += kron(kron(kron(i1, j_ops[1]), j_ops[1]), i2).tocsr()  # Low Low

            # ensure CSR format before appending
            l_ops.append(l_embedded.tocsr())

    if mode not in [0, 1, 2, 3, 4]:
        return 1

    print("Half Done!")

    for j, l_op in enumerate(l_ops):
        not_unitary += kron(l_op, l_op.conjugate())
        not_unitary -= 0.5 * (kron(i, (l_op.getH().dot(l_op)).T) +
                              kron(l_op.getH().dot(l_op), i))
        print(f"{(j + 1)}/{len(l_ops)} done!")

    l_sup_op += gamma * not_unitary
    save_npz(f"Gamma_{str(gamma).split('.')[1]}/Delta_{int(delta*10)}/{noise}_Noise/L_{sites}Q", l_sup_op)

    return


if __name__ == "__main__":

    print(f"Calculating the Liouvillian...\n")

    start = time.time()
    l_calc(sites=10, gamma=0.05, delta=1.0, noise="X_Y_Z")
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
