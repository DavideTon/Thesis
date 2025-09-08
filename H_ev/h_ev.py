import numpy as np
from scipy.sparse import csr_matrix, kron, diags
import re
import time


def h_ev(sites, delta):

    # Simulation parameters
    s_time = 3 * np.pi / 4
    dt = 0.001

    t_steps = int(s_time / dt) + 1
    t_points = np.linspace(0, s_time, t_steps)

    # Get initial state
    while True:
        state_str = input("\nInsert the initial state (press Enter for default):\n")
        if state_str == "":
            state_str = "".join("0" if i % 2 == 0 else "1" for i in range(sites))
            print(f"Default state selected: {state_str}")
            break
        if not re.fullmatch(r"[01]+", state_str) or len(state_str) != sites:
            print("Insert a valid state!\n")
            continue
        else:
            break

    # Construct initial state as a sparse vector
    psi0 = csr_matrix([[1 + 0j]])
    zero_vec = csr_matrix([[1 + 0j], [0 + 0j]])
    one_vec = csr_matrix([[0 + 0j], [1 + 0j]])

    for bit in state_str:
        psi0 = kron(psi0, zero_vec if bit == '0' else one_vec)

    print(f"\nInitial state (sparse shape {psi0.shape}):\n")

    # Load Hamiltonian eigenvalues and eigenvectors
    evals = np.load(f"../Hamiltonians/Delta_{int(delta*10)}/evals_{sites}Q.npy")
    evecs = csr_matrix(np.load(f"../Hamiltonians/Delta_{int(delta*10)}/evecs_{sites}Q.npy"))

    # Compute time evolution operator
    d_exp = diags(np.exp(-1j * evals * dt), 0, format='csr')
    U = evecs.dot(d_exp).dot(evecs.getH())

    # Track entropy over time
    s_matrix = np.zeros((int(sites / 2), t_steps))

    for t in range(t_steps):
        for div in range(1, int(sites / 2) + 1):
            d1, d2 = 2 ** div, 2 ** (sites - div)

            # Reshape sparse state to dense for partial trace
            C = psi0.toarray().reshape(d1, d2)
            rho1 = C @ C.conj().T

            # Entropy (Von Neumann)
            s_vals = np.linalg.eigvalsh(rho1)
            s_vals = s_vals[s_vals > 1e-12]
            s_matrix[div - 1][t] = -np.sum(s_vals * np.log2(s_vals))

        psi0 = U.dot(psi0)  # Evolve state

        if t % (t_steps // 10) == 0 or t == t_steps - 1:
            print(f"\rCompleted: {round((t * 100) / t_steps, 2)}%", end="")

    # Save output
    np.savez(f"Delta_{int(delta*10)}/ev_{sites}Q", t_points=t_points, s_matrix=s_matrix)


if __name__ == "__main__":

    print(f"Calculating entropy...\n")

    start = time.time()
    h_ev(sites=2, delta=1.0)
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
