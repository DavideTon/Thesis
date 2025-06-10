import numpy as np
from scipy.linalg import eigh
import re
import time


def t_ev(h_file):

    # Set the parameter variables

    # Simulation time
    s_time = 3 * np.pi

    # Time Step
    dt = 0.001

    t_steps = int(s_time / dt) + 1
    t_points = np.linspace(0, s_time, t_steps)

    # Extract the number of Qubits

    match = re.search(r'_(\d+)Q', h_file)

    if match:
        n = int(match.group(1))
    else:
        print("Insert a valid file name!")
        return

    # Ask for the initial state

    while True:
        state_str = input("Insert the initial state:\n")
        if not re.fullmatch(r"[01]+", state_str) or len(state_str) != n:
            print("Insert a valid state!\n")
            continue
        else:
            break

    state_n = int(state_str, 2)

    psi0 = np.zeros(2**n, dtype=complex)
    psi0[state_n] = 1. + 0.j

    print(f"\nInitial state: {psi0}\n")

    # Load the hamiltonian and calculate eigenvalues and eigenvectors

    h = np.load(h_file)

    e_vals, e_vecs = eigh(h)

    # Evolve the initial state and calculate the entropy

    d_exp = np.diag(np.exp(-1j * e_vals * dt))
    U = e_vecs @ d_exp @ e_vecs.conj().T

    s_matrix = np.zeros((int(n / 2), t_steps))

    psi_t = psi0.copy()

    for t in range(t_steps):
        for div in range(1, int(n / 2) + 1):
            # Calculate the reduced density matrix

            d1, d2 = 2 ** div, 2 ** (n - div)
            C = psi_t.reshape(d1, d2)
            rho1 = C @ C.conj().T

            # Calculate the Von Neumann Entropy
            s_vals = np.linalg.eigvalsh(rho1)
            s_vals = s_vals[s_vals > 1e-12]
            s_matrix[div - 1][t] = -np.sum(s_vals * np.log2(s_vals))

        psi_t = U @ psi_t

        print(f"Completed at {round((t * 100)/t_steps, 2)} %")

    s_matrix = np.array(s_matrix)

    # Save the results

    np.savez(f"ev_{n}Q.npz", t_points=t_points, s_matrix=s_matrix)

    return


if __name__ == "__main__":
    start = time.time()
    t_ev("H_6Q.npy")
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
