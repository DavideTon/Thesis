import numpy as np
from scipy.linalg import expm
import re
import time


def l_ev(l_file):

    # Set the parameter variables

    # Simulation time
    s_time = 3 * np.pi

    # Time Step
    dt = 0.001

    t_steps = int(s_time / dt) + 1
    t_points = np.linspace(0, s_time, t_steps)

    # Extract the number of Qubits

    match = re.search(r'_(\d+)Q', l_file)
    if not match:
        print("Insert a valid file name!")
        return
    n = int(match.group(1))

    # Ask for the initial state

    while True:
        state_str = input("Insert the initial state:\n")
        if re.fullmatch(r"[01]+", state_str) and len(state_str) == n:
            break
        print("Insert a valid state!\n")

    state_n = int(state_str, 2)

    psi0 = np.zeros(2**n, dtype=complex)
    psi0[state_n] = 1. + 0.j
    print(f"\nInitial state: {psi0}\n")

    rho = np.outer(psi0, psi0.conj())
    rho_vec = rho.flatten(order='F')  # 'F' = column stack

    # Load the L super operators

    L_Z = np.load(l_file + "_Z.npy")

    # Compute time evolution for a time step

    U_Z = expm(L_Z * dt)

    e_arr = []

    rho_vec_t = rho_vec.copy()

    for t in range(t_steps):
        d1, d2 = 2 ** int(n / 2), 2 ** (n - int(n / 2))
        rho_t = rho_vec_t.reshape((d1, d2, d1, d2), order='F')

        # Partial transposition on subsystem B
        rho_t = np.transpose(rho_t, axes=(0, 3, 2, 1))

        # Reshape back to 2D matrix
        rho_t = rho_t.reshape((2 ** n, 2 ** n), order='F')

        # Logarithmic negativity: log2 of trace norm
        e_arr.append(np.log2(np.linalg.norm(rho_t, ord='nuc')))

        rho_vec_t = U_Z @ rho_vec_t

        print(f"\rProgress: {round(100 * t / t_steps, 2)} %", end='')

    # Save results

    np.savez(f"Ent_{n}Q.npz", t_points=t_points, s_matrix=e_arr)


if __name__ == "__main__":
    for i in range(2, 7):
        start = time.time()
        l_ev(f"L_{i}Q")
        end = time.time()

        print(f"\nTime taken: {end - start:.4f} seconds")
