import numpy as np
from scipy.linalg import expm
import re
import time


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


def l_tests(l_file):

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

    L_X = np.load(l_file + "_X.npy")
    L_Y = np.load(l_file + "_Y.npy")
    L_Z = np.load(l_file + "_Z.npy")

    # Compute time evolution for a time step

    U_X = expm(L_X * dt)
    # U_Y = expm(L_Y * dt)
    # U_Z = expm(L_Z * dt)

    s_arr = []
    x_m_arr = []
    y_m_arr = []

    rho_vec_t = rho_vec.copy()

    for i in range(t_steps):
        rho_t = rho_vec_t.reshape((2 ** n, 2 ** n), order='F')

        # d1, d2 = 2 ** int(n / 2), 2 ** (n - int(n / 2))
        d1, d2 = 2, 2 ** (n - 1)
        rho_t = rho_t.reshape(d1, d2, d1, d2)
        rho1 = np.trace(rho_t, axis1=1, axis2=3)

        x_m_arr.append(np.trace(rho1 @ X))
        y_m_arr.append(np.trace(rho1 @ Y))

        """s_vals = np.linalg.eigvalsh(rho1)
        s_vals = s_vals[s_vals > 1e-12]
        entropy = -np.sum(s_vals * np.log2(s_vals))

        s_arr.append(entropy)"""

        rho_vec_t = U_X @ rho_vec_t

        print(f"\rProgress: {round(100 * i / t_steps, 2)} %", end='')

    # Save results
    # np.savez(f"X_{n}Q.npz", t_points=t_points, s_matrix=s_arr)
    np.savez(f"X_{n}Q.npz", t_points=t_points, s_matrix=x_m_arr)
    np.savez(f"Y_{n}Q.npz", t_points=t_points, s_matrix=y_m_arr)


if __name__ == "__main__":
    for i in range(2, 7):
        start = time.time()
        l_tests(f"L_{i}Q")
        end = time.time()

        print(f"\nTime taken: {end - start:.4f} seconds")
