import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def analytical_solution(x):
    p1 = 0.5 * (1 + np.cos(4 * x))
    p2 = 0.5 * (1 - np.cos(4 * x))

    p1 = np.clip(p1, 1e-12, 1.0)
    p2 = np.clip(p2, 1e-12, 1.0)

    s1 = -p1 * np.log2(p1)
    s2 = -p2 * np.log2(p2)

    return s1 + s2


def main():
    # Define Simulation Variables

    # Initial State
    psi0 = np.array([0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j])

    # Simulation time
    time = np.pi

    # Time Step
    dt = 0.001

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

    with open("2Q.txt", "w") as ofile:

        # Calculate the Hamiltonian for Heisenberg Model

        H = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
        ofile.write(f"Hamiltonian:\n{H}\n")

        # Calculate the evolution operator

        e_vals, e_vecs = eigh(H)
        ofile.write(f"\nEigenvalues:\n{e_vals}\n")
        ofile.write(f"\nEigenvectors:\n{e_vecs}\n")

        # Evolve th system

        ofile.write(f"\nInitial State:\n{psi0}\n")

        t_steps = int(time / dt) + 1

        s_arr = []
        norm_arr = []

    def ev(t_step: int) -> float:
        with open("2Q.txt", "a") as ofile_f:

            d_exp = np.diag(np.exp(-1j * e_vals * t_step * dt))
            U = e_vecs @ d_exp @ e_vecs.conj().T

            ofile_f.write("\n------------------------------------------------------")
            ofile_f.write(f"\nTime Step = {t_step}\n")
            ofile_f.write(f"\nTime = {t_step * dt}\n")

            psi = U @ psi0
            norm_arr.append(np.linalg.norm(psi))

            # Compute density matrices
            rho = np.outer(psi, np.conj(psi))
            ofile_f.write(f"\nDensity matrix:\n{rho}\n")

            # Compute first qubit reduced density matrix
            rho = rho.reshape(2, 2, 2, 2)
            rho1 = np.trace(rho, axis1=1, axis2=3)
            ofile_f.write(f"\nReduced Matrix\n{rho1}\n")

            # Compute Von Neumann Entropy
            s = 0
            e_vals1, _ = eigh(rho1)
            for e_val in e_vals1:
                s -= e_val * np.log2(e_val)
            ofile_f.write(f"\nVon Neumann Entropy s = {s}\n")

        return s

    for i in range(t_steps):
        s_arr.append(ev(i))

    # Plot the Entropy
    t_points = np.linspace(0, time, t_steps)
    res = np.array(s_arr) - analytical_solution(t_points)

    plt.figure(1)
    plt.plot(t_points, s_arr, linewidth=2.5, label="Simulated Data")
    plt.plot(t_points, analytical_solution(t_points), linestyle="--",
             label="Analytical Data")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.plot(t_points, res, label="Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.legend(loc='upper left')

    plt.figure(3)
    plt.plot(t_points, norm_arr, label="State Norm")
    plt.xlabel("Time")
    plt.ylabel("Norm")
    plt.legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    main()
