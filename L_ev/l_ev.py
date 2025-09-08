import numpy as np
from scipy.sparse import csr_matrix, kron, load_npz
from scipy.sparse.linalg import expm_multiply
from itertools import combinations
from math import comb
import time


def l_ev(sites, gamma, delta, noise):

    # Set the parameter variables
    s_time = 100
    dt = 0.025

    t_steps = int(s_time / dt) + 1
    t_points = np.linspace(0, s_time, t_steps)

    # Get initial state
    zero_vec = csr_matrix([[1 + 0j], [0 + 0j]])
    one_vec = csr_matrix([[0 + 0j], [1 + 0j]])
    bell = (1/np.sqrt(2)) * csr_matrix([[0 + 0j], [1 + 0j], [-1 + 0j], [0 + 0j]], dtype=complex)

    while True:
        state_str = input("Select the initial state:\nn: Neel state\np_u/d: polarised up/down state"
                          "\ns: for SU(2) sector state\nb: for Combinations of Bell States\nm: for mirror state\n"
                          "c: for cluster state\n")
        if state_str == "n":
            psi0 = csr_matrix([[1 + 0j]])
            for i in range(sites):
                psi0 = kron(psi0, zero_vec if i % 2 == 0 else one_vec)
            print("Neel state selected:")
            o_path = "Neel_State/"
            break
        elif state_str == "p_u":
            psi0 = csr_matrix([[1 + 0j]])
            for _ in range(sites):
                psi0 = kron(psi0, one_vec)
            print("Polarized Up state selected")
            o_path = "Polarized_State/"
            break
        elif state_str == "p_d":
            psi0 = csr_matrix([[1 + 0j]])
            for _ in range(sites):
                psi0 = kron(psi0, zero_vec)
            print("Polarized Down state selected")
            o_path = "Polarized_State/"
            break
        elif state_str == "s":
            psi0 = csr_matrix((2**sites, 1), dtype=complex)
            for positions in combinations(range(sites), int(sites / 2)):
                phi = csr_matrix([[1 + 0j]])
                for i in range(sites):
                    phi = kron(phi, one_vec if i in positions else zero_vec)
                psi0 += phi
            psi0 /= np.sqrt(comb(sites, int(sites/2)))
            print("SU(2) Sector state selected:")
            o_path = "SU(2)_Sec_State/"
            break
        elif state_str == "b":
            psi0 = csr_matrix([[1 + 0j]])
            for _ in range(int(sites/2)):
                psi0 = kron(psi0, bell)
            print("Bell state selected:")
            o_path = "Bell_State/"
            break
        elif state_str == "m":
            psi0 = csr_matrix([[1 + 0j]])
            for _ in range(int(sites / 2)):
                psi0 = kron(psi0, bell)

            psi_dense = psi0.toarray().reshape([2] * sites)
            perm = []
            for i in range(0, sites, 2):
                perm.append(i)
                perm.append(sites - 1 - i)
            psi_dense = np.transpose(psi_dense, axes=perm).reshape(-1)

            psi0 = csr_matrix(psi_dense.reshape(-1, 1))

            print("Mirror state selected:")
            o_path = "Mirror_State/"
            break
        elif state_str == "c":
            psi0 = csr_matrix([[1 + 0j]])
            for i in range(sites):
                psi0 = kron(psi0, zero_vec if i % 4 < 1.5 else one_vec)
                if i % 4 < 1.5:
                    print(0)
                else:
                    print(1)
            print("Cluster state selected:")
            o_path = "Cluster_State/"
            break
        else:
            print("Insert a valid state!\n")
            continue

    rho_vec = kron(psi0, psi0.conjugate())
    o_path += f"Gamma_{str(gamma).split('.')[1]}/Delta_{int(delta*10)}/{noise}_Noise/"

    # Load the L super operators and compute step time evolution
    print(1)
    L = (load_npz(f"../Liouvillians/Gamma_{str(gamma).split('.')[1]}/Delta_{int(delta*10)}/{noise}_Noise/L_{sites}Q.npz").tocsc())
    print(2)
    L_dt = L * dt
    print(3)

    # Compute Von Neumann Entropy and Logarithmic Negativity
    s_arr = []  # Von Neumann entropy
    ln_arr = []  # Logarithmic negativity

    d = 2 ** int(sites / 2)

    for t in range(t_steps):
        # --- Reshape rho_vec to matrix ---
        rho_t = rho_vec.toarray().reshape((2 ** sites, 2 ** sites), order='F')

        # ========== Von Neumann entropy ==========
        # Reshape into bipartite form
        rho_bip = rho_t.reshape(d, d, d, d)
        # Partial trace over subsystem B
        rho1 = np.trace(rho_bip, axis1=1, axis2=3)

        # Eigenvalues of reduced density matrix
        s_vals = np.linalg.eigvalsh(rho1)
        s_vals = s_vals[s_vals > 1e-12]
        entropy = -np.sum(s_vals * np.log2(s_vals))
        s_arr.append(entropy)

        # ========== Logarithmic Negativity ==========
        # Partial transpose on subsystem B
        rho_pt = np.transpose(rho_bip, axes=(0, 3, 2, 1))
        # Reshape back to 2D matrix
        rho_pt = rho_pt.reshape((2 ** sites, 2 ** sites), order='F')
        # Logarithmic negativity: log2(trace norm)
        ln_arr.append(np.log2(np.linalg.norm(rho_pt, ord='nuc')))

        # --- Time evolution ---
        rho_vec = expm_multiply(L_dt, rho_vec)

        print(f"\rProgress: {round(100 * t / t_steps, 4)} %", end='')

    # Save results
    np.savez(f"{o_path}lev_{sites}Q", t_points=t_points, s_arr=s_arr, ln_arr=ln_arr)


if __name__ == "__main__":

    print(f"Calculating entropy...\n")

    start = time.time()
    l_ev(sites=8, gamma=0.05, delta=1.0, noise="Z")
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
