from scipy.sparse import load_npz
from scipy.linalg import eigh
import numpy as np

import time


def eval_dec(sites, delta):

    h = load_npz(f"Delta_{int(delta*10)}/H_{sites}Q.npz").toarray()

    print("Calculating eigenvalues and eigenvectors...")
    evals, evecs = eigh(h)

    print("Eigenvalues and eigenvectors calculated.")

    # Save eigenvalues and eigenvectors as npy files
    np.save(f"Delta_{int(delta*10)}/evals_{sites}Q", evals)
    np.save(f"Delta_{int(delta*10)}/evecs_{sites}Q", evecs)
    print("Eigenvalues and eigenvectors saved!")

    return


if __name__ == "__main__":
    print(f"Calculating eigenvalues...\n")

    start = time.time()
    eval_dec(sites=8, delta=0.5)
    end = time.time()

    print(f"\nTime taken: {end - start:.4f} seconds")
