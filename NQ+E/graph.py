import numpy as np
import re
import matplotlib.pyplot as plt


def graph(ev_file):

    match = re.search(r'_(\d+)Q', ev_file)
    if not match:
        print("Insert a valid file name!")
        return
    n = int(match.group(1))

    # Load Data

    data = np.load(ev_file)
    t_points = data["t_points"]
    entropy = data["s_matrix"]

    # Plot Data

    plt.figure(figsize=(10, 7))

    plt.plot(t_points, entropy, linewidth=3)

    plt.xlabel("Time (1/J)", fontsize=16, labelpad=10)
    plt.xticks(fontsize=16)
    plt.ylabel("Entanglement", fontsize=16, labelpad=10)
    plt.yticks(fontsize=16)

    plt.title(f"Entanglement for {n} Qubits with Noise", fontsize=16, pad=15)

    # plt.legend(fontsize=16)
    plt.grid(True)

    plt.savefig(f"Ent_{n}Q+E.png", format='png', dpi=300)

    plt.show()


if __name__ == "__main__":
    for i in range(2, 7):
        graph(f"Ent_{i}Q.npz")
