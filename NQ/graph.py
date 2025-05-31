import numpy as np
import matplotlib.pyplot as plt


def graph(ev_file):

    # Load Data

    data = np.load(ev_file)
    t_points = data["t_points"]
    s_matrix = data["s_matrix"]

    # Plot Data

    for i, entropy in enumerate(s_matrix):
        plt.plot(t_points, entropy, label=f"Cut at {i + 1}")

    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.title("Entanglement Entropy over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    graph("ev_10Q.npz")
