import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def graph(files):
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage[utf8]{inputenc}
            \usepackage{amsmath,amssymb,amsfonts}
            \usepackage{braket}
            \usepackage{eucal}
        """
    })

    font_size = 25

    # font sizes tuned relative to 11pt LaTeX
    plt.rcParams.update({
        "axes.labelsize": font_size,
        "font.size": font_size,
        "legend.fontsize": 20,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "lines.linewidth": 6,
    })

    # Load Data
    data = [[np.load(file)["t_points"], np.load(file)["s_matrix"]] for file in files]

    # Plot Data

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for i, entropy in enumerate(data[0][1]):
        ax1.plot(data[0][0], entropy, color='C0', alpha=1)
    ax1.set_xlabel(r"Time $[1/J]$", labelpad=10)
    ax1.set_ylabel(r"$S_A$", labelpad=10)
    ax1.set_xlim(data[0][0][0]-0.01, data[0][0][-1]+0.01)
    ax1.set_ylim(np.min(data[0][1][0])-0.01, np.max(data[0][1][0])+0.012)
    ax1.text(-0.2, 1.1, "(a)", transform=ax1.transAxes, fontweight="bold", va='top', ha='left')

    for i, entropy in enumerate(data[1][1]):
        ax2.plot(data[1][0], entropy, label=fr"$|A|={i + 1}$", color='C0', alpha=((2*i+5)/11))
    ax2.set_xlabel(r"Time $[1/J]$", labelpad=10)
    ax2.set_xlim(data[1][0][0]-0.05, data[1][0][-1]+0.1)
    ax2.set_ylim(np.min(data[1][1][-1])-0.01, 3+0.01)
    ax2.text(-0.2, 1.1, "(b)", transform=ax2.transAxes, fontweight="bold", va='top', ha='left')

    ax2.legend(loc="lower right", ncol=2)

    plt.savefig(f"unitary.pdf", format='pdf')

    plt.show()


if __name__ == "__main__":
    delta = 1.0
    sites = 2

    delta_ = delta
    sites_ = 8

    graph([f"Delta_{int(delta*10)}/ev_{sites}Q.npz", f"Delta_{int(delta_*10)}/ev_{sites_}Q.npz"])
