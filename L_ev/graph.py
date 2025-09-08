import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
    data = [[np.load(file)["t_points"], np.load(file)["s_arr"], np.load(file)["ln_arr"]] for file in files]

    # Plot Data

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    a2 = 1.0
    a1 = 0.6

    x_shift = 0.25

    # ===== First Plot ======
    ax1.plot(data[0][0], data[0][1], color='C0', alpha=a1, label=r"$L_j=Z_j$")
    ax1.plot(data[1][0], data[1][1], color='C0', alpha=a2, label=r"$L_j=Y_j$")
    ax1.set_xlabel(r"Time $[1/J]$", labelpad=10)
    ax1.set_ylabel(r"$S_A$", labelpad=10)
    ax1.set_xlim(data[0][0][0]-0.25, data[0][0][-1]+0.2)
    ax1.set_ylim(-0.04, 4.04)

    ax1.text(-0.2, 1.1, "(a)", transform=ax1.transAxes, fontweight="bold", va='top', ha='left')

    ax1.legend(loc="lower right")

    # Create inset axes (zoom)
    axins = inset_axes(ax1, width="40%", height="30%", loc="center right")

    # Plot inside the inset
    axins.plot(data[0][0], data[0][1], color='C0', alpha=a1)
    axins.plot(data[1][0], data[1][1], color='C0', alpha=a2)

    x1, x2, y1, y2 = 10, 30.2, 3.7, 4.04
    axins.hlines(4, x1, x2, color="black", linestyle="--", linewidth=4, alpha=a1)
    axins.hlines(3.885, x1, x2, color="black", linestyle="--", linewidth=4, alpha=a1)

    # Set zoomed-in limits
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_yticks([3.8, 4.0])

    # Connect the inset with the main plot
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # ===== Second Plot ======
    ax2.plot(data[0][0], data[0][2], color='C1', alpha=a1, label=r"$L_j=Z_j$")
    ax2.plot(data[1][0], data[1][2], color='C1', alpha=a2, label=r"$L_j=Y_j$")
    ax2.set_xlabel(r"Time $[1/J]$", labelpad=10)
    ax2.set_ylabel(r"$\mathcal{N}_A$")
    ax2.set_xlim(data[0][0][0]-0.25, data[0][0][-1]+0.2)
    ax2.set_ylim(-0.028, 3.028)

    ax2.text(-0.2, 1.1, "(b)", transform=ax2.transAxes, fontweight="bold", va='top', ha='left')

    ax2.legend(loc="center right")

    plt.savefig(f"Y_vs_Z.pdf", format='pdf')
    plt.show()


if __name__ == "__main__":
    state = "Neel"
    gamma = "05"
    delta = "10"
    noise = "Z"
    n_sites = 8

    state_ = state
    gamma_ = gamma
    delta_ = delta
    noise_ = "Y"

    graph([
        f"{state}_State/Gamma_{gamma}/Delta_{delta}/{noise}_Noise/lev_{n_sites}Q.npz",
        f"{state_}_State/Gamma_{gamma_}/Delta_{delta_}/{noise_}_Noise/lev_{n_sites}Q.npz"
    ])
