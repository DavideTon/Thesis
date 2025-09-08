import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def graph(sites):
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
    files = [f"Bell_State/Gamma_05/Delta_10/XXZ_Noise/lev_{n_sites}Q.npz" for n_sites in sites]
    data = [[np.load(file)["t_points"], np.load(file)["s_arr"], np.load(file)["ln_arr"]] for file in files]

    # Plot Data

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ===== First Plot ======
    stat_ent = np.array([d[2][-1] for d in data])
    m, q = np.polyfit(sites, stat_ent, 1)
    print(f"m = {m}\nq = {q}")

    ax1.plot(sites, m*sites+q, color="black", linestyle="--", linewidth=4, alpha=0.5, zorder=0)
    ax1.scatter(sites, stat_ent, s=50, color="C1", alpha=1, zorder=1)

    ax1.set_xlabel(r"$N$", labelpad=10)
    ax1.set_ylabel(r"$\mathcal{N}_A^{\mathrm{stat}}$", labelpad=10)
    ax1.set_xticks([2, 4, 6, 8, 10, 12])
    ax1.set_xlim(1.9, 12.1)
    ax1.set_ylim(0.992, 1.62)

    ax1.text(-0.2, 1.1, "(a)", transform=ax1.transAxes, fontweight="bold", va='top', ha='left')
    #ax1.legend(loc="upper right")

    # ===== Second Plot ======
    ratio = np.divide(stat_ent, np.divide(sites, 2))
    x_points = np.linspace(sites[0], sites[-1], 100)
    fit_points = np.divide(2*q, x_points) + 2*m

    ax2.plot(x_points, fit_points, color="black", linestyle="--", linewidth=4, alpha=0.5, zorder=0)
    ax2.scatter(sites, ratio, color="C1", s=50, zorder=1)

    ax2.set_xlabel(r"$N$", labelpad=10)
    ax2.set_ylabel(r"$\mathcal{N}_A^{\mathrm{stat}}/N_{\mathrm{Bell}}$", labelpad=10)
    ax2.set_xticks([2, 4, 6, 8, 10, 12])
    ax2.set_xlim(1.9, 12.1)
    ax2.set_ylim(0.257, 1.009)

    #ax2.set_yticks([])

    ax2.text(-0.2, 1.1, "(b)", transform=ax2.transAxes, fontweight="bold", va='top', ha='left')
    #ax2.legend(loc="lower right")

    plt.savefig(f"test.pdf", format='pdf')
    plt.show()


if __name__ == "__main__":

    graph(np.array([2, 6, 8, 10, 12]))
