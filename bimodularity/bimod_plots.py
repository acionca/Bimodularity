# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import gaussian_kde, spearmanr
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.linear_model import LinearRegression
import networkx as nx

import os.path as op
from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import (
    to_rgb,
    to_rgba,
    ListedColormap,
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.patheffects import withStroke
from matplotlib.colorbar import Colorbar
from matplotlib.patches import ArrowStyle

import nibabel as nib
from dipy.viz import window, actor
from dipy.tracking.metrics import spline
from dipy.tracking.streamline import (
    set_number_of_points,
    select_random_set_of_streamlines,
)

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh

from . import dgsp
from . import bundle
from .palettes import (
    CLUSTER,
    CLUSTER_CB,
    CLUSTER_SOFT,
    DIV_RB,
    EXTENDED_NCAR,
    PASTEL,
    DIV_RB_SILVER,
    SHORT_NCAR,
)


def get_all_cmaps():
    all_colors = [
        CLUSTER,
        CLUSTER_CB,
        CLUSTER_SOFT,
        DIV_RB,
        DIV_RB_SILVER,
        EXTENDED_NCAR,
        SHORT_NCAR,
    ]
    cmap_names = [
        "cluster_palette",
        "cluster_palette_cb",
        "cluster_palette_soft",
        "div_rb",
        "div_rb_silver",
        "extended_ncar",
        "short_ncar",
    ]

    fig, axes = plt.subplots(figsize=(8, 2))
    x_plot = np.linspace(0, 1, 256)

    cmaps = {}
    for i, (name, cmap) in enumerate(zip(cmap_names, all_colors)):
        cmap = LinearSegmentedColormap.from_list("colors", cmap)
        cmaps[name] = cmap
        axes.scatter(
            x_plot, np.ones_like(x_plot) * i, c=cmap(x_plot), marker="o", s=100
        )
    axes.set_yticks(np.arange(len(cmap_names)), labels=cmap_names)
    axes.set_xticks([])

    return cmaps


def plot_palette():
    palette_rgb = [to_rgb(color) for color in PASTEL]
    plt.scatter([0, 1, 2, 3], [0] * 4, c=palette_rgb, s=400, edgecolors="k", lw=2)


def get_custom_cmap() -> ListedColormap:

    palette_rgb = [to_rgb(color) for color in PASTEL]

    custom_cmap = ListedColormap([(0, 0, 0), (1, 1, 1)] + palette_rgb)

    return custom_cmap


def add_cbar(fig: Figure, ax: Axes, **kwargs) -> (Figure, Axes, Colorbar):
    """Add a colorbar to an existing figure/axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to get the colorbar from
    ax : matplotlib.axes.Axes
        axes to add the colorbar to

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.colorbar.Colorbar)
        tuple of figure and axes with the colorbar added
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    return fig, ax, cbar


def draw_cbar(
    fig,
    cmap,
    ax_pos,
    ticks=[0, 5, 10],
    labels=["Receiving", "Both", "Sending"],
    fontsize: int = 18,
    orientation: str = "horizontal",
    **kwargs,
):
    cax = fig.add_axes(ax_pos)  # [left, bottom, width, height]

    norm = Normalize(vmin=0, vmax=10)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=cax, orientation=orientation, **kwargs)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels, fontsize=fontsize)

    return cb


def draw_self_loop(
    axes: Axes,
    posx: float,
    posy: float,
    size: float,
    rad: float = 0.4,
    offset: float = 0.01,
    mutation_scale: int = 20,
    onearrow: bool = True,
) -> Axes:

    all_arrows_pos = [
        (posx, posy, posx + size, posy, rad),
        (posx + size - offset, posy - offset, posx + size - offset, posy + size, rad),
        (posx + size, posy + size - offset, posx, posy + size - offset, rad),
        (posx + offset, posy + size, posx + offset, posy - offset, rad),
    ]

    for pos_i, arr_pos in enumerate(all_arrows_pos):
        px, py, sx, sy, rad = arr_pos

        style = "-"
        if onearrow:
            if pos_i == len(all_arrows_pos) - 1:
                style = "-|>"
        elif pos_i % 2:
            style = "-|>"

        arrow = FancyArrowPatch(
            (px, py),
            (sx, sy),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=2,
            color="k",
            connectionstyle=f"arc3,rad={rad}",
        )

        axes.add_patch(arrow)

    return axes


def draw_smaller_self_loop(
    axes: Axes,
    posx: float,
    posy: float,
    size: float,
    rad: float = 0.4,
    offset: float = 0.01,
    mutation_scale: int = 20,
    onearrow: bool = True,
) -> Axes:

    all_arrows_pos = [
        (posx, posy, posx, posy - size, rad),
        (posx - offset, posy - size + offset, posx + size, posy - size, rad),
        (posx + size - offset, posy - size - offset, posx + size, posy, rad),
    ]

    all_arrows_pos = [
        (posx, posy, posx, posy - size, rad),
        (
            posx - offset,
            posy - size + offset,
            posx + size + offset,
            posy - size + offset,
            rad,
        ),
        (posx + size, posy - size, posx + size, posy, rad),
    ]

    for pos_i, arr_pos in enumerate(all_arrows_pos):
        px, py, sx, sy, rad = arr_pos

        style = "-"
        if onearrow:
            if pos_i == len(all_arrows_pos) - 1:
                style = "-|>"
        elif pos_i % 2:
            style = "-|>"

        arrow = FancyArrowPatch(
            (px, py),
            (sx, sy),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=2,
            color="k",
            connectionstyle=f"arc3,rad={rad}",
        )

        axes.add_patch(arrow)

    return axes


def plot_community_scheme(
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    title_letter: str = "",
    fontscale: float = 1.2,
    com_names: Optional[np.ndarray] = None,
    com_names_yoffset: float = 0,
    x_names: [str, str] = ["Sending 1", "Sending 2"],
    y_names: [str, str] = ["Receiving 1", "Receiving 2"],
    arrow_colors: Optional[np.ndarray] = None,
    plot_cycle: bool = False,
    override_title: Optional[str] = None,
) -> Optional[Axes]:
    # First plot (A)
    xy_pos = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    pos = {i: xy_pos[i] for i in range(4)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = "none"
    if use_cmap:
        palette_indices = [3, 2, 1, 0]
        colors = [to_rgba(PASTEL[color_i], alpha=0.5) for color_i in palette_indices]
        # colors = [to_rgba(color, alpha=0.5) for color in PASTEL]

    ax.scatter(
        xy_pos[:, 0],
        xy_pos[:, 1],
        marker="s",
        s=1.5e4,
        color=colors,
        edgecolor="black",
        linewidth=2,
        zorder=1,
    )

    # Definition of arrows
    mid_point = 0.5

    starting_points = [
        (-0.2, -mid_point),
        (mid_point, -0.2),
        (0.2, mid_point),
        (-mid_point, 0.2),
        (0.3, -0.2),
        (-0.3, 0.2),
    ]
    ending_points = [
        (0.2, -mid_point),
        (mid_point, 0.2),
        (-0.2, mid_point),
        (-mid_point, -0.2),
        (-0.2, 0.3),
        (0.2, -0.3),
    ]

    if plot_cycle:
        starting_points = starting_points[:-2]
        ending_points = ending_points[:-2]

    if arrow_colors is None:
        arrow_colors = ["black"] * len(starting_points)

    for start, end, col in zip(starting_points, ending_points, arrow_colors):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=30,
            linewidth=2,
            facecolor=col,
            edgecolor=col,
        )
        ax.add_patch(arrow)

    # Drawing the rectangles (communities)
    if com_names is None:
        com_names = ["$S_4$", "$S_3$", "$S_2$", "$S_1$"]

    for pos, com_name in zip(xy_pos, com_names):
        ax.text(
            pos[0],
            pos[1] + com_names_yoffset,
            com_name,
            fontsize=24 * fontscale,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Ticks parameters
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )

    if override_title is None:
        override_title = "Community structure scheme"
    ax.set_title(override_title, fontsize=20 * fontscale)

    ax.set_xticks(
        np.linspace(-0.5, 0.5, 2),
        labels=x_names,
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.linspace(-0.5, 0.5, 2),
        labels=y_names,
        fontsize=18 * fontscale,
        rotation=90,
        va="center",
    )
    ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Sending communities", fontsize=18 * fontscale)
    ax.set_ylabel("Receiving communities", fontsize=18 * fontscale)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)

    return ax


def plot_adjacency(
    matrix: np.ndarray,
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
) -> Optional[Axes]:

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    plot_adj = matrix.copy()

    n_per_com = matrix.shape[0] // 4

    cmap = "binary_r"
    if use_cmap:
        cmap = get_custom_cmap()
        for i in range(4):
            plot_adj[
                i * n_per_com : (i + 1) * n_per_com, i * n_per_com : (i + 1) * n_per_com
            ] = matrix[
                i * n_per_com : (i + 1) * n_per_com, i * n_per_com : (i + 1) * n_per_com
            ] * (
                i + 2
            )

    ax.imshow(plot_adj, cmap=cmap, interpolation="none")
    # axes[1].imshow(graph, cmap="binary_r")

    # Parameters B
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )
    if override_title is None:
        override_title = "Graph adjacency matrix"
    ax.set_title(override_title, fontsize=20 * fontscale)

    ax.set_xticks(
        np.arange(n_per_com // 2, matrix.shape[0] + 1, n_per_com),
        labels=["$S_1$", "$S_2$", "$S_3$", "$S_4$"],
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.arange(n_per_com // 2, matrix.shape[0] + 1, n_per_com),
        labels=["$S_1$", "$S_2$", "$S_3$", "$S_4$"],
        fontsize=18 * fontscale,
    )
    return ax


def plot_spectrum(
    matrix: np.ndarray,
    vector_id: int = 0,
    show_n_eig: int = 10,
    write_s: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    split_ax: bool = False,
    fix_negative: bool = True,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
    normalize_s: bool = False,
    **kwargs,
) -> Axes:

    # Building the modularity matrix
    modmat = dgsp.modularity_matrix(matrix, null_model="outin")

    U, S, Vh = dgsp.sorted_SVD(modmat, fix_negative=fix_negative)
    V = Vh.T

    r_squared = S[vector_id] ** 2 / (S**2).sum()
    print(f"s (norm) = {S[vector_id] / (2 * matrix.sum())}")
    print(f"R^2 = {r_squared}")

    n_nodes = matrix.shape[0]

    # Starting the plot
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if split_ax:
        gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax, wspace=0.04)
        axes = [fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])]
    else:
        axes = [ax] * 2

    node_colors = "k"
    if not fix_negative:
        node_colors = []
        for i in range(show_n_eig):
            # angle = V[:, i] @ U[:, i].T
            angle = U[:, i] @ V[:, i]
            # node_colors.append(angle)
            col_id = 1 - (np.sign(angle) + 1) // 2
            node_colors.append(PASTEL[col_id.astype(int)])

    # Figure A (eigenvalues)

    # Clear the 1st subplot (in the background)

    if split_ax:
        ax.set_xticks([0], labels=[0], fontsize=16 * fontscale, fontdict={"color": "w"})
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=True,
            labelsize=16 * fontscale,
            color="w",
        )

    if split_ax:
        for _, s in ax.spines.items():
            s.set_visible(False)

    if normalize_s:
        S = S / (2 * matrix.sum())

    # axes[0].plot(S, marker="o", lw=4, ms=10, color="k")
    axes[0].plot(S, marker="o", lw=4, ms=20, markeredgewidth=1, color="k")

    if split_ax:
        axes[1].plot(S, marker="o", lw=4, ms=10, color="k")

    if not fix_negative:
        axes[0].scatter(
            np.arange(show_n_eig),
            S[:show_n_eig],
            color=node_colors,
            # color=np.sign(node_colors),
            # cmap="coolwarm",
            edgecolors="k",
            lw=2,
            s=220,
            zorder=2,
        )

    x_id = np.arange(n_nodes)[vector_id]
    if (vector_id >= 0) and (vector_id < show_n_eig):
        if write_s:
            axes[0].plot(
                x_id,
                S[vector_id],
                marker="s",
                lw=4,
                ms=22,
                # color=PASTEL[0],
                color=node_colors[vector_id],
                markeredgecolor="k",
                markeredgewidth=4,
            )
            axes[0].text(
                x_id + 1,
                S[vector_id],
                f"$\\mu_{{{x_id+1}}}={S[vector_id]:2.2f}, R^2={r_squared:1.2f}$",
                # f"$s_{{{x_id}}}={S[vector_id]:2.2f}, R^2={r_squared:1.2f}$",
                fontsize=18 * fontscale,
                ha="left",
                va="center",
                path_effects=[
                    withStroke(
                        linewidth=3 * fontscale,
                        foreground="w",
                        alpha=0.8,
                    )
                ],
            )
    else:
        axes[1].plot(
            x_id,
            S[vector_id],
            marker="s",
            lw=4,
            ms=14,
            # color=PASTEL[0],
            color=node_colors[vector_id],
            markeredgecolor="k",
            markeredgewidth=2,
        )

    # Plotting the horizontal line at y=0
    ax.set_ylim(axes[0].get_ylim())
    ax.plot([-10, 10], [0] * 2, color="k", ls="--", alpha=0.8, zorder=5)

    if split_ax:
        axes[0].plot([-2, show_n_eig], [0] * 2, color="k", ls="--", alpha=0.8)
        axes[1].plot(
            [len(S) - show_n_eig, len(S) + 2], [0] * 2, color="k", ls="--", alpha=0.8
        )
    else:
        # ax.set_ylim(S[:show_n_eig].min() - 1, S[:show_n_eig].max() + 1)
        ax.set_ylim(0.9 * S[:show_n_eig].min(), 1.1 * S[:show_n_eig].max())

    # From matplotlib examples
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )

    if split_ax:
        axes[0].plot([1], [0], transform=axes[0].transAxes, **kwargs)
        axes[1].plot([0], [0], transform=axes[1].transAxes, **kwargs)

    # Parameters
    if override_title is None:
        override_title = "Eigenspectrum"
    ax.set_title(override_title, fontsize=20 * fontscale)
    axes[0].set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )

    axes[0].set_xlim(-2, show_n_eig - 0.5)

    n_ticks = show_n_eig // 3
    axes[0].set_xticks(
        np.arange(0, show_n_eig, n_ticks), labels=np.arange(0, show_n_eig, n_ticks) + 1
    )

    if split_ax:
        axes[1].set_xlim(len(S) - show_n_eig + 0.5, len(S) + 2)
        axes[1].set_xticks(
            np.arange(len(S) - show_n_eig + n_ticks, len(S) + 2, n_ticks)
        )

    axes[0].spines.right.set_visible(False)

    if split_ax:
        axes[1].spines.left.set_visible(False)

    ax.set_xlabel("Indices $n$", fontsize=18 * fontscale)
    axes[0].set_ylabel("Singular Values $\\mu_n$", fontsize=18 * fontscale)
    # axes[0].set_ylabel("Singular values $s_i$", fontsize=18 * fontscale)

    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[:].set_linewidth(2)

        # ax.set_xticks(np.linspace(-.1, .1, 3), labels=np.linspace(-.1, .1, 3), fontsize=18*fontscale)
        # ax.set_yticks(np.linspace(-.1, .1, 3), labels=np.linspace(-.1, .1, 3), fontsize=18*fontscale)

    axes[0].tick_params(
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelsize=16 * fontscale,
        width=2,
    )

    if split_ax:
        axes[1].tick_params(
            left=False,
            bottom=True,
            labelleft=False,
            labelbottom=True,
            labelsize=16 * fontscale,
        )

    return ax


def plot_graph_embedding(
    matrix: np.ndarray,
    vector_id: int = 0,
    n_com: int = 4,
    ax: Optional[Axes] = None,
    use_cmap: bool = True,
    cmap: str = "plasma",
    static_color: str = "tab:blue",
    write_label: bool = False,
    write_var: bool = False,
    label_lw: int = 3,
    directed_edges: bool = True,
    edge_alpha: float = 0.02,
    fontscale: float = 1.2,
    title_letter: str = "",
    override_title: Optional[str] = None,
    node_clusers: Optional[np.ndarray] = None,
    **kwargs,
) -> Axes:

    # Building the modularity matrix
    modmat = dgsp.modularity_matrix(matrix, null_model="outin")

    U, S, Vh = dgsp.sorted_SVD(modmat, **kwargs)
    V = Vh.T

    n_nodes = matrix.shape[0]
    n_per_com = n_nodes // n_com

    graph_pos = {i: (U[i, vector_id], V[i, vector_id]) for i in range(n_nodes)}
    labels = {i: "" for i in range(n_nodes)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if directed_edges:
        graph = nx.DiGraph(matrix)
    else:
        graph = nx.Graph(matrix)

    nx.draw_networkx_edges(graph, pos=graph_pos, alpha=edge_alpha, ax=ax)

    if use_cmap:
        if node_clusers is None:
            palette_rgb = [to_rgb(color) for color in PASTEL]
            colors = [palette_rgb[i // n_per_com] for i in np.arange(n_nodes)]
        else:
            try:
                cmap = plt.get_cmap(cmap, int(node_clusers.max() + 1))
                colors = [cmap(int(i)) for i in node_clusers]
            except ValueError:
                colors = [cmap] * n_nodes
    else:
        colors = static_color

    ax.scatter(
        U[:, vector_id],
        V[:, vector_id],
        s=200,
        color=colors,
        edgecolor="k",
        linewidth=2,
        zorder=2,
    )

    if write_var:
        expl_var = S[vector_id] ** 2 / np.sum(S**2)
        ax.text(
            0.05,
            0.90,
            f"$R^2={expl_var:1.3f}$",
            transform=ax.transAxes,
            fontsize=16 * fontscale,
            path_effects=[
                withStroke(
                    linewidth=label_lw,
                    foreground="w",
                    alpha=0.8,
                )
            ],
        )

        angle = (V.T @ U)[vector_id, vector_id]
        ax.text(
            0.05,
            0.85,
            f"$\\mathbf{{v}}^T\\mathbf{{u}}={angle:1.2f}$",
            transform=ax.transAxes,
            fontsize=16 * fontscale,
            path_effects=[
                withStroke(
                    linewidth=label_lw,
                    foreground="w",
                    alpha=0.8,
                )
            ],
        )

    if write_label:
        for com_i in range(n_com):
            mean_u = np.mean(U[com_i * n_per_com : (com_i + 1) * n_per_com, vector_id])
            mean_v = np.mean(V[com_i * n_per_com : (com_i + 1) * n_per_com, vector_id])
            ax.text(
                mean_u,
                mean_v,
                f"$S_{com_i+1}$",
                fontsize=26 * fontscale,
                color="k",
                # color="w",
                # color=PASTEL[com_i],
                fontweight="bold",
                path_effects=[
                    withStroke(
                        linewidth=label_lw,
                        foreground="w",
                        alpha=0.8,
                        # foreground="k",
                        # foreground=PASTEL[com_i],
                    )
                ],
                ha="center",
                va="center",
                zorder=4,
            )
            # ax.scatter(
            #    mean_u, mean_v, s=1000, marker="o", color="w", alpha=0.8, zorder=3
            # )

    # Parameters
    ax.set_title(
        title_letter,
        loc="left",
        fontsize=22 * fontscale,
        fontdict={"fontweight": "bold"},
    )
    if override_title is None:
        override_title = f"Bimodularity Embedding $(n={{{vector_id+1}}})$"
    ax.set_title(override_title, fontsize=20 * fontscale)
    ax.tick_params(
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelsize=16 * fontscale,
        width=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[:].set_linewidth(2)

    ax.set_xticks(
        np.linspace(-0.1, 0.1, 3),
        labels=np.linspace(-0.1, 0.1, 3),
        fontsize=18 * fontscale,
    )
    ax.set_yticks(
        np.linspace(-0.1, 0.1, 3),
        labels=np.linspace(-0.1, 0.1, 3),
        fontsize=18 * fontscale,
    )

    ax.set_xlabel(
        f"Left Singular Vector $\\mathbf{{u}}_{{{vector_id+1}}}$",
        fontsize=18 * fontscale,
    )
    ax.set_ylabel(
        f"Right Singular Vector $\\mathbf{{v}}_{{{vector_id+1}}}$",
        fontsize=18 * fontscale,
    )

    return ax


def plot_bicommunity(
    adjacency,
    send_com,
    receive_com,
    fig=None,
    axes=None,
    graph_pos=None,
    lw=1,
    edge_alpha=0.05,
    draw_arrows=False,
    cmap=None,
    s=80,
    s_scale=5 / 8,
):
    if fig is None:
        fig, axes = plt.subplots(figsize=(10, 10))

    if graph_pos is None:
        graph_pos = nx.spring_layout(nx.DiGraph(adjacency))

    node_pos = np.array(list(graph_pos.values())).T

    if draw_arrows:
        nx.draw_networkx_edges(
            nx.DiGraph(adjacency), pos=graph_pos, alpha=edge_alpha, ax=axes
        )
    else:
        nx.draw_networkx_edges(
            nx.Graph(adjacency), pos=graph_pos, alpha=edge_alpha, ax=axes
        )

    is_in_none = np.logical_and(send_com == 0, receive_com == 0)
    is_in_both = np.logical_and(send_com > 0, receive_com > 0)
    send_only = np.logical_and(send_com > 0, receive_com == 0)
    receive_only = np.logical_and(send_com == 0, receive_com > 0)

    if cmap is None:
        cmap = "RdBu_r"

    axes.scatter(
        node_pos[0, is_in_none],
        node_pos[1, is_in_none],
        s=s * s_scale,
        c="tab:gray",
        edgecolor="k",
        linewidth=lw / 2,
        zorder=2,
    )
    axes.scatter(
        node_pos[0, send_only],
        node_pos[1, send_only],
        s=s,
        c=send_com[send_only],
        # cmap="RdBu_r",
        cmap=cmap,
        edgecolor="k",
        marker="s",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )
    axes.scatter(
        node_pos[0, receive_only],
        node_pos[1, receive_only],
        s=s,
        c=-receive_com[receive_only],
        # cmap="RdBu_r",
        cmap=cmap,
        edgecolor="k",
        marker="D",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )
    axes.scatter(
        node_pos[0, is_in_both],
        node_pos[1, is_in_both],
        s=s,
        c=send_com[is_in_both] - receive_com[is_in_both],
        cmap=cmap,
        # c="w",
        edgecolor="k",
        marker="o",
        linewidth=lw,
        zorder=2,
        vmin=-1,
        vmax=1,
    )

    return fig, axes


def plot_all_bicommunity(
    adjacency,
    send_com,
    receive_com,
    fig=None,
    axes=None,
    layout="embedding",
    scatter_only=False,
    titles=None,
    nrows=1,
    ncols=None,
    draw_legend=True,
    legend_on_ax=False,
    cmap=None,
    gspec_wspace=0,
    gspec_hspace=0.05,
    right_pad=None,
    **kwargs,
):
    if ncols is None:
        ncols = len(send_com) // nrows + (len(send_com) % nrows)

    if right_pad is not None:
        com_gs = GridSpecFromSubplotSpec(
            nrows=nrows,
            ncols=ncols + 1,
            # ncols=len(send_com) // nrows,
            subplot_spec=axes.get_subplotspec(),
            wspace=gspec_wspace,
            hspace=gspec_hspace,
            width_ratios=[1] * ncols + [right_pad],
        )
        com_axes = [
            fig.add_subplot(gs)
            for gi, gs in enumerate(com_gs)
            if (gi + 1) % (ncols + 1)
        ]
    else:
        com_gs = GridSpecFromSubplotSpec(
            nrows=nrows,
            ncols=ncols,
            # ncols=len(send_com) // nrows,
            subplot_spec=axes.get_subplotspec(),
            wspace=gspec_wspace,
            hspace=gspec_hspace,
        )
        com_axes = [fig.add_subplot(gs) for gs in com_gs]
    # axes.set_visible(False)
    axes.axis("off")

    if isinstance(layout, dict):
        graph_pos = layout.copy()
    elif layout == "embedding":
        U, _, Vh = dgsp.sorted_SVD(dgsp.modularity_matrix(adjacency))
        V = Vh.T

        if np.allclose(adjacency, adjacency.T):
            graph_pos = {i: (U[i, 0], V[i, 1]) for i, _ in enumerate(adjacency)}
        else:
            graph_pos = {i: (U[i, 0], V[i, 0]) for i, _ in enumerate(adjacency)}
    else:
        graph_pos = nx.spring_layout(nx.DiGraph(adjacency))

    if titles is None:
        titles = [f"Com {i+1}" for i in range(len(send_com))]

    for i, title in enumerate(titles):
        com_axes[i].set_title(title, fontsize=20)

    if cmap is None:
        cmap = plt.get_cmap("RdBu")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap, 5)

    for i, (send, receive) in enumerate(zip(send_com, receive_com)):

        com_axes[i].set_facecolor("none")

        if scatter_only:
            com_axes[i].scatter(send, receive)
        else:
            plot_bicommunity(
                adjacency,
                send,
                receive,
                fig=fig,
                axes=com_axes[i],
                graph_pos=graph_pos,
                cmap=cmap,
                **kwargs,
            )

        com_axes[i].spines[:].set_visible(False)
    com_axes[-1].spines[:].set_visible(False)
    com_axes[-1].tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )

    leg_titles = ["Send", "Both", "Receive"]
    markers = ["s", "o", "D"]

    custom_legend = [
        Line2D(
            [0],
            [0],
            color="w",
            marker=markers[i],
            markeredgecolor="k",
            markerfacecolor=cmap(1 + i),
            markersize=10,
        )
        for i, _ in enumerate(leg_titles)
    ]

    if draw_legend:
        if legend_on_ax:
            axes.legend(
                custom_legend,
                leg_titles,
                loc="lower center",
                ncol=3,
                fontsize=20,
                bbox_to_anchor=(0.5, -0.12),
            )
        else:
            fig.legend(
                custom_legend, leg_titles, loc="lower center", ncol=3, fontsize=20
            )

    return fig, axes


def plot_bicommunity_types(
    sending_communities,
    receiving_communities,
    types,
    type_colors=None,
    titles=None,
    fontsize=14,
    fig=None,
    axes=None,
):

    if type_colors is None:
        cmap = plt.get_cmap("Set1")
        type_colors = {t: cmap(i) for i, t in enumerate(types.unique())}

    if titles is None:
        titles = [f"Community {com_i+1}" for com_i in range(len(sending_communities))]

    if axes is None:
        fig, axes = plt.subplots(figsize=(5 * len(sending_communities), 5))

    types_series = pd.Series(types)

    for com_i, (s, r) in enumerate(zip(sending_communities, receiving_communities)):
        types_send = types_series[s > 0].value_counts()
        types_receive = types_series[r > 0].value_counts()

        axes[com_i].set_visible(False)
        gs_com = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=axes[com_i].get_subplotspec(), hspace=0.05, wspace=0
        )
        axes_com = [fig.add_subplot(gs_com[i]) for i in range(2)]

        axes_com[0].set_ylabel("Sending\n", fontsize=fontsize + 2, labelpad=0)
        axes_com[1].set_ylabel("Receiving\n", fontsize=fontsize + 2, labelpad=0)

        axes_com[0].yaxis.set_label_position("right")
        axes_com[1].yaxis.set_label_position("right")

        axes_com[0].pie(
            types_send,
            colors=[type_colors[t] for t in types_send.index],
            labels=types_send.index,
            labeldistance=0.6,
            # rotatelabels=True,
            wedgeprops={"edgecolor": "w", "linewidth": 2},
            textprops={"fontsize": fontsize, "ha": "center"},
            # textprops={"size": "smaller"},
        )
        axes_com[1].pie(
            types_receive,
            colors=[type_colors[t] for t in types_receive.index],
            labels=types_receive.index,
            labeldistance=0.6,
            # rotatelabels=True,
            wedgeprops={"edgecolor": "w", "linewidth": 2},
            textprops={"fontsize": fontsize, "ha": "center"},
            # textprops={"size": "smaller"},
        )

    return axes


def random_circle_patch(n_samples: int, center_offset: tuple = (0, 0), rmax: float = 1):
    rand_angle = np.random.uniform(0, 2 * np.pi, n_samples)
    rand_radius = np.random.uniform(0, rmax, n_samples)

    rand = (
        np.vstack([np.cos(rand_angle), np.sin(rand_angle)]) * np.sqrt(rand_radius)
        + np.array(center_offset)[:, None]
    )

    return rand


def generate_grid_circle(
    n_samples: int, center_offset: tuple = (0, 0), radius: float = 1.0
):
    n_r = 2
    n_theta = n_samples - 1

    r = np.linspace(0, radius, n_r)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # r, theta = np.meshgrid(r, theta)
    theta, r = np.meshgrid(theta, r)
    x = (r * np.cos(theta)).flatten() + center_offset[0]
    y = (r * np.sin(theta)).flatten() + center_offset[1]
    return np.vstack((x, y))[:, n_samples - 2 :]
    # return x, y


def circular_layout(
    n_blocks: int,
    n_per_com: int,
    radius: float = 1,
    small_radius: float = 0.3,
    return_dict: bool = False,
    even_circles: bool = False,
    offset: Optional[float] = None,
):
    if offset is None:
        offset = np.pi * (n_blocks - 2) / (2 * n_blocks)

    circular_centers = (
        np.array(
            [
                [
                    np.cos(offset + 2 * np.pi * i / n_blocks),
                    np.sin(offset + 2 * np.pi * i / n_blocks),
                ]
                for i in range(n_blocks)
            ]
        )
        * radius
    )

    if even_circles:
        circular_pos = np.array(
            [generate_grid_circle(n_per_com, c, small_radius) for c in circular_centers]
        )
    else:
        circular_pos = np.array(
            [random_circle_patch(n_per_com, c, small_radius) for c in circular_centers]
        )
    circular_pos = np.swapaxes(circular_pos, 0, 1).reshape(2, -1)

    if return_dict:
        circ_dict = {i: (c[0], c[1]) for i, c in enumerate(circular_pos.T)}
        return circular_pos, circ_dict

    return circular_pos


def graph_signal_gif(
    all_recon: np.ndarray,
    graph_pos: dict,
    figure_loc: str,
    gif_name: str,
    savegif: bool = False,
):

    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(10, 10),
        gridspec_kw={"wspace": 0, "height_ratios": [2, 1]},
    )

    def animate(i):
        scat.set_offsets((x[i], 0))
        return (scat,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(x) - 1, interval=50
    )

    if savegif:
        writer = animation.PillowWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save(op.join(figure_loc, gif_name), writer=writer)

    plt.show()


def plot_lobe_lines(
    axes,
    lobe_sizes,
    lobe_labels,
    draw_grid=True,
    plot_labels=True,
    y_only=False,
    x_only=False,
    grid_color="w",
    grid_lw=1,
    fontsize=12,
    no_insula=True,
    x_hemi=True,
):
    lobe_cumsum = np.concatenate([[0], np.cumsum(lobe_sizes)])

    if draw_grid:
        for l_s in lobe_cumsum[1:-1]:
            # axes.axhline(l_s, color=grid_color, lw=grid_lw)
            # axes.axvline(l_s, color=grid_color, lw=grid_lw)
            axes.axhline(l_s - 0.5, color=grid_color, lw=grid_lw)
            axes.axvline(l_s - 0.5, color=grid_color, lw=grid_lw)

    if plot_labels:
        tick_pos = lobe_cumsum[:-1] + (np.diff(lobe_cumsum) / 2)
        tick_pos = tick_pos[:-1]  # removing brainstem

        plot_labs = lobe_labels[:-1]
        plot_labs = [lab.replace("_lobe", "").replace("-", " ") for lab in plot_labs]

        if no_insula:
            plot_labs = [lab if "insul" not in lab else "" for lab in plot_labs]

        if x_hemi:
            labs_no_hemi = [lab.split(" ")[-1].capitalize() for lab in plot_labs]
            axes.set_yticks(tick_pos, labels=labs_no_hemi)

            n_reg = np.sum(lobe_sizes)
            axes.set_xticks([n_reg / 4, 3 * n_reg / 4], labels=["Left", "Right"])
        else:
            if not y_only:
                axes.set_xticks(tick_pos, labels=plot_labs, rotation=-40, ha="right")
            if not x_only:
                axes.set_yticks(tick_pos, labels=plot_labs)
        axes.tick_params(
            labelsize=fontsize,
            labelbottom=False,
            bottom=False,
            labeltop=True,
            top=True,
        )
    return axes


def plot_summary_graph(
    summary,
    labels,
    axes,
    cmap,
    e_strength: float = 5.0,
    scatter_size: float = 400,
    pos: dict = None,
    text: bool = False,
    legend: bool = True,
):
    sum_graph = nx.DiGraph(summary)
    edges = sum_graph.edges()

    e_str = np.array([summary[i, j] for i, j in edges])
    e_str = e_strength * e_str / e_str.max()

    if pos is None:
        pos = nx.spring_layout(sum_graph)
        pos = nx.kamada_kawai_layout(sum_graph)

    pos_array = np.array(list(pos.values()))

    net_colors = cmap.resampled(len(labels) + 1)
    axes.scatter(
        pos_array[:, 0],
        pos_array[:, 1],
        s=scatter_size,
        zorder=3,
        c=np.arange(len(labels)) + 1,
        cmap=net_colors,
        edgecolor="k",
        linewidth=2,
        vmin=0,
    )
    _ = nx.draw_networkx_edges(
        sum_graph,
        pos,
        ax=axes,
        arrowstyle="-|>",
        arrowsize=30,
        edge_color="k",
        edgelist=edges,
        alpha=1,
        width=e_str,
        connectionstyle="arc3,rad=0.3",  # ,angleA=-80,angleB=10",
        # connectionstyle="angle3,angleA=50,angleB=-40",
    )
    #    connectionstyle='arc,angleA=0.2,angleB=0.2,rad=0.5')

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=labels[i],
            markerfacecolor=net_colors(i + 1),
            markersize=10,
        )
        for i in range(len(labels))
    ]

    if text:
        # Handling overlapping text labels
        texty = pos_array[:, 1] + 0.05
        textdiff = texty[None, :] - texty[:, None]
        i_over, j_over = np.array(np.where(np.abs(textdiff) < 0.05))
        for i, j in zip(i_over, j_over):
            if i != j:
                texty[j] += 0.02 * np.sign(textdiff[i, j])

        for i, label in enumerate(labels):
            axes.text(
                pos_array[i, 0],
                texty[i],
                label,
                fontsize=14,
                # fontweight="bold",
                ha="center",
                va="center",
                color=net_colors(i + 1),
                path_effects=[
                    withStroke(
                        linewidth=2,
                        foreground="k",
                        alpha=0.8,
                    )
                ],
            )
    if legend and not text:
        axes.legend(handles=handles, ncols=2, fontsize=14)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def styled_dendrogram(
    Z,
    k=None,
    cut_height=None,
    top_color="k",
    cmap=None,
    p=0,
    lw=2,
    dyn_lw=True,
    line_styles=None,
    ax=None,
    **kwargs,
):
    """
    Draw a dendrogram with custom per-branch style.
    line_styles: list of dicts, same length as number of merges.
    Each dict can have keys like 'color', 'linewidth', 'linestyle'.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    d = dendrogram(Z, no_plot=True, **kwargs)
    n_leaves = len(d["leaves"])

    ordered_labs = None
    if k is not None:
        labs = fcluster(Z, t=k, criterion="maxclust")
        ordered_labs = labs[d["leaves"]]
    elif cut_height is not None:
        labs = fcluster(Z, t=cut_height, criterion="distance")
        ordered_labs = labs[d["leaves"]]

    print(labs.min(), labs.max())

    if cmap is None:
        cmap = plt.get_cmap("tab20")

    cluster_colors = {}
    if ordered_labs is not None:
        uniq = np.unique(ordered_labs)
        if len(uniq) == 1:
            cluster_colors[uniq[0]] = cmap(1)
        else:
            for i, u in enumerate(uniq):
                cluster_colors[u] = cmap((i + 1) / (len(uniq)))

    for i, (xs, ys) in enumerate(zip(d["icoord"], d["dcoord"])):
        height = max(ys)

        if height < p:
            continue
        # height = min(ys)
        if (cut_height is not None) and (height > cut_height):
            color = top_color
            l_lw = lw / 2 if dyn_lw else lw
        elif ordered_labs is None:
            # fallback to provided style or black
            color = (
                line_styles[i].get("color")
                if line_styles and i < len(line_styles) and "color" in line_styles[i]
                else "k"
            )
        else:
            # map branch x-range to leaf index range in the leaf order
            left_x, right_x = min(xs), max(xs)
            left_idx = int(np.clip(np.floor((left_x - 5) / 10 + 1e-9), 0, n_leaves - 1))
            right_idx = int(
                np.clip(np.ceil((right_x - 5) / 10 - 1e-9), 0, n_leaves - 1)
            )
            branch_labels = ordered_labs[left_idx : (right_idx + 1)]
            # choose majority cluster for that branch
            vals, counts = np.unique(branch_labels, return_counts=True)
            maj = vals[np.argmax(counts)]
            color = cluster_colors.get(maj, top_color)
            l_lw = lw

        style = line_styles[i] if line_styles and i < len(line_styles) else {}
        # ensure we don't pass 'color' from style (we override it)
        style = {k: v for k, v in style.items() if k != "color"}
        ax.plot(xs, ys, color=color, lw=l_lw, **style)

    return d, ax


def get_camera_pos(view="transverse"):
    if (view == "transverse-vertical") or ("verti" in view):
        yoffset = -15
        position = (0, yoffset, 400)
        focal_point = (0, yoffset, 0)
        view_up = (0.0, 0.0, 0.0)
    elif (view == "transverse") or ("tra" in view):
        xoffset = 8
        yoffset = -15
        position = (xoffset, yoffset, 350)
        focal_point = (xoffset, yoffset, 0)
        view_up = (-0.5, 0.0, 0.0)
    elif (view == "sagittal-left") or ("left" in view):
        yoffset = -18
        position = (-350, yoffset, 0)
        focal_point = (0, yoffset, 0)
        view_up = (0.0, 0.0, 1.0)
    elif (view == "sagittal-right") or ("right" in view):
        yoffset = -18
        position = (350, yoffset, 0)
        focal_point = (0, yoffset, 0)
        view_up = (0.0, 0.0, 1.0)
    elif (view == "coronal-back") or ("back" in view):
        zoffset = 10
        position = (0, -330, zoffset)
        focal_point = (0, 0, zoffset)
        view_up = (0.0, 0.0, 1.0)
    elif (view == "coronal-front") or ("front" in view):
        zoffset = 10
        position = (0, 330, zoffset)
        focal_point = (0, 0, zoffset)
        view_up = (0.0, 0.0, 1.0)
    elif view == "custom":
        angle = np.deg2rad(-45)  # -35 # -45
        dist = 300
        posx = dist * np.sin(angle)
        posy = dist * np.cos(angle)
        position = (posx, posy, 80)  # z=80
        # position = (-350, 350, 0)
        focal_point = (30, -50, 0)
        view_up = (0.0, 0.0, 1.0)
    elif view == "custom2":
        yoffset = -18
        zoffset = 10
        position = (-250, 250 + yoffset, zoffset)
        focal_point = (0, yoffset, zoffset)
        view_up = (0.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown view: {view}")

    return position, focal_point, view_up


def get_brain_actors(opacity=0.3):
    fsaverage = fetch_surf_fsaverage()

    brain_actors = []
    for surf in ["pial_left", "pial_right"]:
        coords, faces = load_surf_mesh(fsaverage[surf])
        # surf_act = actor.surface(coords, faces, smooth="loop")
        surf_act = actor.surface(coords, faces=faces, smooth="loop")
        surf_act.GetProperty().SetOpacity(opacity)
        brain_actors.append(surf_act)

    return brain_actors


def get_tube_actor(
    tractogram,
    cmap,
    n_centroids=1,
    linewidth=0.4,
    upsample=None,
    bidir_col=None,
    bundle_list=None,
    alpha=1,
):
    if upsample is not None:
        space = int(400 / (12 * upsample))
        sl_interp = [spline(sl, k=3, s=0, nest=1) for sl in tractogram.streamlines]
        slines = [sl[::space] for sl in sl_interp]
    else:
        slines = tractogram.streamlines

    n_segments = len(slines[0])
    sline_cmap = cmap.resampled(n_segments)

    col_val = np.array(
        [
            np.linspace(dir_col[0], dir_col[-1], len(sl))
            for dir_col, sl in zip(tractogram.data_per_point["dir_col"], slines)
        ]
    )

    colors = np.array([sline_cmap((1 + cv) / 2) for cv in col_val])

    if bundle_list is not None:
        colors = np.repeat(bidir_col[bundle_list][:, None, :], n_segments, axis=1)
    else:
        if bidir_col is None:
            bidir_col = (0.5, 0.5, 0.8, 1)

        bidir_col = np.array(bidir_col)
        for i, col in enumerate(colors):
            if np.allclose(col[0], col[-1]):
                colors[i, :] = bidir_col

    if colors.ndim > 3:
        colors = colors[:, :, 0]

    if alpha != 1:
        colors[..., -1] = alpha

    # Add actors to the scene
    # stream_actor = actor.line(streamlines, colors=np.asarray(colors, dtype=object), linewidth=2)

    return actor.streamtube(
        slines,
        colors=np.asarray(colors, dtype=object),
        linewidth=linewidth,
    )


def apply_actor_parameters(tube_actor, brain_actors, gloss_brain=False):

    prop = tube_actor.GetProperty()

    prop.SetAmbient(0.4)  # base light
    prop.SetDiffuse(0.6)  # directional light response
    prop.SetSpecular(0.05)  # shininess amount
    prop.SetSpecularPower(5)  # shininess tightness

    for act in brain_actors:
        p = act.GetProperty()
        p.SetLighting(True)
        p.SetInterpolationToPhong()

        if gloss_brain:
            p.SetAmbient(0.15)
            p.SetDiffuse(0.6)
            p.SetSpecular(0.3)
            p.SetSpecularPower(25)
        else:
            p.SetAmbient(0.3)
            p.SetDiffuse(0.6)
            p.SetSpecular(0.01)
            p.SetSpecularPower(5)

    return tube_actor, brain_actors


def plot_bundle_surf(
    tube_actor,
    brain_actors,
    view=None,
    axes=None,
    overlay_slines=False,
    scene_size=(2000, 2000),
):

    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 8))
    scene = window.Scene()

    # White background
    scene.SetBackground((1, 1, 1))

    tube_actor, brain_actors = apply_actor_parameters(tube_actor, brain_actors)

    for surf_act in brain_actors:
        scene.add(surf_act)

    if not overlay_slines:
        # scene.add(stream_actor)
        scene.add(tube_actor)

    if view is not None:
        position, focal_point, view_up = get_camera_pos(view=view)
    else:
        position = (0, 0, 350)
        focal_point = (0, -15, 0)
        view_up = (0.0, 0.0, 0.0)

    scene.set_camera(position=position, focal_point=focal_point, view_up=view_up)

    try:
        win = window.snapshot(scene, size=scene_size, offscreen=True)
        scene.clear()
        scene = None
    except Exception as e:
        print(f"Rendering failed: {e}. Returning empty axes.")
        return axes

    win = np.ascontiguousarray(win)

    alpha_mask = ((win == (255, 255, 255)).sum(axis=-1) < 3).astype(float)
    win = np.concatenate([win, 255 * alpha_mask[:, :, None]], axis=-1).astype(int)
    axes.imshow(win)

    if overlay_slines:
        scene2 = window.Scene()
        scene2.SetBackground((1, 1, 1))
        scene2.add(tube_actor)

        if view is not None:
            position, focal_point, view_up = get_camera_pos(view=view)
        else:
            position = (0, 0, 350)
            focal_point = (0, -15, 0)
            view_up = (0.0, 0.0, 0.0)

        scene2.set_camera(position=position, focal_point=focal_point, view_up=view_up)
        win = window.snapshot(scene2, size=scene_size, offscreen=True)
        scene2.clear()
        scene2 = None
        win = np.ascontiguousarray(win)

        win_mask = ((win == (255, 255, 255)).sum(axis=-1) < 3).astype(float)
        win = np.concatenate([win, 255 * win_mask[:, :, None]], axis=-1).astype(int)

        axes.imshow(win)
    return axes


def plot_kde(
    data,
    data_range=None,
    ax=None,
    bw_adjust=1,
    fill=False,
    color="silver",
    alpha=0.8,
    normalize=True,
    vertical=False,
):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if data_range is None:
        data_range = np.linspace(data.min(), data.max(), 100)

    kde = gaussian_kde(data, bw_method=bw_adjust)
    y_val = kde(data_range)

    if normalize:
        y_val = y_val / y_val.max()

    if vertical:
        x_plot = y_val
        y_plot = data_range
    else:
        x_plot = data_range
        y_plot = y_val

    ax.plot(x_plot, y_plot, lw=2, color=color, alpha=alpha)
    if fill:
        ax.fill_between(
            x_plot, y_plot, lw=2, color=color, alpha=alpha / 2, edgecolor="none"
        )


def plot_actors(
    tube_actor,
    brain_actors,
    cmap="RdBu_r",
    overlay_slines=True,
    view="custom",
    plot_cbar=False,
    cbar_pos=None,
    fig=None,
    axes=None,
    cbar_fontsize=18,
):
    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 8))

    axes.axis("off")
    if "ortho" not in view:
        axes = plot_bundle_surf(
            tube_actor,
            brain_actors,
            view=view,
            axes=axes,
            overlay_slines=overlay_slines,
        )
    else:
        if fig is None:
            raise ValueError("fig must be provided when using orthogonal views")

        gs1 = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=axes.get_subplotspec(), wspace=0, hspace=0
        )
        ax_tra = fig.add_subplot(gs1[:, 0])
        b_axes = [ax_tra] + [fig.add_subplot(gs1[i, 1]) for i in range(2)]

        for ax, view in zip(b_axes, ["transverse-vertical", "back", "right"]):
            scene_size = (2000, 2000)
            if "tra" in view:
                scene_size = (2000, 3000)
            plot_bundle_surf(
                tube_actor,
                brain_actors,
                view=view,
                axes=ax,
                overlay_slines=overlay_slines,
                scene_size=scene_size,
            )
            ax.axis("off")

    if plot_cbar:
        if cbar_pos is None:
            cbar_pos = [0.15, 0.15, 0.8, 0.02]

        draw_cbar(
            fig,
            cmap,
            ax_pos=cbar_pos,
            ticks=[0, 10],
            labels=["Receiving", "Sending"],
            fontsize=cbar_fontsize,
        )

    return axes


def plot_bicom_tracts(
    bicom_id,
    edge_clusters_mat,
    labels,
    scale,
    cmap="RdBu_r",
    atlas_dir="/Users/acionca/data/atlas_data",
    n_centroids=2,
    upsample=None,
    brain_opacity=1,
    overlay_slines=True,
    linewidth=0.4,
    slines_alpha=1,
    bidir_col=(0.5, 0.5, 0.8, 1),
    view="custom",
    plot_cbar=False,
    cbar_pos=None,
    cbar_fontsize=18,
    fig=None,
    axes=None,
):

    centroid_dir = op.join(
        atlas_dir, "centroids", f"scale{scale}", f"group_centroids_scale{scale}"
    )

    selected_bundles, selected_bundles_dir = bundle.get_bicom_bundles(
        bicom_id, edge_clusters_mat, labels, scale=scale
    )

    centroid_tractogram = bundle.get_bundle_centroid(
        centroid_dir,
        scale,
        selected_bundles,
        selected_bundles_dir,
        n_centroids=n_centroids,
    )

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    brain_actors = get_brain_actors(opacity=brain_opacity)
    tube_actor = get_tube_actor(
        centroid_tractogram,
        cmap,
        n_centroids=n_centroids,
        upsample=upsample,
        linewidth=linewidth,
        alpha=slines_alpha,
        bidir_col=bidir_col,
    )

    axes = plot_actors(
        tube_actor,
        brain_actors,
        cmap=cmap,
        overlay_slines=overlay_slines,
        view=view,
        plot_cbar=plot_cbar,
        cbar_pos=cbar_pos,
        cbar_fontsize=cbar_fontsize,
        fig=fig,
        axes=axes,
    )

    return axes


def plot_trk(
    path_to_trk,
    cmap="RdBu_r",
    n_slines=None,
    upsample=None,
    brain_opacity=1,
    overlay_slines=True,
    linewidth=0.4,
    slines_alpha=1,
    trk_color_list=None,
    view="custom",
    plot_cbar=False,
    fig=None,
    axes=None,
):
    if isinstance(path_to_trk, list):
        print(f"Merging {len(path_to_trk)} tractogram files...")
        all_sls = []
        bundle_index = []
        n_segments = 20
        for i, p in enumerate(path_to_trk):
            trk = nib.streamlines.load(p)

            # if i == 0:
            #     n_segments = len(trk.streamlines[0])

            if n_slines is not None:
                rng = np.random.default_rng(seed=42)
                sel_sls = select_random_set_of_streamlines(
                    trk.streamlines, n_slines, rng=rng
                )
            else:
                sel_sls = trk.streamlines

            for sl in sel_sls:
                # all_sls.append(sl)
                # if len(sl) == n_segments:
                all_sls.append(np.array(sl))
                bundle_index.append(i)

        all_sls = set_number_of_points(all_sls, nb_points=20)
        centroid_trk = nib.streamlines.Tractogram(all_sls, affine_to_rasmm=np.eye(4))

        # n_segments = len(centroid_trk.streamlines[0])

        dircol = [np.zeros(len(sl)) for sl in centroid_trk.streamlines]
        centroid_trk.data_per_point = {
            "dir_col": dircol
            # (len(centroid_trk.streamlines), n_segments), dtype=float)
        }
    else:
        centroid_trk = nib.streamlines.load(path_to_trk)
        slines = centroid_trk.streamlines
        n_segments = len(slines[0])

        centroid_trk.data_per_point = {
            "dir_col": np.zeros(
                (centroid_trk.streamlines._data.shape[0], n_segments), dtype=float
            )
        }
        bundle_index = None

    tube_actor = get_tube_actor(
        centroid_trk,
        cmap=cmap,
        upsample=upsample,
        linewidth=linewidth,
        alpha=slines_alpha,
        bidir_col=trk_color_list,
        bundle_list=bundle_index,
    )
    brain_actors = get_brain_actors(opacity=brain_opacity)

    axes = plot_actors(
        tube_actor,
        brain_actors,
        cmap=cmap,
        overlay_slines=overlay_slines,
        view=view,
        plot_cbar=plot_cbar,
        fig=fig,
        axes=axes,
    )
    return axes


def get_loo_curves(e_ratio, f_ratio, x_plot=None):
    if x_plot is None:
        x_plot = np.linspace(0, 1, 100)

    all_yvals = []
    all_corrs = []
    for k, _ in enumerate(f_ratio):
        loo_mask = ~np.eye(len(f_ratio), dtype=bool)[k]
        ftract_loo = f_ratio[loo_mask]
        ec_loo = e_ratio[loo_mask]

        # all_corrs.append(np.corrcoef(ec_loo, ftract_loo)[0, 1])
        all_corrs.append(spearmanr(ec_loo, ftract_loo)[0])

        reg = LinearRegression().fit(ec_loo.reshape(-1, 1), ftract_loo.reshape(-1, 1))
        reg.score(ec_loo.reshape(-1, 1), ftract_loo.reshape(-1, 1))

        y_vals = reg.intercept_[0] + reg.coef_[0] * x_plot
        all_yvals.append(y_vals)

    return np.array(all_yvals), np.array(all_corrs)


def plot_yeo_summary(
    subnet_mat,
    yeo_labels,
    cmap,
    fig=None,
    axes=None,
    label_space=1.2,
    width_mod=4,
    r_lab_only=False,
    l_lab_only=False,
    manual_arrows=False,
    net_alpha=False,
):
    angles_raw = np.linspace(-np.pi / 4, np.pi / 4, len(yeo_labels))
    all_angles = np.concatenate([np.pi + angles_raw, -angles_raw])

    nodal_pos = np.array([(np.cos(angle), np.sin(angle)) for angle in all_angles])

    zero_block = np.zeros((len(yeo_labels), len(yeo_labels)))
    bipartite_mat = np.block([[zero_block, subnet_mat], [zero_block, zero_block]])

    bipartite_graph = nx.DiGraph(bipartite_mat)
    edges = bipartite_graph.edges()

    strengths = np.array([bipartite_graph[u][v]["weight"] for u, v in edges])
    strengths /= strengths.max()

    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 8))

    if manual_arrows:
        for e_i, e in enumerate(edges):
            arrow = FancyArrowPatch(
                nodal_pos[e[0]],
                nodal_pos[e[1]] * np.array([0.9, 1]),
                mutation_scale=3,
                facecolor="k",
                edgecolor="none",
                # mutation_scale=10, facecolor="tab:purple", edgecolor="none",
                arrowstyle=ArrowStyle(
                    "Simple",
                    tail_width=width_mod * strengths[e_i],
                    head_width=1 + 2 * width_mod * strengths[e_i],
                    head_length=2 + strengths[e_i],
                ),
                alpha=strengths[e_i],
                connectionstyle="arc3",
            )
            axes.add_patch(arrow)
    else:
        nx.draw_networkx_edges(
            bipartite_graph,
            pos={i: nodal_pos[i] for i in range(len(nodal_pos))},
            ax=axes,
            edge_color="k",
            alpha=strengths,
            # arrowsize=10,
            arrowsize=20,
            width=width_mod * strengths,
        )

    axes.scatter(
        nodal_pos[:, 0],
        nodal_pos[:, 1],
        s=100,
        color="w",
        edgecolors="none",
    )

    if net_alpha:
        norm_send = subnet_mat.sum(axis=1)
        norm_recv = subnet_mat.sum(axis=0)
        norm_send /= norm_send.max()
        norm_recv /= norm_recv.max()

        node_alpha = np.concatenate([norm_send, norm_recv])
        node_alpha = node_alpha / node_alpha.max()
        node_alpha = 0.5 + 0.5 * node_alpha
    else:
        node_alpha = 1

    axes.scatter(
        nodal_pos[:, 0],
        nodal_pos[:, 1],
        s=100,
        c=np.concatenate([norm_send, -norm_recv]),
        cmap=cmap,
        # color="k",
        edgecolors="none",
        alpha=node_alpha,
    )

    axes.set_xlim(-2.5, 1.5)
    if not r_lab_only:
        for i, net in enumerate(yeo_labels):
            axes.text(
                label_space * nodal_pos[i, 0],
                nodal_pos[i, 1],
                net,
                fontsize=12,
                ha="right",
                va="center",
            )
    else:
        # axes.set_xlim(-1.2, 1.5)
        axes.set_xlim(-1.2, 3)

    if not l_lab_only:
        for i, net in enumerate(yeo_labels):
            axes.text(
                label_space * nodal_pos[len(yeo_labels) + i, 0],
                nodal_pos[len(yeo_labels) + i, 1],
                net,
                fontsize=12,
                ha="left",
                va="center",
            )

    # axes.set_ylim(-1.1, 0.9)
    axes.set_ylim(-1.15, 0.9)

    return axes


def get_2d_colormap(n=256, center_strength=0.5, white_val=1):
    c_send_uni = np.array([240, 50, 50]) / 255
    c_send_trans = np.array([240, 240, 50]) / 255
    c_recv_uni = np.array([80, 80, 240]) / 255
    c_recv_trans = np.array([80, 240, 80]) / 255

    c_send_uni = np.array(to_rgb("#F8C768"))
    c_send_trans = np.array(to_rgb("#A50721"))
    c_recv_uni = np.array(to_rgb("#7DD0ED"))
    c_recv_trans = np.array(to_rgb("#19367E"))

    x = np.linspace(0, 1, n)  # send → receive
    y = np.linspace(0, 1, n)  # unimodal → transmodal

    WHITE = np.array([1, 1, 1]) * white_val

    cmap = np.zeros((n, n, 3))

    for i, yi in enumerate(y):
        # interpolate vertically (unimodal → transmodal)
        left = (1 - yi) * c_recv_trans + yi * c_recv_uni
        right = (1 - yi) * c_send_trans + yi * c_send_uni

        # interpolate horizontally (send → receive)
        for j, xj in enumerate(x):
            base_color = (1 - xj) * left + xj * right

            # distance to center (0.5, 0.5)
            # d = np.sqrt((xj - 0.5) ** 2 + (yi - 0.5) ** 2)
            # d /= np.sqrt(2 * (0.5**2))  # normalize to [0,1]
            d = np.abs(xj - 0.5) * 2

            # nonlinear emphasis of center
            w = d**center_strength

            cmap[i, j] = (1 - w) * WHITE + w * base_color
            # cmap[i, j] = (1 - xj) * left + xj * right

    return cmap


def get_2d_color(send, rec, cmap2d):

    send_norm = (send - send.min()) / (send.max() - send.min())
    recv_norm = (rec - rec.min()) / (rec.max() - rec.min())

    n = cmap2d.shape[0]
    i = (send_norm * (n - 1)).astype(int)
    j = (recv_norm * (n - 1)).astype(int)
    return cmap2d[j, i]


def plot_node_summary(
    e_clust_mask,
    node_pos,
    gradient,
    fig=None,
    axes=None,
    label_space=1.2,
    width_mod=4,
    r_lab_only=False,
    l_lab_only=False,
    manual_arrows=False,
    net_alpha=False,
):
    n_nodes = e_clust_mask.shape[0]
    angles_raw = np.linspace(-np.pi / 4, np.pi / 4, n_nodes)
    all_angles = np.concatenate([np.pi + angles_raw, -angles_raw])

    nodal_pos = np.array([(np.cos(angle), np.sin(angle)) for angle in all_angles])

    nodal_pos = node_pos.copy()  # [:, [1, 2]]

    # zero_block = np.zeros((n_nodes, n_nodes))
    # bipartite_mat = np.block([[zero_block, e_clust_mask], [zero_block, zero_block]])

    # bipartite_graph = nx.DiGraph(bipartite_mat)
    bipartite_graph = nx.DiGraph(e_clust_mask)
    edges = bipartite_graph.edges()

    strengths = np.array([e_clust_mask[u, v] for u, v in edges])
    strengths /= strengths.max()

    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 8))

    # if manual_arrows:
    #     for e_i, e in enumerate(edges):
    #         arrow = FancyArrowPatch(
    #             nodal_pos[e[0]],
    #             nodal_pos[e[1]],  # * np.array([0.9, 1]),
    #             mutation_scale=1,
    #             facecolor="k",
    #             edgecolor="none",
    #             # mutation_scale=10, facecolor="tab:purple", edgecolor="none",
    #             arrowstyle=ArrowStyle(
    #                 "Simple",
    #                 tail_width=width_mod * strengths[e_i],
    #                 head_width=5 + 2 * width_mod * strengths[e_i],
    #                 head_length=5 + strengths[e_i],
    #             ),
    #             # alpha=strengths[e_i],
    #             alpha=0.2 + 0.2 * strengths[e_i],
    #             connectionstyle="arc3",
    #         )
    #         axes.add_patch(arrow)
    # else:
    #     nx.draw_networkx_edges(
    #         bipartite_graph,
    #         pos={i: nodal_pos[i] for i in range(len(nodal_pos))},
    #         ax=axes,
    #         edge_color="k",
    #         alpha=strengths,
    #         # arrowsize=10,
    #         arrowsize=10,
    #         width=width_mod * strengths,
    #     )

    norm_send = e_clust_mask.sum(axis=1)
    norm_recv = e_clust_mask.sum(axis=0)
    norm_send = norm_send / norm_send.max()
    norm_recv = norm_recv / norm_recv.max()

    if net_alpha:
        # node_alpha = np.concatenate([norm_send, norm_recv])
        node_alpha = (norm_send + norm_recv) / 2
        node_alpha = node_alpha / node_alpha.max()

        # node_alpha = 0.2 + 0.8 * node_alpha
    else:
        node_alpha = 1

    sort_by_alpha = np.argsort(node_alpha)

    node_s = 20 + 180 * node_alpha
    axes.scatter(
        nodal_pos[:, 0],
        nodal_pos[:, 1],
        s=node_s,
        color="w",
        edgecolors="k",
        lw=0.5,
        alpha=0.5 + 0.5 * (node_alpha > 0),
    )

    # node_color = np.concatenate([norm_send, -norm_recv])
    node_color = norm_send - norm_recv

    node_color = nodal_pos[:, 0]

    cmap2d = get_2d_colormap()
    node_color = get_2d_color(norm_send - norm_recv, gradient, cmap2d)
    # node_color = gradient / gradient.max()

    axes.scatter(
        nodal_pos[sort_by_alpha, 0],
        nodal_pos[sort_by_alpha, 1],
        s=node_s[sort_by_alpha],
        c=node_color[sort_by_alpha],
        edgecolors="k",
        lw=0.5,
        # cmap="RdBu_r",
        # cmap="turbo_r",
        alpha=(node_alpha[sort_by_alpha] > 0).astype(float),
        # alpha=node_alpha[sort_by_alpha],
    )

    # print(axes.get_xlim())
    if axes.get_xlim()[0] < -90:
        axes.set_xlim(-110, 80)
    else:
        axes.set_xlim(-80, 110)
    axes.set_ylim(-80, 80)

    return axes


def plot_gradient_legend(
    axes=None, aspect="equal", labels_out=False, arrowscale=30, label_fs=16
):

    if axes is None:
        fig, axes = plt.subplots(figsize=(5, 5))

    mid_color = 255 / 2
    all_a_pos = [
        [[100, mid_color], [255, mid_color]],
        [[200, mid_color], [0, mid_color]],
        [[mid_color, 100], [mid_color, 255]],
        [[mid_color, 200], [mid_color, 0]],
    ]
    for a_pos in all_a_pos:
        arrow = FancyArrowPatch(
            a_pos[0],
            a_pos[1],
            mutation_scale=arrowscale,
            facecolor="k",
            edgecolor="none",
            arrowstyle="simple",
        )
        axes.add_patch(arrow)

    cmap2d = get_2d_colormap()
    axes.imshow(cmap2d, origin="lower", aspect=aspect)

    if labels_out:
        axes.text(
            0.5,
            1.2,
            "Node Colors: Gradient Value",
            fontsize=16,
            ha="center",
            va="bottom",
            transform=axes.transAxes,
        )

        axes.text(
            0.5,
            -0.03,
            "Transmodal",
            fontsize=label_fs,
            ha="center",
            va="top",
            transform=axes.transAxes,
        )
        axes.text(
            0.5,
            1,
            "Unimodal",
            fontsize=label_fs,
            ha="center",
            va="bottom",
            transform=axes.transAxes,
        )
        axes.text(
            -0.02,
            0.5,
            "Receiving",
            fontsize=label_fs,
            ha="right",
            va="center",
            transform=axes.transAxes,
        )
        axes.text(
            1.01,
            0.5,
            "Sending",
            fontsize=label_fs,
            ha="left",
            va="center",
            transform=axes.transAxes,
        )
    else:
        axes.text(110, 5, "Transmodal", fontsize=label_fs, ha="right", va="bottom")
        axes.text(140, 250, "Unimodal", fontsize=label_fs, ha="left", va="top")
        axes.text(10, 110, "Receiving", fontsize=label_fs, ha="left", va="top")
        axes.text(240, 140, "Sending", fontsize=label_fs, ha="right", va="bottom")

    # axes.set_xlabel("Sending  →  Receiving")
    # axes.set_ylabel("Unimodal  →  Transmodal")
    axes.set_xticks([])
    axes.set_yticks([])
    # axes.axis("off")
