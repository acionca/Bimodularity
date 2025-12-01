# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import numpy as np
from scipy.linalg import block_diag
import networkx as nx

import os.path as op
from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import to_rgb, to_rgba, ListedColormap, LinearSegmentedColormap
from matplotlib.patheffects import withStroke

from dipy.viz import window, actor
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh

import dgsp
from palettes import CLUSTER, CLUSTER_CB, CLUSTER_SOFT, DIV_RB, EXTENDED_NCAR, PASTEL


def get_all_cmaps():
    all_colors = [
        CLUSTER,
        CLUSTER_CB,
        CLUSTER_SOFT,
        DIV_RB,
        EXTENDED_NCAR,
    ]
    cmap_names = [
        "cluster_palette",
        "cluster_palette_cb",
        "cluster_palette_soft",
        "div_rb",
        "extended_ncar",
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


def add_cbar(fig: Figure, ax: Axes, **kwargs) -> Axes:
    """Add a colorbar to an existing figure/axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to get the colorbar from
    ax : matplotlib.axes.Axes
        axes to add the colorbar to

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        tuple of figure and axes with the colorbar added
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    return fig, ax, cbar


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
        colors = "tab:blue"

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
    grid_color="w",
    grid_lw=1,
    fontsize=12,
    no_insula=False,
):
    lobe_cumsum = np.concatenate([[0], np.cumsum(lobe_sizes)])

    if draw_grid:
        for l_s in lobe_cumsum[1:-1]:
            axes.axhline(l_s, color=grid_color, lw=grid_lw)
            axes.axvline(l_s, color=grid_color, lw=grid_lw)
            # axes.axhline(l_s - 0.5, color=grid_color, lw=grid_lw)
            # axes.axvline(l_s - 0.5, color=grid_color, lw=grid_lw)

    if plot_labels:
        tick_pos = lobe_cumsum[:-1] + (np.diff(lobe_cumsum) / 2)
        tick_pos = tick_pos[:-1]  # removing brainstem

        plot_labs = lobe_labels[:-1]
        plot_labs = [lab.replace("_lobe", "").replace("-", " ") for lab in plot_labs]

        if no_insula:
            plot_labs = [lab if "insul" not in lab else "" for lab in plot_labs]

        if not y_only:
            axes.set_xticks(tick_pos, labels=plot_labs, rotation=-40, ha="right")
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
                        alpha=1,
                    )
                ],
            )
    else:
        axes.legend(handles=handles, ncols=2, fontsize=14)


def get_camera_pos(view="transverse"):
    if (view == "transverse") or ("tra" in view):
        yoffset = -15
        position = (0, yoffset, 350)
        focal_point = (0, yoffset, 0)
        view_up = (0.0, 0.0, 0.0)
    elif (view == "sagittal") or ("sag" in view):
        yoffset = -18
        position = (350, yoffset, 0)
        focal_point = (0, yoffset, 0)
        view_up = (0.0, 0.0, 1.0)
    elif (view == "coronal") or ("cor" in view):
        zoffset = 10
        position = (0, -310, zoffset)
        focal_point = (0, 0, zoffset)
        view_up = (0.0, 0.0, 1.0)
    elif view == "custom":
        angle = np.deg2rad(-35)
        dist = 300
        posx = dist * np.sin(angle)
        posy = dist * np.cos(angle)
        position = (posx, posy, 80)
        # position = (-350, 350, 0)
        focal_point = (30, -50, 0)
        view_up = (0.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown view: {view}")

    return position, focal_point, view_up


def get_brain_actors(opacity=0.3):
    fsaverage = fetch_surf_fsaverage()

    brain_actors = []
    for surf in ["pial_left", "pial_right"]:
        coords, faces = load_surf_mesh(fsaverage[surf])
        surf_act = actor.surface(coords, faces, smooth="loop")
        surf_act.GetProperty().SetOpacity(opacity)
        brain_actors.append(surf_act)

    return brain_actors


def get_tube_actor(tractogram, cmap, n_kept=1, n_colors=13, linewidth=0.5):

    sline_cmap = cmap.resampled(n_colors)

    colors = np.array(
        [
            sline_cmap((6 * (1 + sl)).astype(int))
            for sl in tractogram.data_per_point["send"]
        ]
    )
    if n_kept == 1:
        colors = colors[:, 0, :3]
    else:
        colors = colors[:, :, 0, :3]

    # Add actors to the scene
    # stream_actor = actor.line(streamlines, colors=np.asarray(colors, dtype=object), linewidth=2)

    return actor.streamtube(
        tractogram.streamlines,
        colors=np.asarray(colors, dtype=object),
        linewidth=linewidth,
    )


def plot_bundle_surf(
    tube_actor, brain_actors, view=None, axes=None, overlay_slines=False
):

    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 8))
    scene = window.Scene()

    # White background
    scene.SetBackground((1, 1, 1))

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
    win = window.snapshot(scene, size=(2000, 2000), offscreen=True)
    win = np.ascontiguousarray(win)

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
        win = window.snapshot(scene2, size=(2000, 2000), offscreen=True)
        win = np.ascontiguousarray(win)

        win_mask = ((win == (255, 255, 255)).sum(axis=-1) < 3).astype(float)
        win = np.concatenate([win, 255 * win_mask[:, :, None]], axis=-1).astype(int)

        axes.imshow(win)
