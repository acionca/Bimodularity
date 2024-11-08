from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from dgsp import configuration_null, modularity_matrix, modularity_quadratic, sorted_SVD


def community_pos(n_nodes, seed=123, n_nodes_per_com=None):

    if n_nodes_per_com is None:
        half_node = int(n_nodes / 2)
        n_nodes_per_com = [half_node] * 2

    com_spaces = np.linspace(-0.5, 0.5, len(n_nodes_per_com))

    np.random.seed(seed)

    pos_com = {}
    k = 0
    for com_id, n_in_com in enumerate(n_nodes_per_com):

        scatters = np.random.normal(np.zeros((n_in_com, 2)), 0.1)
        pos_com.update(
            {
                k + i: np.array([scatters[i, 0], com_spaces[com_id] + scatters[i, 1]])
                for i in range(n_in_com)
            }
        )
        k = len(pos_com)

    return pos_com


def square_community_pos(nodes_per_com: int, n_com: int, seed=1234):
    np.random.seed(seed)

    random_scatter = np.random.normal(0, 0.1, (nodes_per_com, 2))

    com_square = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

    pos_com = {}
    for com_id in np.arange(n_com):
        pos_com.update(
            {
                com_id * nodes_per_com
                + i: np.array(
                    [
                        com_square[com_id, 0] + random_scatter[i, 0],
                        com_square[com_id, 1] + random_scatter[i, 1],
                    ]
                )
                for i in range(nodes_per_com)
            }
        )
    return pos_com


def draw_graph(
    G,
    pos,
    ax: Optional[plt.Axes] = None,
    arrows: Optional[bool] = None,
    with_labels: bool = True,
    node_kwds: dict = {},
    edge_kwds: dict = {},
    label_kwds: dict = {},
):
    """_summary_

    Parameters
    ----------
    G : _type_
        _description_
    pos : _type_
        _description_
    ax : Optional[plt.Axes], optional
        _description_, by default None
    arrows : Optional[bool], optional
        _description_, by default None
    with_labels : bool, optional
        _description_, by default True
    node_kwds : dict, optional
        _description_, by default {}
    edge_kwds : dict, optional
        _description_, by default {}
    label_kwds : dict, optional
        _description_, by default {}
    """

    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kwds)
    nx.draw_networkx_edges(G, pos, arrows=arrows, ax=ax, **edge_kwds)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, **label_kwds)


def custom_draw(
    axis,
    a_mat: np.ndarray,
    position: str = "com",
    community_vector: Optional[np.ndarray] = None,
    node_size: int = 300,
    edge_alpha: float = 0.1,
    cm_scale: int = 20,
    labels: Optional[dict] = None,
    colors: Optional[np.ndarray] = None,
    direct: bool = True,
    **kwargs,
):
    n_nodes = len(a_mat)

    edge_kw = {"alpha": edge_alpha, "connectionstyle": "arc3,rad=0.2"}
    if direct:
        mytoy_graph = nx.DiGraph(a_mat)
        edge_kw.update({"arrowsize": 20})
    else:
        mytoy_graph = nx.Graph(a_mat)

    pos = nx.spring_layout(mytoy_graph)
    if isinstance(position, dict):
        pos = position
    else:
        if "com" in position:
            pos = community_pos(n_nodes, **kwargs)
        elif "bip" in position:
            pos = nx.bipartite_layout(mytoy_graph, **kwargs)
        elif "square" in position:
            pos = square_community_pos(n_nodes // 4, 4)

    node_color = "tab:blue"
    if community_vector is not None:
        if isinstance(community_vector[0], float):
            mycm = plt.cm.get_cmap("coolwarm", cm_scale)
            node_color = mycm(
                (
                    cm_scale // 2
                    + np.trunc(cm_scale * community_vector)
                    + np.sign(community_vector)
                ).astype(int)
            )
        else:
            if colors is None:
                mycm = plt.cm.get_cmap("tab10")
                node_color = mycm(community_vector)
            else:
                node_color = [colors[i] for i in community_vector]

    # nx.draw(
    #    mytoy_graph,
    #    pos=pos,
    #    with_labels=True,
    #    ax=axis,
    #    arrowsize=20,
    #    node_color=node_color,
    #    # connectionstyle="arc3,rad=0.1",
    # )

    draw_graph(
        mytoy_graph,
        pos=pos,
        ax=axis,
        node_kwds={"node_size": node_size, "node_color": node_color},
        edge_kwds=edge_kw,
        label_kwds={"labels": labels},
    )

    return axis


def plot_graph(
    a_mat: np.ndarray,
    position: str = "com",
    draw_half_line: bool = False,
    community_vector: Optional[np.ndarray] = None,
    edge_alpha: float = 0.1,
    labels: Optional[dict] = None,
    colors: Optional[np.ndarray] = None,
    **kwargs,
):
    n_nodes = len(a_mat)

    fig, ax = plt.subplots(ncols=3, figsize=(20, 8))

    ax[0].imshow(a_mat)

    if draw_half_line:
        ax[0].plot([-0.5, n_nodes - 0.5], [n_nodes / 2 - 0.5] * 2, c="w")
        ax[0].plot([n_nodes / 2 - 0.5] * 2, [-0.5, n_nodes - 0.5], c="w")

    mytoy_graph = nx.Graph(a_mat)

    pos = nx.spring_layout(mytoy_graph)
    if isinstance(position, dict):
        pos = position
    else:
        if "com" in position:
            pos = community_pos(n_nodes, **kwargs)
        elif "bip" in position:
            pos = nx.bipartite_layout(mytoy_graph, **kwargs)
        elif "square" in position:
            pos = square_community_pos(n_nodes // 4, 4)

    node_color = "tab:blue"
    if community_vector is not None:
        if isinstance(community_vector[0], float):
            mycm = plt.cm.get_cmap("coolwarm", 20)
            # node_color = mycm(1 + np.sign(np.around(community_vector, 3)))
            node_val = (10 + 30 * community_vector).astype(int)
            node_color = mycm(node_val)
        else:
            if colors is None:
                mycm = plt.cm.get_cmap("tab10")
                node_color = mycm(community_vector)
            else:
                node_color = [colors[i] for i in community_vector]

    if labels is None:
        labels = {i: i for i in range(n_nodes)}

    draw_graph(
        mytoy_graph,
        pos=pos,
        ax=ax[1],
        node_kwds={"node_color": node_color},
        edge_kwds={"alpha": edge_alpha},
        label_kwds={"labels": labels},
    )

    mytoy_graph = nx.DiGraph(a_mat)

    draw_graph(
        mytoy_graph,
        pos=pos,
        ax=ax[2],
        node_kwds={"node_color": node_color},
        edge_kwds={"alpha": edge_alpha, "arrowsize": 20, "connectionstyle": "arc3"},
        label_kwds={"labels": labels},
    )

    ax[1].set_axis_off()
    ax[2].set_axis_off()

    return fig, ax


def plot_graph_community(
    a_mat: np.ndarray,
    community_vector: np.ndarray,
    position: str = "com",
    draw_half_line: bool = False,
    **kwargs,
):
    n_nodes = len(a_mat)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 8))

    mytoy_graph = nx.Graph(a_mat)

    pos = nx.spring_layout(mytoy_graph)
    if "com" in position:
        pos = community_pos(n_nodes, **kwargs)
    elif "bip" in position:
        pos = nx.bipartite_layout(mytoy_graph, **kwargs)

    nx.draw(mytoy_graph, pos=pos, with_labels=True, ax=ax[0])

    for ax_i, com_vec in enumerate(community_vector):
        if isinstance(com_vec[0], float):
            cmap = "coolwarm"
            max_val = np.max(np.abs(com_vec))
            vmin = -max_val
            vmax = max_val
        else:
            cmap = "tab10"
            vmin = com_vec.min()
            vmax = com_vec.max()

        ax[1 + ax_i].set_title(f"Community {ax_i+1}")
        mytoy_graph = nx.DiGraph(a_mat)
        nx.draw(
            mytoy_graph,
            pos=pos,
            with_labels=True,
            ax=ax[1 + ax_i],
            arrowsize=20,
            node_color=com_vec,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    return fig, ax


def plot_bipartite_SVD(
    adj: np.ndarray,
    U: np.ndarray,
    Vh: np.ndarray,
    vector_id: int = 0,
    node_size: int = 300,
    edge_alpha: float = 0.1,
    clusters: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[str] = None,
    ax=None,
    **kwargs,
):

    # Building bipartite graph
    graph_bip = np.hstack([np.zeros_like(adj), adj])
    graph_bip = np.vstack([graph_bip, np.zeros_like(graph_bip)])

    # Sorting rows by U and columns by Vh
    sort_by_U = np.argsort(U[:, vector_id])
    sort_by_V = np.argsort(Vh[vector_id])

    row_ids = np.concatenate([sort_by_U, len(sort_by_U) + sort_by_U])
    col_ids = np.concatenate([sort_by_V, len(sort_by_V) + sort_by_V])

    sorted_graph_bip = graph_bip[row_ids][:, col_ids]

    if clusters is None:
        com_vec = np.concatenate([U[sort_by_U, vector_id], Vh[vector_id, sort_by_V]])
    else:
        com_vec = np.concatenate([clusters, clusters])
        # com_vec = np.concatenate([clusters[sort_by_U], clusters[sort_by_V]])

    no_labels = {i: "" for i in range(graph_bip.shape[0])}

    pos_x = [
        ((2 * i) // (graph_bip.shape[0])) * 2 - 1 for i in range(graph_bip.shape[0])
    ]
    pos_y = np.concatenate([U[:, vector_id], Vh[vector_id]])

    pos = {i: (x, y) for i, (x, y) in enumerate(zip(pos_x, pos_y))}
    # pos = {i: (((2 * i) // (graph_bip.shape[0])) * 2 - 1, ) for i in range(graph_bip.shape[0])}

    fig, axes = plt.subplots(figsize=(8, 10))
    axes = custom_draw(
        axes,
        sorted_graph_bip,
        position=pos,
        # position="bipartite",
        nodes=np.arange(graph_bip.shape[0] // 2),
        community_vector=com_vec,
        node_size=node_size,
        edge_alpha=edge_alpha,
        labels=no_labels,
        colors=colors,
        **kwargs,
    )

    if labels is not None:
        side_x = np.array([-1.1, 1.1])

        for side_id, (side, vect) in enumerate(zip([sort_by_U, sort_by_V], [U, Vh.T])):
            for i, label in enumerate(labels[side]):
                axes.text(
                    side_x[side_id],
                    vect[side[i], vector_id],
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="k",
                )

    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    return fig, ax


def plot_cluster_bipartite(
    adj: np.ndarray,
    U: np.ndarray,
    Vh: np.ndarray,
    clusters: np.ndarray,
    vector_id: int = 0,
    node_size: int = 300,
    edge_alpha: float = 0.1,
    colors: Optional[np.ndarray] = None,
    labels: Optional[str] = None,
    ax=None,
    **kwargs,
):

    # Building bipartite graph
    graph_bip = np.hstack([np.zeros_like(adj), adj])
    graph_bip = np.vstack([graph_bip, np.zeros_like(graph_bip)])

    sorting_df = pd.DataFrame(
        {
            "cluster": clusters,
            "U": np.sign((U[:, vector_id])),
            "V": np.sign((Vh[vector_id])),
            "label": labels,
        }
    )
    sort_by_U = sorting_df.sort_values(["U", "cluster"]).index.to_numpy()
    sort_by_V = sorting_df.sort_values(["V", "cluster"]).index.to_numpy()

    # Sorting rows by U and columns by Vh
    sort_by_com = np.argsort(clusters)

    # row_ids = np.concatenate([sort_by_com, len(sort_by_com) + sort_by_com])
    row_ids = np.concatenate([sort_by_U, len(sort_by_U) + sort_by_U])
    col_ids = np.concatenate([sort_by_V, len(sort_by_V) + sort_by_V])

    sorted_graph_bip = graph_bip[row_ids][:, col_ids]

    # com_vec = np.concatenate([U[sort_by_com, vector_id], Vh[vector_id, sort_by_com]])
    com_vec = np.concatenate([U[sort_by_U, vector_id], Vh[vector_id, sort_by_V]])

    no_labels = {i: "" for i in range(graph_bip.shape[0])}

    pos_y = [
        -(((2 * i) // (graph_bip.shape[0])) * 2 - 1) for i in range(graph_bip.shape[0])
    ]

    pos_x = np.concatenate([np.arange(graph_bip.shape[0] // 2)] * 2)
    # pos_x = np.concatenate([sort_by_U, sort_by_V])

    # space = 5
    # pos_y = np.concatenate([np.arange(0, space * graph_bip.shape[0] // 2, space)] * 2)

    pos = {i: (x, y) for i, (x, y) in enumerate(zip(pos_x, pos_y))}

    # fig, axes = plt.subplots(figsize=(8, 30))
    fig, axes = plt.subplots(figsize=(30, 8))
    axes = custom_draw(
        axes,
        sorted_graph_bip,
        position=pos,
        # position="bipartite",
        nodes=np.arange(graph_bip.shape[0] // 2),
        community_vector=com_vec,
        node_size=node_size,
        edge_alpha=edge_alpha,
        cm_scale=3,
        labels=no_labels,
        direct=False,
        **kwargs,
    )

    if labels is not None:

        # for i, label in enumerate(labels[sort_by_com]):
        #     axes.text(
        #         -1.08,
        #         pos_y[i],
        #         label,
        #         ha="right",
        #         va="center",
        #         fontsize=8,
        #         color="k",
        #     )

        side_x = np.array([-1.08, 1.08])
        va = ["top", "bottom"]
        for side_id in [0, 1]:
            for i, label in enumerate(labels[[sort_by_V, sort_by_U][side_id]]):
                axes.text(
                    # side_x[side_id],
                    # pos_y[i],
                    pos_x[i],
                    side_x[side_id],
                    label,
                    ha="center",
                    va=va[side_id],
                    rotation=90,
                    fontsize=6,
                    color="k",
                )

    scatter_pos = np.array(list(pos.values())).T

    plt.scatter(
        scatter_pos[0],
        1.05 * scatter_pos[1],
        c=np.concatenate([clusters[sort_by_U], clusters[sort_by_V]]),
        s=30,
        marker="s",
        cmap="Set1",
        vmax=9,
    )

    cumsum_start = -0.5
    cluster_series = pd.Series(clusters)

    # for n_type, c in zip(np.unique(clusters), colors):
    #
    #    cumsum_end = cumsum_start + cluster_series.value_counts()[n_type]
    #    # axes.plot([-1, 1], [cumsum_end] * 2, lw=4, color="tab:green", alpha=0.5)
    #    # axes.plot([-1.05] * 2, [cumsum_start, cumsum_end], lw=6, color=c, alpha=0.8)
    #    # axes.plot([1.05] * 2, [cumsum_start, cumsum_end], lw=6, color=c, alpha=0.8)
    #
    #    axes.plot([cumsum_start, cumsum_end], [-1.05] * 2, lw=6, color=c, alpha=0.8)
    #    axes.plot([cumsum_start, cumsum_end], [1.05] * 2, lw=6, color=c, alpha=0.8)
    #    cumsum_start = cumsum_end

    # plt.axis([-1.2, 1.2, -1, len(clusters) + 1])
    plt.axis([-1, len(clusters) + 1, -1.3, 1.3])
    # plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    return fig, ax


def plot_flow(
    adj: np.ndarray,
    U: np.ndarray,
    Vh: np.ndarray,
    clusters: np.ndarray,
    vector_id: int = 0,
    node_size: int = 300,
    edge_alpha: float = 0.1,
    colors: Optional[np.ndarray] = None,
    labels: Optional[str] = None,
    ax=None,
    **kwargs,
):
    pos_x = []

    for i, cluster_val in enumerate(np.unique(clusters)):
        cluster_mask = clusters == cluster_val

    # Sorting rows by U and columns by Vh
    sort_by_com = np.argsort(clusters)

    row_ids = np.concatenate([sort_by_com, len(sort_by_com) + sort_by_com])

    sorted_graph_bip = graph_bip[row_ids][:, row_ids]

    com_vec = np.concatenate([U[sort_by_com, vector_id], Vh[vector_id, sort_by_com]])

    no_labels = {i: "" for i in range(graph_bip.shape[0])}

    pos_x = [
        ((2 * i) // (graph_bip.shape[0])) * 2 - 1 for i in range(graph_bip.shape[0])
    ]

    pos_y = np.concatenate([np.arange(graph_bip.shape[0] // 2)] * 2)
    # space = 5
    # pos_y = np.concatenate([np.arange(0, space * graph_bip.shape[0] // 2, space)] * 2)

    pos = {i: (x, y) for i, (x, y) in enumerate(zip(pos_x, pos_y))}

    fig, axes = plt.subplots(figsize=(8, 30))
    axes = custom_draw(
        axes,
        sorted_graph_bip,
        position=pos,
        # position="bipartite",
        nodes=np.arange(graph_bip.shape[0] // 2),
        community_vector=com_vec,
        node_size=node_size,
        edge_alpha=edge_alpha,
        cm_scale=3,
        labels=no_labels,
        **kwargs,
    )

    if labels is not None:

        for i, label in enumerate(labels[sort_by_com]):
            axes.text(
                -1.08,
                pos_y[i],
                label,
                ha="right",
                va="center",
                fontsize=8,
                color="k",
            )

    cumsum_start = -0.5
    cluster_series = pd.Series(clusters)

    for n_type, c in zip(np.unique(clusters), colors):

        cumsum_end = cumsum_start + cluster_series.value_counts()[n_type]
        # axes.plot([-1, 1], [cumsum_end] * 2, lw=4, color="tab:green", alpha=0.5)
        axes.plot([-1.05] * 2, [cumsum_start, cumsum_end], lw=6, color=c, alpha=0.8)
        axes.plot([1.05] * 2, [cumsum_start, cumsum_end], lw=6, color=c, alpha=0.8)
        cumsum_start = cumsum_end

    # plt.axis([-1.2, 1.2, -1, len(clusters) + 1])
    # plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    return fig, ax


def SVD_summary(
    a_mat,
    nulls_names: Optional[list] = None,
    reconstructions_ids: list = [0, 1, -1],
    show_mod: bool = True,
    vector_subsample: int = 5,
    sort_by_q: bool = False,
    add_undirected: list = [],
) -> list[tuple]:
    # Overview of SVD decomposition of the modularity matrix

    all_nulls_names = nulls_names.copy()
    if nulls_names is None:
        all_nulls_names = ["z_in", "z_out", "z_inout", "z_outin"]

    if isinstance(add_undirected, str):
        add_undirected = [add_undirected]

    all_nulls_names += ["z_in"] * len(add_undirected)

    n_reco = len(reconstructions_ids)

    mod_id = 1
    if show_mod:
        mod_id = 2

    _, ax = plt.subplots(
        nrows=len(all_nulls_names),
        ncols=mod_id + 3 + n_reco,
        figsize=(5 * (n_reco + 2), 4 * len(all_nulls_names)),
    )

    # Make it work when all_nulls_names has only 1 element
    ax = np.atleast_2d(ax)

    all_svds = []

    undir_i = 0

    # for i_null, (null, null_name) in enumerate(zip(all_nulls, all_nulls_names)):
    for i_null, null_name in enumerate(all_nulls_names):

        if i_null == len(all_nulls_names) - len(add_undirected) + undir_i:
            if add_undirected[undir_i] == "sum":
                a_undir = a_mat + a_mat.T
                ax[i_null, 0].set_title("$A+A^T$")
            elif add_undirected[undir_i] == "ATA":
                a_undir = a_mat.T @ a_mat
                ax[i_null, 0].set_title(r"$A^T\dot A$")
            elif add_undirected[undir_i] == "AAT":
                a_undir = a_mat @ a_mat.T
                ax[i_null, 0].set_title(r"$A\dot A^T$")
            undir_i += 1

            null = configuration_null(a_undir, null_model=null_name.split("_")[-1])
            mod_mat = a_undir - null / a_undir.sum()

            ax[i_null, 0].matshow(a_undir)

        else:
            null = configuration_null(a_mat, null_model=null_name.split("_")[-1])
            mod_mat = a_mat - null / a_mat.sum()

            ax[i_null, 0].matshow(a_mat)
            ax[i_null, 0].set_title("$A$")

        ax[i_null, 1].matshow(null)
        ax[i_null, 1].set_title(null_name)

        if show_mod:
            ax[i_null, mod_id].matshow(mod_mat)
            ax[i_null, mod_id].set_title("$M=A-Z$")

        U, S, Vh = sorted_SVD(mod_mat, sort_by_q=sort_by_q)

        all_svds.append((U, S, Vh))

        # SVD Reconstructions
        for i_rec, rec_id in enumerate(reconstructions_ids):
            s_vec = np.zeros(len(a_mat))
            s_vec[rec_id] = S[rec_id]

            rec = U @ np.diag(s_vec) @ Vh

            q = modularity_quadratic(mod_mat, U[:, rec_id])

            ax[i_null, mod_id + 1 + i_rec].matshow(
                rec, vmin=-0.1, vmax=0.1, cmap="coolwarm"
            )
            ax[i_null, mod_id + 1 + i_rec].set_title(
                f"i={rec_id}, S={S[rec_id]:1.2f}, Q={q:1.2f}"
            )

        # SVD Vectors
        maxval = 0.5

        u_quad = modularity_quadratic(mod_mat, U[:, 0])
        v_quad = modularity_quadratic(mod_mat, Vh[0])

        u_id = mod_id + 1 + n_reco
        ax[i_null, u_id].imshow(
            np.vstack([S[:vector_subsample], U[:, :vector_subsample]]),
            cmap="coolwarm",
            vmin=-maxval,
            vmax=maxval,
        )
        ax[i_null, u_id].plot(
            [-0.5, len(S[:vector_subsample]) - 0.5], [0.5, 0.5], lw=2, c="k"
        )
        ax[i_null, u_id].set_title(f"$U$, $Q={u_quad:1.5f}$")

        v_id = u_id + 1
        ax[i_null, v_id].imshow(
            np.hstack([S.reshape((-1, 1))[:vector_subsample], Vh[:vector_subsample]]),
            cmap="coolwarm",
            vmin=-maxval,
            vmax=maxval,
        )
        ax[i_null, v_id].plot(
            [0.5, 0.5], [-0.5, len(S[:vector_subsample]) - 0.5], lw=2, c="k"
        )
        ax[i_null, v_id].set_title(f"$V^H$, $Q={v_quad:1.5f}$")
    return all_svds


def plot_benchmark_results(
    benchmark_results: np.ndarray,
    e_prob_range: list,
    con_prob_range: list,
    out_prob_range: list,
    null_model: str = "z_in",
):
    fig, axes = plt.subplots(
        1, len(out_prob_range), figsize=(5 * (1 + len(out_prob_range)), 5)
    )

    fig.suptitle(
        f"SVD prediction with different output probabilities - Null model: {null_model}"
    )

    for ax_i, ax in enumerate(axes):
        image = ax.imshow(
            benchmark_results.mean(axis=0)[:, :, ax_i, 0], vmin=0, vmax=10
        )
        plt.colorbar(image, ax=ax)

        ax.set_title(f"Prediction errors (#) - out_prob={out_prob_range[ax_i]}")
        ax.set_xlabel("connection probability")
        ax.set_xticks(
            np.arange(len(con_prob_range)), labels=[f"{i:1.2f}" for i in con_prob_range]
        )
        ax.set_ylabel("edge probability")
        ax.set_yticks(
            np.arange(len(e_prob_range)), labels=[f"{i:1.2f}" for i in e_prob_range]
        )

    return


def plot_s_and_q(a_mat, null_model="in", fix_negative=True):
    mod_mat = modularity_matrix(a_mat, null_model=null_model)
    U, S, Vh = sorted_SVD(mod_mat, fix_negative=fix_negative)

    q_ui = np.array([modularity_quadratic(mod_mat, U[:, i]) for i in range(len(S))])
    q_vi = np.array([modularity_quadratic(mod_mat, Vh[i]) for i in range(len(S))])

    fig, axes = plt.subplots(figsize=(8, 8))
    axes.plot(S, label="Singular values", lw=4)
    axes.plot(q_ui, label="Modularity index (U)", lw=2, ls="--")
    axes.plot(q_vi, label="Modularity index (V)", lw=2, ls=":")

    axes.set_xlabel("Singular value/vector index")
    axes.set_ylabel("Value")

    axes.legend()

    return


def plot_singular_against_modality(a_mat, null_model="in", fix_negative=True):

    mod_mat = modularity_matrix(a_mat, null_model=null_model)
    U, S, Vh = sorted_SVD(mod_mat, fix_negative=fix_negative)

    mod_values = np.zeros_like(S)

    for rec_id in np.arange(mod_mat.shape[0]):
        ui = U[:, rec_id]
        mod_values[rec_id] = modularity_quadratic(mod_mat, ui)

    _, axes = plt.subplots(figsize=(8, 8))

    axes.scatter(S, mod_values, c=np.arange(len(S)), cmap="plasma", s=100)
    axes.plot([S.min(), S.max()], [S.min(), S.max()], c="k", ls="--")
    axes.set_xlabel("Singular values")
    axes.set_ylabel("Modularity index")


def plot_svect_evolution(
    u_list: np.ndarray,
    v_list: np.ndarray,
    dir_edge_list: np.ndarray,
    n_subset: int = 10,
    offset: float = 0.15,
) -> None:

    lowest_sub = min([n_subset, len(u_list)])

    _, ax_solo = plt.subplots(1, 1, figsize=(25, 5 + lowest_sub))

    line_styles = [":", "--"]
    line_labels = ["$u_1$", "$v_1$"]
    scatter_mark = ["$u$", "$v$"]

    for vec_id, vect in enumerate([u_list, np.moveaxis(v_list, 1, 2)]):
        colors = plt.colormaps["plasma"](np.linspace(0, 1, lowest_sub))

        for i, u in enumerate(vect[1:lowest_sub, :, 0]):
            if offset > 0:
                ax_solo.plot(vect[0, :, 0] - i * offset, lw=1, c="tab:green")

            ax_solo.scatter(
                dir_edge_list[i][vec_id, 0],
                u[dir_edge_list[i][vec_id, 0]] - i * offset,
                s=150,
                marker=scatter_mark[vec_id],
                color=colors[i],
            )
            ax_solo.plot(
                u - i * offset, line_styles[vec_id], lw=3, c=colors[i], alpha=0.8
            )

    custom_legend = [
        Line2D([0], [0], ls=style, color=colors[0], lw=2) for style in line_styles
    ]

    ax_solo.legend(custom_legend, line_labels, handlelength=6)


def plot_svect_evolution_stem(
    u_list: np.ndarray,
    v_list: np.ndarray,
    dir_edge_list: np.ndarray,
    n_subset: int = 10,
    offset: float = 0.15,
) -> None:

    lowest_sub = min([n_subset, len(u_list)])

    _, ax_solo = plt.subplots(1, 1, figsize=(25, 5 + lowest_sub))

    line_styles = [":", "--"]
    line_labels = ["$u_1$", "$v_1$"]
    scatter_mark = ["o", "s"]

    for vec_id, vect in enumerate([u_list, np.moveaxis(v_list, 1, 2)]):
        colors = plt.colormaps["plasma"](np.linspace(0, 1, lowest_sub))

        for i, u in enumerate(vect[1:lowest_sub, :, 0]):
            diff = vect[0, :, 0] - u
            mark, stem, base = ax_solo.stem(
                diff - i * offset,
                bottom=-i * offset,
                markerfmt="",
                linefmt=line_styles[vec_id],
            )
            base.set_color(colors[i])
            stem.set_color(colors[i])
            stem.set_linewidth(2)
            ax_solo.scatter(
                dir_edge_list[i][vec_id, 0],
                diff[dir_edge_list[i][vec_id, 0]] - i * offset,
                s=50,
                marker=scatter_mark[vec_id],
                edgecolor=colors[i],
                color="none",
            )

    custom_legend = [
        Line2D([0], [0], ls=style, color=colors[0], lw=2) for style in line_styles
    ]

    ax_solo.legend(custom_legend, line_labels, handlelength=6)

    ax_solo.set_yticks(
        -offset * np.arange(lowest_sub - 1), labels=np.arange(lowest_sub - 1) + 1
    )
    ax_solo.set_ylabel("Number of directed edges")


def add_cbar(fig, ax, **kwargs):
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
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical", **kwargs)
    return fig, ax
