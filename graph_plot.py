import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_brain_graph(
    graph, node_signal=None, edge_signal=None, layout="transverse", fig=None, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    return fig, ax


def plot_centroids(
    edge_centroids,
    adj,
    labels,
    clust_cmap,
    node_pos,
    e_cmap_name="turbo",
    plot_sag=False,
):
    n_clusters = edge_centroids.shape[0]
    if plot_sag:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=n_clusters,
            # figsize=(20, 10),
            figsize=(n_clusters * 3, 13),
            gridspec_kw={"hspace": 0, "wspace": 0.02, "height_ratios": [1, 1, 0.5]},
        )
    else:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=n_clusters,
            # figsize=(20, 10),
            figsize=(n_clusters * 3, 10),
            gridspec_kw={"hspace": 0, "wspace": 0.02},
        )

    edge_list = nx.DiGraph(adj).edges()
    edge_cmap = plt.get_cmap(e_cmap_name)

    for ax in axes.flat:
        ax.axis("off")

    for k in range(n_clusters):
        centroid_mat = np.zeros_like(adj, dtype=float)
        centroid_mat[adj != 0] = edge_centroids[k]

        perc = np.sum(labels == k) / len(labels)

        maxval = np.abs(centroid_mat).max()

        axes[0, k].imshow(
            centroid_mat,
            cmap=e_cmap_name,
            interpolation="none",
            vmin=-maxval,
            vmax=maxval,
        )
        axes[0, k].plot([-0.5, len(adj) - 0.5], [-3, -3], color=clust_cmap(k), lw=4)
        # axes[0, k].set_title(f"Cluster {k+1} $({100*perc:2.2f}\%)$", fontsize=16)
        axes[0, k].axis("off")

        alpha = np.abs(edge_centroids[k])  # **2
        alpha = alpha / np.max(alpha)

        e_order = np.argsort(np.abs(edge_centroids[k]))

        axes[1, k].set_title(f"Cluster {k+1} $({100*perc:2.2f}\%)$", fontsize=16)
        axes[1, k].scatter(
            node_pos[:, 0],
            node_pos[:, 1],
            s=20,
            color="silver",
            edgecolors="k",
            linewidths=1,
            zorder=2,
        )

        graph_pos = {
            i: (x, y) for i, (x, y) in enumerate(zip(node_pos[:, 0], node_pos[:, 1]))
        }
        nx.draw_networkx_edges(
            nx.Graph(centroid_mat),
            pos=graph_pos,
            ax=axes[1, k],
            alpha=alpha[e_order],
            edge_color=edge_centroids[k][e_order],
            edgelist=np.array(edge_list)[e_order],
            edge_cmap=edge_cmap,
            edge_vmin=-maxval,
            edge_vmax=maxval,
        )

        if plot_sag:
            axes[2, k].scatter(
                node_pos[:, 1],
                node_pos[:, 2],
                s=20,
                color="silver",
                edgecolors="k",
                linewidths=1,
                zorder=2,
            )
            graph_pos = {
                i: (x, y)
                for i, (x, y) in enumerate(zip(node_pos[:, 1], node_pos[:, 2]))
            }
            nx.draw_networkx_edges(
                nx.Graph(centroid_mat),
                pos=graph_pos,
                ax=axes[2, k],
                alpha=alpha[e_order],
                edge_color=edge_centroids[k][e_order],
                edgelist=np.array(edge_list)[e_order],
                edge_cmap=edge_cmap,
                edge_vmin=-maxval,
                edge_vmax=maxval,
            )

    return fig, axes
