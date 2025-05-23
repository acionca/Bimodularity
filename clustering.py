import numpy as np

from tqdm.notebook import tqdm

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from joblib import Parallel, delayed

from dgsp import modularity_matrix, sorted_SVD


def get_op_SVD(adj, operator="modularity", norm=False):
    if "mod" in operator:
        op_mat = modularity_matrix(adj)
    elif "lap" in operator:
        op_mat = np.diag(adj.sum(axis=1)) - adj
        if norm:
            op_mat = op_mat / adj.sum(axis=1)
    elif "adj" in operator:
        op_mat = adj

    U, S, Vh = sorted_SVD(op_mat)
    V = Vh.T

    if "lap" in operator:
        U = np.flip(U, axis=1)
        S = np.flip(S)
        V = np.flip(V, axis=1)

    return U, S, V


def get_recon_mat(adj, comp_range, operator="modularity", **kwargs):
    U, _, V = get_op_SVD(adj, operator=operator, **kwargs)

    uk = U[:, comp_range[0] : comp_range[1]]
    vk = V[:, comp_range[0] : comp_range[1]]

    # Project and reconstruct
    C_mat = np.block([[uk, np.zeros_like(vk)], [np.zeros_like(uk), vk]])
    C_pinv = np.linalg.pinv(C_mat)

    rec_mat = C_mat @ C_pinv

    return rec_mat


def edge_signal_product(
    adj, x_concat, comp_range, operator="modularity", sqrt=False, **kwargs
):
    n_nodes = adj.shape[0]
    if "naive" in operator:
        rec_mat = np.eye(2 * len(adj))
    else:
        rec_mat = get_recon_mat(adj, comp_range, operator=operator, **kwargs)

    C_rec = rec_mat @ x_concat.T

    # Outer product and edge masking
    # outer1 = np.array([np.outer(rec[:n_nodes], rec[n_nodes:]) for rec in C_rec.T])
    # edge_assignments_vec = outer[..., adj != 0]
    outer = C_rec[:n_nodes, None, :] * C_rec[None, n_nodes:, :]

    if sqrt:
        outer = np.sign(outer) * np.sqrt(np.abs(outer))
    edge_assignments_vec = outer[adj != 0].T

    return edge_assignments_vec


def norm_clustering(x_proj, k_clust, norm=True):

    norm_val = 1
    if norm:
        norm_val = np.linalg.norm(x_proj, axis=1)[:, None]

    kmeans = KMeans(n_clusters=k_clust).fit(x_proj / norm_val)

    return kmeans.labels_, kmeans.cluster_centers_


def prepare_benchmark(
    graph,
    x_test,
    x_retest,
    all_comps=np.arange(2, 10, 1),
    all_operators=["modularity", "laplacian"],
    start_comp=0,
    sqrt=False,
):
    all_edge_test = np.zeros(
        (len(all_comps), len(all_operators), len(x_test), np.sum(graph != 0))
    )
    all_edge_retest = np.zeros(
        (len(all_comps), len(all_operators), len(x_retest), np.sum(graph != 0))
    )

    if isinstance(start_comp, int):
        start_comp = [start_comp] * len(all_operators)

    for n, n_comp in enumerate(all_comps):
        print(f"Computing for {n_comp} components")
        for i, operator in enumerate(all_operators):
            e_both = edge_signal_product(
                graph,
                np.concatenate([x_test, x_retest]),
                (start_comp[i], start_comp[i] + n_comp),
                operator=operator,
                norm=True,
                sqrt=sqrt,
            )
            all_edge_test[n, i] = e_both[: len(x_test)]
            all_edge_retest[n, i] = e_both[len(x_test) :]

    return all_edge_test, all_edge_retest


def benchmark_clustering(
    graph,
    x_test,
    x_retest,
    all_comps=np.arange(2, 10, 1),
    all_k_clusters=np.arange(2, 15, 1),
    all_operators=["modularity", "laplacian"],
    start_comp=0,
    norm=True,
):

    test_labels = np.zeros(
        (len(all_k_clusters), len(all_comps), len(all_operators), len(x_test))
    )
    retest_labels = np.zeros(
        (len(all_k_clusters), len(all_comps), len(all_operators), len(x_retest))
    )

    test_centroids = [
        np.zeros((len(all_comps), len(all_operators), k, np.sum(graph != 0)))
        for k in all_k_clusters
    ]
    retest_centroids = [
        np.zeros((len(all_comps), len(all_operators), k, np.sum(graph != 0)))
        for k in all_k_clusters
    ]

    if isinstance(start_comp, int):
        start_comp = [start_comp] * len(all_operators)

    pbar = tqdm(total=len(all_comps) * len(all_operators) * len(all_k_clusters))

    for n, n_comp in enumerate(all_comps):
        print(f"Computing for {n_comp} components")
        for i, operator in enumerate(all_operators):
            edge_test = edge_signal_product(
                graph,
                x_test,
                (start_comp[i], start_comp[i] + n_comp),
                operator=operator,
                norm=True,
            )
            edge_retest = edge_signal_product(
                graph,
                x_retest,
                (start_comp[i], start_comp[i] + n_comp),
                operator=operator,
                norm=True,
            )

            for k, k_clust in enumerate(all_k_clusters):
                test_labels[k, n, i], test_centroids[k][n, i] = norm_clustering(
                    edge_test, k_clust, norm=norm
                )
                retest_labels[k, n, i], retest_centroids[k][n, i] = norm_clustering(
                    edge_retest, k_clust, norm=norm
                )
                pbar.update(1)
    pbar.close()

    return test_labels, retest_labels, test_centroids, retest_centroids


def benchmark_clustering_fast(
    stat_test_edge,
    stat_retest_edge,
    all_k_clusters=np.arange(2, 15, 1),
    norm=True,
):
    n_comp, n_op, n_t_test, n_edges = stat_test_edge.shape
    n_t_retest = stat_retest_edge.shape[2]

    test_labels = np.zeros((len(all_k_clusters), n_comp, n_op, n_t_test))
    retest_labels = np.zeros((len(all_k_clusters), n_comp, n_op, n_t_retest))

    test_centroids = [np.zeros((n_comp, n_op, k, n_edges)) for k in all_k_clusters]
    retest_centroids = [np.zeros((n_comp, n_op, k, n_edges)) for k in all_k_clusters]

    pbar = tqdm(total=n_comp * n_op * len(all_k_clusters))
    for n in range(n_comp):
        for i in range(n_op):
            for k, k_clust in enumerate(all_k_clusters):
                test_labels[k, n, i], test_centroids[k][n, i] = norm_clustering(
                    stat_test_edge[n, i], k_clust, norm=norm
                )
                retest_labels[k, n, i], retest_centroids[k][n, i] = norm_clustering(
                    stat_retest_edge[n, i], k_clust, norm=norm
                )
                pbar.update(1)
    pbar.close()

    return test_labels, retest_labels, test_centroids, retest_centroids


def get_centroid_similarities(t_centroids, rt_centroids):
    all_ks = [cent.shape[2] for cent in t_centroids]
    comp_n = t_centroids[0].shape[0]
    op_n = t_centroids[0].shape[1]

    all_cent_match = [np.zeros((comp_n, op_n, k, k)) for k in all_ks]
    all_diag_means = np.zeros((len(all_ks), comp_n, op_n), dtype=float)
    all_off_diag_means = np.zeros((len(all_ks), comp_n, op_n), dtype=float)

    for k, k_clust in enumerate(all_ks):
        for n in range(comp_n):
            for i in range(op_n):
                corr_input = np.vstack([t_centroids[k][n, i], rt_centroids[k][n, i]])
                corr = np.corrcoef(corr_input)[:k_clust][:, k_clust:]

                row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
                all_cent_match[k][n, i] = corr[row_ind][:, col_ind]

                all_diag_means[k, n, i] = np.diag(all_cent_match[k][n, i]).mean()
                # all_off_diag_means[k, n, i] = all_cent_match[k][n, i][~np.eye(k_clust, dtype=bool)].mean()
                all_off_diag_means[k, n, i] = np.abs(
                    all_cent_match[k][n, i][~np.eye(k_clust, dtype=bool)]
                ).mean()
    return all_cent_match, all_diag_means, all_off_diag_means
