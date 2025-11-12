# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from typing import Optional, Union
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from graph_examples import toy_random


# Some directed modularity tools
def configuration_null(a_mat: np.ndarray, null_model: str = "outin"):
    k_in = a_mat.sum(axis=0).reshape((1, -1))
    k_out = a_mat.sum(axis=1).reshape((1, -1))

    if null_model == "in":
        z = k_in.T @ k_in
    elif null_model == "out":
        z = k_out.T @ k_out
    elif null_model == "inout":
        z = k_in.T @ k_out
    elif null_model == "outin":
        z = k_out.T @ k_in
    elif null_model == "avg":
        z = (k_in.T @ k_in + k_out.T @ k_out) / 2
    elif null_model == "send":
        z = send_receive_probability(a_mat)[0]
    elif null_model == "receive":
        z = send_receive_probability(a_mat)[1]

    return z / a_mat.sum()


def send_receive_probability(adj: np.ndarray):
    n_edges = adj.sum()

    in_deg = np.atleast_2d(adj.sum(axis=0))
    out_deg = np.atleast_2d(adj.sum(axis=1))

    in_deg_squared = (in_deg**2).sum()
    out_deg_squared = (out_deg**2).sum()

    send_prob = np.outer(out_deg, out_deg) * in_deg_squared / n_edges**2
    receive_prob = np.outer(in_deg, in_deg) * out_deg_squared / n_edges**2

    return send_prob, receive_prob


def modularity_matrix(a_mat: np.ndarray, null_model: str = "outin"):

    z = configuration_null(a_mat, null_model)

    return a_mat - z


def modularity_quadratic(mod_mat, signal):

    signal = signal.reshape((-1, 1))
    mod_value = signal.T @ mod_mat @ signal

    return mod_value[0, 0]


def sorted_SVD(matrix: np.ndarray, fix_negative: bool = False, sort_by_q: bool = False):

    U, S, Vh = np.linalg.svd(matrix)
    # U, S, Vh = svd(matrix)

    if fix_negative:
        for i, _ in enumerate(matrix):
            if U[:, i].T @ Vh[i] < 0:
                Vh[i] *= -1
                S[i] *= -1

    sort_id = np.flip(np.argsort(S))
    if sort_by_q:
        q_s = S * np.diag(Vh @ U)
        sort_id = np.flip(np.argsort(q_s))

    S = S[sort_id]
    U = U[:, sort_id]
    Vh = Vh[sort_id]

    return U, S, Vh


def make_directed(
    a_mat: np.ndarray,
    n_dir_edge: int,
    p: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:

    triu_idx = np.array(np.triu_indices_from(a_mat, k=1))

    lined_triu = a_mat[*triu_idx]
    lined_tril = a_mat.T[*triu_idx]

    undir_edges_id = np.where(np.logical_and(lined_triu, lined_tril))[0]

    np.random.seed(seed)
    to_direct = np.random.choice(undir_edges_id, n_dir_edge, replace=False)

    a_dir = a_mat.copy()

    dir_in = np.random.choice([True, False], p=[p, 1 - p])

    if dir_in:
        a_dir[*triu_idx[:, to_direct]] = 0
    else:
        a_dir.T[*triu_idx[:, to_direct]] = 0

    return a_dir, triu_idx[:, to_direct]


def random_imbalance(adjacency: np.ndarray, sigma: float = 0.1) -> np.ndarray:

    non_zero = adjacency != 0

    imbalance = np.zeros_like(adjacency, dtype=float)
    imbalance[non_zero] = np.random.normal(0, sigma, non_zero.sum(dtype=int))

    return adjacency + imbalance


def random_edge_decrease(
    adjacency: np.ndarray, sigma: float = 0.1, chosen: Optional[tuple] = None
) -> np.ndarray:

    non_zero = adjacency != 0
    if chosen is None:
        non_zero_id = np.array(np.where(non_zero)).T
        chosen = non_zero_id[np.random.choice(len(non_zero_id), replace=False)]

    adj_imbalance = adjacency.copy()
    adj_imbalance[*chosen] -= sigma

    return adj_imbalance, chosen


def incremental_directed_SVD(
    adjacency: np.ndarray,
    null_model: str = "in",
    initial_seed: int = 123,
    fix_negative: bool = True,
    sort_by_q: bool = False,
    store_adjacency: bool = False,
    **kwargs,
):
    s_list = []
    q_list = []
    u_list = []
    v_list = []
    dir_id_list = []

    adj_list = []

    rand_adj = adjacency.copy()

    while not (rand_adj + rand_adj.T == adjacency).all():

        mod_mat = modularity_matrix(rand_adj, null_model=null_model)
        U, S, Vh = sorted_SVD(mod_mat, fix_negative=fix_negative, sort_by_q=sort_by_q)

        u_list.append(U)
        v_list.append(Vh)
        s_list.append(S)
        q_list.append([modularity_quadratic(mod_mat, U[:, i]) for i, _ in enumerate(S)])

        rand_adj, dir_id = make_directed(
            rand_adj, n_dir_edge=1, seed=initial_seed, **kwargs
        )
        adj_list.append(rand_adj)
        dir_id_list.append(dir_id)

        initial_seed += 1

    if store_adjacency:
        return (
            np.array(u_list),
            np.array(v_list),
            np.array(s_list),
            np.array(q_list),
            dir_id_list,
            np.array(adj_list),
        )
    return (
        np.array(u_list),
        np.array(v_list),
        np.array(s_list),
        np.array(q_list),
        dir_id_list,
    )


def SVD_prediction(
    a_mat: np.ndarray,
    null_model: str = "in",
    predictor_id: int = 0,
    y_true: np.ndarray = None,
    return_dict: bool = True,
) -> Union[dict, np.ndarray]:

    # Get SVD decomposition of the modularity matrix
    mod_mat = modularity_matrix(a_mat, null_model=null_model)
    U, S, Vh = sorted_SVD(mod_mat)

    # Set small values to 0
    tol = 1e-5
    U[np.abs(U) < tol] = 0
    Vh[np.abs(Vh) < tol] = 0

    # Get the modularity value for the selected singular vectors
    u_quad = modularity_quadratic(mod_mat, U[:, predictor_id])
    v_quad = modularity_quadratic(mod_mat, Vh[predictor_id])

    # Find consensus between predictors
    U_pred = np.sign(U[:, predictor_id])
    Vh_pred = np.sign(Vh[predictor_id])

    # Ensure the first element is positive
    Vh_pred = Vh_pred * np.sign(U_pred[0] * Vh_pred[0])

    disagree = U_pred * Vh_pred < 0

    consensus = np.where(
        np.abs(U[:, predictor_id]) > np.abs(Vh[predictor_id]), U_pred, Vh_pred
    )

    # Run the prediction
    if y_true is None:
        y_true = np.hstack([np.ones(len(a_mat) // 2), -np.ones(len(a_mat) // 2)])

    n_error_pos = sum(y_true - consensus != 0)
    n_error_neg = sum(y_true + consensus != 0)

    n_error = min(n_error_pos, n_error_neg)

    if return_dict:
        results = {
            "pred_error": n_error,
            "disagree": sum(disagree),
            "u_quad": u_quad,
            "v_quad": v_quad,
        }

        return results
    return n_error, sum(disagree), u_quad, v_quad


def SVD_benchmark(
    e_prob_range: list,
    con_prob_range: list,
    out_prob_range: list,
    n_nodes: int = 10,
    n_repeats: int = 10,
    **kwargs,
):

    results = np.zeros(
        (n_repeats, len(e_prob_range), len(con_prob_range), len(out_prob_range), 4)
    )
    for i, e_prob in enumerate(e_prob_range):
        for j, con_prob in enumerate(con_prob_range):
            for k, out_prob in enumerate(out_prob_range):
                for rep in range(n_repeats):
                    a_mat = toy_random(
                        n_nodes=n_nodes,
                        edge_prob=e_prob,
                        con_prob=con_prob,
                        out_prob=out_prob,
                        directed=True,
                    )

                    results[rep, i, j, k] = SVD_prediction(
                        a_mat, return_dict=False, **kwargs
                    )

    return results


# Community detection part


def edge_bicommunities(
    adjacency,
    U,
    V,
    n_components,
    n_kmeans=10,
    verbose=False,
    scale_S=None,
    assign_only=False,
    return_centroids=False,
    max_k=20,
    undirected=False,
    return_kmeans=False,
    kwargs_kmeans1={},
    kwargs_kmeans2={},
    clust_only=False,
    d_thresh=0.,
    **kwargs,
) -> tuple:

    n_nodes = adjacency.shape[0]

    if n_components > len(U):
        print(
            f"Warning: `n_components` too large, fixing to {len(U)} (was {n_components})."
        )
        n_components = len(U)

    if scale_S is None:
        scale_S = np.ones(n_components)

    u_features = U[:, :n_components] * scale_S[:n_components]
    v_features = V[:, :n_components] * scale_S[:n_components]

    # Slower method
    # edge_out = np.array([u_features] * n_nodes).T
    # edge_in = np.array([v_features] * n_nodes)
    # edge_in = np.moveaxis(edge_in, -1, 0)
    # edge_assignments = np.concatenate([edge_out, edge_in], axis=0)
    # edge_assignments_vec = edge_assignments[:, adjacency != 0].T

    rows, cols = np.where(adjacency != 0)
    edge_assignments_vec = np.hstack([u_features[rows], v_features[cols]])

    if assign_only:
        return edge_assignments_vec

    if n_kmeans is None:
        n_kmeans = get_best_k(edge_assignments_vec, verbose=verbose, max_k=max_k)

    kmeans = KMeans(n_clusters=n_kmeans, **kwargs_kmeans1).fit(
            edge_assignments_vec
        )
        
    if undirected:
        edge_clusters_inter = kmeans.labels_ + 1

        # Slower method
        # angle_signs = np.sign([v.T @ u for u, v in zip(U[:, :n_components].T, V[:, :n_components].T)])
        # flipped_centers = np.concatenate([kmeans.cluster_centers_[:, n_components:],
        #                                   kmeans.cluster_centers_[:, :n_components]] * angle_signs, axis=1)
        angle_signs = np.sign(np.sum(U[:, :n_components] * V[:, :n_components], axis=0))
        flipped_centers = np.concatenate([
            kmeans.cluster_centers_[:, n_components:],
            kmeans.cluster_centers_[:, :n_components]
        ], axis=1)
        flipped_centers = flipped_centers * np.concatenate([angle_signs]*2)[None, :]

        dist_mat = cdist(kmeans.cluster_centers_, flipped_centers)
        row, col = linear_sum_assignment(dist_mat)
        updated_centroids = (kmeans.cluster_centers_[row] + flipped_centers[col])/2

        if d_thresh > 0:
            cent_to_keep = np.ones(updated_centroids.shape[0], dtype=bool)
            is_self = np.array(row == col)
            row_col = np.vstack([row[~is_self], col[~is_self]])
            row_col = np.unique(np.array([[r, c] if r < c else [c, r] for r, c in row_col.T]).T, axis=1)

            for r, c in row_col.T:
                dist = np.linalg.norm(updated_centroids[r] - updated_centroids[c])

                if dist < d_thresh:
                    updated_centroids[r] = (updated_centroids[r] + updated_centroids[c])/2
                    cent_to_keep[c] = False
            updated_centroids = updated_centroids[cent_to_keep]

            if verbose:
                print(f"Removed {n_kmeans - len(updated_centroids)} centroids after merging !")

        # added_centroids = (updated_centroids[row_col[0]] + updated_centroids[row_col[1]])/2
        # print(np.allclose(added_centroids[0, :n_components], added_centroids[0, n_components:]*angle_signs))
        # updated_centroids = np.vstack([updated_centroids, added_centroids])
        added_centroids = []

        # max_self_dist = dist_mat[is_self, is_self].max()
        # print(max_self_dist)

        kmeans = KMeans(n_clusters=len(updated_centroids), init=updated_centroids, **kwargs_kmeans2).fit(
            edge_assignments_vec
        )

        # dist_mat = cdist(kmeans.cluster_centers_[row], kmeans.cluster_centers_[col])
        # dist_mat2 = cdist(updated_centroids[row], updated_centroids[col])
        # for i in range(n_kmeans):
        #     print(row[i], col[i], dist_mat[i, i], dist_mat2[i, i], dist_mat[i, i] < max_self_dist)

    if return_kmeans:
        # return kmeans, edge_assignments[:, adjacency != 0].T
        return kmeans, edge_assignments_vec

    edge_clusters = kmeans.labels_ + 1

    if clust_only and undirected:
        return edge_clusters_inter, edge_clusters
    
    n_clusters = edge_clusters.max()
    if verbose:
        print(f"Found {n_clusters} clusters !")

    edge_clusters_mat = np.zeros((n_nodes, n_nodes), dtype=int)
    edge_clusters_mat[adjacency != 0] = edge_clusters

    if return_centroids:
        # cluster_centroids = np.zeros((n_clusters, 2, n_components))
        cluster_centroids = kmeans.cluster_centers_.reshape(
            (n_clusters, 2, n_components)
        )

        return edge_clusters, edge_clusters_mat, cluster_centroids

    return edge_clusters, edge_clusters_mat


def get_best_k(X, max_k=10, verbose=False, return_silhouette=False):
    print(f"Running silhouette analysis for k = 2 to {max_k} ...")
    n_clusters = np.arange(2, max_k)
    silhouette = np.zeros(n_clusters.shape[0])

    for i, n in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)
        silhouette[i] = silhouette_score(X, kmeans.labels_)

        if verbose:
            print(f"Silhouette score for K={n} is : {silhouette[i]:1.3f}")

    print(
        f"Best average silhouette_score is : {np.max(silhouette):1.2f} for K={n_clusters[np.argmax(silhouette)]}"
    )
    if return_silhouette:
        return silhouette
    return n_clusters[np.argmax(silhouette)]


def get_node_clusters(
    edge_clusters,
    edge_clusters_mat,
    method="bimod",
    scale=True,
):
    n_nodes = edge_clusters_mat.shape[0]
    n_clusters = np.max(edge_clusters)

    if method == "probability":
        # Aggregate edges to nodes using cluster probability (number of edges)
        n_per_cluster = np.zeros((n_nodes, n_clusters))
        for cluster_id in np.arange(1, np.max(edge_clusters_mat) + 1):
            n_per_cluster[:, cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )
            n_per_cluster[:, cluster_id - 1] += np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )

        cluster_prob = n_per_cluster / n_per_cluster.sum(axis=1)[:, None]

        cluster_maxprob = np.argmax(cluster_prob, axis=1) + 1

        return cluster_maxprob, cluster_prob
    if "bimod" in method:
        sending_communities = np.zeros((n_clusters, n_nodes))
        receiving_communities = np.zeros((n_clusters, n_nodes))

        for cluster_id in np.arange(1, np.max(edge_clusters_mat) + 1):
            sending_communities[cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=1
            )
            receiving_communities[cluster_id - 1] = np.sum(
                edge_clusters_mat == cluster_id, axis=0
            )

        if scale:
            sending_communities = np.divide(
                sending_communities,
                np.sum(edge_clusters_mat > 0, axis=1),
                where=np.sum(edge_clusters_mat > 0, axis=1) != 0,
                out=np.zeros_like(sending_communities),
            )
            receiving_communities = np.divide(
                receiving_communities,
                np.sum(edge_clusters_mat > 0, axis=0),
                where=np.sum(edge_clusters_mat > 0, axis=0) != 0,
                out=np.zeros_like(receiving_communities),
            )

        return sending_communities, receiving_communities


def bimod_index_edges(adjacency, edge_clusters_mat, scale=False):
    n_clusters = np.max(edge_clusters_mat)

    null_model = configuration_null(adjacency, null_model="outin")

    bimod_indices = np.zeros(n_clusters)

    for cluster_id in np.arange(n_clusters):
        adj_contrib = adjacency[edge_clusters_mat == cluster_id + 1]
        null_contrib = null_model[edge_clusters_mat == cluster_id + 1]

        bimod_indices[cluster_id] = np.sum(adj_contrib - null_contrib)

        if scale:
            bimod_indices[cluster_id] /= np.sum(adj_contrib > 0)

    return bimod_indices


def bimod_index_nodes(adjacency, send_com, receive_com, scale=False):
    n_clusters = len(send_com)

    null_model = configuration_null(adjacency, null_model="outin")

    bimod_indices = np.zeros(n_clusters)

    for cluster_id in np.arange(n_clusters):
        send_fltr = send_com[cluster_id] > 0
        receive_fltr = receive_com[cluster_id] > 0

        adj_contrib = adjacency[send_fltr][:, receive_fltr]
        null_contrib = null_model[send_fltr][:, receive_fltr]

        bimod_indices[cluster_id] = np.sum(adj_contrib - null_contrib)

        if scale:
            # all_edges = np.sum(np.atleast_2d(send_fltr).T @ np.atleast_2d(receive_fltr))
            all_edges = np.sum(adjacency)
            bimod_indices[cluster_id] /= all_edges
        # / np.sum(adj_contrib > 0)

    return bimod_indices


def bimod_index_quad(adjacency, send_com, receive_com, scale=None):
    n_clusters = len(send_com)

    modmat = modularity_matrix(adjacency, null_model="outin")

    bimod_indices = np.zeros(n_clusters)

    for cluster_id in np.arange(n_clusters):
        quad_form = send_com[cluster_id] @ modmat @ receive_com[cluster_id]
        # quad_form = (send_com[cluster_id] > 0) @ modmat @ (receive_com[cluster_id] > 0)

        if scale:
            bimod_indices[cluster_id] = 1 / (2 * adjacency.sum()) * quad_form
        else:
            bimod_indices[cluster_id] = quad_form

    return bimod_indices


def benchmark_bimod(
    adjacency, k_max=10, n_vec_max=10, use_nodes=False, scale_features=True
):

    U, S, Vh = sorted_SVD(modularity_matrix(adjacency, null_model="outin"))
    V = Vh.T

    if scale_features:
        scale_factor = S
    else:
        scale_factor = np.ones(S.shape[0])

    if use_nodes:
        bimod_func = bimod_index_nodes
    else:
        bimod_func = bimod_index_quad

    all_results = []
    for vector_id_max in np.arange(1, n_vec_max + 1):
        all_per_vectors = []
        for n_kmeans in np.arange(2, k_max + 1):

            edge_clusters, edge_clusters_mat = edge_bicommunities(
                adjacency,
                U,
                V,
                vector_id_max,
                method="kmeans",
                n_kmeans=n_kmeans,
                scale_S=scale_factor[:vector_id_max],
            )

            sending_communities, receiving_communities = get_node_clusters(
                edge_clusters, edge_clusters_mat, method="bimodularity"
            )

            bimod = bimod_func(
                adjacency, sending_communities, receiving_communities, scale=True
            )
            # sum_power = 2
            # bimod = bimod**sum_power / np.sum(bimod**sum_power)
            sorted = np.flip(np.argsort(np.abs(bimod)))

            all_per_vectors.append(bimod[sorted])
        all_results.append(all_per_vectors)

    return all_results


def get_c_pinv(
    adjacency: np.ndarray,
    n_vec_max: int,
    n_kmeans: int,
    sort_bimod: bool = True,
    normalize: bool = True,
    ones: bool = False,
    return_clusters: bool = False,
    **kwargs,
):
    B = modularity_matrix(adjacency)
    U, S, Vh = sorted_SVD(B, fix_negative=False)
    V = Vh.T

    scale_factor = S[:n_vec_max]
    edge_clusters, edge_clusters_mat = edge_bicommunities(
        adjacency,
        U,
        V,
        n_vec_max,
        method="kmeans",
        n_kmeans=n_kmeans,
        scale_S=scale_factor,
        **kwargs,
    )

    c_out, c_in = get_node_clusters(edge_clusters, edge_clusters_mat)

    bimod_idx = bimod_index_nodes(adjacency, c_out, c_in, scale=True)
    if sort_bimod:
        sorting_array = np.flip(np.argsort(bimod_idx))
        bimod_idx = bimod_idx[sorting_array]

        edge_clusters_mat_sorted = edge_clusters_mat.copy()
        edge_clusters_sorted = edge_clusters.copy()
        for i_new, i in enumerate(sorting_array):
            edge_clusters_mat_sorted[edge_clusters_mat == i + 1] = i_new + 1
            edge_clusters_sorted[edge_clusters == i + 1] = i_new + 1

        edge_clusters_mat = edge_clusters_mat_sorted
        edge_clusters = edge_clusters_sorted

        c_out = c_out[sorting_array]
        c_in = c_in[sorting_array]

    C_mat_out = c_out.T
    C_mat_in = c_in.T

    if ones:
        C_mat_out = (C_mat_out > 0).astype(float)
        C_mat_in = (C_mat_in > 0).astype(float)

    if normalize:
        C_mat_out /= np.linalg.norm(C_mat_out, axis=0)
        C_mat_in /= np.linalg.norm(C_mat_in, axis=0)

    c_pinv_out = np.linalg.pinv(C_mat_out)
    c_pinv_in = np.linalg.pinv(C_mat_in)

    if return_clusters:
        return (
            C_mat_out,
            c_pinv_out,
            C_mat_in,
            c_pinv_in,
            bimod_idx,
            edge_clusters,
            edge_clusters_mat,
        )

    return C_mat_out, c_pinv_out, C_mat_in, c_pinv_in, bimod_idx


def columnwise_corr(pred, truth):
    # Standardize columns (z-scoring)
    pred_std = (pred - pred.mean(axis=0)) / pred.std(axis=0)
    truth_std = (truth - truth.mean(axis=0)) / truth.std(axis=0)

    # Correlation = normalized dot product
    corr = pred_std.T @ truth_std / (pred.shape[0] - 1)
    return corr


def reorder_corr(c_mat, sort_rows=False, assign="linear", pca_comp=0):
    n_rows, n_columns = c_mat.shape
    size = max(n_rows, n_columns)
    cost = np.zeros((size, size))

    cost[:n_rows, :n_columns] = c_mat

    if "linear" in assign:
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    elif "PCA" in assign:
        pca = PCA(n_components=pca_comp + 1)

        components = pca.fit_transform(cost)[:, pca_comp]
        row_ind = np.argsort(components)
        col_ind = np.arange(n_columns)

    sort = np.argsort(col_ind)
    if sort_rows:
        sort = np.argsort(row_ind)

    row_ind = row_ind[sort]
    col_ind = col_ind[sort]
    if n_rows < n_columns:
        row_ind = row_ind[row_ind < n_rows]
    else:
        col_ind = col_ind[col_ind < n_columns]

    return cost[row_ind][:, col_ind], row_ind, col_ind


def get_ideal_conjugates(send, rec, normalize=True):
    send_rec = np.concatenate([send, rec], axis=1)
    conj = np.concatenate([rec, send], axis=1)
    if normalize:
        send_rec = send_rec / np.linalg.norm(send_rec, axis=1, keepdims=True)
        conj = conj / np.linalg.norm(conj, axis=1, keepdims=True)

    return send_rec, conj

def get_unique_conjugates(row, col):
    row_col = np.vstack([row, col])
    row_col = np.unique(
        np.array([[r, c] if r <= c else [c, r] for r, c in row_col.T]).T,
        axis=1
        )
    return row_col


def get_conjugates_matching(send, rec, unique=True, return_matrix=False):
    send_rec, conj = get_ideal_conjugates(send, rec, normalize=True)

    dist = cdist(send_rec, conj, metric="cosine")#'euclidean')
    row_ind, col_ind = linear_sum_assignment(dist)

    if unique:
        return get_unique_conjugates(row_ind, col_ind)
    
    if return_matrix:
        return row_ind, col_ind, dist

    return row_ind, col_ind


def get_asym_ratio(adj, bicom_masks=None, edge_clusters_mat=None):
    if bicom_masks is None:
        bicom_masks = np.array([edge_clusters_mat == (i+1) for i in range(edge_clusters_mat.max())])

    # sum_per_com = (adj * bicom_masks).sum(axis=(1, 2))
    # count_per_com = (adj * bicom_masks > 0).sum(axis=(1, 2))
    # mean_per_com = sum_per_com / count_per_com

    ratio = np.zeros(len(bicom_masks), dtype=float)
    for i, mask in enumerate(bicom_masks):
        ratio[i] = (adj * mask).mean() / ((adj * mask.T).mean() + (adj * mask).mean())
    
    return ratio


def get_conjugate_ratio(adj, row_ind, col_ind, bicom_masks=None, edge_clusters_mat=None):
    if bicom_masks is None:
        bicom_masks = np.array([edge_clusters_mat == (i+1) for i in range(edge_clusters_mat.max())])

    sum_per_com = (adj * bicom_masks).sum(axis=(1, 2))
    count_per_com = (adj * bicom_masks > 0).sum(axis=(1, 2))
    mean_per_com = sum_per_com / count_per_com

    ratio = np.zeros_like(row_ind, dtype=float)
    for i, _ in enumerate(row_ind):
        ratio[i] = mean_per_com[row_ind[i]] / (mean_per_com[col_ind[i]] + mean_per_com[row_ind[i]])
    
    return ratio

# Permutation Testing

def shuffle_edges(adj, perm_prop, n_shuffle=1, edge_mask=None):
    if edge_mask is None:
        edge_mask = adj > 0
    n_zer_edges = adj[edge_mask]

    n_perm_edges = int(perm_prop * len(n_zer_edges))
    # shuffled = np.zeros_like(adj)
    shuffled = np.zeros((n_shuffle, *adj.shape))

    for s in range(n_shuffle):
        selected_edges = np.random.choice(len(n_zer_edges), size=n_perm_edges, replace=False)
        perm_edges = n_zer_edges.copy()
        perm_edges[selected_edges] = np.random.permutation(n_zer_edges[selected_edges])

        shuffled[s, edge_mask] = perm_edges

    return shuffled

def shuffle_edges_sym(adj, perm_prop, n_shuffle=1):
    triu = np.array(np.triu_indices_from(adj, k=1)).T
    n_triu = len(triu)
    n_perm_edges = int(perm_prop * n_triu)

    shuffled = np.repeat(adj[np.newaxis, :, :], n_shuffle, axis=0)

    for s in range(n_shuffle):
        selected_edges = np.random.choice(n_triu, size=n_perm_edges, replace=False)
        selected_triu = triu[selected_edges].T

        shuffled[s, selected_triu[0], selected_triu[1]] = adj[selected_triu[1], selected_triu[0]]
        shuffled[s, selected_triu[1], selected_triu[0]] = adj[selected_triu[0], selected_triu[1]]

    return shuffled