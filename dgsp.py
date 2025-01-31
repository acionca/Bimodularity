# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

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

    U, S, Vh = np.linalg.svd(matrix, full_matrices=True)

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
    method="partition",
    n_kmeans=10,
    verbose=False,
    scale_S=None,
    assign_only=False,
    **kwargs,
) -> tuple:

    n_nodes = adjacency.shape[0]

    if scale_S is None:
        scale_S = np.ones(n_components)

    u_features = U[:, :n_components] * np.sqrt(scale_S)
    v_features = V[:, :n_components] * np.sqrt(scale_S)

    if method in ["partition", "sign"]:
        u_features = np.sign(u_features).astype(int)
        v_features = np.sign(v_features).astype(int)

    edge_out = np.array([u_features] * n_nodes).T
    edge_in = np.array([v_features] * n_nodes)
    edge_in = np.moveaxis(edge_in, -1, 0)

    # edge-based clustering
    edge_assignments = np.concatenate([edge_out, edge_in], axis=0)
    edge_assignments_vec = edge_assignments.reshape((2 * n_components, -1)).T

    edge_assignments_vec = edge_assignments_vec[(adjacency != 0).reshape(-1)]

    if assign_only:
        return edge_assignments_vec

    if method in ["partition", "sign"]:
        clusters = np.unique(edge_assignments_vec, axis=0)
        n_clusters = clusters.shape[0]
        if verbose:
            print(f"Found {n_clusters} clusters !")

        cluster2num = {tuple(c): i + 1 for i, c in enumerate(clusters)}

        edge_clusters = np.array([cluster2num[tuple(c)] for c in edge_assignments_vec])

    elif method == "kmeans":
        if n_kmeans is None:
            n_kmeans = get_best_k(edge_assignments_vec, verbose=verbose, **kwargs)

        kmeans = KMeans(n_clusters=n_kmeans, random_state=0, n_init="auto").fit(
            edge_assignments_vec
        )
        edge_clusters = kmeans.labels_ + 1

        n_clusters = edge_clusters.max()
        if verbose:
            print(f"Found {n_clusters} clusters !")
    else:
        raise ValueError(
            "Method not recognized (possible values: partition, sign, kmeans)"
        )

    edge_clusters_mat = np.zeros((n_nodes, n_nodes), dtype=int)
    edge_clusters_mat[adjacency != 0] = edge_clusters

    return edge_clusters, edge_clusters_mat


def get_best_k(X, max_k=10, verbose=False):
    print(f"Running silhouette analysis for k = 2 to {max_k} ...")
    n_clusters = np.arange(2, max_k)
    silhouette = np.zeros(n_clusters.shape[0])

    for i, n in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)
        silhouette[i] = silhouette_score(X, kmeans.labels_)

        if verbose:
            print(f"Silhouette score for K={n} is : {silhouette[i]:1.2f}")

    print(
        f"Best average silhouette_score is : {np.max(silhouette):1.2f} for K={n_clusters[np.argmax(silhouette)]}"
    )
    return n_clusters[np.argmax(silhouette)]


def get_node_clusters(edge_clusters, edge_clusters_mat, method="bimod", scale=True):
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
            sending_communities = np.nan_to_num(
                sending_communities / np.sum(edge_clusters_mat > 0, axis=1),
                posinf=0,
                neginf=0,
            )
            receiving_communities = np.nan_to_num(
                receiving_communities / np.sum(edge_clusters_mat > 0, axis=0),
                posinf=0,
                neginf=0,
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
            all_edges = np.sum(np.atleast_2d(send_fltr).T @ np.atleast_2d(receive_fltr))
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
        scale_factor = S**2 / (S**2).sum()
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
