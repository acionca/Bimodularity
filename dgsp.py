# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

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

    return z


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

    return a_mat - z / a_mat.sum()


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
