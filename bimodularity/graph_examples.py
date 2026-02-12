from typing import Optional, Union
import numpy as np


# Toy examples of directed graphs and communities
def make_directed_clique(n_nodes, dir_out=False, weighted=False, density=1):
    a_mat = np.zeros((n_nodes, n_nodes))

    indices = np.triu_indices_from(a_mat, k=1)
    n_indices = len(indices[0])
    n_sparse = int((1 - density) * n_indices)

    sample_val = np.ones(n_indices)
    if weighted:
        sample_val = np.random.normal(np.ones(n_indices), 0.2)
    sample_val[np.random.choice(np.arange(n_indices), size=n_sparse, replace=False)] = 0
    a_mat[indices] = sample_val

    if dir_out:
        a_mat = a_mat.T
    return a_mat


def toy_bifurc(
    n_nodes,
    nodes_per_clique,
    n_connecting_edges,
    directed=False,
    out_nodes=False,
    **kwargs,
):
    n_connecting_nodes = n_nodes - 2 * nodes_per_clique

    C1 = make_directed_clique(nodes_per_clique, **kwargs)
    C2 = make_directed_clique(nodes_per_clique, dir_out=directed, **kwargs)
    L = np.hstack(
        [
            np.ones((n_connecting_nodes, n_connecting_edges)),
            np.zeros((n_connecting_nodes, nodes_per_clique - n_connecting_edges)),
        ]
    )

    L_out1 = L
    L_out2 = L
    L_in1 = np.zeros_like(L).T
    L_in2 = np.zeros_like(L).T

    if directed:
        L_out2 = np.zeros_like(L)
        L_in2 = L.T
    elif out_nodes:
        L_in1 = L.T
        L_in2 = L.T
        L_out1 = np.zeros_like(L)
        L_out2 = np.zeros_like(L)

    all_zeros = np.zeros((nodes_per_clique, nodes_per_clique))

    toy_mat = np.block(
        [
            [C1, L_in1, all_zeros],
            [L_out1, np.zeros((n_connecting_nodes, n_connecting_nodes)), L_out2],
            [all_zeros, L_in2, C2],
        ]
    )
    return toy_mat


def toy_fully(n_nodes, density=1, out_prob=0.5, seed=None):
    half_nodes = int(n_nodes / 2)
    fully_directed = np.zeros((n_nodes, n_nodes))

    triu_idx = np.array(np.triu_indices(half_nodes, k=1))
    triu_idx = np.hstack([triu_idx, triu_idx + half_nodes])
    tril_idx = np.flip(triu_idx, axis=0)

    np.random.seed(seed)
    randomized_indices = np.random.normal(np.zeros(triu_idx.shape[-1])) > 0

    all_idx = np.hstack(
        [triu_idx[:, randomized_indices], tril_idx[:, ~randomized_indices]]
    )

    n_sparse = int(all_idx.shape[1] * (1 - density))

    sparse_ones = np.ones_like(all_idx[0])
    sparse_ones[np.random.choice(len(sparse_ones), n_sparse)] = 0
    fully_directed[*all_idx] = sparse_ones

    return fully_directed


def random_directed_edges_id(
    ids: list,
    edge_prob: float = 0.5,
    directed: bool = True,
    out_prob: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random edges from a list of ids

    Parameters
    ----------
    ids : list
        list of edge ids
    edge_prob : float, optional
        probability to select an edge, by default 0.5
    directed : bool, optional
        condition for directed connections, by default True
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    seed : Optional[int], optional
        seed for reproducibility, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ids of the selected edges (tuple of outgoing and incoming if directed)
    """

    n_edges = np.shape(ids)[-1]

    np.random.seed(seed)
    edge_ids = np.random.choice(
        np.arange(n_edges), int(n_edges * edge_prob), replace=False
    )

    if directed:
        out_ids = np.random.choice(
            edge_ids, int(len(edge_ids) * out_prob), replace=False
        )
        in_ids = np.array([i for i in edge_ids if i not in out_ids])

        return ids[:, list(out_ids)], ids[:, list(in_ids)]
    return ids[:, edge_ids]


def random_graph(
    n_nodes: int, edge_prob=0.5, directed=False, seed: Optional[int] = None
) -> np.ndarray:
    """Generate a random graph with a given number of nodes

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default False
    seed : Optional[int], optional
        seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        adjacency matrix of the generated graph
    """

    a_mat = np.zeros((n_nodes, n_nodes), dtype=int)

    all_ids = np.array(np.triu_indices_from(a_mat, k=1))

    edge_ids = random_directed_edges_id(
        all_ids, edge_prob=edge_prob, directed=directed, out_prob=0.5, seed=seed
    )

    if directed:
        edge_ids = np.hstack([edge_ids[0], np.flip(edge_ids[1], axis=0)])
    else:
        edge_ids = np.hstack([edge_ids, np.flip(edge_ids, axis=0)])

    a_mat[*edge_ids] = 1

    return a_mat


def random_connector(
    n_nodes: int,
    edge_prob: float = 0.5,
    directed: bool = True,
    out_prob: float = 0.5,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Generate a random "connector" graph with a given number of nodes

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default True
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    seed : Optional[float], optional
        seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        adjacency matrix (two matrices in the directed case) of the generated
        "connector" graph
    """

    all_ids = np.array([(i, j) for i in range(n_nodes) for j in range(n_nodes)]).T

    edge_ids = random_directed_edges_id(
        all_ids, edge_prob=edge_prob, directed=directed, out_prob=out_prob, seed=seed
    )

    if directed:
        a_mat = np.zeros((2, n_nodes, n_nodes), dtype=int)
        a_mat[0][*edge_ids[0]] = 1
        # Transpose the in edges
        a_mat[1][edge_ids[1][1], edge_ids[1][0]] = 1
    else:
        a_mat = np.zeros((n_nodes, n_nodes), dtype=int)
        a_mat[edge_ids] = 1

    return a_mat


def toy_random(
    n_nodes: int,
    out_prob=0.5,
    edge_prob=0.5,
    directed=False,
    con_prob: Optional[float] = None,
) -> np.ndarray:
    """Generate a random graph with a given number of nodes. The graph will have two
    densily connected communities (density 'edge_prob') with a random number of
    connecting edges (desntiy 'con_prob').

    Parameters
    ----------
    n_nodes : int
        number of nodes
    out_prob : float, optional
        probability for an out-going edge (only in the directed case), by default 0.5
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    directed : bool, optional
        condition for directed connections, by default False
    con_prob : Optional[float], optional
        density of connecting edges, by default None

    Returns
    -------
    np.ndarray
        graph adjacency matrix
    """
    half_nodes = int(n_nodes / 2)

    com_1 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob, directed=directed)
    com_2 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob, directed=directed)

    if con_prob is None:
        con_prob = edge_prob / 2
    n_connect = half_nodes**2
    edge_idx = np.random.choice(n_connect, int(n_connect * con_prob), replace=False)

    if directed:
        chosen_out_edges = np.random.choice(
            edge_idx,
            int(len(edge_idx) * out_prob),
            replace=False,
        )
        edge_out_idx = np.array([i in chosen_out_edges for i in edge_idx])
        edge_out = edge_idx[edge_out_idx]
        edge_in = edge_idx[~edge_out_idx]
    else:
        edge_out = edge_idx.copy()
        edge_in = edge_idx.copy()

    connect_out = np.zeros(n_connect)
    connect_in = np.zeros(n_connect)

    connect_out[edge_out] = 1
    connect_in[edge_in] = 1

    a_mat = np.block(
        [
            [com_1, connect_out.reshape((half_nodes, -1))],
            [connect_in.reshape((half_nodes, -1)).T, com_2],
        ]
    )
    return a_mat


def toy_random_seeded(
    n_nodes,
    seed,
    out_prob=0.5,
    edge_prob=0.5,
    directed=False,
    con_prob=None,
    verbose=False,
):
    half_nodes = int(n_nodes / 2)

    seeds = [seed + i for i in range(4)]
    com_1 = random_graph(
        n_nodes=half_nodes, edge_prob=edge_prob, directed=directed, seed=seeds[0]
    )
    com_2 = random_graph(
        n_nodes=half_nodes, edge_prob=edge_prob, directed=directed, seed=seeds[1]
    )

    # connect_com = np.zeros(half_nodes**2)
    if con_prob is None:
        con_prob = edge_prob / 2
    n_connect = half_nodes**2
    np.random.seed(seeds[2])
    edge_idx = np.random.choice(n_connect, int(n_connect * con_prob), replace=False)
    if verbose:
        print(f"There will be {len(edge_idx)} connecting edges")

    if directed:
        np.random.seed(seeds[3])
        chosen_out_edges = np.random.choice(
            edge_idx,
            int(len(edge_idx) * out_prob),
            replace=False,
            # len(edge_idx), int(len(edge_idx) * out_prob), replace=False
        )
        edge_out_idx = np.array([i in chosen_out_edges for i in edge_idx])
        edge_out = edge_idx[edge_out_idx]
        edge_in = edge_idx[~edge_out_idx]
    else:
        edge_out = edge_idx.copy()
        edge_in = edge_idx.copy()

    connect_out = np.zeros(n_connect)
    connect_in = np.zeros(n_connect)

    connect_out[edge_out] = 1
    connect_in[edge_in] = 1

    if verbose:
        print(f"With {connect_out.sum()} out edges and {connect_in.sum()} in edges.")

    a_mat = np.block(
        [
            [com_1, connect_out.reshape((half_nodes, -1))],
            [connect_in.reshape((half_nodes, -1)).T, com_2],
        ]
    )

    if verbose:
        print(
            f"Average degrees are in: {a_mat.sum(axis=0).mean():1.3f}"
            f", out: {a_mat.sum(axis=1).mean():1.3f}"
        )
    return a_mat


def toy_random_old(n_nodes, densities=0.5, seed=None):
    if isinstance(densities, list):
        half_nodes = int(n_nodes / 2)
        blocks = [
            toy_random_old(n_nodes=half_nodes, densities=dens, seed=seed + i)
            for i, dens in enumerate(densities)
        ]
        random_directed = np.block([blocks[:2], blocks[2:]])
        return random_directed

    np.random.seed(seed)
    random_directed = np.random.normal(np.zeros((n_nodes, n_nodes)), 1)
    thresh = np.percentile(random_directed.flatten(), 100 * densities, method="linear")

    random_directed = (random_directed < thresh).astype(int)

    random_directed -= np.diag(np.diag(random_directed))

    return random_directed


def toy_magnetic(n_nodes, connect_center=False):
    a_mag = np.zeros((n_nodes, n_nodes))

    a_mag[0, 1:-1] = 1
    a_mag[1:-1, -1] = 1

    if connect_center:
        a_mag[-1, 0] = 1

    return a_mag


def toy_bipartite(n_nodes, out_prop=0.5, density=1, seed=None):
    half_nodes = int(n_nodes / 2)

    n_max_con = half_nodes**2
    n_con = int(density * n_max_con)

    if seed is None:
        seed = int(np.random.normal(10000, 100))

    np.random.seed(seed=seed)
    sparse_edges_id = np.random.choice(n_max_con, n_con, replace=False)
    np.random.seed(seed=seed + 1)
    directed_edges_id = np.random.choice(
        sparse_edges_id, int(out_prop * n_con), replace=False
    )

    out_con = np.zeros(n_max_con)
    in_con = np.zeros(n_max_con)

    out_con[directed_edges_id] = 1
    in_con[[i for i in sparse_edges_id if i not in directed_edges_id]] = 1

    a_bip = np.block(
        [
            [
                np.zeros((half_nodes, half_nodes)),
                out_con.reshape((half_nodes, half_nodes)),
            ],
            [
                in_con.reshape((half_nodes, half_nodes)).T,
                np.zeros((half_nodes, half_nodes)),
            ],
        ]
    )

    return a_bip


def toy_n_communities(
    nodes_per_com: int,
    n_com: int = 3,
    com_density: float = 0.5,
    connect_density: list[float] = 0.5,
    connect_out_prob: list[float] = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:

    if seed is None:
        seed = int(np.random.normal(10000, 100))

    if isinstance(connect_density, (int, float)):
        connect_density = [connect_density] * (n_com * (n_com - 1) // 2)

    if isinstance(connect_out_prob, (int, float)):
        connect_out_prob = [connect_out_prob] * (n_com * (n_com - 1) // 2)

    n_connectors = n_com * (n_com - 1) // 2

    communities = [
        random_graph(
            nodes_per_com, edge_prob=com_density, directed=True, seed=seed + seed_i
        )
        for seed_i in range(n_com)
    ]

    if isinstance(connect_out_prob, (int, float)):
        connect_out_prob = [connect_out_prob] * n_connectors

    if isinstance(connect_density, (int, float)):
        connect_density = [connect_density] * n_connectors

    connectors = [
        random_connector(
            nodes_per_com,
            edge_prob=p_edge,
            directed=True,
            out_prob=p_out,
            seed=seed + seed_i,
        )
        for seed_i, (p_out, p_edge) in enumerate(zip(connect_out_prob, connect_density))
    ]

    if n_com == 3:
        adj = np.block(
            [
                [communities[0], connectors[0][0], connectors[1][0]],
                [connectors[0][1], communities[1], connectors[2][0]],
                [connectors[1][1], connectors[2][1], communities[2]],
            ]
        )
    if n_com == 4:
        adj = np.block(
            [
                [communities[0], connectors[0][0], connectors[1][0], connectors[2][0]],
                [connectors[0][1], communities[1], connectors[3][0], connectors[4][0]],
                [connectors[1][1], connectors[3][1], communities[2], connectors[5][0]],
                [connectors[2][1], connectors[4][1], connectors[5][1], communities[3]],
            ]
        )

    return adj


def block_cycle(
    nodes_per_com: int,
    n_blocks: int = 3,
    com_density: float = 0.5,
    connect_density: Union[list[float], float] = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:

    if seed is None:
        seed = int(np.random.normal(10000, 100))

    if isinstance(connect_density, (int, float)):
        connect_density = [connect_density] * n_blocks

    communities = [
        random_graph(
            nodes_per_com, edge_prob=com_density, directed=True, seed=seed + seed_i
        )
        for seed_i in range(n_blocks)
    ]

    connectors = [
        random_connector(
            nodes_per_com,
            edge_prob=p_edge,
            directed=True,
            out_prob=1,
            seed=seed + seed_i,
        )[0]
        for seed_i, p_edge in enumerate(connect_density)
    ]

    zero_com = np.zeros_like(communities[0])
    row_blocks = np.array(
        [
            [communities[i], connectors[i]] + [zero_com] * (n_blocks - 2)
            for i in range(n_blocks)
        ]
    )

    for i in range(n_blocks):
        row_blocks[i] = np.roll(row_blocks[i], i, axis=0)

    rows_conc = [np.concatenate(row_blocks[i], axis=1) for i in range(n_blocks)]
    adj = np.concatenate(rows_conc, axis=0)

    return adj
