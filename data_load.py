import os
import os.path as op

import pandas as pd
import numpy as np
from scipy.io import loadmat

import pickle
import h5py

import nibabel as nib


def save(pickle_filename: str, iterable: object) -> None:
    """
    Pickle an object to a file.

    Parameters
    ----------
    pickle_filename : str
        Path to the file where the object will be pickled.
    iterable : object
        The object to be pickled.

    Returns
    -------
    None
    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(iterable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(pickle_filename: str) -> object:
    """
    Load a pickled object from the specified file.

    Parameters
    ----------
    pickle_filename : str
        The filename of the pickled object to load.

    Returns
    -------
    object
        The loaded object.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b


def get_aggprop(h5dict: h5py._hl.files.File, property: str):
    """
    Get the bundles statistics on whole brain level from the HDF5 file.

    Parameters
    ----------
    h5dict : h5py._hl.files.File
        The opened HDF5 file.
    property : str
        The property to extract from the HDF5 file.

    Returns
    -------
    ret : np.arrasy
        The array containing the requested property values.
    """

    try:
        ret = np.array(h5dict.get("matrices").get(property))
    except:
        print("Not valid property OR h5 not opened")
    return ret


def load_celegans_graph(
    path_to_worm_data: str = "data",
    path_to_worm_matrices: str = "data/celegans-bullmore",
    thresh: float = 0,
    no_sex: bool = False,
    gap_junc: bool = False,
):

    adj_filename = "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
    neurons_types_filename = "NeuronFixedPoints.xls"

    matrix_filename = "Worm279dir.mat"

    neuron_adj_df = pd.ExcelFile(op.join(path_to_worm_data, adj_filename))

    chem_df = neuron_adj_df.parse("hermaphrodite chemical")

    # Getting neuron informations
    # Dropping Nans
    neuron_def_df = chem_df.iloc[:, :3].dropna(how="all").ffill()
    neuron_def_df.columns = ["Type", "Detail", "Neuron"]
    # Dropping Pharynx neurons
    neuron_def_df = neuron_def_df[neuron_def_df.Type != "PHARYNX"].reset_index(
        drop=True
    )

    lab_to_id = {lab: i for i, lab in enumerate(neuron_def_df["Neuron"].values)}

    # Adding landmark information (average position on antero-posterior axis)
    path_to_neurons_types = op.join(path_to_worm_data, neurons_types_filename)
    neuron_AvgLandmarkPos = (
        pd.read_excel(path_to_neurons_types)
        .groupby("Neuron")["Landmark Position"]
        .mean()
    )

    neuron_def_df = neuron_def_df.merge(neuron_AvgLandmarkPos, on="Neuron", how="left")

    # Order Neurons by Type and Landmark Position
    # neuron_def_df = neuron_def_df.sort_values(["Type", "Landmark Position"])

    type_to_num = {t: i for i, t in enumerate(neuron_def_df["Type"].unique())}
    neuron_def_df["Type_num"] = [type_to_num[t] for t in neuron_def_df["Type"]]

    all_labels = neuron_def_df["Neuron"].values.tolist()
    print(f"There are {len(all_labels)} neurons in the dataset")

    fixed_details = [
        lab.replace(" MOTOR NEURONS", "") if "MOTOR NEURONS" in str(lab) else lab
        for lab in neuron_def_df.loc[:, "Detail"]
    ]
    neuron_def_df.loc[:, "Detail"] = fixed_details

    neuron_def_df["Type_long"] = (
        neuron_def_df["Type"] + "_" + neuron_def_df["Detail"].astype(str)
    )

    type_long_to_num = {t: i for i, t in enumerate(neuron_def_df["Type_long"].unique())}
    neuron_def_df["TypeLong_num"] = [
        type_long_to_num[t] for t in neuron_def_df["Type_long"]
    ]

    print("All neuron sub-types are:")
    print(neuron_def_df.loc[:, "Type_long"].unique())

    worm_dict = loadmat(op.join(path_to_worm_matrices, matrix_filename))

    worm_adj_gap = worm_dict["Worm279_ejunct_matrix_dir"]
    worm_adj_chem = worm_dict["Worm279_synapse_matrix_dir"]

    worm_label = np.array([lab[0][0] for lab in worm_dict["Worm279_labels"]])
    worm_type = [
        neuron_def_df[neuron_def_df["Neuron"] == lab]["Type"].values[0]
        for lab in worm_label
    ]
    worm_type_num = [
        neuron_def_df[neuron_def_df["Neuron"] == lab]["Type_num"].values[0]
        for lab in worm_label
    ]
    worm_landmark = [
        neuron_def_df[neuron_def_df["Neuron"] == lab]["Landmark Position"].values[0]
        for lab in worm_label
    ]

    worm_pos = worm_dict["Worm279_positions"]

    # REMOVE SEX SPECIFIC NEURONS
    if no_sex:
        no_sex_spec = np.array(worm_type_num, dtype=int) < 3
        worm_type = np.array(worm_type)[no_sex_spec]
        worm_type_num = np.array(worm_type_num)[no_sex_spec]
        worm_label = np.array(worm_label)[no_sex_spec]
        worm_pos = np.array(worm_pos)[no_sex_spec]
        worm_adj_gap = np.array(worm_adj_gap)[no_sex_spec][:, no_sex_spec]
        worm_adj_chem = np.array(worm_adj_chem)[no_sex_spec][:, no_sex_spec]

    if gap_junc:
        wiring_adj = worm_adj_gap + worm_adj_chem
    else:
        wiring_adj = worm_adj_chem
    wiring_adj -= np.diag(np.diag(wiring_adj))

    wiring_sym = (wiring_adj > thresh).astype(int)

    worm_df = pd.DataFrame(
        {
            "Neuron": worm_label,
            "Type": worm_type,
            "Type_num": worm_type_num,
            "Position x": [p[0] for p in worm_pos],
            "Position y": [p[1] for p in worm_pos],
        }
    )

    return wiring_sym, worm_df


def normalize_slines_vol(
    mat: np.ndarray,
    atlas_path: str,
    atlas_fname: str = "roi_atlas-ftract-scale1-GM.nii.gz",
    labels: list = None,
) -> np.ndarray:

    atlas_data = nib.load(op.join(atlas_path, atlas_fname)).get_fdata()
    n_roi = int(atlas_data.max())

    volumes = np.zeros(n_roi, dtype=int)
    for i in range(n_roi):
        volumes[i] = np.sum(atlas_data == (i + 1))

    # for vol_i in np.argsort(volumes)[:6]:
    # print(labels[vol_i], volumes[vol_i])

    # Normalizing by average volume of each pair of regions
    vol_matrix = (volumes[:, None] + volumes[None, :]) / 2

    # print(vol_matrix.shape)

    return mat / vol_matrix


def load_brain_graph(
    path_to_data="./data/brain",
    data_suffix="",
    delay_max=100,
    scale=1,
    b_prob_threshold=0.0,
    f_prob_threshold=0.0,
    slines_theshold=0,
    k_threshold=0.9,
    undirected=True,
    use_delay=True,
    normalize_slines=False,
    log_slines=False,
    gamma_dir=1,
    verbose=False,
):

    b_prob_fname = f"{data_suffix}bundle_probability_atlas-scale{scale}.pkl"
    bundle_prob = load(op.join(path_to_data, b_prob_fname))
    # bundle_prob = bundle_prob[:-2][:, :-2]
    np.fill_diagonal(bundle_prob, val=0)

    slines_fname = f"{data_suffix}bundle_streamlines_atlas-scale{scale}.pkl"
    slines_mat = load(op.join(path_to_data, slines_fname))
    # slines_mat = np.nan_to_num(slines_mat[:-2][:, :-2])
    slines_mat = np.nan_to_num(slines_mat)
    np.fill_diagonal(slines_mat, val=0)
    slines_mat = (slines_mat + slines_mat.T) // 2

    if not undirected:
        f_prob_fname = (
            f"{data_suffix}adj_probability_ftract-d{delay_max}-scale{scale}.pkl"
        )
        ftract_prob = load(op.join(path_to_data, f_prob_fname))
        ftract_prob = ftract_prob[:-2][:, :-2]

        delay_fname = f"{data_suffix}adj_delay_ftract-d{delay_max}-scale{scale}.pkl"
        ftract_delays = load(op.join(path_to_data, delay_fname))
        ftract_delays = ftract_delays[:-2][:, :-2]

        # Mask for existing f-tract prob and delay (not NaN or 0)
        noprob = np.logical_or(ftract_prob == 0, np.isnan(ftract_prob))
        noprob = np.logical_or(ftract_delays == 0, noprob)
        noprob = np.logical_or(np.isnan(ftract_delays), noprob)

        ftract_delays = np.divide(
            1,
            ftract_delays,
            where=np.logical_not(noprob),
            out=np.zeros_like(ftract_delays),
        )

    node_centers = load(
        op.join(path_to_data, f"{data_suffix}roi_centers-ftract-scale{scale}.pkl")
    )  # [:82]

    try:
        labels = np.genfromtxt(
            op.join(path_to_data, f"{data_suffix}brain_labels.csv"), dtype=str
        )
    except FileNotFoundError:
        labels = np.genfromtxt(
            op.join(path_to_data, f"{data_suffix}brain_labels-scale{scale}.csv"),
            dtype=str,
        )

    # print(labels[-5:])
    # for i, lab in enumerate(labels):
    #     print(i + 1, lab)

    # s_mat = bundle_prob
    # f_mat = ftract_prob

    if use_delay:
        s_mat = np.zeros_like(slines_mat)
        if log_slines:
            s_mat[slines_mat > 1] = np.log(slines_mat[slines_mat > 1])
        else:
            s_mat = slines_mat.copy()

        if not undirected:
            f_mat = ftract_delays
    else:
        s_mat = np.ones_like(slines_mat)
        s_mat = bundle_prob.copy()

    # s_mat = np.ones_like(slines_mat)

    s_mat[bundle_prob < b_prob_threshold] = 0
    s_mat[slines_mat < slines_theshold] = 0

    if not undirected:
        f_mat[ftract_prob < f_prob_threshold] = 0

    if verbose:
        print(f"There are {len(labels)} nodes in the graph")
        print(
            f"{(s_mat > 0).sum()/(slines_mat > 0).sum():.2%} of connections remain after thresholding"
        )

    if normalize_slines and use_delay:
        s_mat = normalize_slines_vol(
            s_mat,
            atlas_path=path_to_data,
            atlas_fname=f"{data_suffix}roi_atlas-ftract-scale{scale}-GM.nii.gz",
            labels=labels,
        )

    if gamma_dir != 1:
        f_mat = f_mat**gamma_dir

    if undirected:
        k_matrix = s_mat.copy()
        # k_matrix = bundle_prob + bundle_prob.T
    else:
        # k_matrix = (2 * s_mat) * (f_mat / (f_mat + f_mat.T))
        k_matrix = s_mat * (f_mat / (f_mat + f_mat.T))
        k_matrix = np.nan_to_num(k_matrix)

    if k_threshold > 0:
        k_matrix = (k_matrix >= k_threshold).astype(int)

    if verbose:
        print("Is it undirected ?", np.allclose(k_matrix, k_matrix.T))

    return k_matrix, labels, node_centers


def load_bundle_graph(
    path_to_data="./data/brain",
    data_suffix="",
    scale=1,
    b_prob_threshold=0.0,
    slines_theshold=0,
    normalize_slines=False,
    log_slines=False,
    verbose=False,
):

    b_prob_fname = f"{data_suffix}bundle_probability_atlas-scale{scale}.pkl"
    bundle_prob = load(op.join(path_to_data, b_prob_fname))
    np.fill_diagonal(bundle_prob, val=0)

    slines_fname = f"{data_suffix}bundle_streamlines_atlas-scale{scale}.pkl"
    slines_mat = load(op.join(path_to_data, slines_fname))
    np.fill_diagonal(np.nan_to_num(slines_mat), val=0)

    # Ensure symmetry
    slines_mat = (slines_mat + slines_mat.T) // 2

    node_centers = load(
        op.join(path_to_data, f"{data_suffix}roi_centers-ftract-scale{scale}.pkl")
    )

    try:
        labels = np.genfromtxt(
            op.join(path_to_data, f"{data_suffix}brain_labels.csv"), dtype=str
        )
    except FileNotFoundError:
        labels = np.genfromtxt(
            op.join(path_to_data, f"{data_suffix}brain_labels-scale{scale}.csv"),
            dtype=str,
        )

    s_mat = np.zeros_like(slines_mat)
    if log_slines:
        s_mat[slines_mat > 1] = np.log(slines_mat[slines_mat > 1])
    else:
        s_mat = slines_mat.copy()

    s_mat[bundle_prob < b_prob_threshold] = 0
    s_mat[slines_mat < slines_theshold] = 0

    if verbose:
        print(f"There are {len(labels)} nodes in the graph")
        print(
            f"{(s_mat > 0).sum()/(slines_mat > 0).sum():.2%} of connections remain after thresholding"
        )

    if normalize_slines:
        s_mat = normalize_slines_vol(
            s_mat,
            atlas_path=path_to_data,
            atlas_fname=f"{data_suffix}roi_atlas-ftract-scale{scale}-GM.nii.gz",
            labels=labels,
        )

    return s_mat, labels, node_centers


def load_nodal_fmri(
    path_to_atlased="/Users/acionca/data/HCP-MIP/atlased",
    atlas="Laus2008_smth6_lp0.15",
    task="rest1_dir-LR",
    concat=True,
):
    path_to_fmri = op.join(path_to_atlased, atlas)

    # task = "motor"
    # task = "rest1_dir-LR"
    # task = "rest1"
    # task = "emotion"
    # task = "gambling"
    # task = "language"

    all_fnames = sorted([f for f in os.listdir(path_to_fmri) if task in f])
    # all_fnames = [f for f in all_fnames if f"{sub}" in f]
    print(f"Found {len(all_fnames)} matching files")

    all_nodals = [
        np.genfromtxt(op.join(path_to_fmri, f), delimiter=",") for f in all_fnames
    ]

    if not concat:
        return all_nodals
    else:
        nodal_fmri = np.concatenate(all_nodals, axis=0)
        print("Nodal fMRI has shape:", nodal_fmri.shape)

        all_lengths = [len(n) for n in all_nodals]

        # TODO: Implement the paradygm loading and concatenation
        # if "rest" not in task:
        #     path_to_paradygm = op.join(
        #         path_to_fmri, fname.replace("timeseries", "regressor")
        #     )
        #     paradygm = np.genfromtxt(path_to_paradygm, delimiter=",").astype(int)
        # else:
        #     paradygm = np.zeros(len(nodal_fmri), dtype=int)

        return nodal_fmri, all_lengths  # , paradygm


def load_nodal_mat(
    path_to_hcp: str,
    task: str = "rest1",
    dir="LR",
    atlas="Glasser360",
    mat_suffix="bp_z",
    n_subset: int = None,
    concat: bool = True,
):
    all_subs = sorted(os.listdir(path_to_hcp))

    task_suffix = "r" if "rest" in task else "t"
    if "dir" in task:
        task_dir = f"{task_suffix}fMRI_{task.split('_dir')[0].upper()}_{dir}"
    else:
        task_dir = f"{task_suffix}fMRI_{task.upper()}_{dir}"

    fname_suffix = f"{atlas}S_{mat_suffix}.mat"

    if n_subset is not None:
        all_subs = all_subs[:n_subset]

    all_nodals = []
    for sub in all_subs:
        path_to_mat = op.join(path_to_hcp, sub, task_dir, atlas)
        fnames = sorted([f for f in os.listdir(path_to_mat) if fname_suffix in f])

        if len(fnames) > 1:
            print("More than one matching file, taking the first one only")

        mat = loadmat(op.join(path_to_mat, fnames[0]))
        all_nodals.append(mat["TS"].T)

    if not concat:
        return all_nodals
    else:
        nodal_fmri = np.concatenate(all_nodals, axis=0)
        print("Nodal fMRI has shape:", nodal_fmri.shape)

        all_lengths = [len(n) for n in all_nodals]

        # TODO: Implement the paradygm loading and concatenation
        # if "rest" not in task:
        #     path_to_paradygm = op.join(
        #         path_to_fmri, fname.replace("timeseries", "regressor")
        #     )
        #     paradygm = np.genfromtxt(path_to_paradygm, delimiter=",").astype(int)
        # else:
        #     paradygm = np.zeros(len(nodal_fmri), dtype=int)

        return nodal_fmri, all_lengths  # , paradygm


def get_lobe_info(scale, labels, path_to_lobe="./results/atlas_correspondence"):
    lobe_info = load(op.join(path_to_lobe, f"Laus2018_LobeMNI-scale{scale}OneThal.pkl"))
    lobe_labels = lobe_info["labels"]
    maxlobe = np.argmax(lobe_info["dice"], axis=0)

    lobe_df = pd.DataFrame(
        {
            "roi_id": np.arange(len(labels)),
            "roi_name": labels,
            "lobe_id": maxlobe,
            "lobe_label": [lobe_labels[ll] for ll in maxlobe],
        }
    )

    is_subc = ["subc-rh" in n for n in lobe_df["roi_name"].values]
    lobe_df.loc[is_subc, "lobe_id"] = 17
    lobe_df.loc[is_subc, "lobe_label"] = "rh-subcortical"
    lobe_labels[17] = "rh-subcortical"

    is_subc = ["subc-lh" in n for n in lobe_df["roi_name"].values]
    lobe_df.loc[is_subc, "lobe_id"] = 8
    lobe_df.loc[is_subc, "lobe_label"] = "lh-subcortical"
    lobe_labels[8] = "lh-subcortical"

    lobe_df.loc[lobe_df.roi_name == "Brain_Stem", "lobe_id"] = 18
    lobe_df.loc[lobe_df.roi_name == "Brain_Stem", "lobe_label"] = "brainstem"
    lobe_labels.append("brainstem")

    new_order = [2, 5, 4, 7, 3, 8, 11, 14, 13, 16, 12, 17, 18]
    reorder_ids = {old_id: new_id for new_id, old_id in enumerate(new_order)}

    lobe_df["lobe_id_reorder"] = lobe_df["lobe_id"].map(reorder_ids)
    order_by_lobe = np.argsort(lobe_df["lobe_id_reorder"].values)
    lobe_sizes = lobe_df["lobe_id_reorder"].value_counts().sort_index().values
    lobe_labels = [lobe_labels[old_id] for old_id in new_order]

    return order_by_lobe, lobe_sizes, lobe_labels, lobe_df
