import os
import os.path as op

import pandas as pd
import numpy as np

import h5py
import nibabel as nib

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mdf

from tqdm.notebook import tqdm


def fix_thalamus(
    labels,
    matrix=None,
    atlas_data=None,
    pos=None,
    timeseries=None,
    use_max=False,
):

    # Average the Thalamus SubRegions
    is_r_thal = ["thal-rh-" in lab for lab in labels]
    is_l_thal = ["thal-lh-" in lab for lab in labels]

    r_thal_idx = np.where(is_r_thal)[0]
    l_thal_idx = np.where(is_l_thal)[0]

    remove_dupl = np.logical_or(is_r_thal, is_l_thal)
    remove_dupl[r_thal_idx.min()] = False
    remove_dupl[l_thal_idx.min()] = False

    new_labels = labels.copy()
    new_labels[r_thal_idx] = "subc-rh-thalamus"
    new_labels[l_thal_idx] = "subc-lh-thalamus"
    new_labels = new_labels[~remove_dupl]

    if pos is None and matrix is None and atlas_data is None and timeseries is None:
        return new_labels

    if pos is not None:
        new_pos = pos.copy()
        new_pos[r_thal_idx] = pos[r_thal_idx].mean(axis=0, keepdims=True)
        new_pos[l_thal_idx] = pos[l_thal_idx].mean(axis=0, keepdims=True)
        new_pos = new_pos[~remove_dupl]

        if matrix is None and atlas_data is None:
            return new_labels, new_pos

    if matrix is not None:
        new_matrix = matrix.copy()

        if use_max:
            new_matrix[is_r_thal, :] = matrix[is_r_thal, :].max(axis=0, keepdims=True)
            new_matrix[is_l_thal, :] = matrix[is_l_thal, :].max(axis=0, keepdims=True)
            new_matrix[:, is_r_thal] = matrix[:, is_r_thal].max(axis=1, keepdims=True)
            new_matrix[:, is_l_thal] = matrix[:, is_l_thal].max(axis=1, keepdims=True)
        else:
            new_matrix[is_r_thal, :] = matrix[is_r_thal, :].mean(axis=0, keepdims=True)
            new_matrix[is_l_thal, :] = matrix[is_l_thal, :].mean(axis=0, keepdims=True)
            new_matrix[:, is_r_thal] = matrix[:, is_r_thal].mean(axis=1, keepdims=True)
            new_matrix[:, is_l_thal] = matrix[:, is_l_thal].mean(axis=1, keepdims=True)

        new_matrix = matrix[~remove_dupl, :][:, ~remove_dupl]
        if atlas_data is None:
            return new_labels, new_matrix

    if atlas_data is not None:
        new_atlas_data = atlas_data.copy()

        for i in r_thal_idx:
            new_atlas_data[atlas_data == i + 1] = r_thal_idx.min() + 1

        r_remove = len(r_thal_idx) - 1
        new_atlas_data[new_atlas_data > r_thal_idx.min() + 1] -= r_remove
        for i in l_thal_idx:
            new_atlas_data[atlas_data == i + 1] = l_thal_idx.min() + 1 - r_remove
        new_atlas_data[new_atlas_data > l_thal_idx.min() + 1] -= len(l_thal_idx) - 1
        if matrix is None:
            return new_labels, new_atlas_data, new_pos
        if pos is None:
            return new_labels, new_matrix, new_atlas_data

    if timeseries is not None:
        new_ts = timeseries.copy()
        new_ts[is_r_thal, :] = timeseries[is_r_thal, :].mean(axis=0, keepdims=True)
        new_ts[is_l_thal, :] = timeseries[is_l_thal, :].mean(axis=0, keepdims=True)

        new_ts = timeseries[~remove_dupl]
        return new_labels, new_ts

    return new_labels, new_matrix, new_atlas_data, new_pos


def get_bundle_data(
    path_to_data, path_to_ressources, scale=1, path_to_bundles="./data/bundles"
):
    b_atlas = op.join(path_to_data, "atlas_data", "fiber_atlas")

    # connFilename = op.join(path_to_bundles, f'wm.connatlas.scale{scale}.h5')
    connFilename = op.join(b_atlas, "probconnatlas", f"wm.connatlas.scale{scale}.h5")
    hf = h5py.File(connFilename, "r")
    atlas_dict = hf.get("atlas")

    # b_count_img = nib.load(op.join(path_to_bundles, f"wmatlas.scale{scale}.bundcount.nii.gz"))
    b_count_img = nib.load(
        op.join(b_atlas, f"bundcount_and_tdi", f"wmatlas.scale{scale}.bundcount.nii.gz")
    )

    # gm_rois = nib.load(op.join(path_to_ressources, f"roi_atlas-ftract-scale{scale}-GM.nii.gz"))
    gm_rois = nib.load(
        op.join(path_to_ressources, f"Laus2018_roi_atlas-ftract-scale{scale}-GM.nii.gz")
    )
    gm_rois_array = gm_rois.get_fdata().astype(int)

    return atlas_dict, b_count_img, gm_rois_array


def bundle_bicom(
    edge_clusters_mat,
    c_out,
    c_in,
    adj_matrix,
    b_count_img,
    gm_rois_array,
    atlas_dict,
    prob=True,
    weight=True,
    bundle_only=False,
):
    n_clusters = edge_clusters_mat.max()

    match_indexes = np.arange(c_out.shape[1])
    mni_array = np.zeros(
        (b_count_img.shape[0], b_count_img.shape[1], b_count_img.shape[2], n_clusters),
        dtype=float,
    )
    matching_str = (match_indexes + 1).astype(str)

    if bundle_only:
        mni_send_receive = None
    else:
        mni_send_receive = np.zeros(
            (
                b_count_img.shape[0],
                b_count_img.shape[1],
                b_count_img.shape[2],
                n_clusters,
                2,
            ),
            dtype=float,
        )

    for k in range(n_clusters):
        cluster_edges_ids = np.argwhere(edge_clusters_mat == k + 1)

        idx0, idx1 = cluster_edges_ids[:, 0], cluster_edges_ids[:, 1]
        mask_flip = match_indexes[idx0] > match_indexes[idx1]
        cluster_edges_ids[mask_flip] = np.stack(
            [idx1[mask_flip], idx0[mask_flip]], axis=1
        )

        mask_valid = matching_str[cluster_edges_ids[:, 0]] != "0"
        if not all(mask_valid):
            print(
                f"Warning: cluster {k+1} has {np.sum(~mask_valid)} edges with 0 index, ignored."
            )
        b_ids = (
            matching_str[cluster_edges_ids[:, 0]]
            + "_"
            + matching_str[cluster_edges_ids[:, 1]]
        )

        cluster_edges_ids = cluster_edges_ids[mask_valid]

        if not bundle_only:
            roi_val = c_out[k, i].copy()
            for i in np.where(c_out[k] > 0)[0]:
                mni_send_receive[gm_rois_array == i + 1, k, 0] = roi_val
            roi_val = c_in[k, i].copy()
            for i in np.where(c_in[k] > 0)[0]:
                mni_send_receive[gm_rois_array == i + 1, k, 1] = roi_val

        for edge, bundle_id in zip(cluster_edges_ids, b_ids):

            fill = 1.0
            if prob:
                fill = atlas_dict.get(bundle_id)[:, -1].astype(float)

            if weight:
                fill *= adj_matrix[edge[0], edge[1]]

            coords = atlas_dict.get(bundle_id)[:, :3].T
            mni_array[coords[0], coords[1], coords[2], k] += fill

    mni_array = mni_array / mni_array.max(axis=(0, 1, 2), keepdims=True)

    return mni_array, mni_send_receive


def get_thalamus_bundles_info(labels, scale):
    if scale != 2:
        raise NotImplementedError(
            "Thalamus bundles info (and centroid data) is only implemented for scale 2"
        )

    thal_ids = np.array(np.where(["thalamus" in l for l in labels])[0]) + 1
    n_thal_sub = 7

    thal_ids_scale = {2: [np.arange(58, 65), np.arange(128, 135)]}
    thal_ids_bundles = thal_ids_scale[scale]

    return thal_ids, thal_ids_bundles, n_thal_sub


def get_bicom_bundles(bicom_id, edge_clusters_mat, labels, scale=2):
    # print(f"Bicom {bicom_id} selected")

    thal_ids, thal_ids_bundles, n_thal_sub = get_thalamus_bundles_info(labels, scale)

    e_array = np.array(np.where(edge_clusters_mat == bicom_id)) + 1

    selected_bundles = []
    selected_bundles_dir = []
    for e in e_array.T:

        s_r_list = []
        for s_r in e:
            if s_r < thal_ids[0]:
                s_r_list.append([s_r])
            elif s_r == thal_ids[0]:
                s_r_list.append(thal_ids_bundles[0])
            elif s_r < thal_ids[1]:
                s_r_list.append([s_r + n_thal_sub - 1])
            elif s_r == thal_ids[1]:
                s_r_list.append(thal_ids_bundles[1])
            else:
                s_r_list.append([s_r + n_thal_sub * 2 - 2])

        # fixed_e = [f"{s}_{r}" for s in s_r_list[0] for r in s_r_list[1]]
        fixed_e = [[s, r] for s in s_r_list[0] for r in s_r_list[1]]
        dir_e = [s > r for s, r in fixed_e]

        fixed_e = [
            f"{s}_{r}" if not d else f"{r}_{s}" for (s, r), d in zip(fixed_e, dir_e)
        ]

        selected_bundles.extend(fixed_e)
        selected_bundles_dir.extend(dir_e)

    selected_bundles_dir = np.array(selected_bundles_dir)
    selected_bundles = np.array(selected_bundles)
    return selected_bundles, selected_bundles_dir


def get_bundle_centroid(
    centroid_dir, scale, selected_bundles, selected_bundles_dir, n_centroids=2
):
    # Location of the atlas
    # group_bund_cent = "/media/COSAS/Yasser/Work2Do/Connectome_Atlas/Results/Concatenated/scale2/group_centroids/"
    # centroid_dir = (
    #     "/Users/acionca/data/atlas_data/centroids/scale2/group_centroids_scale2"
    # )

    h5_centroids_file = os.path.join(
        centroid_dir, f"wm.connatlas.scale{scale}.centroids.h5"
    )

    # Reading the HDF5 file
    hf = h5py.File(h5_centroids_file, "r")
    bundles = list(hf.get("centroids").keys())

    # Creating the new h5 file
    # tempVar = hf.get("header/affine")
    # affine = np.array(tempVar)
    # print("Affine:", affine)

    # Select only the selected bundles
    # bundles = [b for b in bundles if b in selected_bundles]
    bundles_dir = [
        d for b, d in zip(selected_bundles, selected_bundles_dir) if b in bundles
    ]
    bundles = [b for b in selected_bundles if b in bundles]

    # print(
    #     f"Plotting {len(bundles)} bundles out of {len(selected_bundles)} selected bundles !"
    # )

    stl_coord_list = []
    stl_bund_id = []
    stl_dir_list = []

    already_seen = []

    # n_bidir = 0
    for bund_id, (bund, inv_dir) in enumerate(zip(bundles, bundles_dir)):
        # Read the group data
        data = hf.get("centroids/" + bund)
        streamlines_matrix = np.array(data)

        sl_coords = []
        sl_id = []
        sl_dir = []

        # Handling bidirectional bundles
        if bund in already_seen:
            where = np.where(np.array(already_seen) == bund)[0][0]
            stl_dir_list[n_centroids * where : n_centroids * (where + 1)] = [
                np.zeros_like(d)
                for d in stl_dir_list[n_centroids * where : n_centroids * (where + 1)]
            ]
            # n_bidir += 1
            continue

        # Do a loop along the streamlines
        stl_ids = np.unique(streamlines_matrix[:, 3])
        for stl_id in stl_ids:
            ind = streamlines_matrix[:, 3] == stl_id
            stl_coord = streamlines_matrix[ind, :3]
            stl_send = streamlines_matrix[ind, 4]
            stl_rec = streamlines_matrix[ind, 5]

            # Append coordinates
            sl_coords.append(stl_coord)

            # For data_per_point, each streamline needs its own array
            # Reshape to (n_points, 1) for TRK format
            sl_id.append(np.ones((len(stl_coord), 1), dtype=np.float32) * bund_id)
            # sl_send.append(stl_send.reshape(-1, 1).astype(np.float32))
            # sl_rec.append(stl_rec.reshape(-1, 1).astype(np.float32))

            if inv_dir:
                dir = stl_rec.reshape(-1, 1).astype(np.float32)
            else:
                dir = stl_send.reshape(-1, 1).astype(np.float32)

            if bund in already_seen:
                print("THIS SHOULD NEVER HAPPEN")
                dir = np.zeros_like(dir)

            sl_dir.append(dir)

            # sl_dir.append(stl_rec.reshape(-1, 1).astype(np.float32))

            # # Append coordinates
            # stl_coord_list.append(stl_coord)

            # # For data_per_point, each streamline needs its own array
            # # Reshape to (n_points, 1) for TRK format
            # stl_bund_id.append(np.ones((len(stl_coord), 1), dtype=np.float32) * bund_id)
            # stl_send_list.append(stl_send.reshape(-1, 1).astype(np.float32))
            # stl_rec_list.append(stl_rec.reshape(-1, 1).astype(np.float32))

        mean_sl = np.median(sl_coords, axis=0)

        # if n_centroids == 1:
        #     stl_coord_list.extend([mean_sl])
        #     stl_bund_id.extend([(np.ones(len(mean_sl)) * bund_id).reshape(-1, 1)])
        # stl_dir_list.extend(np.linspace(-1, 1, len(mean_sl), endpoint=True).reshape(-1, 1))

        d_mat = np.pow(np.array(sl_coords) - mean_sl, 2).sum(axis=(1, 2))
        kept_sl = np.argsort(d_mat)[:n_centroids]

        stl_coord_list.extend(np.array(sl_coords)[kept_sl])
        stl_bund_id.extend(np.array(sl_id)[kept_sl])
        stl_dir_list.extend(np.array(sl_dir)[kept_sl])

        already_seen.append(bund)

    # print(f"Number of bidirectional bundles found: {n_bidir} (out of {len(bundles)})")

    # Close the HDF5 file
    hf.close()

    # Create tractogram with the affine matrix
    centroid_tractogram = nib.streamlines.Tractogram(
        stl_coord_list, affine_to_rasmm=np.eye(4)
    )

    # Add data per_point to the tractogram
    centroid_tractogram.data_per_point["bundle_id"] = stl_bund_id
    centroid_tractogram.data_per_point["dir_col"] = stl_dir_list

    # centroid_file = os.path.join(
    #     centroid_dir,
    #     f"saved_centroids-scale{scale}",
    #     f"wm.connatlas.scale{scale}.bicom{selected_bicom}of{edge_clusters_mat.max()}.trk",
    # )
    # os.makedirs(os.path.dirname(centroid_file), exist_ok=True)
    # Save with a reference image if available, or just with the tractogram
    # nib.streamlines.save(
    #     centroid_tractogram,
    #     centroid_file,
    # )

    return centroid_tractogram


path_to_data = "/Users/acionca/data"
path_to_bundles = "./data/bundles"
b_atlas = op.join(path_to_data, "atlas_data", "fiber_atlas")


def get_edge_to_bundle(
    graph: np.ndarray, scale: str, labels: pd.DataFrame, overlap_thresh: float = 0.1
):
    # connFilename = op.join(path_to_bundles, f'wm.connatlas.scale{scale}.h5')
    connFilename = op.join(b_atlas, "probconnatlas", f"wm.connatlas.scale{scale}.h5")
    hf = h5py.File(connFilename, "r")
    atlas_dict = hf.get("atlas")
    atlas_list = list(atlas_dict.keys())

    # b_count_img = nib.load(op.join(path_to_bundles, f"wmatlas.scale{scale}.bundcount.nii.gz"))
    b_count_img = nib.load(
        op.join(b_atlas, f"bundcount_and_tdi", f"wmatlas.scale{scale}.bundcount.nii.gz")
    )

    # Loading anatomical bundle atlas
    path_to_a_bundles = "/Users/acionca/data/atlas_SCIL/atlas"

    bundle_fname = "probability_maps_all_0.1.nii.gz"
    a_bundles = nib.load(op.join(path_to_a_bundles, bundle_fname))
    a_bundles_data = a_bundles.get_fdata()
    a_bundles_labels = pd.read_csv(op.join(path_to_a_bundles, "bundle_names.csv"))
    a_bundles_labels = a_bundles_labels.rename(columns={"Unnamed: 0": "BundleNum"})
    a_bundles_labels["Both"] = (
        a_bundles_labels["BundleNum"].astype(str) + "_" + a_bundles_labels["BundleName"]
    )

    b_id_mat = np.zeros_like(graph, dtype=int)
    b_id_mat[graph > 0] = np.arange(1, np.sum(graph > 0) + 1)

    edge_to_bundle = np.zeros((b_id_mat.max(), a_bundles_data.shape[-1]), dtype=float)
    a_masks = np.zeros(
        (
            a_bundles_data.shape[-1],
            a_bundles_data.shape[0] * a_bundles_data.shape[1] * a_bundles_data.shape[2],
        ),
        dtype=int,
    )

    for anat_i in range(a_bundles_data.shape[-1]):
        a_masks[anat_i] = (
            (a_bundles_data[:, :, :, anat_i] > overlap_thresh).astype(int).flatten()
        )

    print(f"Estimatinge edge to bundle overlap ...")
    for e_id in tqdm(range(b_id_mat.max())):
        b, _ = get_bicom_bundles(e_id + 1, b_id_mat, labels)

        mni_array = np.zeros(
            (b_count_img.shape[0], b_count_img.shape[1], b_count_img.shape[2]),
            dtype=float,
        )

        for b_i in b:
            if b_i in atlas_list:
                fill = atlas_dict[b_i][:, -1].astype(float)
                coords = atlas_dict[b_i][:, :3].T
                mni_array[coords[0], coords[1], coords[2]] += fill

        mni_array = (
            (mni_array / np.max(mni_array) > overlap_thresh).astype(int).flatten()
        )
        intersect = a_masks @ mni_array

        edge_to_bundle[e_id] = intersect / np.sum(mni_array)

    return np.nan_to_num(edge_to_bundle)


def compute_edge_to_bundle_distance(
    graph,
    labels,
    scale,
    path_to_data="/Users/acionca/data/",
    a_bundles_labels=None,
    edge_n_points=12,
):
    path_to_a_bundles = op.join(path_to_data, "atlas_SCIL")
    centroid_dir = op.join(
        path_to_data,
        "atlas_data",
        "centroids",
        f"scale{scale}",
        f"group_centroids_scale{scale}",
    )

    if a_bundles_labels is None:
        a_bundles_labels = pd.read_csv(
            op.join(path_to_a_bundles, "atlas", "bundle_names.csv")
        )
        a_bundles_labels = a_bundles_labels.rename(columns={"Unnamed: 0": "BundleNum"})

    all_a_cent = []
    for a_i, a_name in enumerate(a_bundles_labels.BundleName):
        # print(a_i, a_name)
        b_cent_fname = op.join(path_to_a_bundles, "centroids", f"{a_name}_centroid.trk")
        a_slines = nib.streamlines.load(b_cent_fname).streamlines
        a_slines = set_number_of_points(a_slines, nb_points=edge_n_points)
        all_a_cent.append(a_slines)

    b_id_mat = np.zeros_like(graph, dtype=int)
    b_id_mat[graph > 0] = np.arange(1, np.sum(graph > 0) + 1)

    edge_to_bundle_Bmdf = np.zeros((np.sum(graph > 0), len(all_a_cent)))
    all_e_cent = []
    for b_id in tqdm(np.arange(1, np.sum(graph > 0) + 1)):
        selected_bundles, selected_bundles_dir = get_bicom_bundles(
            b_id, b_id_mat, labels, scale=scale
        )
        tract = get_bundle_centroid(
            centroid_dir,
            scale,
            selected_bundles=selected_bundles,
            selected_bundles_dir=selected_bundles_dir,
            n_centroids=1,
        )

        e_cent = tract.streamlines

        for a_i, a_cent in enumerate(all_a_cent):
            dist = bundles_distances_mdf(a_cent, e_cent)
            edge_to_bundle_Bmdf[b_id - 1, a_i] = dist.min(axis=1).mean()

    edge_to_bundle_Bmdf = 1 / edge_to_bundle_Bmdf

    return edge_to_bundle_Bmdf
