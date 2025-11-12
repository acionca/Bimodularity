from hmac import new
import os.path as op

import numpy as np

import h5py
import nibabel as nib


def fix_thalamus(labels, matrix=None, atlas_data=None, pos=None, use_max=False):

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

    if pos is None and matrix is None and atlas_data is None:
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
