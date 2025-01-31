import os.path as op

import pandas as pd
import numpy as np
from scipy.io import loadmat


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
