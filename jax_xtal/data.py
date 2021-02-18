import json
import os
from glob import glob
from typing import List, Mapping, Any
from logging import getLogger
from functools import partial

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pymatgen.core import Structure, PeriodicSite
from tqdm import tqdm


Batch = Mapping[str, Any]


logger = getLogger("cgcnn")


class AtomFeaturizer:
    """
    Featurize each atom in structure

    Parameters
    ----------
    atom_features_json: path to the precomputed "atom_init.json"
    """

    def __init__(self, atom_features_json: str):
        if not os.path.exists(atom_features_json):
            raise FileNotFoundError("atom_features_json not found:", atom_features_json)
        with open(atom_features_json, "r") as f:
            self._lookup_table = json.load(f)

    def _get_atom_feature(self, number: int):
        return self._lookup_table[str(number)]

    @property
    def num_initial_atom_features(self):
        return len(self._lookup_table["1"])  # '1' means hydrogen

    def __call__(self, structure: Structure):
        """
        Parameters
        ---------
        structure: structure to be calculated atomic features

        Returns
        -------
        atom_features: (structure.num_sites, self.num_atom_features)
        """
        atom_features = np.array([self._get_atom_feature(site.specie.Z) for site in structure])
        return atom_features


class BondFeaturizer:
    """
    Expands the bond distance by Gaussian basis.

    Parameters
    ----------
    dmin: min bond distance for consideration
    dmax: max bond distance for consideration
    num_filters: number of gaussian filters
    blur: variance of gaussian filters
    """

    def __init__(self, dmin: float = 0.7, dmax: float = 5.2, num_filters: int = 10, blur=None):
        assert dmin < dmax
        assert num_filters >= 2
        self._dmin = dmin
        self._dmax = dmax
        self._num_filters = num_filters
        self._filter = np.linspace(dmin, dmax, num_filters, endpoint=True)

        if blur is None:
            self._blur = (dmax - dmin) / num_filters
        else:
            self._blur = blur

    @property
    def num_bond_features(self):
        return self._num_filters

    def _expand_by_basis(self, distances):
        # (max_num_neighbors, num_bond_features) is returned
        return np.exp(-(((distances[:, None] - self._filter[None, :]) / self._blur) ** 2))

    def __call__(self, all_neighbors: List[List[PeriodicSite]], max_num_neighbors: int):
        """
        Parameters
        ----------
        graph: Structure graph
        max_num_neighbors:
            If number of neighbors within a cutoff radius is less than `max_num_neighbors`,
            the remains are filled zeros.

        Returns
        -------
        bond_features: (num_atoms, max_num_neighbors, self.num_bond_features)
        neighbor_indices: (num_atoms, max_num_neighbors)
        """
        bond_features = []
        neighbor_indices = []
        # for i in range(graph.structure.num_sites):
        for neighbors in all_neighbors:
            # graph.get_connected_sites returns sorted list by distance
            nn = [(site.nn_distance, site.index) for site in neighbors]
            if len(nn) < max_num_neighbors:
                num_ghosts = max_num_neighbors - len(nn)
                # set some large distance
                padded_neighbors = nn + [(2 * self._dmax, 0) for _ in range(num_ghosts)]
            else:
                padded_neighbors = nn[:max_num_neighbors]

            bond_features.append(
                self._expand_by_basis(np.array([pn[0] for pn in padded_neighbors]))
            )
            neighbor_indices.append([pn[1] for pn in padded_neighbors])

        bond_features = np.array(bond_features)
        neighbor_indices = np.array(neighbor_indices)
        return bond_features, neighbor_indices


def create_dataset(
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: BondFeaturizer,
    structures_dir: str,
    targets_csv_path: str = "",
    is_training: bool = True,
    max_num_neighbors: int = 12,
    cutoff: float = 6.0,
    seed=0,
    n_jobs=1,
):
    """
    Parameters
    ----------
    atom_featurizer:
        convert atom into one-hot vector
    bond_featurizer:
        convert bond infomation into one-hot vector
    neighbor_strategy:
        inherit pymatgen's NearNeighbors class, which find neighbors of each atom with some criterion
    max_num_neighbors:
        maximum number of neighbors for each atom in cosideration
    cutoff: cutoff radius for graph
    structures_dir:
        path for json files of pymatgen's structures
    targets_csv_path:
        comma delimiter, no header
    is_training: bool
        if set False, targets_csv_path is not read.
    seed: int

    Returns
    -------
    dataset: list of dict with the following keys
        - neighbor_indices
        - atom_features
        - bond_features
        - (Optional) target
    ids: list of names of samples
    """
    if not os.path.exists(structures_dir):
        raise FileNotFoundError(f"structures_dir does not exist: {structures_dir}")

    if is_training:
        if not os.path.exists(targets_csv_path):
            raise FileNotFoundError(f"targets_csv_path does not exist: {targets_csv_path}")
        targets_csv_path = targets_csv_path

        # load targets and shuffle indices
        targets = pd.read_csv(targets_csv_path, sep=",", header=None, names=["id", "target"])
        rng_np = np.random.default_rng(seed)
        targets = targets.iloc[rng_np.permutation(len(targets))].reset_index(drop=True)
        ids = targets["id"].tolist()
    else:
        ids = list(
            [
                os.path.basename(path).split(".")[0]
                for path in glob(os.path.join(structures_dir, "*.json"))
            ]
        )

    # precompute datate
    logger.info("Preprocessing dataset")
    inputs = Parallel(n_jobs, verbose=1)(
        delayed(_create_inputs)(
            ids[idx], structures_dir, atom_featurizer, bond_featurizer, max_num_neighbors, cutoff,
        )
        for idx in range(len(ids))
    )

    dataset = [{**inputs[idx]} for idx in range(len(ids))]
    if is_training:
        for idx in range(len(ids)):
            target = targets.iloc[idx]["target"]
            dataset[idx]["target"] = target

    return dataset, ids


def _create_inputs(
    basename, structures_dir, atom_featurizer, bond_featurizer, max_num_neighbors, cutoff
):
    # load structure
    structure_json_basename = basename + ".json"
    structure_json_path = os.path.join(structures_dir, structure_json_basename)
    with open(structure_json_path, "r") as f:
        structure = Structure.from_dict(json.load(f))

    initial_atom_features = atom_featurizer(structure)

    # Padding neighbors might cause artificial effect, see https://github.com/txie-93/cgcnn/pull/16
    neighbors = structure.get_all_neighbors(r=cutoff)
    bond_features, neighbor_indices = bond_featurizer(neighbors, max_num_neighbors)

    inputs = {
        "neighbor_indices": neighbor_indices,  # (num_atoms, max_num_neighbors)
        "atom_features": initial_atom_features,  # (num_atoms, num_atom_features)
        "bond_features": bond_features,  # (num_atoms, max_num_neighbors, num_bond_features)
    }

    return inputs


def split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Returns
    -------
    train_dataset
    val_dataset
    test_dataset
    """
    assert train_ratio + val_ratio + test_ratio <= 1.0
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)

    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : (train_size + val_size)]
    test_indices = indices[-test_size:]

    train_dataset = [dataset[idx] for idx in train_indices]
    val_dataset = [dataset[idx] for idx in val_indices]
    test_dataset = [dataset[idx] for idx in test_indices]

    return train_dataset, val_dataset, test_dataset


def collate_pool(samples, train=True) -> Batch:
    """
    relabel atoms in a batch

    Parameters
    ----------
    samples: list of datum from CrystalDataset

    Returns
    -------
    batch_data: has a following entries
        - "neighbor_indices": (N, max_num_neighbors)
        - "atom_features": (N, num_atom_features)
        - "bond_features": (N, max_num_neighbors, num_bond_features)
        - "atom_indices"
        - "target": (batch_size, 1)
        where N is the total number of atoms in the samples
    """
    batch_neighbor_indices = []
    batch_atom_features = []
    batch_bond_features = []
    batch_targets = []
    atom_indices = []

    index_offset = 0
    for data in samples:
        batch_atom_features.append(data["atom_features"])
        batch_bond_features.append(data["bond_features"])
        batch_neighbor_indices.append(data["neighbor_indices"] + index_offset)
        if train:
            batch_targets.append(data["target"])

        num_atoms_i = data["atom_features"].shape[0]
        atom_indices.append(np.arange(num_atoms_i) + index_offset)

        index_offset += num_atoms_i

    batch_data = {
        "neighbor_indices": np.concatenate(batch_neighbor_indices, axis=0).astype(np.int32),
        "atom_features": np.concatenate(batch_atom_features, axis=0).astype(np.float32),
        "bond_features": np.concatenate(batch_bond_features, axis=0).astype(np.float32),
        "atom_indices": atom_indices,
    }
    if train:
        batch_data["target"] = np.array(batch_targets)[:, None]

    return batch_data
