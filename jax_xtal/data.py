import json
import os
from functools import lru_cache
from glob import glob

import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.analysis.graphs import StructureGraph
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


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
        Paraeters
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

    def __call__(self, graph: StructureGraph, max_num_neighbors: int):
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
        for i in range(graph.structure.num_sites):
            # graph.get_connected_sites returns sorted list by distance
            neighbors = [(site.dist, site.index) for site in graph.get_connected_sites(i)]
            if len(neighbors) < max_num_neighbors:
                num_ghosts = max_num_neighbors - len(neighbors)
                # set some large distance
                padded_neighbors = neighbors + [(2 * self._dmax, 0) for _ in range(num_ghosts)]
            else:
                padded_neighbors = neighbors[:max_num_neighbors]

            bond_features.append(
                self._expand_by_basis(np.array([pn[0] for pn in padded_neighbors]))
            )
            neighbor_indices.append([pn[1] for pn in padded_neighbors])

        bond_features = np.array(bond_features)
        neighbor_indices = np.array(neighbor_indices)
        return bond_features, neighbor_indices


class CutoffNN(NearNeighbors):
    """
    An extremely simple NN class only using cutoff radius

    # TODO: procedure to create graph in official implementation differs from that in SI.Sec.I.A.
    """

    def __init__(self, cutoff: float = 6.0):
        self._cutoff = cutoff

    @property
    def structures_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return True

    @property
    def extend_structure_molecules(self):
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed == False.
        """
        return True

    def get_nn_info(self, structure, n):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.
        Args:
            structure (Structure): input structure.
            n (integer): index of site for which to determine near-neighbor
                sites.
        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        site = structure[n]

        neighs_dists = structure.get_neighbors(site, self._cutoff)

        nn_info = []
        for nn in neighs_dists:
            n_site = nn
            dist = nn.nn_distance

            nn_info.append(
                {
                    "site": n_site,
                    "image": self._get_image(structure, n_site),
                    "weight": dist,
                    "site_index": self._get_original_site(structure, n_site),
                }
            )

        return nn_info


class CrystalDataset(Dataset):
    """
    Parameters
    ----------
    atom_featurizer:
        convert atom into one-hot vector
    bond_featurizer:
        convert bond infromation into one-hot vector
    neighbor_strategy:
        inherite pymatgen's NearNeighbors class, which find neighbors of each atom with some criterion
    max_num_neighbors:
        maximum number of neighbors for each atom in cosideration
    structures_dir:
        path for json files of pymatgen's structures
    targets_csv_path:
        comma delimiter, no header
    train: bool
        if set False, targets_csv_path is not read.
    seed: int
    """

    def __init__(
        self,
        atom_featurizer: AtomFeaturizer,
        bond_featurizer: BondFeaturizer,
        neighbor_strategy: NearNeighbors,
        structures_dir: str,
        targets_csv_path: str = "",
        train: bool = True,
        max_num_neighbors: int = 12,
        seed=0,
    ):
        if not os.path.exists(structures_dir):
            raise FileNotFoundError(f"structures_dir does not exist: {structures_dir}")
        self._structures_dir = structures_dir

        self._atom_featurizer = atom_featurizer
        self._bond_featurizer = bond_featurizer
        self._neighbor_strategy = neighbor_strategy
        self._max_num_neighbors = max_num_neighbors

        self._train = train
        if self._train:
            if not os.path.exists(targets_csv_path):
                raise FileNotFoundError(f"targets_csv_path does not exist: {targets_csv_path}")
            self._targets_csv_path = targets_csv_path

            # load targets and shuffle indices
            _targets = pd.read_csv(
                self._targets_csv_path, sep=",", header=None, names=["id", "target"]
            )
            rng_np = np.random.default_rng(seed)
            self._targets = _targets.iloc[rng_np.permutation(len(_targets))].reset_index(drop=True)
            self._ids = self._targets["id"].tolist()
        else:
            self._ids = list(
                [
                    os.path.basename(path).split(".")[0]
                    for path in glob(os.path.join(structures_dir, "*.json"))
                ]
            )

    def __len__(self):
        return len(self._ids)

    @property
    def num_initial_atom_features(self):
        return self._atom_featurizer.num_initial_atom_features

    def get_id_list(self):
        return self._ids

    @lru_cache()
    def __getitem__(self, idx):
        # load structure
        structure_json_basename = self._ids[idx] + ".json"
        structure_json_path = os.path.join(self._structures_dir, structure_json_basename)
        with open(structure_json_path, "r") as f:
            structure = Structure.from_dict(json.load(f))

        initial_atom_features = self._atom_featurizer(structure)

        # Padding neighbors might cause artificial effect, see https://github.com/txie-93/cgcnn/pull/16
        graph = StructureGraph.with_local_env_strategy(structure, self._neighbor_strategy)
        bond_features, neighbor_indices = self._bond_featurizer(graph, self._max_num_neighbors)

        data = {
            "id": self._ids[idx],
            "neighbor_indices": neighbor_indices,  # (num_atoms, max_num_neighbors)
            "atom_features": initial_atom_features,  # (num_atoms, num_atom_features)
            "bond_features": bond_features,  # (num_atoms, max_num_neighbors, num_bond_features)
        }

        if self._train:
            target = self._targets.iloc[idx]["target"]
            data["target"] = target

        return data


def get_dataloaders(
    dataset: CrystalDataset,
    batch_size=1,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    num_workers=0,
    pin_memory=False,
):
    assert train_ratio + val_ratio + test_ratio <= 1.0
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)

    indices = list(range(total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size : (train_size + val_size)])
    test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def collate_pool(samples, train=True):
    """
    relabel atoms in a batch

    Parameters
    ----------
    samples: list of datum from CrystalDataset

    Returns
    -------
    batch_data: has a following entries
        - "id": (batch_size, )
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
        atom_indices.append(jnp.arange(num_atoms_i) + index_offset)

        index_offset += num_atoms_i

    batch_data = {
        "neighbor_indices": jnp.concatenate(batch_neighbor_indices, axis=0),
        "atom_features": jnp.concatenate(batch_atom_features, axis=0),
        "bond_features": jnp.concatenate(batch_bond_features, axis=0),
        "atom_indices": atom_indices,
    }
    if train:
        batch_data["target"] = jnp.array(batch_targets)[:, None]

    return batch_data
