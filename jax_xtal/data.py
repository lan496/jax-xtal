import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.analysis.graphs import StructureGraph
from torch.utils.data import Dataset


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
        with open(atom_features_json, 'r') as f:
            self._lookup_table = json.load(f)

    def _get_atom_feature(self, number: int):
        return self._lookup_table[str(number)]

    @property
    def num_atom_features(self):
        return len(self._lookup_table['1'])  # '1' means hydrogen

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
    def __init__(self, dmin: float, dmax: float, num_filters: int, blur=None):
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
        return np.exp(-((distances[:, None] - self._filter[None, :]) / self._blur) ** 2)

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

            bond_features.append(self._expand_by_basis(np.array([pn[0] for pn in padded_neighbors])))
            neighbor_indices.append([pn[1] for pn in padded_neighbors])

        bond_features = np.array(bond_features)
        neighbor_indices = np.array(neighbor_indices)
        return bond_features, neighbor_indices


class CutoffNN(NearNeighbors):
    """
    An extremely simple NN class only using cutoff radius
    """
    def __init__(self, cutoff: float):
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
    structures_dir:
    targets_csv_path: comma delimiter, no header
    """
    def __init__(
        self,
        structures_dir: str,
        targets_csv_path: str,
        atom_featurizer: AtomFeaturizer,
        bond_featurizer: BondFeaturizer,
        neighbor_strategy: NearNeighbors,
        max_num_neighbors=12,
        seed=0
    ):
        if not os.path.exists(structures_dir):
            raise FileNotFoundError(f"structures_dir does not exist: {structures_dir}")
        if not os.path.exists(targets_csv_path):
            raise FileNotFoundError(f"targets_csv_path does not exist: {targets_csv_path}")
        self._structures_dir = structures_dir
        self._targets_csv_path = targets_csv_path

        # load targets and shuffle indices
        _targets = pd.read_csv(
            self._targets_csv_path,
            sep=',',
            header=None,
            names=['id', 'target']
        )
        rng_np = np.random.default_rng(seed)
        self._targets = _targets.iloc[rng_np.permutation(len(_targets))].reset_index(drop=True)

        self._atom_featurizer = atom_featurizer
        self._bond_featurizer = bond_featurizer
        self._neighbor_strategy = neighbor_strategy
        self._max_num_neighbors = max_num_neighbors

    def __len__(self):
        return len(self._targets)

    @lru_cache()
    def __getitem__(self, idx):
        # load structure
        structure_json_basename = self._targets.iloc[idx]['id'] + '.json'
        structure_json_path = os.path.join(self._structures_dir, structure_json_basename)
        with open(structure_json_path, 'r') as f:
            structure = Structure.from_dict(json.load(f))

        initial_atom_features = self._atom_featurizer(structure)

        # Padding neighbors might cause artificial effect, see https://github.com/txie-93/cgcnn/pull/16
        graph = StructureGraph.with_local_env_strategy(structure, self._neighbor_strategy)
        bond_features, neighbor_indices = self._bond_featurizer(graph, self._max_num_neighbors)

        target = self._targets.iloc[idx]['target']

        data = {
            'id': self._targets.iloc[idx]['id'],
            'neighbor_indices': neighbor_indices,  # (num_atoms, max_num_neighbors)
            'initial_atom_features': initial_atom_features,  # (num_atoms, num_atom_features)
            'bond_features': bond_features,  # (num_atoms, max_num_neighbors, num_bond_features)
            'target': target,
        }
        return data


if __name__ == '__main__':
    neighbor_strategy = CutoffNN(cutoff=6.0)
    atom_featurizer = AtomFeaturizer('../data/atom_init.json')
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=10)
    dataset = CrystalDataset(
        '../data/structures_dummy',
        '../data/targets_dummy.csv',
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        neighbor_strategy=neighbor_strategy
    )
    print(dataset[0])
