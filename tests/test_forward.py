import os

from jax.random import PRNGKey

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    collate_pool,
)
from jax_xtal.model import CGCNN


def test_cgcnn_forward():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    atom_featurizer = AtomFeaturizer(os.path.join(root_dir, "data", "atom_init.json"))
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=10)
    dataset, _ = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=os.path.join(root_dir, "data", "structures_dummy"),
        targets_csv_path=os.path.join(root_dir, "data", "targets_dummy.csv"),
        max_num_neighbors=12,
        cutoff=6.0,
    )

    rng = PRNGKey(0)

    batch_size = 2
    batch = collate_pool(dataset[:batch_size])

    neighbor_indices = batch["neighbor_indices"]
    atom_features = batch["atom_features"]
    bond_features = batch["bond_features"]
    atom_indices = batch["atom_indices"]

    model = CGCNN(num_atom_features=2, num_convs=1, num_hidden_layers=1, num_hidden_features=64)
    params = model.init(rng, neighbor_indices, atom_features, bond_features, atom_indices)
    _ = model.apply(
        params,
        neighbor_indices,
        atom_features,
        bond_features,
        atom_indices,
        mutable=["batch_stats"],
    )
