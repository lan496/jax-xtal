import os

from jax.random import PRNGKey

from jax_xtal.data import CutoffNN, AtomFeaturizer, BondFeaturizer, CrystalDataset, get_dataloaders
from jax_xtal.model import CGCNN


def test_cgcnn_forward():
    neighbor_strategy = CutoffNN(cutoff=6.0)
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    atom_featurizer = AtomFeaturizer(os.path.join(root_dir, "data", "atom_init.json"))
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=10)
    dataset = CrystalDataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        neighbor_strategy=neighbor_strategy,
        structures_dir=os.path.join(root_dir, "data", "structures_dummy"),
        targets_csv_path=os.path.join(root_dir, "data", "targets_dummy.csv"),
        max_num_neighbors=12,
    )

    rng = PRNGKey(0)

    train_loader, _, _ = get_dataloaders(dataset, batch_size=2)
    batch = next(iter(train_loader))

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
