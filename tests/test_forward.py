import os

import haiku as hk
import jax

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    collate_pool,
)
from jax_xtal.model import get_model_fn_t


def test_cgcnn_forward():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    num_bond_features = 10
    atom_featurizer = AtomFeaturizer(os.path.join(root_dir, "data", "atom_init.json"))
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=num_bond_features)
    max_num_neighbors = 12
    dataset, _ = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=os.path.join(root_dir, "data", "structures_dummy"),
        targets_csv_path=os.path.join(root_dir, "data", "targets_dummy.csv"),
        max_num_neighbors=max_num_neighbors,
        cutoff=6.0,
        is_training=True,
    )

    rng = jax.random.PRNGKey(0)

    # Define model
    batch_size = 2
    model_fn = get_model_fn_t(
        num_initial_atom_features=atom_featurizer.num_initial_atom_features,
        num_atom_features=2,
        num_bond_features=num_bond_features,
        num_convs=3,
        num_hidden_layers=5,
        num_hidden_features=7,
        max_num_neighbors=max_num_neighbors,
        batch_size=batch_size,
    )
    model = hk.without_apply_rng(model_fn)

    # Initialize model
    batch = collate_pool(dataset[:batch_size], False)
    rng, init_rng = jax.random.split(rng)
    params, state = model.init(init_rng, batch, is_training=True)

    _ = model.apply(params, state, batch, is_training=True)
    _ = model.apply(params, state, batch, is_training=False)
