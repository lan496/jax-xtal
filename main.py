import os
from functools import partial

import jax
from jax.random import PRNGKey
import torch

from jax_xtal.data import CutoffNN, AtomFeaturizer, BondFeaturizer, CrystalDataset, get_dataloaders
from jax_xtal.model import CGCNN
from jax_xtal.train_utils import (
    create_train_state,
    train_one_step,
    eval_one_step,
    train_one_epoch,
    eval_model,
    multi_step_lr,
)


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    rng = PRNGKey(seed)

    max_num_neighbors = 12
    num_initial_atom_features = 2
    num_bond_features = 10
    num_epochs = 30
    learning_rate = 1e-5
    milestones = 10
    l2_reg = 1e-8
    batch_size = 2

    neighbor_strategy = CutoffNN(cutoff=6.0)
    root_dir = os.path.dirname(__file__)
    atom_featurizer = AtomFeaturizer(os.path.join(root_dir, "data", "atom_init.json"))
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=num_bond_features)
    dataset = CrystalDataset(
        os.path.join(root_dir, "data", "structures_dummy"),
        os.path.join(root_dir, "data", "targets_dummy.csv"),
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        neighbor_strategy=neighbor_strategy,
        max_num_neighbors=max_num_neighbors,
    )
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=batch_size)

    model = CGCNN(num_atom_features=2, num_convs=1, num_hidden_layers=1, num_hidden_features=64)
    rng, rng_state = jax.random.split(rng)
    state = create_train_state(
        rng=rng_state,
        model=model,
        max_num_neighbors=max_num_neighbors,
        num_initial_atom_features=dataset.num_initial_atom_features,
        num_bond_features=num_bond_features,
        learning_rate=learning_rate,
    )

    # MultiStepLR scheduelr for learning rate
    steps_per_epoch = len(train_loader) // batch_size
    learning_rate_fn = partial(
        multi_step_lr,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        milestones=milestones,
    )

    train_step_fn = jax.jit(
        partial(
            train_one_step, apply_fn=model.apply, learning_rate_fn=learning_rate_fn, l2_reg=l2_reg
        )
    )
    val_step_fn = jax.jit(partial(eval_one_step, apply_fn=model.apply))

    epoch_metrics = []
    for epoch in range(1, num_epochs + 1):
        state, train_summary = train_one_epoch(train_step_fn, state, train_loader, epoch)
        train_loss = train_summary["loss"]
        train_mae = train_summary["mae"]
        print("Training - epoch: %2d, loss: %.2f, MAE: %.2f" % (epoch, train_loss, train_mae))
        val_summary = eval_model(val_step_fn, state, val_loader)
        val_loss = val_summary["loss"]
        val_mae = val_summary["mae"]
        print("Testing  - epoch: %2d, loss: %.2f, MAE: %.2f" % (epoch, val_loss, val_mae))
