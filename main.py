import os
from functools import partial
import argparse

import jax
from jax.random import PRNGKey
import torch
import torch.multiprocessing as multiprocessing

from jax_xtal.data import AtomFeaturizer, BondFeaturizer, CrystalDataset, get_dataloaders
from jax_xtal.model import CGCNN
from jax_xtal.train_utils import (
    create_train_state,
    train_one_step,
    eval_one_step,
    train_one_epoch,
    eval_model,
    multi_step_lr,
    save_checkpoint,
)
from jax_xtal.config import load_config


if __name__ == "__main__":
    # see https://github.com/google/jax/issues/3382
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path for json config")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config.seed
    torch.manual_seed(seed)
    rng = PRNGKey(seed)

    root_dir = os.path.dirname(__file__)
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.dmax, num_filters=config.num_bond_features
    )
    dataset = CrystalDataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=config.structures_dir,
        targets_csv_path=config.targets_csv_path,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        seed=seed,
        n_jobs=config.n_jobs,
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = CGCNN(
        num_atom_features=config.num_atom_features,
        num_convs=config.num_convs,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_features=config.num_hidden_features,
    )
    rng, rng_state = jax.random.split(rng)
    state = create_train_state(
        rng=rng_state,
        model=model,
        max_num_neighbors=config.max_num_neighbors,
        num_initial_atom_features=dataset.num_initial_atom_features,
        num_bond_features=config.num_bond_features,
        learning_rate=config.learning_rate,
    )

    # MultiStepLR scheduelr for learning rate
    train_size = int(len(dataset) * config.train_ratio)
    steps_per_epoch = train_size // config.batch_size
    learning_rate_fn = partial(
        multi_step_lr,
        steps_per_epoch=steps_per_epoch,
        learning_rate=config.learning_rate,
        milestones=config.milestones,
    )

    train_step_fn = jax.jit(
        partial(
            train_one_step,
            apply_fn=model.apply,
            learning_rate_fn=learning_rate_fn,
            l2_reg=config.l2_reg,
        )
    )
    val_step_fn = jax.jit(partial(eval_one_step, apply_fn=model.apply))

    print("Start training")
    epoch_metrics = []
    for epoch in range(1, config.num_epochs + 1):
        state, train_summary = train_one_epoch(train_step_fn, state, train_loader, epoch)
        train_loss = train_summary["loss"]
        train_mae = train_summary["mae"]
        print("Training - epoch: %2d, loss: %.2f, MAE: %.2f" % (epoch, train_loss, train_mae))
        val_summary = eval_model(val_step_fn, state, val_loader)
        val_loss = val_summary["loss"]
        val_mae = val_summary["mae"]
        print("Testing  - epoch: %2d, loss: %.2f, MAE: %.2f" % (epoch, val_loss, val_mae))

    test_summary = eval_model(val_step_fn, state, test_loader)
    test_loss = test_summary["loss"]
    test_mae = test_summary["mae"]
    print("Testing  -            loss: %.2f, MAE: %.2f" % (test_loss, test_mae))

    print("Save checkpoint")
    workdir = config.checkpoint_dir
    os.makedirs(workdir, exist_ok=True)
    save_checkpoint(state, workdir)
