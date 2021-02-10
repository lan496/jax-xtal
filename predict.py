import os
from functools import partial
import argparse

import jax
from jax.random import PRNGKey
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    CrystalDataset,
    get_dataloaders,
    collate_pool,
)
from jax_xtal.model import CGCNN
from jax_xtal.train_utils import (
    create_train_state,
    predict_one_step,
    predict_dataset,
    restore_checkpoint,
)
from jax_xtal.config import load_config


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="pre-trained parameters")
    parser.add_argument("--config", required=True, type=str, help="json config used for training")
    parser.add_argument(
        "--structures_dir", required=True, type=str, help="directory of json files to be predicted"
    )
    parser.add_argument("--output", required=True, type=str, help="path to output predictions")
    args = parser.parse_args()

    config = load_config(args.config)

    batch_size = 2
    num_workers = 1
    pin_memory = False

    root_dir = os.path.dirname(__file__)
    atom_init_features_path = os.path.join(root_dir, "data", "atom_init.json")
    structures_dir = (os.path.join(root_dir, "data", "structures_dummy"),)

    seed = config.seed
    torch.manual_seed(seed)
    rng = PRNGKey(seed)

    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.dmax, num_filters=config.num_bond_features
    )
    dataset = CrystalDataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=args.structures_dir,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        train=False,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate_pool, train=False),
        pin_memory=pin_memory,
    )

    # load checkpoint
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
    state = restore_checkpoint(args.checkpoint, state)

    # prediction
    pred_step_fn = jax.jit(partial(predict_one_step, apply_fn=model.apply))
    predictions = predict_dataset(pred_step_fn, state, dataloader)
    list_ids = dataset.get_id_list()
    with open(args.output, "w") as f:
        for i, idx in enumerate(list_ids):
            f.write(f"{idx}, {predictions[i][0]}\n")
