import os
from functools import partial
import argparse

import jax
from jax.random import PRNGKey

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
)
from jax_xtal.model import CGCNN
from jax_xtal.train_utils import (
    create_train_state,
    predict_one_step,
    predict_dataset,
    restore_checkpoint,
    Normalizer,
)
from jax_xtal.config import load_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="pre-trained parameters")
    parser.add_argument("--config", required=True, type=str, help="json config used for training")
    parser.add_argument(
        "--structures_dir", required=True, type=str, help="directory of json files to be predicted"
    )
    parser.add_argument("--output", required=True, type=str, help="path to output predictions")
    args = parser.parse_args()

    config = load_config(args.config)

    root_dir = os.path.dirname(__file__)
    atom_init_features_path = os.path.join(root_dir, "data", "atom_init.json")
    structures_dir = (os.path.join(root_dir, "data", "structures_dummy"),)

    seed = config.seed
    rng = PRNGKey(seed)

    # prepare dataset
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.dmax, num_filters=config.num_bond_features
    )
    dataset, list_ids = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=config.structures_dir,
        targets_csv_path=config.targets_csv_path,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        seed=seed,
        n_jobs=config.n_jobs,
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
        num_initial_atom_features=atom_featurizer.num_initial_atom_features,
        num_bond_features=config.num_bond_features,
        learning_rate=config.learning_rate,
        normalizer=Normalizer(0, 0),  # dummy normalizer instance
    )
    state = restore_checkpoint(args.checkpoint, state)

    # prediction
    predictions = predict_dataset(
        apply_fn=model.apply, state=state, dataset=dataset, batch_size=config.batch_size
    )

    # denormalize predictions
    normalizer = Normalizer(state.sample_mean, state.sample_std)
    denormed_preds = normalizer.denormalize(predictions)

    with open(args.output, "w") as f:
        for i, idx in enumerate(list_ids):
            f.write(f"{idx}, {denormed_preds[i][0]}\n")
