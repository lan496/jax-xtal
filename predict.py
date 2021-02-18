import os
import argparse

import haiku as hk
import jax
import jax.numpy as jnp

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    Batch,
    collate_pool,
)
from jax_xtal.model import get_model_fn_t
from jax_xtal.train_utils import restore_checkpoint, Normalizer
from jax_xtal.config import load_config, Config


def main(config: Config, ckpt_path: str, structures_dir: str, output: str):
    # Prepare test dataset
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.dmax, num_filters=config.num_bond_features
    )
    dataset, list_ids = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=structures_dir,
        targets_csv_path="",
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        is_training=False,
        seed=config.seed,
        n_jobs=config.n_jobs,
    )

    # Define model
    model_fn_t = get_model_fn_t(
        num_atom_features=config.num_atom_features,
        num_convs=config.num_convs,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_features=config.num_hidden_features,
        max_num_neighbors=config.max_num_neighbors,
    )
    model = hk.without_apply_rng(model_fn_t)

    # Load checkpoint
    params, state, normalizer = restore_checkpoint(ckpt_path)

    @jax.jit
    def predict_one_step(batch: Batch) -> jnp.ndarray:
        predictions, _ = model.apply(params, state, batch, is_training=False)
        predictions = jnp.squeeze(predictions, axis=-1)
        return predictions

    # Prediction
    batch_size = config.batch_size_prediction
    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    predictions = []
    for i in range(steps_per_epoch):
        batch = collate_pool(dataset, False)  # train=False
        preds = predict_one_step(batch)
        predictions.append(preds)
    predictions = jnp.concatenate(predictions)  # (len(dataset), )

    # denormalize predictions
    denormed_preds = normalizer.denormalize(predictions)

    with open(args.output, "w") as f:
        for i, idx in enumerate(list_ids):
            f.write(f"{idx},{denormed_preds[i]}\n")


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

    main(config, args.checkpoint, args.structures_dir, args.output)
