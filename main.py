import os
import argparse

import haiku as hk

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    split_dataset,
)
from jax_xtal.train_utils import (
    Normalizer,
    save_checkpoint,
    seed_everything,
    get_module_logger,
)
from jax_xtal.config import load_config, Config
from jax_xtal.train import train_and_eval


def main(config: Config):
    # logger
    os.makedirs(config.log_dir, exist_ok=True)
    log_basename = os.path.basename(args.config)
    log_path = os.path.join(config.log_dir, f"{log_basename}.log")
    logger = get_module_logger("cgcnn", log_path)

    seed = config.seed
    seed_everything(seed)

    # Prepare dataset
    logger.info("Load dataset")
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.cutoff, num_filters=config.num_bond_features
    )
    dataset, _ids = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=config.structures_dir,
        targets_csv_path=config.targets_csv_path,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        is_training=True,
        seed=seed,
        n_jobs=config.n_jobs,
    )

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    del dataset

    # Normalize target value
    num_norm_samples = min(500, len(train_dataset))
    normalizer = Normalizer.from_targets(
        [train_dataset[idx]["target"] for idx in range(num_norm_samples)]
    )
    logger.info(f"Normalize target value: Mean={normalizer.mean:.3f} Std={normalizer.std:.3f}")
    train_dataset = normalizer.normalize_dataset(train_dataset)
    val_dataset = normalizer.normalize_dataset(val_dataset)
    test_dataset = normalizer.normalize_dataset(test_dataset)

    rng_seq = hk.PRNGSequence(seed)
    eval_model_fn, params, state, val_summary = train_and_eval(
        config,
        atom_featurizer.num_initial_atom_features,
        train_dataset,
        val_dataset,
        normalizer,
        rng_seq,
    )

    test_summary = eval_model_fn(params, state, test_dataset)
    test_loss = test_summary["mse"]
    test_mae = normalizer.denormalize_MAE(test_summary["mae"])
    logger.info("[Test] loss: %.4f, MAE: %.4f eV/atom" % (test_loss, test_mae))

    ##

    logger.info("Save checkpoint")
    workdir = config.checkpoint_dir
    os.makedirs(workdir, exist_ok=True)
    save_checkpoint(params, state, normalizer, workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path for json config")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
