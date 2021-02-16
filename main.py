import os
from functools import partial
import argparse
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import jax.profiler
import jax
from jax.random import PRNGKey
from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    split_dataset,
)
from jax_xtal.model import CGCNN
from jax_xtal.train_utils import (
    create_train_state,
    train_one_step,
    eval_one_step,
    train_one_epoch,
    eval_model,
    multi_step_lr,
    save_checkpoint,
    Normalizer,
)
from jax_xtal.config import load_config


def get_module_logger(modname, log_path):
    logger = getLogger(modname)

    log_fmt = Formatter(
        "%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s "
    )
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_path, "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path for json config")
    args = parser.parse_args()

    config = load_config(args.config)

    # logger
    os.makedirs(config.log_dir, exist_ok=True)
    log_basename = os.path.basename(args.config)
    log_path = os.path.join(config.log_dir, f"{log_basename}.log")
    logger = get_module_logger("cgcnn", log_path)

    seed = config.seed
    rng = PRNGKey(seed)

    # prepare dataset
    logger.info("Load dataset")
    root_dir = os.path.dirname(__file__)
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.dmax, num_filters=config.num_bond_features
    )
    dataset, _ = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=config.structures_dir,
        targets_csv_path=config.targets_csv_path,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
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

    # normalize target value
    num_norm_samples = min(500, len(train_dataset))
    normalizer = Normalizer.from_targets(
        [train_dataset[idx]["target"] for idx in range(num_norm_samples)]
    )
    train_dataset = normalizer.normalize_dataset(train_dataset)
    val_dataset = normalizer.normalize_dataset(val_dataset)
    test_dataset = normalizer.normalize_dataset(test_dataset)

    # initialize model
    logger.info("Initialize model")
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
        normalizer=normalizer,
    )

    # MultiStepLR scheduelr for learning rate
    train_size = len(train_dataset)
    steps_per_epoch = train_size // config.batch_size
    learning_rate_fn = partial(
        multi_step_lr,
        steps_per_epoch=steps_per_epoch,
        learning_rate=config.learning_rate,
        milestones=config.milestones,
    )

    logger.info("Start training")
    batch_size = config.batch_size
    epoch_metrics = []
    for epoch in range(1, config.num_epochs + 1):
        rng, rng_train = jax.random.split(rng)
        state, train_summary = train_one_epoch(
            apply_fn=model.apply,
            state=state,
            train_dataset=train_dataset,
            batch_size=batch_size,
            learning_rate_fn=learning_rate_fn,
            l2_reg=config.l2_reg,
            epoch=epoch,
            rng=rng_train,
            print_freq=config.print_freq,
        )
        train_loss = train_summary["loss"]
        train_mae = normalizer.denormalize_MAE(train_summary["mae"])
        logger.info(
            "Training - epoch: %2d, loss: %.2f, MAE: %.2f eV/atom" % (epoch, train_loss, train_mae)
        )
        val_summary = eval_model(
            apply_fn=model.apply, state=state, val_dataset=val_dataset, batch_size=batch_size
        )
        val_loss = val_summary["loss"]
        val_mae = normalizer.denormalize_MAE(val_summary["mae"])
        logger.info(
            "Testing  - epoch: %2d, loss: %.2f, MAE: %.2f eV/atom" % (epoch, val_loss, val_mae)
        )
        jax.profiler.save_device_memory_profile(f"memory_{epoch}.prof")

    test_summary = eval_model(
        apply_fn=model.apply, state=state, val_dataset=test_dataset, batch_size=batch_size,
    )
    test_loss = test_summary["loss"]
    test_mae = normalizer.denormalize_MAE(test_summary["mae"])
    logger.info("Testing  -            loss: %.2f, MAE: %.2f eV/atom" % (test_loss, test_mae))

    logger.info("Save checkpoint")
    workdir = config.checkpoint_dir
    os.makedirs(workdir, exist_ok=True)
    save_checkpoint(state, workdir)
