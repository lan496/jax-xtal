import os
from functools import partial
import argparse
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from typing import Mapping, Any, Tuple
from time import time

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from jax_xtal.data import (
    AtomFeaturizer,
    BondFeaturizer,
    create_dataset,
    split_dataset,
    collate_pool,
    Batch,
)
from jax_xtal.model import get_model_fn_t
from jax_xtal.train_utils import (
    mean_squared_error,
    mean_absolute_error,
    Normalizer,
    Metrics,
    get_metrics_mean,
    save_checkpoint,
)
from jax_xtal.config import load_config, Config

OptState = Any


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


def main(config: Config):
    # logger
    os.makedirs(config.log_dir, exist_ok=True)
    log_basename = os.path.basename(args.config)
    log_path = os.path.join(config.log_dir, f"{log_basename}.log")
    logger = get_module_logger("cgcnn", log_path)

    seed = config.seed
    rng = jax.random.PRNGKey(seed)

    # Prepare dataset
    logger.info("Load dataset")
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

    # Normalize target value
    num_norm_samples = min(500, len(train_dataset))
    normalizer = Normalizer.from_targets(
        [train_dataset[idx]["target"] for idx in range(num_norm_samples)]
    )
    logger.info(f"Normalize target value: Mean={normalizer.mean:.3f} Std={normalizer.std:.3f}")
    train_dataset = normalizer.normalize_dataset(train_dataset)
    val_dataset = normalizer.normalize_dataset(val_dataset)
    test_dataset = normalizer.normalize_dataset(test_dataset)

    # Define model and optimizer
    model_fn_t = get_model_fn_t(
        num_atom_features=config.num_atom_features,
        num_convs=config.num_convs,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_features=config.num_hidden_features,
        max_num_neighbors=config.max_num_neighbors,
    )
    model = hk.without_apply_rng(model_fn_t)
    optimizer = optax.sgd(config.learning_rate)

    # Initialize model and optimizer
    logger.info("Initialize model and optimizer")
    init_batch = collate_pool(train_dataset[:2], False)
    rng, init_rng = jax.random.split(rng)
    params, state = model.init(init_rng, init_batch, train=False)
    opt_state = optimizer.init(params)
    num_params = hk.data_structures.tree_size(params)
    byte_size = hk.data_structures.tree_bytes(params)  # size with f32
    logger.info(f"{num_params} params, size={byte_size/1e6:.2f}MB")

    # Loss function
    @jax.jit
    def loss_fn(params: hk.Params, state: hk.State, batch: Batch) -> Tuple[jnp.ndarray, hk.State]:
        predictions, state = model.apply(params, state, batch, True)  # train=True
        predictions = jnp.squeeze(predictions, axis=-1)

        mse = mean_squared_error(predictions, batch["target"])
        weight_l2 = 0.5 * sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(params)])
        return mse + config.l2_reg * weight_l2, state

    # Evaluate metrics
    @jax.jit
    def compute_metrics(params: hk.Params, state: hk.Params, batch: Batch) -> Metrics:
        predictions, _ = model.apply(params, state, batch, train=False)
        predictions = jnp.squeeze(predictions, axis=-1)
        mse = mean_squared_error(predictions, batch["target"])
        mae = mean_absolute_error(predictions, batch["target"])
        metrics = {
            "mse": mse,
            "mae": mae,
        }
        return metrics

    # Update params and state
    @jax.jit
    def update(
        params: hk.Params, state: hk.State, opt_state: OptState, batch: Batch
    ) -> Tuple[hk.Params, hk.State, OptState, jnp.float32]:
        (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, state, opt_state, loss

    def train_one_epoch(
        params: hk.Params, state: hk.State, opt_state: OptState, rng_train
    ) -> Tuple[hk.Params, hk.State, OptState, Metrics]:
        # shuffle training data
        batch_size = config.batch_size
        train_size = len(train_dataset)
        steps_per_epoch = train_size // batch_size
        perms = jax.random.permutation(rng_train, train_size)
        perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        train_metrics = []

        lap = time()
        for i, perm in enumerate(perms):
            batch = collate_pool([train_dataset[idx] for idx in perm], True)  # train=True
            params, state, opt_state, _ = update(params, state, opt_state, batch)
            metrics = compute_metrics(params, state, batch)
            train_metrics.append(metrics)

            time_step = time() - lap
            lap = time()
            if (i + 1) % config.print_freq == 0:
                logger.info(
                    f"Epoch [{epoch}][{i + 1}/{steps_per_epoch}]: Loss={metrics['mse']:.4f}, Time={time_step:.2f} sec/step"
                )

        train_summary = get_metrics_mean(train_metrics)
        return params, state, opt_state, train_summary

    def eval_model(params: hk.Params, state: hk.State, dataset) -> Metrics:
        batch_size = config.batch_size_prediction
        eval_metrics = []
        steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
        for i in range(steps_per_epoch):
            batch = collate_pool(
                dataset[i * batch_size : (i + 1) * batch_size], True
            )  # train=True to get batch['target']
            metrics = compute_metrics(params, state, batch)
            eval_metrics.append(metrics)
        eval_summary = get_metrics_mean(eval_metrics)
        return eval_summary

    # Train/eval loop
    logger.info("Start training")
    for epoch in range(1, config.num_epochs + 1):
        rng, rng_train = jax.random.split(rng)
        params, state, opt_state, train_summary = train_one_epoch(
            params, state, opt_state, rng_train
        )
        train_loss = train_summary["mse"]
        train_mae = normalizer.denormalize_MAE(train_summary["mae"])
        logger.info(
            "[Train] epoch: %2d, loss: %.2f, MAE: %.2f eV/atom" % (epoch, train_loss, train_mae)
        )

        val_summary = eval_model(params, state, val_dataset)
        val_loss = val_summary["mse"]
        val_mae = normalizer.denormalize_MAE(val_summary["mae"])
        logger.info(
            "[Eval] epoch: %2d, loss: %.2f, MAE: %.2f eV/atom" % (epoch, val_loss, val_mae)
        )
        jax.profiler.save_device_memory_profile(f"memory{epoch}.prof")

    test_summary = eval_model(params, state, test_dataset)
    test_loss = test_summary["mse"]
    test_mae = normalizer.denormalize_MAE(test_summary["mae"])
    logger.info("[Test] loss: %.2f, MAE: %.2f eV/atom" % (test_loss, test_mae))

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
