from logging import getLogger
from time import time
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from joblib import Parallel, delayed

from jax_xtal.config import Config
from jax_xtal.model import get_model_fn_t
from jax_xtal.data import collate_pool, Batch
from jax_xtal.train_utils import (
    Metrics,
    Normalizer,
    mean_squared_error,
    mean_absolute_error,
    get_metrics_mean,
)


logger = getLogger("cgcnn")


OptState = Any


def train_and_eval(
    config: Config,
    num_initial_atom_features: int,
    train_dataset,
    val_dataset,
    normalizer: Normalizer,
    rng_seq,
):
    """
    Returns
    -------
    eval_model
    test_summary: Metrics
    params: hk.Params
    state: hk.State
    """
    # Define model and optimizer
    model_fn_t = get_model_fn_t(
        num_initial_atom_features=num_initial_atom_features,
        num_atom_features=config.num_atom_features,
        num_bond_features=config.num_bond_features,
        num_convs=config.num_convs,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_features=config.num_hidden_features,
        max_num_neighbors=config.max_num_neighbors,
        batch_size=config.batch_size,
    )
    model = hk.without_apply_rng(model_fn_t)
    optimizer = optax.sgd(config.learning_rate)

    # Initialize model and optimizer
    logger.info("Initialize model and optimizer")
    if len(train_dataset) < config.batch_size:
        raise ValueError(
            f"Prepare train dataset ({len(train_dataset)}) more than batch_size ({config.batch_size})"
        )
    if len(val_dataset) < config.batch_size:
        raise ValueError(
            f"Prepare validation dataset ({len(val_dataset)}) more than batch_size ({config.batch_size})"
        )
    init_batch = collate_pool(train_dataset[: config.batch_size], have_targets=True)
    params, state = model.init(next(rng_seq), init_batch, is_training=True)
    opt_state = optimizer.init(params)
    num_params = hk.data_structures.tree_size(params)
    byte_size = hk.data_structures.tree_bytes(params)  # size with f32
    logger.info(f"{num_params} params, size={byte_size/1e6:.2f}MB")

    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    # Loss function
    @jax.jit
    def loss_fn(
        params: hk.Params, state: hk.State, batch: Batch
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
        predictions, state = model.apply(params, state, batch, True)  # is_training=True

        mse = mean_squared_error(predictions, batch["target"])
        return mse, (predictions, state)

    # Evaluate metrics
    @jax.jit
    def compute_metrics(predictions: jnp.ndarray, batch: Batch) -> Metrics:
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
    ) -> Tuple[hk.Params, hk.State, OptState, Metrics]:
        (_loss, (predictions, state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, batch
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        metrics = compute_metrics(predictions, batch)
        return params, state, opt_state, metrics

    @jax.jit
    def eval_one_step(
        params: hk.Params, state: hk.State, batch: Batch
    ) -> Tuple[jnp.ndarray, Metrics]:
        predictions, state = model.apply(params, state, batch, False)  # is_training=False
        metrics = compute_metrics(predictions, batch)
        return predictions, metrics

    def train_one_epoch(
        params: hk.Params, state: hk.State, opt_state: OptState
    ) -> Tuple[hk.Params, hk.State, OptState, Metrics]:
        # shuffle training data
        batch_size = config.batch_size
        train_size = len(train_dataset)
        steps_per_epoch = train_size // batch_size
        perms = jax.random.permutation(next(rng_seq), train_size)
        perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        # Prepare batches
        batches = Parallel(n_jobs=config.n_jobs)(
            delayed(collate_pool)([train_dataset[idx] for idx in perm], True) for perm in perms
        )

        train_metrics = []

        lap = time()
        for i, batch in enumerate(batches):
            params, state, opt_state, metrics = update(params, state, opt_state, batch)
            train_metrics.append(metrics)

            time_step = time() - lap
            lap = time()
            if (i + 1) % config.print_freq == 0:
                logger.debug(
                    f"Epoch [{epoch}][{i + 1}/{steps_per_epoch}]: Loss={metrics['mse']:.4f}, MAE={normalizer.denormalize_MAE(metrics['mae']):.4f}, Time={time_step:.2f} sec/step"
                )

        train_summary = get_metrics_mean(train_metrics)
        return params, state, opt_state, train_summary

    def eval_model(params: hk.Params, state: hk.State, dataset) -> Metrics:
        batch_size = config.batch_size
        eval_metrics = []
        steps_per_epoch = len(dataset) // batch_size

        # Prepare batches
        # train=True to get batch['target']
        batches = Parallel(n_jobs=config.n_jobs)(
            [
                delayed(collate_pool)(
                    [dataset[ii] for ii in range(i * batch_size, (i + 1) * batch_size)], True,
                )
                for i in range(steps_per_epoch)
            ]
        )

        for i, batch in enumerate(batches):
            _predictions, metrics = eval_one_step(params, state, batch)
            eval_metrics.append(metrics)
        eval_summary = get_metrics_mean(eval_metrics)
        return eval_summary

    # Train/eval loop
    logger.info("Start training")
    for epoch in range(1, config.num_epochs + 1):
        params, state, opt_state, train_summary = train_one_epoch(params, state, opt_state)
        train_loss = train_summary["mse"]
        train_mae = normalizer.denormalize_MAE(train_summary["mae"])
        logger.info(
            "[Train] epoch: %2d, loss: %.4f, MAE: %.4f eV/atom" % (epoch, train_loss, train_mae)
        )
        # jax.profiler.save_device_memory_profile(f"memory.{epoch}.prof")

    val_summary = eval_model(params, state, val_dataset)
    val_loss = val_summary["mse"]
    val_mae = normalizer.denormalize_MAE(val_summary["mae"])
    logger.info("[Eval] epoch: %2d, loss: %.4f, MAE: %.4f eV/atom" % (epoch, val_loss, val_mae))

    return eval_model, params, state, val_summary
