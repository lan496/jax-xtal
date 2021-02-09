import os
from typing import Any

import jax
import jax.numpy as jnp
import flax
from flax import optim
from flax import serialization

from jax_xtal.model import CGCNN
from jax_xtal.data import get_dataloaders


@flax.struct.dataclass
class TrainState:
    step: int
    optimizer: optim.Optimizer
    model_state: Any


def create_optimizer(params, learning_rate) -> optim.Optimizer:
    """
    Create an Adam optimizer
    """
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


def mean_squared_error(predictions, targets):
    return jnp.mean(jnp.square(predictions - targets), axis=None)


def mean_absolute_error(predictions, targets):
    return jnp.mean(jnp.abs(predictions - targets), axis=None)


def compute_metrics(predictions, targets):
    mse = mean_squared_error(predictions, targets)
    mae = mean_absolute_error(predictions, targets)
    metrics = {
        "loss": mse,
        "mae": mae,
    }
    return metrics


def multi_step_lr(step: int, steps_per_epoch, learning_rate: float, milestones: int, gamma=0.1):
    decayed = learning_rate * (gamma ** (step // (milestones * steps_per_epoch)))
    return decayed


def train_one_step(apply_fn, batch, state: TrainState, learning_rate_fn, l2_reg):
    """
    Parameters
    ----------
    apply_fn: nn.Module.apply
    batch: batch samples
    state: store training state
    learning_rate_fn: takes step and return decayed learning rate
    l2_reg: weight
    """

    def loss_fn(params):
        variables = {"params": params, **state.model_state}
        predictions, new_model_state = apply_fn(
            variables,
            batch["neighbor_indices"],
            batch["atom_features"],
            batch["bond_features"],
            batch["atom_indices"],
            mutable=["batch_stats"],  # for BatchNorm
        )

        loss = mean_squared_error(predictions, batch["target"])
        weight_penalty_params = jax.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])
        weight_penalty = l2_reg * weight_l2
        loss = loss + weight_penalty

        return loss, (new_model_state, predictions)

    step = state.step
    learning_rate = learning_rate_fn(step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    optimizer = state.optimizer
    (_, (new_model_state, predictions)), grads = grad_fn(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grads, learning_rate=learning_rate)
    metrics = compute_metrics(predictions, batch["target"])
    metrics["learning_rate"] = learning_rate

    new_state = state.replace(step=step + 1, optimizer=new_optimizer, model_state=new_model_state)

    return new_state, metrics


def eval_one_step(apply_fn, batch, state: TrainState):
    params = state.optimizer.target
    variables = {"params": params, **state.model_state}
    predictions = apply_fn(
        variables,
        batch["neighbor_indices"],
        batch["atom_features"],
        batch["bond_features"],
        batch["atom_indices"],
        train=False,
        mutable=False,
    )
    return compute_metrics(predictions, batch["target"])


def train_one_epoch(train_step_fn, state: TrainState, train_loader, epoch):
    train_metrics = []
    for batch in train_loader:
        batch.pop("id")  # pop unused entry
        state, metrics = train_step_fn(batch=batch, state=state)
        train_metrics.append(metrics)
    train_metrics = jax.device_get(train_metrics)
    train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)[0]
    # TODO: add learning_rate to train_summary
    return state, train_summary


def eval_model(val_step_fn, state: TrainState, val_loader):
    eval_metrics = []
    for batch in val_loader:
        batch.pop("id")  # pop unused entry
        metrics = val_step_fn(batch=batch, state=state)
        eval_metrics.append(metrics)
    eval_metrics = jax.device_get(eval_metrics)
    eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)[0]
    return eval_summary


def initialize_model(key, model, max_num_neighbors, num_initial_atom_features, num_bond_features):
    # TODO: jit model.init, https://github.com/google/flax/blob/master/examples/imagenet/train.py
    dtype = jnp.float32
    neighbor_indices = jnp.zeros((1, max_num_neighbors), dtype=int)
    atom_features = jnp.zeros((1, num_initial_atom_features), dtype=dtype)
    bond_features = jnp.zeros((1, max_num_neighbors, num_bond_features), dtype=dtype)
    atom_indices = [jnp.array([0])]

    variables = model.init(
        key, neighbor_indices, atom_features, bond_features, atom_indices, train=False
    )
    model_state, params = variables.pop("params")
    return params, model_state


def create_train_state(
    rng, model, max_num_neighbors, num_initial_atom_features, num_bond_features, learning_rate
) -> TrainState:
    rng, rng_model = jax.random.split(rng)
    params, model_state = initialize_model(
        key=rng_model,
        model=model,
        max_num_neighbors=max_num_neighbors,
        num_initial_atom_features=num_initial_atom_features,
        num_bond_features=num_bond_features,
    )
    optimizer = create_optimizer(params, learning_rate=learning_rate)
    state = TrainState(step=0, optimizer=optimizer, model_state=model_state)
    return state


def save_checkpoint(optimizer: optim.Optimizer, workdir: str, step: int):
    ckpt_path = os.path.join(workdir, f"checkpoint_{step}.flax")
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(optimizer.target))


# def restore_checkpoint(ckpt_path: str, )
