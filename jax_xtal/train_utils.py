import os
from typing import Any

import jax
import jax.numpy as jnp
import flax
from flax import optim
from flax import serialization
from tqdm import tqdm

from jax_xtal.model import CGCNN
from jax_xtal.data import collate_pool


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
            jax.device_put(batch["neighbor_indices"]),
            jax.device_put(batch["atom_features"]),
            jax.device_put(batch["bond_features"]),
            jax.device_put(batch["atom_indices"]),
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


def predict_one_step(apply_fn, batch, state: TrainState):
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
    return predictions


def train_one_epoch(train_step_fn, state: TrainState, train_dataset, batch_size, epoch, rng):
    # shuffle training data
    train_size = len(train_dataset)
    steps_per_epoch = train_size // batch_size
    perms = jax.random.permutation(rng, train_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    train_metrics = []
    for perm in tqdm(perms):
        batch = collate_pool([train_dataset[idx] for idx in perm])

        state, metrics = train_step_fn(batch=batch, state=state)
        train_metrics.append(metrics)
    train_metrics = jax.device_get(train_metrics)
    train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)[0]
    # TODO: add learning_rate to train_summary
    return state, train_summary


def eval_model(val_step_fn, state: TrainState, val_dataset):
    batch = collate_pool(val_dataset)
    eval_summary = val_step_fn(batch=batch, state=state)
    eval_summary = jax.device_get(eval_summary)
    return eval_summary


def predict_dataset(test_step_fn, state: TrainState, dataset):
    batch = collate_pool(dataset)
    predictions = test_step_fn(batch=batch, state=state)  # (len(dataset), 1)
    return predictions


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


def save_checkpoint(state: TrainState, workdir: str):
    step = state.step
    ckpt_path = os.path.join(workdir, f"checkpoint_{step}.flax")
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(state))


def restore_checkpoint(ckpt_path: str, state: TrainState) -> TrainState:
    with open(ckpt_path, "rb") as f:
        restored_state = serialization.from_bytes(state, f.read())
    return restored_state
