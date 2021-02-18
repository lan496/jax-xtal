import os
from typing import Any
from time import time
from logging import getLogger
from functools import partial
from typing import Mapping, List, Tuple
import pickle

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp


Metrics = Mapping[str, jnp.ndarray]


logger = getLogger("cgcnn")


class Normalizer:
    """
    normalize target value
    """

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def normalize(self, target):
        return (target - self._mean) / self._std

    def normalize_dataset(self, dataset):
        for idx in range(len(dataset)):
            dataset[idx]["target"] = self.normalize(dataset[idx]["target"])
        return dataset

    def denormalize(self, normed):
        return self._std * normed + self._mean

    def denormalize_MAE(self, normed_mae):
        return self._std * normed_mae

    @classmethod
    def from_targets(cls, targets):
        mean = np.mean(targets)
        std = np.std(targets)
        return cls(mean, std)


def mean_squared_error(predictions, targets):
    return jnp.mean(jnp.square(predictions - targets), axis=None)


def mean_absolute_error(predictions, targets):
    return jnp.mean(jnp.abs(predictions - targets), axis=None)


def save_checkpoint(params: hk.Params, state: hk.State, normalizer: Normalizer, workdir: str):
    ckpt_path = os.path.join(workdir, f"checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump((params, state, normalizer.mean, normalizer.std), f)


def restore_checkpoint(ckpt_path: str) -> Tuple[hk.Params, hk.State, Normalizer]:
    with open(ckpt_path, "rb") as f:
        params, state, mean, std = pickle.load(f)
    normalizer = Normalizer(mean, std)
    return params, state, normalizer


def get_metrics_mean(list_metrics: List[Metrics]) -> Metrics:
    summary = jax.device_put(list_metrics)
    summary = jax.tree_map(lambda x: x.mean(), list_metrics)[0]
    return summary
