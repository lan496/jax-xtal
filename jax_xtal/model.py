from functools import partial
from typing import Any, Mapping

import haiku as hk
from haiku import BatchNorm, Linear
import jax
import jax.numpy as jnp


Batch = Mapping[str, jnp.ndarray]


class CGConv(hk.Module):
    """
    Convolutional layer in Eq. (5)
    """

    def __init__(self, num_atom_features: int, max_num_neighbors: int, name: str = None):
        super().__init__(name=name)
        self._num_atom_features = num_atom_features
        self._max_num_neighbors = max_num_neighbors

        self._cgweight = Linear(2 * self._num_atom_features, name="cgweight")

    def __call__(
        self,
        neighbor_indices: jnp.ndarray,
        atom_features: jnp.ndarray,
        bond_features: jnp.ndarray,
        train: bool = True,
    ):
        """
        Let the total number of atoms in the batch be N,
        neighbor_indices: (N, max_num_neighbors)
        atom_features: (N, num_atom_features)
        bond_features: (N, max_num_neighbors, num_bond_features)
        """
        num_atoms_batch = neighbor_indices.shape[0]

        # (N, max_num_neighbors, num_atom_features)
        atom_neighbor_features = atom_features[neighbor_indices, :]
        total_neighbor_features = jnp.concatenate(
            [
                jnp.broadcast_to(
                    atom_features[:, None, :],
                    (num_atoms_batch, self._max_num_neighbors, self._num_atom_features),
                ),
                atom_neighbor_features,
                bond_features,
            ],
            axis=2,
        )
        total_gated_features = self._cgweight(total_neighbor_features)
        total_gated_features = BatchNorm(
            create_scale=True, create_offset=True, decay_rate=1.0, name="bn_1"
        )(
            total_gated_features.reshape(-1, 2 * self._num_atom_features),
            is_training=train,
            test_local_stats=True,
        ).reshape(
            num_atoms_batch, self._max_num_neighbors, 2 * self._num_atom_features
        )  # TODO: why reshape here?

        neighbor_filter, neighbor_core = jnp.split(total_gated_features, 2, axis=2)
        neighbor_filter = jax.nn.sigmoid(neighbor_filter)
        neighbor_core = jax.nn.softplus(neighbor_core)

        neighbor_summed = jnp.sum(
            neighbor_filter * neighbor_core, axis=1
        )  # (N, num_atom_features)
        neighbor_summed = BatchNorm(
            create_scale=True, create_offset=True, decay_rate=1.0, name="bn_2"
        )(neighbor_summed, is_training=train, test_local_stats=True)
        out = jax.nn.softplus(atom_features + neighbor_summed)  # TODO: defer from Eq. (5) ?
        return out


class CGPooling(hk.Module):
    """
    average-pooling over each crystal in a batch
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, atom_features: jnp.ndarray, atom_indices: jnp.ndarray):
        """
        Parameters
        ----------
        atom_features: (N, num_atom_features)
        atom_indices: list with batch_size length

        Returns
        -------
        averaged_features: (batch_size, 1)
        """
        averaged_features = [
            jnp.mean(atom_features[indices, :], axis=0, keepdims=True) for indices in atom_indices
        ]
        averaged_features = jnp.concatenate(averaged_features, axis=0)
        return averaged_features


class CGCNN(hk.Module):
    """
    Crystal Graph Convolutional Neural Network
    """

    def __init__(
        self,
        num_atom_features: int,
        num_convs: int,
        num_hidden_layers: int,
        num_hidden_features: int,
        max_num_neighbors: int,
        name=None,
    ):
        super().__init__(name=name)
        self._num_atom_features = num_atom_features
        self._num_convs = num_convs
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_features = num_hidden_features
        self._max_num_neighbors = max_num_neighbors

        self._embedding = Linear(self._num_atom_features, name="embedding")
        self._cgconvs = [
            CGConv(self._num_atom_features, self._max_num_neighbors, name=f"cgconv_{i}")
            for i in range(self._num_convs)
        ]
        self._cgpooling = CGPooling(name="cgpooling")
        self._fcs = [
            Linear(self._num_hidden_features, name=f"fc_{i}")
            for i in range(self._num_hidden_layers)
        ]
        self._fc_last = Linear(1, name="fc_last")

    def __call__(
        self,
        neighbor_indices: jnp.ndarray,
        atom_features: jnp.ndarray,
        bond_features: jnp.ndarray,
        atom_indices: jnp.ndarray,
        train: bool = True,
    ):
        atom_features = self._embedding(atom_features)
        for i in range(self._num_convs):
            atom_features = self._cgconvs[i](neighbor_indices, atom_features, bond_features, train)

        crystal_features = self._cgpooling(atom_features, atom_indices)
        crystal_features = jax.nn.softplus(crystal_features)
        for i in range(self._num_hidden_layers):
            crystal_features = self._fcs[i](crystal_features)
            crystal_features = jax.nn.softplus(crystal_features)
        out = self._fc_last(crystal_features)
        return out


def get_model_fn_t(
    num_atom_features: int,
    num_convs: int,
    num_hidden_layers: int,
    num_hidden_features: int,
    max_num_neighbors: int,
):
    def model_fn(batch: Batch, train: bool) -> jnp.ndarray:
        model = CGCNN(
            num_atom_features, num_convs, num_hidden_layers, num_hidden_features, max_num_neighbors
        )
        neighbor_indices = batch["neighbor_indices"]
        atom_features = batch["atom_features"]
        bond_features = batch["bond_features"]
        atom_indices = batch["atom_indices"]
        return model(neighbor_indices, atom_features, bond_features, atom_indices, train)

    return hk.transform_with_state(model_fn)
