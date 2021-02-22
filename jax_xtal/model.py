from functools import partial
from typing import Any, Mapping, List
import math

import haiku as hk
from haiku import BatchNorm, Linear
import jax
import jax.numpy as jnp


Batch = Mapping[str, jnp.ndarray]


class CGConv(hk.Module):
    """
    Convolutional layer in Eq. (5)
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        max_num_neighbors: int,
        name: str = None,
    ):
        super().__init__(name=name)
        self._num_atom_features = num_atom_features
        self._num_bond_features = num_bond_features
        self._max_num_neighbors = max_num_neighbors

        in_features_cgweight = 2 * self._num_atom_features + self._num_bond_features
        stdv_cgweight = 1.0 / math.sqrt(in_features_cgweight)
        w_init_cgweight = hk.initializers.RandomUniform(-stdv_cgweight, stdv_cgweight)
        b_init_cgweight = hk.initializers.RandomUniform(-stdv_cgweight, stdv_cgweight)
        self._cgweight = Linear(
            2 * self._num_atom_features,
            w_init=w_init_cgweight,
            b_init=b_init_cgweight,
            name="cgweight",
        )
        self._bn1 = BatchNorm(True, True, 0.9, name="bn_1")  # momemtum=0.1
        self._bn2 = BatchNorm(True, True, 0.9, name="bn_2")  # momemtum=0.1

    def __call__(
        self,
        neighbor_indices: jnp.ndarray,
        atom_features: jnp.ndarray,
        bond_features: jnp.ndarray,
        is_training: bool,
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
        total_gated_features = self._bn1(
            total_gated_features.reshape(-1, 2 * self._num_atom_features),
            is_training=is_training,
        ).reshape(
            num_atoms_batch, self._max_num_neighbors, 2 * self._num_atom_features
        )  # TODO: why reshape here?

        neighbor_filter, neighbor_core = jnp.split(total_gated_features, 2, axis=2)
        neighbor_filter = jax.nn.sigmoid(neighbor_filter)
        neighbor_core = jax.nn.softplus(neighbor_core)

        neighbor_summed = jnp.sum(
            neighbor_filter * neighbor_core, axis=1
        )  # (N, num_atom_features)
        neighbor_summed = self._bn2(neighbor_summed, is_training=is_training)
        out = jax.nn.softplus(atom_features + neighbor_summed)  # TODO: defer from Eq. (5) ?
        return out


class CGPooling(hk.Module):
    """
    average-pooling over each crystal in a batch
    """

    def __init__(self, batch_size: int, name=None):
        super().__init__(name=name)
        self._batch_size = batch_size

    def __call__(self, atom_features: jnp.ndarray, segment_ids: jnp.ndarray, num_atoms: jnp.ndarray):
        """
        Parameters
        ----------
        atom_features: (N, num_atom_features)
        segment_ids: (N, )
        num_atoms: (batch_size, )

        Returns
        -------
        averaged_features: (batch_size, 1)
        """
        averaged_features = jnp.mean(atom_features, axis=1)  # (N, )
        # sum over each graph
        averaged_features = jax.ops.segment_sum(
            averaged_features,
            segment_ids,
            num_segments=self._batch_size,
            indices_are_sorted=True,
            unique_indices=False
        )  # (batch_size, )
        averaged_features = averaged_features / num_atoms
        averaged_features = jnp.expand_dims(
            averaged_features,
            axis=1
        )  # (batch_size, 1)
        return averaged_features


class CGCNN(hk.Module):
    """
    Crystal Graph Convolutional Neural Network
    """

    def __init__(
        self,
        num_initial_atom_features: int,
        num_atom_features: int,
        num_bond_features: int,
        num_convs: int,
        num_hidden_layers: int,
        num_hidden_features: int,
        max_num_neighbors: int,
        batch_size: int,
        name=None,
    ):
        super().__init__(name=name)
        self._num_initial_atom_features = num_initial_atom_features
        self._num_atom_features = num_atom_features
        self._num_bond_features = num_bond_features
        self._num_convs = num_convs
        self._num_hidden_layers = num_hidden_layers
        self._num_hidden_features = num_hidden_features
        self._max_num_neighbors = max_num_neighbors
        self._batch_size = batch_size

        stdv_embedding = 1.0 / math.sqrt(self._num_initial_atom_features)
        w_init_embedding = hk.initializers.RandomUniform(-stdv_embedding, stdv_embedding)
        b_init_embedding = hk.initializers.RandomUniform(-stdv_embedding, stdv_embedding)
        self._embedding = Linear(
            self._num_atom_features,
            w_init=w_init_embedding,
            b_init=b_init_embedding,
            name="embedding",
        )
        self._cgconvs = [
            CGConv(
                num_atom_features=self._num_atom_features,
                num_bond_features=self._num_bond_features,
                max_num_neighbors=self._max_num_neighbors,
                name=f"cgconv_{i}",
            )
            for i in range(self._num_convs)
        ]
        self._cgpooling = CGPooling(batch_size=self._batch_size, name="cgpooling")

        stdv_fc_first = 1.0 / math.sqrt(self._num_atom_features)
        w_init_fc_first = hk.initializers.RandomUniform(-stdv_fc_first, stdv_fc_first)
        b_init_fc_first = hk.initializers.RandomUniform(-stdv_fc_first, stdv_fc_first)
        stdv_fc_second = 1.0 / math.sqrt(self._num_hidden_features)
        w_init_fc_second = hk.initializers.RandomUniform(-stdv_fc_second, stdv_fc_second)
        b_init_fc_second = hk.initializers.RandomUniform(-stdv_fc_second, stdv_fc_second)
        self._fcs = [
            Linear(
                self._num_hidden_features,
                w_init=w_init_fc_first,
                b_init=b_init_fc_first,
                name="fc_0",
            )
        ]
        for i in range(1, self._num_hidden_layers):
            self._fcs.append(
                Linear(
                    self._num_hidden_features,
                    w_init=w_init_fc_second,
                    b_init=b_init_fc_second,
                    name=f"fc_{i}",
                )
            )
        self._fc_last = Linear(1, w_init=w_init_fc_second, b_init=b_init_fc_second, name="fc_last")

    def __call__(
        self,
        neighbor_indices: jnp.ndarray,
        atom_features: jnp.ndarray,
        bond_features: jnp.ndarray,
        num_atoms: jnp.ndarray,
        segment_ids: jnp.ndarray,
        is_training: bool,
    ):
        atom_features = self._embedding(atom_features)
        for i in range(self._num_convs):
            atom_features = self._cgconvs[i](
                neighbor_indices, atom_features, bond_features, is_training
            )

        crystal_features = self._cgpooling(atom_features, segment_ids, num_atoms)
        crystal_features = jax.nn.softplus(crystal_features)

        for i in range(self._num_hidden_layers):
            crystal_features = self._fcs[i](crystal_features)
            crystal_features = jax.nn.softplus(crystal_features)
        out = self._fc_last(crystal_features)
        return out


def get_model_fn_t(
    num_initial_atom_features: int,
    num_atom_features: int,
    num_bond_features: int,
    num_convs: int,
    num_hidden_layers: int,
    num_hidden_features: int,
    max_num_neighbors: int,
    batch_size: int
):
    def model_fn(batch: Batch, is_training: bool) -> jnp.ndarray:
        model = CGCNN(
            num_initial_atom_features=num_initial_atom_features,
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            num_convs=num_convs,
            num_hidden_layers=num_hidden_layers,
            num_hidden_features=num_hidden_features,
            max_num_neighbors=max_num_neighbors,
            batch_size=batch_size,
            name="cgcnn",
        )
        neighbor_indices = batch["neighbor_indices"]
        atom_features = batch["atom_features"]
        bond_features = batch["bond_features"]
        num_atoms = batch['num_atoms']
        segment_ids = batch["segment_ids"]
        return model(neighbor_indices, atom_features, bond_features, num_atoms, segment_ids, is_training)

    return hk.transform_with_state(model_fn)
