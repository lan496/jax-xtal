from functools import partial

from flax import linen as nn
from flax.linen import Dense, Embed, BatchNorm, softplus, sigmoid
import jax
import jax.numpy as jnp


class CGConv(nn.Module):
    """
    Convolutional layer in Eq. (5)
    """

    @nn.compact
    def __call__(self, neighbor_indices, atom_features, bond_features, train: bool = True):
        """
        Let the total number of atoms in the batch be N,
        neighbor_indices: (N, max_num_neighbors)
        atom_features: (N, num_atom_features)
        bond_features: (N, max_num_neighbors, num_bond_features)
        """
        norm = partial(BatchNorm, use_running_average=not train)

        num_atoms_batch, max_num_neighbors = neighbor_indices.shape
        num_atom_features = atom_features.shape[1]

        # (N, max_num_neighbors, num_atom_features)
        atom_neighbor_features = atom_features[neighbor_indices, :]
        total_neighbor_features = jnp.concatenate(
            [
                jnp.tile(atom_features[:, None, :], (1, max_num_neighbors, 1)),
                atom_neighbor_features,
                bond_features,
            ],
            axis=2,
        )
        total_gated_features = Dense(2 * num_atom_features)(total_neighbor_features)
        total_gated_features = norm(name="bn_1")(
            total_gated_features.reshape(-1, 2 * num_atom_features)
        ).reshape(
            num_atoms_batch, max_num_neighbors, 2 * num_atom_features
        )  # TODO: why reshape here?

        neighbor_filter, neighbor_core = jnp.split(total_gated_features, 2, axis=2)
        neighbor_filter = sigmoid(neighbor_filter)
        neighbor_core = softplus(neighbor_core)

        neighbor_summed = jnp.sum(
            neighbor_filter * neighbor_core, axis=1
        )  # (N, num_atom_features)
        neighbor_summed = norm(name="bn_2")(neighbor_summed)
        out = softplus(atom_features + neighbor_summed)  # TODO: defer from Eq. (5) ?
        return out


class CGPooling(nn.Module):
    """
    average-pooling over each crystal in a batch
    """

    @nn.compact
    def __call__(self, atom_features, atom_indices):
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


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network
    """

    num_atom_features: int
    num_convs: int
    num_hidden_layers: int
    num_hidden_features: int

    @nn.compact
    def __call__(
        self, neighbor_indices, atom_features, bond_features, atom_indices, train: bool = True
    ):
        atom_features = nn.Dense(self.num_atom_features, name="embedding")(atom_features)
        for i in range(self.num_convs):
            atom_features = CGConv(name=f"conv_{i}")(
                neighbor_indices, atom_features, bond_features, train
            )

        crystal_features = CGPooling()(atom_features, atom_indices)
        crystal_features = softplus(crystal_features)
        for i in range(self.num_hidden_layers):
            crystal_features = nn.Dense(self.num_hidden_features, name=f"fc_{i}")(crystal_features)
            crystal_features = softplus(crystal_features)
        out = nn.Dense(1, name="fc_last")(crystal_features)
        return out
