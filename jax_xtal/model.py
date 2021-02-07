from functools import partial

from flax import linen as nn
from flax.linen import Dense, sigmoid, BatchNorm, softplus
import jax
import jax.numpy as jnp


class MultiNeighborConv(nn.Module):
    """
    Convolutional layer in Eq. (5)
    """

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        norm = partial(BatchNorm, use_running_average=not train)

        # let the total number of atoms in the batch be N,
        # neighbor_indices: (N, max_num_neighbors)
        # atom_features: (N, num_atom_features)
        # bond_features: (N, max_num_neighbors, num_bond_features)
        neighbor_indices = inputs["neighbor_indices"]
        atom_features = inputs["atom_features"]
        bond_features = inputs["bond_features"]
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


if __name__ == "__main__":
    from data import CutoffNN, AtomFeaturizer, BondFeaturizer, CrystalDataset, get_dataloaders
    import numpy as np

    neighbor_strategy = CutoffNN(cutoff=6.0)
    atom_featurizer = AtomFeaturizer("../data/atom_init.json")
    bond_featurizer = BondFeaturizer(dmin=0.7, dmax=5.2, num_filters=10)
    dataset = CrystalDataset(
        "../data/structures_dummy",
        "../data/targets_dummy.csv",
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        neighbor_strategy=neighbor_strategy,
        max_num_neighbors=12,
    )

    rng = jax.random.PRNGKey(0)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=2)
    for batch in train_loader:
        model = MultiNeighborConv()
        inputs = {
            "neighbor_indices": batch["neighbor_indices"],
            "atom_features": batch["atom_features"],
            "bond_features": batch["bond_features"],
        }
        params = model.init(rng, inputs)
        out = model.apply(params, inputs, mutable=["batch_stats"])

        import pdb

        pdb.set_trace()
