import os
from dataclasses import dataclass
import json

from dataclasses_json import dataclass_json


DEFAULT_ATOM_INIT_FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "atom_init.json"
)


@dataclass_json
@dataclass
class Config:
    # dataset path
    structures_dir: str
    targets_csv_path: str
    # model
    num_atom_features: int = 64
    num_convs: int = 2
    num_hidden_layers: int = 1
    num_hidden_features: int = 128
    # training
    num_epochs: int = 30
    learning_rate: float = 1e-2
    l2_reg: float = 1e-8
    batch_size: int = 256
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    # prediction
    batch_size_prediction: int = 1024
    # preprocessing
    atom_init_features_path: str = DEFAULT_ATOM_INIT_FEATURES_PATH
    cutoff: float = 6.0
    dmin: float = 0.7
    dmax: float = 5.2
    num_bond_features: int = 10
    max_num_neighbors: int = 12
    # misc
    print_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "log"
    seed: int = 0
    n_jobs: int = 1  # workers for preprocessing


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = json.load(f)
    return Config.from_dict(data)
