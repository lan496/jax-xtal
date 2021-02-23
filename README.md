# jax-xtal
jax/haiku implementation of [Crystal Graph Convolutional Neural Networks (CGCNN)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)

## Installation
There are two choices for installing this repository, (nvidia-)docker and pip in local environment.

I have tested this repository with the following environment:
- CUDA 11.1
- CuDNN 8
- jax 0.2.9
- jaxlib 0.1.61
- haiku 0.0.4.dev0
- python 3.8

### Docker
If you use a GPU, make sure you have installed [a NVIDIA driver and NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#).
```shell
git clone git@github.com:lan496/jax-xtal.git
cd jax-xtal
docker build -t jax-xtal -f docker/Dockerfile .
docker run -it --gpus all -v $(pwd):/workspace --name jax-xtal jax-xtal
docker attach jax-xtal
```
In the `jax-xtal` container:
```shell
cd workspace
python -m pip install -e .
```

### Pip in local
If you use a GPU, first follow [these instructions](https://github.com/google/jax#installation) to install JAX.

Then, install this repository with pip:
```shell
git clone git@github.com:lan496/jax-xtal.git
cd jax-xtal
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage
You can predict (currently only) formation energies of your prepared structures with a pre-trained model and train a CGCNN model from your dataset for a regression task.

### Prepare dataset
The following files are required to train or evaluate CGCNN:
- `data/structures/*.json`: directory for [JSON files of structures with pymatgen-format](https://pymatgen.org/usage.html#side-note-as-dict-from-dict)
- (only for training) `data/target.csv`:
  CSV file for a target property of each structure.
  This file is required to have two columns.
  The first column records a basename of a corresponding JSON file without `.json`.
  The second column records a target property.

Example
```
# ls data/structures
mp-754118.json mp-978908.json ...

# head data/target.csv
mp-754118,-2.2429051615740736
mp-978908,-0.9040630391666671
...
```

If you plan to prepare your dataset from [Materials Project](https://materialsproject.org/), this snippet may be useful:
```python
m = MPRester(your_api_key) 
material_id = "mp-1265"
formation_energy_per_atom = m.get_data(material_id, prop='formation_energy_per_atom')[0]['formation_energy_per_atom']
structure = m.get_structure_by_material_id(material_id)
```


### Predicting formation energy with a pre-trained network
To use a pre-trained model, you are required to specify a config file (`configs/default.json`) and a pickled model (`checkpoints/pre_trained.formation_energy.pkl`).

```shell
python predict.py --config configs/default.json --checkpoint checkpoints/checkpoint.default.pkl --structures_dir ./data/structures --output out.csv
```

### Train a CGCNN model
To train a model, a JSON config file for hyperparameters is needed.
An example config is as follows:
```json
# cat configs/your_config.json
{
    "structures_dir": "data/structures",
    "targets_csv_path": "data/target.csv",
    "n_jobs": 10
}
```
`structures_dir` specifies a path for pymatgen-formot structure JSON files, and `target_csv_path` specifies a path for a CSV file of target properties.
Other keys for the config file is defined in [jax_xtal.config.Config](jax_xtal/config.py) dataclass.

After setting your config file, you can train a CGCNN model with
```shell
python main.py --config configs/your_config.json
```
By default, a log file is created under `./log` and a final checkpoint is stored under `./checkpoints`.

## Benchmark
A pre-trained model, [checkpoint/checkpoint.default.pkl](checkpoint/checkpoint.default.pkl), is trained with a dataset from [Materials Project](https://github.com/txie-93/cgcnn/tree/master/data/material-data).
The mean absolute error of formation energies for the test set is 116 meV/atom.

## Misc

### Prepare initial one-hot vectors for atomic features
`data/atom_init.json` is created by the following command.
```shell
python prepare_atom_features.py
```

## Other implementations
- [Official implementation (pytorch)](https://github.com/txie-93/cgcnn)
- [Open Catalyst (pytorch + torch_geometric)](https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/models/cgcnn.py)
- [CGConv layer (torch_geometric)](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv)
