# jax-xtal
jax/flax implementation of [Crystal Graph Convolutional Neural Networks (CGCNN)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)

- CUDA 11.1
- CuDNN 8
- jax 0.2.9
- jaxlib 0.1.59
- flax 0.3.0
- python 3.8
- Ubuntu 18.04

## Installation

### Docker
```script
docker build -t jax-xtal -f docker/Dockerfile .
docker run -it --gpus all -v $(pwd):/workspace --name jax-xtal jax-xtal
docker attach jax-xtal
```

### local

## Usage

### Prepare a custom dataset

#### Prepare initial one-hot vectors for atomic features
`data/atom_init.json` is created by the following command.
```bash
python prepare_atom_features.py
```

### Predicting formation energy with a pre-trained network
```bash
python predict.py --config configs/debug.json --checkpoint checkpoints/checkpoint_30.flax --structures_dir data/structures_dummy --output out.csv
```

### Train a CGCNN model
```bash
python main.py --config configs/debug.json
```

## Benchmark
- Reproduce dataset

## Other implementations
- official(pytorch), [https://github.com/txie-93/cgcnn](https://github.com/txie-93/cgcnn)
- open catalyst(pytorch+torch_geometric), [https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/models/cgcnn.py](https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/models/cgcnn.py)
    - torch_geometric, [https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.CGConv)
