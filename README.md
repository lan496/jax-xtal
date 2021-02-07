# jax-xtal
jax implementation of Crystal Graph Convolutional Neural Networks (CGCNN)

- CUDA 11.1
- CuDNN 8
- jax 0.2.9
- jaxlib 0.1.59
- flax 0.3.0
- torch 1.7.1+cpu
- python 3.8
- Ubuntu 18.04

## Docker environment
```
docker build -t jax-xtal -f docker/Dockerfile .
docker run -it --gpus all -v $(pwd):/workspace --name jax-xtal jax-xtal
docker attach jax-xtal
```
