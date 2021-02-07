# use devel
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# configure for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# install python build dependencies
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git

# pyenv
ARG PYTHON_VERSION=3.8.2
ENV HOME=/root
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && eval "$(pyenv init -)"
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}
RUN pyenv rehash

# install jax
RUN python -m pip install --upgrade pip
ARG JAX_VERSION=0.2.9
ARG JAXLIB_VERSION=0.1.59+cuda111
RUN python -m pip install --upgrade jax==${JAX_VERSION} jaxlib==${JAXLIB_VERSION} -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install jupyter
RUN python -m pip install notebook

RUN mkdir /workspace
ENTRYPOINT ["/bin/bash"]
