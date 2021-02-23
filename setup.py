from os.path import splitext, basename
from setuptools import setup, find_packages
from glob import glob


__version__ = "0.0.1"


setup(
    name="jax_xtal",
    version=__version__,
    license="MIT",
    description="jax implementation of Crystal Graph Convolutional Neural Network (CGCNN)",
    long_description="",
    author="Kohei Shinohara",
    author_email="kohei19950508@gmail.com",
    packages=find_packages("jax_xtal"),
    py_modules=[splitext(basename(path))[0] for path in glob("jax_xtal/*.py")],
    python_requires=">=3.7",
    install_requires=[
        "setuptools",
        "jax",
        "jaxlib",
        "dm-haiku",
        "optax",
        "numpy",
        "pymatgen",
        "optuna",
    ],
    tests_require=["pytest"],
    include_package_data=True,
    extras_requires={},
    zip_safe=False,
)
