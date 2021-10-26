from setuptools import setup

install_requires = [
    "numpy",
    "scipy>=1.3",
    "sympy>=1.5",
    "tensorflow>=2.1",
    "tensorflow_addons",
    "tqdm",
]

setup(
    name="dimenet",
    version="1.0",
    description="Directional Message Passing for Molecular Graphs",
    packages=["dimenet"],
    install_requires=install_requires,
    zip_safe=False,
)
