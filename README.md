# FEM

`FEM` is a python library for solving various combinatorial optimization problems
using gradient based mean-field annealing. The current build-in problem types are:
* maximum cut
* balance minimum cut
* maximum k-satisfactory
* vertex cover
* qubo-like instances
Some examples can be found on [example notebook](examples/build_in.ipynb).

You can also use `FEM` to solve your own optimization problems as long as 
the expectation value of the target function can be written as some function of 
the marginal probability. Please refer to the [customize examples](examples/customize.ipynb) 
for further details.

You can run the jupyter notebooks in [benchmarks](/benchmarks) to reproduce the main results presented in the paper.


## Installation

One can use conda to install the package with the following commands:
```bash
conda env create -f environment.yml
```
this will create an environment named `fem` with all the dependencies except for the pytorch, then activate the environment with `conda activate fem`.

Then `pytorch` have to be installed manually with 
```bash
pip3 install torch torchvision torchaudio
```
see the [pytorch website](https://pytorch.org/) for more details.
