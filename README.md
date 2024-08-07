# Fitting dynamical properties
Here, we provide two examples of fitting [diffusion coefficient](./diffusion-fitting) and [IR spectrum](./IR-fitting), which are the cases in paper [Refining Potential Energy Surface through Dynamical Properties via
  Differentiable Molecular Simulation](http://arxiv.org/abs/2406.18269). The core code of implementing the adjoint method is in the [`difftraj.py`](./diffusion-fitting/difftraj.py), which is also the DiffTraj module in [DMFF](https://github.com/deepmodeling/DMFF/tree/devel/dmff).

The trained potential energy surface (EANN model) and dipole surface (Deep Dipole model) is provided as `params_eann4.pickle` and `dipole.pb` in [IR-fitting](./IR-fitting) folder.

