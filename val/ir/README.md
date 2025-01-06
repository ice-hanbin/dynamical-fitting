# Use DMFF and difftraj to get IR spectrum

This is an example of using the optimized EANN potential to obtain an IR spectrum. Both NVT and NVE simulations were performed using self-implemented code.

## `IR.py`

The main function to calculate the IR spectrum. `potential` and `efunc` give the energy. The `make_dipole` function uses the DeepDipole model to compute the total dipole and its gradient with respect to the atomic positions. The `nvt_sample` function performs the NVT simulation, while the `nve_reweight` function conducts the NVE simulation to obtain dipoles and subsequently generate the IR spectrum.

## `cov_map`

The covalent map file used in nblist is for 216 water molecules, same as [here](https://github.com/ice-hanbin/dynamical-fitting/edit/main/val/density/README.md)

## `difftraj.py`

The code to perform a differentiable NVE simulation, in [DMFF](https://github.com/deepmodeling/DMFF/blob/devel/docs/user_guide/4.8DiffTraj.md).

## `dipole.pb`

DeepDipole model

## `eann.py`

The EANN model in jax (writen by Junmin Chen, in [DMFF](https://github.com/deepmodeling/DMFF/blob/master/docs/user_guide/4.4MLForce.md))

## `exp_IR_water`

Experimental IR spectrum data of water.

## `job.json`

Submit file in [Bohrium](https://bohrium.dp.tech/).

## `nblist.py`

The neighborlist used here, from `jax_md`.

## `params-57.pickle`, `params-61.pickle`, `params-92.pickle`, `params-95.pickle`, `params-96.pickle`

EANN potential parameters from 5 indenpent optimizations. To open this file, ensure your environment includes jax (version 0.4.14) and jaxlib (version 0.4.14+cuda11.cudnn86).

## `utils.py`

Self-implemented code to perform NVT simulation and get IR spectrum.

## `water64.py`

input pdb file for 64 water molecules.
