# Use i-PI and DMFF to run NPT simulation

Here is an example of using the optimized EANN potential to perform an NPT simulation of liquid water to calculate its density. If you are interested in running an NVT simulation using i-PI, you can find the code [here](https://github.com/plumbum082/water_classical_md/tree/main)

## `client_dmff.py`

This interface demonstrates how to use the force and virial from DMFF to run i-PI. The neighborlist used here is from  `jax_md` and is defined in `nblist.py`. The required input files include a PDB file (`water_new.pdb`), a force field file (if needed), a residues file (if needed), and an ML potential file (`params-92.pickle`). Since my force field does not include a classical force field, DMFF cannot generate a covalent map that used in nblist. Therefore, as an example, I have provided a pre-generated covalent map file (`cov_map`) for 216 water molecules.

## `cov_map`

The covalent map file used in nblist is for 216 water molecules. You can generate it using the following code:

```
pdb = PDBFile('water_new.pdb')
ff = Hamiltonian('qspc-fw.xml')
pots = ff.createPotential(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.6*nanometer, rigidWater=False)
cov_map = pots.meta['cov_map']
```

## `driver.py`

i-PI driver

## `eann.py`

The EANN model in jax (writen by Junmin Chen, in [DMFF](https://github.com/deepmodeling/DMFF/blob/master/docs/user_guide/4.4MLForce.md))

## `input.xml`

input file for i-PI

## `nblist.py`

The neighborlist used here, from `jax_md`.

## `params-57.pickle`, `params-61.pickle`, `params-92.pickle`, `params-95.pickle`, `params-96.pickle`

EANN potential parameters from 5 indenpent optimizations. To open this file, ensure your environment includes jax (version 0.4.14) and jaxlib (version 0.4.14+cuda11.cudnn86).

## `run_client_dmff.sh`

bash file to run `client_dmff.py`

## `run_server.sh`

bash file to run i-PI

## `sub.sh`

bash file to run, after preparing all files, use `sbatch sub.sh` to run.

## `water_new_init.pdb`

input pdb file for i-PI

## `water_new.pdb`

input pdb file for DMFF.

