import sys
from deepmd.infer import DeepDipole
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
from jax import jit, value_and_grad, vmap, pmap, grad, vjp, tree_util, random, custom_jvp
import jax.numpy as jnp
import dmff
dmff.PRECISION = 'float'
dmff.update_jax_precision(dmff.PRECISION)
from dmff.api import Hamiltonian
from openmm import *
from openmm.app import * 
from openmm.unit import *
from nblist import NeighborList
from mbar_r import MBAREstimator, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
from dmff.optimize import MultiTransform, genOptimizer
import optax
import pickle
import mdtraj as md
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from utils import NVT_langevin, calculate_corr_vdipole, calculate_ir
from difftraj import Loss_Generator
from eann import EANNForce
from scipy.interpolate import interp1d

seed = 0
T = 298.15 # temperature
rc = 0.6 # nm
gamma = 5.0
dt = 0.0005 # ps
nout_nvt = 1000
nsteps_nvt = 320000
nsteps = 10000
corr_length = 4000
nout = 1
batch_size = 80
init_stru = 'water64.pdb'
exp_spec = np.loadtxt('exp_IR_water')
spec_exp = interp1d(exp_spec[:, 0], exp_spec[:, 1], 'cubic')
key = random.PRNGKey(seed)
metadata = []

pdb = PDBFile(init_stru)
# add by Junmin, eann calculator in JAX, May 27, 2023
atomtype = ['H', 'O']
n_elem = len(atomtype)
species = []
# Loop over all atoms in the topology
for atom in pdb.topology.atoms():
    # Get the element of the atom
    element = atom.element.symbol
    mass = atom.element.mass
    species.append(atomtype.index(atom.element.symbol))
elem_indices = jnp.array(species)
eann_force = EANNForce(n_elem, elem_indices, n_gto=12, rc=6, sizes=(64, 64))
params = {}
with open('params-92.pickle', 'rb') as ifile:
    params = pickle.load(ifile)
dipole_model = DeepDipole('dipole.pb')

pos = jnp.array(pdb.getPositions()._value)
natoms = len(pos)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)
cov_map = jnp.array(np.loadtxt('cov_map')[:192,:192], dtype=jnp.int32)
nbl = NeighborList(box, rc, cov_map)
nbl.allocate(pos)

m_O = 15.99943
m_H = 1.007947
mass = jnp.tile(jnp.array([m_O, m_H, m_H]), natoms//3)

@jit
def efunc(pos, box, pairs, params):
    return jnp.array(eann_force.get_energy(pos*10, box*10, pairs, params['energy']))
@jit
def potential(pos, box, params):
    nblist = nbl.nblist.update(pos)
    pairs = nblist.idx.T
    return efunc(pos, box, pairs, params)

def make_dipole(pos, box, dipole_model):
    atype = np.tile([0, 1, 1], natoms//3)
    box = np.tile(box.reshape([1, 3, 3]), (len(pos), 1, 1))
    wanniers = []
    wannier_grads = []
    for i in range(len(pos)//40):
        wannier, wannier_grad, _ = dipole_model.eval_full(pos[40*i:40*(i+1)]*10, box[40*i:40*(i+1)]*10, atype)
        wanniers.append(wannier)
        wannier_grads.append(wannier_grad)
    wannier = np.concatenate(wanniers)
    wannier_grad = np.concatenate(wannier_grads)
    dipole = np.sum(pos, axis=1) - 3*np.sum(pos[:, ::3], axis=1) -2*wannier/100
    dipole_grad = np.zeros_like(wannier_grad)
    dipole_grad[:, 0, :, 0] = np.tile(np.tile([-2, 1, 1], natoms//3).reshape([1, natoms]), (len(pos), 1))
    dipole_grad[:, 1, :, 1] = np.tile(np.tile([-2, 1, 1], natoms//3).reshape([1, natoms]), (len(pos), 1))
    dipole_grad[:, 2, :, 2] = np.tile(np.tile([-2, 1, 1], natoms//3).reshape([1, natoms]), (len(pos), 1))
    dipole_grad = dipole_grad - 2*wannier_grad/100
    return dipole, dipole_grad

@custom_jvp
def f_nout(state):
    return jnp.array(make_dipole(state['pos'], box, dipole_model)[0])
@f_nout.defjvp
def _f_nout_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out, dipole_grad = make_dipole(x['pos'], box, dipole_model)
    tangent_out = jnp.einsum('ijkl,ikl->ij', dipole_grad, x_dot['pos'])
    return jnp.array(primal_out), tangent_out

def nvt_sample(init_stru, params, key, dcdfile):
    Sample_func = NVT_langevin(init_stru, potential)
    key, subkey = random.split(key)
    state = Sample_func.get_init_state(T, subkey)
    Sample_func.set_condition(gamma, T, dt, nout_nvt, nsteps_nvt)
    key, subkey = random.split(key)
    traj = Sample_func.nvt_nout(state, params, subkey)
    state['pos'] = traj['pos'][-1]
    state['vel'] = traj['vel'][-1]
    key, subkey = random.split(key)
    traj = Sample_func.nvt_nout(state, params, subkey)
    Sample_func.save_dcd(traj, dcdfile)
    return traj, key


def nve_reweight(state_init, params):
    Generator = Loss_Generator(f_nout, box, state_init['pos'][0], mass, dt, nsteps, nout, cov_map, rc, efunc)
    state_splits = [{key: state_init[key][i*batch_size:(i+1)*batch_size] for key in state_init} for i in range(len(state_init['vel'])//batch_size)]
    traj = []
    final_states = []
    for state_split in state_splits:
        fs, tj = Generator.ode_fwd(state_split, params)
        final_states.append(fs)
        traj.append(tj['state'])
    traj = jnp.concatenate(traj, axis=1)
    dipoles = traj/jnp.sqrt(jnp.linalg.det(box))
    np.save('dipoles', dipoles)
    corr = vmap(calculate_corr_vdipole, in_axes=(1, None, None))(dipoles, dt, corr_length)
    corr = jnp.average(corr, axis=0)
    wavenum, lineshape = calculate_ir(corr, 200, 0.0005, 298.15, 8000)
    id = jnp.where(wavenum< 4000)[0][1:]
    lineshape_ref = spec_exp(wavenum[id])   

    return wavenum[id], lineshape_ref, lineshape[id]


state_init, key = nvt_sample(init_stru, params, key, 'init.dcd')

wavenum, lineshape_ref, lineshape = nve_reweight(state_init, params)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(wavenum, lineshape_ref, label='exp')
ax.plot(wavenum, lineshape, label='initial')
ax.set_xlim((0, 4000))
ax.set_ylim((-500, 20000))
ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
ax.set_ylabel(r'n($\omega$)$\alpha$($\omega$) (cm$^{-1}$)')
ax.legend()
plt.savefig('compare.png', dpi=800)
plt.close()

np.save('lineshape', np.stack([wavenum, lineshape_ref, lineshape], axis=1))





