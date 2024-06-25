from deepmd.infer import DeepDipole
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
from jax import jit, value_and_grad, vmap, random, custom_jvp, tree_util
import jax.numpy as jnp
from openmm import *
from openmm.app import * 
from openmm.unit import *
from nblist import NeighborListFreud, NeighborList
from mbar_r import MBAREstimator, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
from dmff.optimize import MultiTransform, genOptimizer
import dmff
dmff.PRECISION = 'float'
dmff.update_jax_precision(dmff.PRECISION)
import optax
import pickle
import mdtraj as md
from functools import partial
import matplotlib.pyplot as plt
from utils import NVT_langevin, calculate_corr_vdipole, calculate_ir
from difftraj import Loss_Generator
from eann import EANNForce
from scipy.interpolate import interp1d

seed = 1234
T = 298.15 # temperature
rc = 0.6 # nm
gamma = 1.0
dt = 0.0005 # ps
nout_nvt = 400
nsteps_nvt = 128000
nsteps = 400
corr_length = 400
nout = 1
batch_size = 80
init_stru1 = 'water64.pdb'
init_stru2 = 'water_new.pdb'
exp_spec = np.loadtxt('exp_IR_water')
spec_exp = interp1d(exp_spec[:, 0], exp_spec[:, 1], 'cubic')
key = random.PRNGKey(seed)

pdb1 = PDBFile(init_stru1)
# add by Junmin, eann calculator in JAX, May 27, 2023
atomtype = ['H', 'O']
n_elem = len(atomtype)
species1 = []
# Loop over all atoms in the topology
for atom in pdb1.topology.atoms():
    # Get the element of the atom
    element = atom.element.symbol
    mass = atom.element.mass
    species1.append(atomtype.index(atom.element.symbol))
elem_indices1 = jnp.array(species1)
eann_force1 = EANNForce(n_elem, elem_indices1, n_gto=12, rc=6, sizes=(64, 64))

pdb2 = PDBFile(init_stru2)
# add by Junmin, eann calculator in JAX, May 27, 2023
atomtype = ['H', 'O']
n_elem = len(atomtype)
species2 = []
# Loop over all atoms in the topology
for atom in pdb2.topology.atoms():
    # Get the element of the atom
    element = atom.element.symbol
    mass = atom.element.mass
    species2.append(atomtype.index(atom.element.symbol))
elem_indices2 = jnp.array(species2)
eann_force2 = EANNForce(n_elem, elem_indices2, n_gto=12, rc=6, sizes=(64, 64))

params = {}
with open('params_eann4.pickle', 'rb') as ifile:
    params['energy'] = pickle.load(ifile)

dipole_model = DeepDipole('dipole.pb')

pos1 = jnp.array(pdb1.getPositions()._value)
natoms1 = len(pos1)
box1 = jnp.array(pdb1.topology.getPeriodicBoxVectors()._value)
cov_map1 = jnp.array(np.loadtxt('cov_map')[:192, :192], dtype=jnp.int32)
nbl1 = NeighborList(box1, rc, cov_map1)
nbl1.allocate(pos1)

m_O = 15.99943
m_H = 1.007947
mass = jnp.tile(jnp.array([m_O, m_H, m_H]), natoms1//3)

pos2 = jnp.array(pdb2.getPositions()._value)
natoms2 = len(pos2)
box2 = jnp.array(pdb2.topology.getPeriodicBoxVectors()._value)
cov_map2 = jnp.array(np.loadtxt('cov_map'), dtype=jnp.int32)
nbl2 = NeighborListFreud(box2, rc, cov_map2)
nbl2.allocate(pos2)


@jit
def efunc1(pos, box, pairs, params):
    return jnp.array(eann_force1.get_energy(pos*10, box*10, pairs, params['energy']))
@jit
def efunc2(pos, box, pairs, params):
    return jnp.array(eann_force2.get_energy(pos*10, box*10, pairs, params['energy']))

@jit
def potential1(pos, box, params):
    nblist = nbl1.nblist.update(pos)
    pairs = nblist.idx.T
    return efunc1(pos, box, pairs, params)


def potential2(pos, box, params):
    nbl2.update(pos, box)
    pairs = nbl2.pairs
    return efunc2(pos, box, pairs, params)


def make_dipole(pos, box, dipole_model):
    atype = np.tile([0, 1, 1], natoms1//3)
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
    dipole_grad[:, 0, :, 0] = np.tile(np.tile([-2, 1, 1], natoms1//3).reshape([1, natoms1]), (len(pos), 1))
    dipole_grad[:, 1, :, 1] = np.tile(np.tile([-2, 1, 1], natoms1//3).reshape([1, natoms1]), (len(pos), 1))
    dipole_grad[:, 2, :, 2] = np.tile(np.tile([-2, 1, 1], natoms1//3).reshape([1, natoms1]), (len(pos), 1))
    dipole_grad = dipole_grad - 2*wannier_grad/100
    return dipole, dipole_grad

@custom_jvp
def f_nout(state):
    return jnp.array(make_dipole(state['pos'], box1, dipole_model)[0])
@f_nout.defjvp
def _f_nout_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out, dipole_grad = make_dipole(x['pos'], box1, dipole_model)
    tangent_out = jnp.einsum('ijkl,ikl->ij', dipole_grad, x_dot['pos'])
    return jnp.array(primal_out), tangent_out

def nvt_sample(init_stru, params, key, dcdfile):
    Sample_func = NVT_langevin(init_stru, potential1)
    key, subkey = random.split(key)
    state = Sample_func.get_init_state(T, subkey)
    Sample_func.set_condition(gamma, T, dt, nout_nvt, nsteps_nvt)
    key, subkey = random.split(key)
    traj = Sample_func.nvt_nout(state, params, subkey)
    Sample_func.save_dcd(traj, dcdfile)
    return traj, key

def L(traj):
    dipoles_reverse = traj[:, :(traj.shape[1]//2)]
    dipoles = traj[:, (traj.shape[1]//2):]
    dipoles = jnp.concatenate([dipoles_reverse[::-1], dipoles], axis=0)
    dipoles = dipoles/jnp.sqrt(jnp.linalg.det(box1))
    corr = vmap(calculate_corr_vdipole, in_axes=(1, None, None))(dipoles, dt, corr_length)
    corr = jnp.average(corr, axis=0)
    wavenum, lineshape = calculate_ir(corr, 200, 0.0005, 298.15, 8000)
    wavenum = jax.lax.stop_gradient(wavenum)
    id1 = jnp.where(wavenum< 1000)[0][1:]
    id2 = jnp.where((wavenum> 1000)&(wavenum< 4000))[0]
    lineshape_ref1 = spec_exp(wavenum[id1])
    lineshape_ref2 = spec_exp(wavenum[id2])
    return 10*jnp.sum((lineshape[id1] - lineshape_ref1)**2) + jnp.sum((lineshape[id2] - lineshape_ref2)**2), (jnp.concatenate([wavenum[id1], wavenum[id2]]), jnp.concatenate([lineshape_ref1, lineshape_ref2]), jnp.concatenate([lineshape[id1], lineshape[id2]]))


def nve_reweight(state_init, params):
    Generator = Loss_Generator(f_nout, box1, state_init['pos'][0], mass, dt, nsteps, nout, cov_map1, rc, efunc1)
    state = {}
    state['pos'] = jnp.concatenate([vmap(Generator.regularize_pos)(state_init['pos'] - state_init['vel']*dt), state_init['pos']])
    state['vel'] = jnp.concatenate([-state_init['vel'], state_init['vel']])
    state_splits = [{key: state[key][i*batch_size:(i+1)*batch_size] for key in state} for i in range(len(state['vel'])//batch_size)]
    traj = []
    final_states = []
    for state_split in state_splits:
        fs, tj = Generator.ode_fwd(state_split, params)
        final_states.append(fs)
        traj.append(tj['state'])
    traj = jnp.concatenate(traj, axis=1)
    (err, (wavenum, lineshape_ref, lineshape)), gradient_traj = value_and_grad(L, has_aux=True)(traj)
    gradient_trajs = jnp.array_split(gradient_traj, len(state['vel'])//batch_size, axis=1)
    gradient = tree_util.tree_map(jnp.zeros_like, params)
    for final_state, gradient_traj in zip(final_states, gradient_trajs):
        ads, gd = Generator._ode_bwd(final_state, params, gradient_traj)
        gradient = tree_util.tree_map(lambda p, u: p + u, gradient, gd)
    return err, gradient, wavenum, lineshape_ref, lineshape

bonds = []
for i in range(len(cov_map2)):
    bonds.append(jnp.concatenate([jnp.array([i]), jnp.where(cov_map2[i] > 0)[0]]))

@jit
def regularize_pos(pos, box):
    cpos = jnp.stack([jnp.sum(pos[bond], axis=0)/len(bond) for bond in bonds])
    box_inv = jnp.linalg.inv(box)
    spos = cpos.dot(box_inv)
    spos -= jnp.floor(spos)
    shift = spos.dot(box) - cpos
    return pos + shift

def cleanfile(dcdfile, seed):
    os.system("sed -i 's/<seed>%s/<seed>%s/g' input.xml"%(seed, seed+100))
    seed = seed + 100
    os.system('bash sub.sh')
    u = md.load_xyz('simulation.pos_0.xyz', top='water_new.pdb')
    cells = []
    with open('simulation.pos_0.xyz', 'r') as f:
        for line in f:
            words = line.split()
            if "CELL(abcABC):" in words:
                cells.append(np.eye(3)*float(words[2]))
    cells = np.array(cells)
    u.unitcell_vectors = cells/10
    u = u[-100:]
    pos = jnp.array(u.xyz)
    box = jnp.array(u.unitcell_vectors)
    pos = vmap(regularize_pos, in_axes=(0, 0))(pos, box)
    u.xyz = pos
    u.save_dcd(dcdfile)
    density_ref = np.loadtxt('simulation.out')[-100:, -1]
    os.mkdir('./%s'%dcdfile[:-4])
    os.system('mv simulation* ./%s'%dcdfile[:-4])
    os.system('mv RESTART ./%s'%dcdfile[:-4])
    os.system('mv logfile ./%s'%dcdfile[:-4])
    os.system('mv water_new* ./%s'%dcdfile[:-4])
    u[-1].save_pdb('water_new.pdb')
    os.system('python make_init_pdb.py water_new.pdb > water_new_init.pdb')
    return density_ref, seed

def Loss2(params, density):
    target_energy_function = buildTrajEnergyFunction(efunc2, cov_map2, rc, ensemble='npt', useFreud=True)
    target_state = TargetState(T, target_energy_function)
    weight, utarget = estimator.estimate_weight(target_state, parameters=params)
    density = jnp.average(density, weights=weight)
    err = 1e12*(density - 1.0)**2
    return err, (density, utarget)


multiTrans1 = MultiTransform(params)
multiTrans1["energy/w"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1["energy/b"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1["energy/c"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1["energy/rs"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1["energy/inta"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1["energy/initpot"] = genOptimizer(learning_rate=0.00005, clip=0.01, nonzero=False)
multiTrans1.finalize()
grad_transform1 = optax.multi_transform(multiTrans1.transforms, multiTrans1.labels)
opt_state1 = grad_transform1.init(params)


nloops = 100
estimator = MBAREstimator()
for nloop in range(nloops):

    if len(estimator.states) < 1:
        ifile = open('params.pickle', 'wb')
        pickle.dump(params['energy'], ifile)
        ifile.close()
        density_ref, seed = cleanfile(f"loop-{nloop}.dcd", seed)
        os.system('mv params.pickle ./loop-%s'%nloop)
        traj = md.load(f"loop-{nloop}.dcd", top="water_new.pdb")
        state_name = f"loop-{nloop}"
        state = OpenMMSampleState(state_name, potential2, params, temperature=T, pressure=1.0)
        sample = Sample(traj, state_name)
        estimator.add_state(state)
        estimator.add_sample(sample)

        # estimator need to be reconverged whenenver new samples or states are added
        estimator.optimize_mbar()

    (loss2, (density, utarget)), gradient2 = value_and_grad(Loss2, argnums=(0), has_aux=True)(params, density_ref)

    ieff = estimator.estimate_effective_sample(utarget, decompose=True)
    print('Effective sample sizes:')
    for k, v in ieff.items():
        print(f'{k}: {v}')

    for k, v in ieff.items():
        if v < 60 and k != "Total":
            estimator.remove_state(k)
 
    if len(estimator.states) >= 1:

        state_init, key = nvt_sample(init_stru1, params, key, f'ir-{nloop}.dcd')
        u = md.load_dcd(f'ir-{nloop}.dcd', top='water64.pdb')
        u[-1].save_pdb('water64.pdb')
        loss1, gradient1, wavenum, lineshape_ref, lineshape = nve_reweight(state_init, params)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(wavenum, lineshape_ref, label='exp')
        ax.plot(wavenum, lineshape, label=f'loop-{nloop}')
        ax.set_xlim((0, 4000))
        ax.set_ylim((-500, 20000))
        ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax.set_ylabel(r'n($\omega$)$\alpha$($\omega$) (cm$^{-1}$)')
        ax.legend()
        plt.savefig(f'compare-{nloop}.png', dpi=800)
        plt.close()


        loss = loss1+loss2
        gradient = tree_util.tree_map(lambda p, u: p + u, gradient1, gradient2)
        print('Loop: ', nloop)
        print('Loss1:', loss1)
        print('Loss2:', loss2)
        print('Loss:', loss)
        print('density:', density)
        print('gradient1:', gradient1)
        print('gradient2:', gradient2)
        print('gradient:', gradient)

        updates, opt_state1 = grad_transform1.update(gradient, opt_state1, params=params)
        params = optax.apply_updates(params, updates)
        ifile = open(f'params-{nloop}.pickle', 'wb')
        pickle.dump(params, ifile)
        ifile.close()


