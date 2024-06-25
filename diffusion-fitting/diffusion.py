from functools import partial
import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax
from jax import jit, value_and_grad, vmap, pmap, grad, random
import jax.numpy as jnp
from dmff1.api import Hamiltonian
from openmm import *
from openmm.app import * 
from openmm.unit import *
from dmff1.mbar import MBAREstimator, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
from dmff1.optimize import MultiTransform, genOptimizer
import dmff1
dmff1.PRECISION = 'float'
dmff1.update_jax_precision(dmff1.PRECISION)
import optax
import mdtraj as md
import pickle
from difftraj import Loss_Generator

seed = 0
T = 94.4 
KB = 1.380649e-23
NA = 6.0221408e23
kT = KB * T * NA
rc = 1.0
init_stru = 'opt.pdb'
parameter = 'forcefield.xml'
dt = 0.001
nout = 2
nsteps = 2000

ff = Hamiltonian(parameter)
pdb = PDBFile(init_stru)
pots = ff.createPotential(pdb.topology, nonbondedMethod=CutoffPeriodic, nonbondedCutoff=rc*nanometer)
cov_map = pots.meta['cov_map']
efunc = jit(pots.getPotentialFunc())
params = ff.getParameters().parameters
pos = jnp.array(pdb.getPositions()._value)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)
m_Ar = 39.948
mass = jnp.tile(jnp.array([m_Ar]), len(pos))


def nvt_sample(init_stru, parameter, trajectory, key):
    pdb = PDBFile(init_stru)
    forcefield = ForceField(parameter)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffPeriodic,
            nonbondedCutoff=rc*nanometer)
    integrator = LangevinMiddleIntegrator(T*kelvin, 0.1/picosecond, dt*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator,
                            platform=Platform.getPlatformByName('CUDA'), platformProperties={'Precision':'mixed'})
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(T*kelvin)
    simulation.step(100000)
    simulation.reporters.append(DCDReporter(trajectory, 2000))
    simulation.step(200000)

    key, subkey = random.split(key)
    u = md.load_dcd(trajectory, top = init_stru)
    positions = jnp.array(u.xyz)
    state_init = {}
    state_init['pos'] = positions
    state_init['vel'] = jnp.einsum('ijk,j->ijk', random.normal(subkey, shape=positions.shape), jnp.sqrt(kT/mass * 1e3)) / 1e3
    return state_init, key

@jit
def f_nout(state, vel0):
    return jnp.average(jnp.sum(vel0 * state['vel'], axis=-1), axis=-1)


def L(traj, params):
    def l_out(traj, params):
        target_energy_function = buildTrajEnergyFunction(efunc, cov_map, rc, ensemble='nvt', useFreud=True)
        target_state = TargetState(T, target_energy_function)
        weight, utarget = estimator.estimate_weight(target_state, parameters=params)
        tcf = jnp.average(traj, weights=weight, axis=-1)/3
        diffusion = jnp.trapz(tcf, dx=dt*nout)
        err = 1e6*(diffusion-0.00236)**2
        return err, (tcf, utarget)
    v, g = value_and_grad(l_out, argnums=(1), has_aux=True)(traj, params)
    return v[0], (v[1][0], v[1][1], g)


metadata = []
def nve_reweight(state_init, params):

    vel0 = state_init['vel']
    Generator = Loss_Generator(partial(f_nout, vel0=vel0), box, state_init['pos'][0], mass, dt, nsteps, nout, cov_map, rc, efunc)
    Loss = Generator.generate_Loss(partial(L, params=params), has_aux=True, metadata=metadata)
    err, gradient = value_and_grad(Loss, argnums=(1))(state_init, params)
    utarget = metadata[-1]['aux_data'][1]
    nvt_g = metadata[-1]['aux_data'][-1]
    return err, gradient, nvt_g, utarget


params_choose = {}
params_choose = params
multiTrans = MultiTransform(params_choose)
multiTrans["NonbondedForce/sigma"] = genOptimizer(learning_rate=0.0002, clip=0.001, nonzero=False)
multiTrans["NonbondedForce/epsilon"] = genOptimizer(learning_rate=0.01, clip=0.05, nonzero=False)
multiTrans.finalize()
grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
opt_state = grad_transform.init(params_choose)

key = random.PRNGKey(seed)
state_init, key = nvt_sample(init_stru, parameter, 'init.dcd', key)

estimator = MBAREstimator()
state_name = 'diffusion'
state = OpenMMSampleState(state_name, parameter, init_stru, temperature=T,
                            nonbondedMethod=CutoffPeriodic, nonbondedCutoff=rc*nanometer)
traj = md.load('init.dcd', top='opt.pdb')
sample = Sample(traj, state_name)
estimator.add_state(state)
estimator.add_sample(sample)
estimator.optimize_mbar()


nloops = 50
for nloop in range(1, nloops):
    print('LOOP:', nloop)
    print('params:', params)
    loss, gradient, nvt_g, utarget = nve_reweight(state_init, params_choose)
    print('Loss:', loss)
    print('nve gradient:', gradient)
    print('nvt gradient:', nvt_g)
    gradient = jax.tree_util.tree_map(lambda p, u: p + u, gradient, nvt_g)
    print('total gradient:', gradient)
    ieff = estimator.estimate_effective_sample(utarget, decompose=True)

    gradient_choose = {}
    gradient_choose = gradient

    updates, opt_state = grad_transform.update(gradient_choose, opt_state, params=params_choose)
    params_choose = optax.apply_updates(params_choose, updates)
    params = params_choose
    ff.getParameters().parameters = params
    ff.renderXML(f'loop-{nloop}.xml')
    parameter = f'loop-{nloop}.xml'

    print('Effective sample sizes:')
    for k, v in ieff.items():
        print(f'{k}: {v}')

    for k, v in ieff.items():
        if v < 100 and k != "Total":
            estimator.remove_state(k)

    if len(estimator.states) < 1:
        print("Add", f"loop-{nloop}")
        # get new sample using the current state
        u = md.load(init_stru)
        u.xyz = state_init['pos'][-1]
        u.save_pdb(f"loop-{nloop}.pdb")
        init_stru = f"loop-{nloop}.pdb"
        state_init, key = nvt_sample(init_stru, parameter, f"loop-{nloop}.dcd", key)
        traj = md.load(f"loop-{nloop}.dcd", top="opt.pdb")
        state = OpenMMSampleState(f"loop-{nloop}", parameter, init_stru, temperature=T,
                            nonbondedMethod=CutoffPeriodic, nonbondedCutoff=rc*nanometer)
        sample = Sample(traj, f"loop-{nloop}")
        estimator.add_state(state)
        estimator.add_sample(sample)

        # estimator need to be reconverged whenenver new samples or states are added
        estimator.optimize_mbar()

ifile = open('metadata.pickle', 'wb')
pickle.dump(metadata, ifile)
ifile.close()

