import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from typing import Dict, List, Mapping, Optional, Tuple, Union, IO
import numpy as np
import jax
from jax import jit, value_and_grad, vmap, pmap, grad, vjp, tree_util, random
import jax.numpy as jnp
from openmm import *
from openmm.app import * 
from openmm.unit import *
import mdtraj as md
from functools import partial

KB = 1.380649e-23
NA = 6.0221408e23
m_O = 15.99943
m_H = 1.007947
c = 299792458

class NVT_langevin:
    def __init__(self, init_stru, potential):
        self.init_stru = init_stru
        self.potential = potential
        self.pdb = PDBFile(self.init_stru)
        self.pos = jnp.array(self.pdb.getPositions()._value)
        self.box = jnp.array(self.pdb.topology.getPeriodicBoxVectors()._value)
        self.natoms = len(self.pos)
        self.m = jnp.repeat(jnp.tile(jnp.array([[m_O, m_H, m_H]]), self.natoms//3), 3, axis=0).T
        idh = jnp.delete(jnp.arange(0, self.natoms), jnp.arange(0, self.natoms//3)*3)
        ido = jnp.repeat(jnp.arange(0, self.natoms)[::3], 2)
        self.bonds = [idh, ido]

        @partial(jit, static_argnums=())
        def regularize_pos(pos):  
            box_inv = jnp.linalg.inv(self.box)
            spos = pos.dot(box_inv)
            spos -= jnp.floor(spos)
            idh, ido = self.bonds
            dr = spos[idh] - spos[ido]
            dr -=jnp.floor(dr + 0.5)
            spos = spos.at[idh].set(dr+spos[ido])
            return spos.dot(self.box)
        self.regularize_pos = regularize_pos
        return

    def get_init_state(self, T, key, pos=None):
        kT = KB * T * NA
        state = {}
        if pos == None:
            state['pos'] = self.pos
        else:
            state['pos'] = pos
        state['vel'] = random.normal(key, shape=(self.natoms, 3))* jnp.sqrt(kT/self.m * 1e3) / 1e3
        return state

    def set_condition(self, gamma, T, dt, nout, nsteps):
        self.gamma = gamma
        self.T = T
        self.dt = dt
        self.nout = nout
        self.nsteps = nsteps
        self.kT = KB * self.T * NA

        @partial(jit, static_argnums=())
        def vv_step(state, params, key):
            x0 = state['pos']
            v0 = state['vel']
            c1 = jnp.exp(-self.gamma*(self.dt/2))
            c2 = jnp.sqrt((1-c1**2)*self.kT/self.m * 1e3) / 1e3
            key, subkey = random.split(key)
            v0 = c1*v0 + c2*random.normal(subkey, shape=(len(v0),3))
            f0 = -grad(self.potential, argnums=(0))(x0, self.box, params)
            a0 = f0 / self.m
            x1 = x0 + v0 * self.dt + a0 * self.dt **2 /2
            x1 = self.regularize_pos(x1)
            f1 = -grad(self.potential, argnums=(0))(x1, self.box, params)
            a1 = f1 / self.m
            v1 = v0 + (a0 + a1) * self.dt /2 
            key, subkey = random.split(key)
            v1 = c1*v1 + c2*random.normal(subkey, shape=(len(v1),3))
            return {'pos': x1, 'vel':v1}
        
        def vv_nout(state, params, key):
            subkeys = random.split(key, num=self.nout)
            def body_fun(i, carry):
                state = carry
                return vv_step(state, params, subkeys[i])
            return jax.lax.fori_loop(0, nout, body_fun, (state))
        
        self.vv_step = vv_step
        self.vv_nout = vv_nout

        return

    def nvt_nout(self, state, params, key):
        subkeys = random.split(key, num=self.nsteps//self.nout)
        def ode_fwd(i, carry):
            state, traj = carry
            state = self.vv_nout(state, params, subkeys[i])
            traj['pos'] = traj['pos'].at[i].set(state['pos'])
            traj['vel'] = traj['vel'].at[i].set(state['vel'])
            return state, traj
        traj = {}
        traj['pos'] = jnp.zeros([self.nsteps//self.nout, self.natoms, 3])
        traj['vel'] = jnp.zeros([self.nsteps//self.nout, self.natoms, 3])
        state, traj = jax.lax.fori_loop(0, self.nsteps//self.nout, ode_fwd, (state, traj))
        return traj
    
    def save_dcd(self, traj, dcdfile):
        u = md.load(self.init_stru)
        uu = u
        for pos in traj['pos']:
            uu.xyz = pos
            u = u.join(uu)
        u = u[1:]
        u.save_dcd(dcdfile)
        return 

    def run_nvt(self, state, params, key, logfile, dcdfile):
        traj = {}
        traj['pos'] = []
        traj['vel'] = []
        ifile = open(logfile, 'w')
        u = md.load(self.init_stru)
        uu = u
        print('#"Time (ps)","Potential Energy (kJ/mole)","Kinetic Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)"', file=ifile)
        subkeys = random.split(key, num=self.nsteps//self.nout)
        for i in range(self.nsteps//self.nout):
            potE = self.potential(state['pos'], self.box, params)
            kinE = jnp.sum(self.m*state['vel']**2/2)
            temp = (jnp.std(state['vel']*jnp.sqrt(self.m), ddof=-1)*1e3)**2/(1e3*KB*NA)
            print('%s, %s, %s, %s, %s'%(self.nout*self.dt*i, potE, kinE, potE+kinE, temp), file=ifile)
            state = self.vv_nout(state, params, subkeys[i])
            traj['pos'].append(state['pos'])
            traj['vel'].append(state['vel'])
            uu.xyz = state['pos']
            u = u.join(uu)
        ifile.close()
        traj['pos'] = jnp.array(traj['pos'])
        traj['vel'] = jnp.array(traj['vel'])
        u = u[1:]
        u.save_dcd(dcdfile)
        return traj


def calculate_corr(A: jnp.ndarray, B: jnp.ndarray, NMAX: int):
    """
    Calculate the correlation function: `corr(t) = <A(0) * B(t)>`. 
    Here, `A(t)` and `B(t)` are arrays of the same dimensions, 
    and `A(t) * B(t)` is the element-wise multiplication. 
    The esenmble average `< >` is estimated by moving average.

    Parameters
    -----
    A, B: jnp.ndarray, in shape of (num_t, ...).
        The first dimension refers to the time steps, and its size can be different.
        The remaining dimensions (if present) must be the same.
    
    NMAX: int.
        Maximal time steps. Calculate `corr(t)` with `0 <= t <= NMAX`.
    
    window: int, optional.
        The width of window to do the moving average. 

        `<A(0) * B(t)> = 1 / window * \sum_{i = 0}^{window - 1} A(i) * B(t + i)`. 

    Return
    -----
    corr: np.ndarray, in shape of (NMAX + 1, ...).
        `corr(t) = <A(0) * B(t)>`
    """

    window = min(A.shape[0], B.shape[0] - NMAX)
    # Prepare for convolution
    v1 = A[:window][::-1]; v2 = B[:window + NMAX]
    pad_width = [(0, 0)] * A.ndim
    pad_width[0] = (0, NMAX)
    v1 = jnp.pad(v1, pad_width, "constant", constant_values = 0)
    # Convolve by FFT
    corr = jnp.fft.ifft(jnp.fft.fft(v1, axis = 0) * jnp.fft.fft(v2, axis = 0), axis = 0).real # type: ignore
    # Moving average
    corr = corr[window - 1:window + NMAX] / window
    return corr



def calculate_corr_vdipole(dipole, dt_ps, window):
    v_dipole = (dipole[1:] - dipole[:-1]) / dt_ps # type: ignore
    v_dipole -= jnp.mean(v_dipole, axis = 0, keepdims = True)
    corr = jnp.sum(calculate_corr(v_dipole, v_dipole, window), axis = -1)
    return corr

def apply_gussian_filter(corr: jnp.ndarray, width: float):
    """
    Apply gaussian filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    return corr * jnp.exp(-.5 * (0.5 * width * jnp.arange(nmax + 1) / nmax)**2)

def FT(DT: float, C: jnp.ndarray, M: Optional[int] = None) -> jnp.ndarray:
    """
    The same as FILONC while `DOM = 2\pi / (M * DT)` (or `OMEGA_MAX = 2\pi / DT`).
    This is implemented by FFT.

    Parameters
    -----
    C: ndarray, the correlation function.
    DT: float, time interval between points in C.
    M: Optional[int], number of intervals on the frequency axis.
    `M = NMAX` by default.

    Return
    -----
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d cosine transform.
    """
    NMAX = C.shape[0] - 1
    assert NMAX % 2 == 0, 'NMAX is not even!'
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    DTH = 2 * jnp.pi / M
    NU = jnp.arange(M + 1)
    THETA = NU * DTH
    SINTH = jnp.sin(THETA)
    COSTH = jnp.cos(THETA)
    SINSQ = jnp.square(SINTH)
    COSSQ = jnp.square(COSTH)
    THSQ  = jnp.square(THETA)
    THCUB = THSQ * THETA
    ALPHA = 1. * ( THSQ + THETA * SINTH * COSTH - 2. * SINSQ )
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    ALPHA = ALPHA.at[0].set(0.)
    BETA = BETA.at[0].set(2. / 3.)
    GAMMA = GAMMA.at[0].set(4. / 3.)
    ALPHA = ALPHA.at[1:].divide(THCUB[1:])
    BETA = BETA.at[1:].divide(THCUB[1:])
    GAMMA = GAMMA.at[1:].divide(THCUB[1:])
    CE, CO = _FFT_OE(C, DTH, M)
    CE -= 0.5 * (C[0] + C[NMAX] * jnp.cos(THETA * NMAX))
    CHAT = 2.0 * (ALPHA * C[NMAX] * jnp.sin ( THETA * NMAX ) + BETA * CE + GAMMA * CO) * DT
    freq = 1 / (M * DT)
    return freq, CHAT

def _FFT_OE(C: np.ndarray, DTH: float, M: int):
    NMAX = C.shape[0] - 1
    NU = jnp.arange(M + 1)
    THETA = NU * DTH
    # Even coordinates
    CE = _range_fft(C[:-1:2], n = int(M / 2)).real # type: ignore
    CE = jnp.concatenate([CE, CE, CE[0:1]]) + C[NMAX] * jnp.cos(THETA * NMAX)
    # Odd coordinates
    CO = (_range_fft(C[1::2], n = int(M / 2)) * jnp.exp(-THETA[:int(M / 2)] * 1j)).real # type: ignore
    CO = jnp.concatenate([CO, -CO, CO[0:1]])
    return CE, CO

def _range_fft(a: jnp.ndarray, n: Optional[int] = None, axis: int = -1):
    """
    Compute `a_hat[..., l, ...] = \sum_{k=1}^{a.shape[axis]} a[..., k, ...]e^{-(2kl\pi/n)}`
    """
    axis %= a.ndim
    l = a.shape[axis]
    if n is None:
        n = l
    if n >= l:
        return jnp.fft.fft(a, n, axis)
    num_n = int(l / n)
    l0 = n * num_n
    new_shape = list(a.shape)
    new_shape[axis] = n
    new_shape.insert(axis, num_n)
    a_main = jnp.sum(a.take(range(l0), axis).reshape(new_shape), axis)
    a_tail = a.take(range(l0, l), axis)
    return jnp.fft.fft(a_main, n, axis) + jnp.fft.fft(a_tail, n, axis)

def calculate_ir(corr: jnp.ndarray, width: float, dt_ps: float, temperature: float, 
                 M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    filter_type = filter_type.lower().strip()
    print("nmax         =", nmax)
    print("dt   (ps)    =", dt_ps)
    print("tmax (ps)    =", tmax)
    print("Filter type  =", filter_type)
    print("Smooth width =", width)
    width = width * tmax / 100.0 * 3
    C = apply_gussian_filter(corr, width)
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = _change_unit(freq_ps, CHAT, temperature)
    return jnp.arange(CHAT.shape[0]) * d_omega, CHAT

def _change_unit(freq_ps, CHAT: jnp.ndarray, temperature: float):
    a0 = 1e-9  # m
    cc = 2.99792458e8;      # m/s
    kB = 1.38064852*1.0e-23 # J/K
    h = 6.62607015e-34      # J*s
    beta = 1.0 / (kB * temperature); 
	# 1 Debye = 0.20819434 e*Angstrom
	# 1 e = 1.602*1.0e-19 C
	# change unit to C*m for M(0)
    unit_basic = 1.602176565 * 1.0e-19 * a0
	# change unit to ps for dM(0)/dt
    unitt = unit_basic / 1
	# because dot(M(0))*dot(M(t)) change unit to C^2 * m^2 / ps^2
    unit2 = unitt**2
    epsilon0 = 8.8541878e-12 # F/m = C^2 / (J * m)
    unit_all = beta / (3.0 * cc * a0 ** 3) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-2; # ps to s, m-1 to cm-1
    CHAT *= unit_all
    d_omega = freq_ps / cc     # Wavenumber
    d_omega *= 1e10         # cm^-1
    return d_omega, CHAT








