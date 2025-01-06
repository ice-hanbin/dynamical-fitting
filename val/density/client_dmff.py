#!/usr/bin/env python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
import sys
import driver
import numpy as np
import jax
#import jax._src.array
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from dmff.utils import jit_condition
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
import pickle
import nblist

from jax.config import config
config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

import dmff
from eann import EANNForce

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, f_xml, r_xml, psr, socktype, device='cpu'):
        addr = addr + '_%s'%os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)

        pdb = app.PDBFile(pdb)
        rc = 6

        # construct inputs
        positions = jnp.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10

        # params = {
        #         'pos': positions,
        #         'box': box,
        #         }
        # with open('pos_box_0.pickle', 'wb') as f:
        #     pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

        # neighbor list
        cov_map = jnp.array(np.loadtxt('cov_map'), dtype=jnp.int64)
        nbl = nblist.NeighborList(box, rc, cov_map)
        nbl.allocate(positions)
        
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
        with open(psr, 'rb') as ifile:
            params_eann = pickle.load(ifile)['energy']

        def admp_calculator(positions, L):
            # print('admp_calculator')
            box = jnp.array([[L,0,0],[0,L,0],[0,0,L]])          
            # add by Junmin, May 27, 2023

            nblist = nbl.nblist.update(positions, box=box)
            pairs = nblist.idx.T
            nbond = cov_map[pairs[:, 0], pairs[:, 1]]
            pairs_b = jnp.concatenate([pairs, nbond[:, None]], axis=1)

            E_eann = jnp.array(eann_force.get_energy(positions, box, pairs_b, params_eann))
            
            return E_eann

        self.tot_force = jit(jax.value_and_grad(admp_calculator,argnums=(0,1)))
        # print('jit')
        L = box[0][0]
        # compile tot_force function
        energy, (grad, virial) = self.tot_force(positions, L)
        # print('tot_force')
        # print(energy)
        # print(grad)
        # print(virial)


    def grad(self, crd, cell): # receive SI input, return SI values
        # print('grad')
        positions = jnp.array(crd*1e10) # convert to angstrom
        box = jnp.array(cell*1e10)      # convert to angstrom
        
        # params = {
        #         'pos': positions,
        #         'box': box,
        #         }
        # with open('pos_box_0.pickle', 'wb') as f:
        #     pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL) 

        # nb list
        L = box[0][0]
        energy, (grad, virial) = self.tot_force(positions, L)
        # print(energy)
        # print(grad)
        # print(virial)
        virial = np.diag((-grad * positions).sum(axis=0) - virial*L/3).ravel()

        # convert to SI
        energy = np.array(energy * 1000 / 6.0221409e+23) # kj/mol to Joules
        grad = np.array(grad * 1000 / 6.0221409e+23 * 1e10) # convert kj/mol/A to joule/m
        virial = np.array(virial * 1000 / 6.0221409e+23) # kj/mol to Joules
        return energy, grad, virial


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
    f_xml = sys.argv[2] # xml file that defines the force field
    r_xml = sys.argv[3] # xml file that defines residues
    psr = sys.argv[4]
    addr = sys.argv[5]
    port = int(sys.argv[6])
    socktype = sys.argv[7] 
    driver_dmff = DMFFDriver(addr, port, fn_pdb, f_xml, r_xml, psr, socktype)
    while True:
        driver_dmff.parse()


