#!/usr/bin/env python3
import os
import sys
import numpy as np
import MDAnalysis as mda

ifn = sys.argv[1]

u = mda.Universe(ifn)
a, b, c, alpha, beta, gamma = u.dimensions

print('TITLE cell{angstrom} positions{angstrom}') # everything in angstrom
print('CRYST%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1          1'%(a,b,c,alpha,beta,gamma))
i_atom = 1
for atom in u.atoms:
    if atom.mass > 1e-6: # not virtuals
        r = atom.position
        print('ATOM%7d%5s   1     1%12.3f%8.3f%8.3f  0.00  0.00            0'%(i_atom, atom.type, r[0], r[1], r[2]))
        i_atom += 1
print('END')


