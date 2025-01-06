#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use(['science','no-latex', 'nature'])

datafile = sys.argv[1]
data = np.loadtxt(datafile)

#data = np.loadtxt(datafile+'simulation_8p.md')
#data = np.loadtxt(datafile+'simulation.md')

step = data.T[0]
t = data.T[1]
# T = data.T[2]
# e = data.T[3]
T = data.T[3]
e = data.T[-1]
rou = data.T[-1]
ave = np.mean(rou[int(len(rou)*3/4):])
std = np.std(rou[int(len(rou)*3/4):])

# Create a figure with two subplots (arranged vertically)
fig, axs = plt.subplots(1, 1, figsize=(5, 5*0.618))

# Plot the first subplot (top)
axs.plot(t, rou, color='crimson')
axs.axhline(y=0.99713, color='blue', linestyle='--', label='Experiment')
axs.set_xlabel('Time (ps)')
axs.set_ylabel('Density (g/cm$^3$)')
axs.set_title('Density in 25 Degree')
axs.legend()
axs.text(0.98, 0.05, 'Density = %.4f ± %.4f (g/mL)'%(ave,std), verticalalignment='bottom', horizontalalignment='right',
        transform=axs.transAxes)
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('res_density.png', dpi=300)
plt.show()
