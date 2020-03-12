# python 2.7
'''
JCollin 01-2020

Script to plot mysfit function

@todo : 
    - plot best mysfit 
'''
#-----------------------------------------------------------------------------
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
NCPATH = '/home2/scratch/jcollin/MT3D_CPSO/xstart_xi_Mackie_33param/'

nc = Dataset(NCPATH + 'cpso_mackie_33_param.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
nc.close()

# --- BEST MISFIT

best_fit = np.array(np.min(energy, 0))

plt.plot(best_fit)
#plt.ylim((0, 100))
plt.xlabel('iteration')
plt.ylabel('best fit')
plt.title('Mackie 33 param')
plt.show()

# --- BEST fit for each model

model_bfit = np.array(np.min(energy, 1))

plt.plot(model_bfit)
plt.xlabel('swarm')
plt.ylabel('best fit')
plt.title('Mackie 33 param')
plt.show()

# --- BEST models

indx = np.argmin(energy, 1)
best_models = np.zeros((models.shape[0], models.shape[1]))

for iswarm in np.arange(models.shape[0]):
    best_models[iswarm, :] = models[iswarm, : , indx[iswarm]]

fig = plt.contourf(best_models)
cbar = plt.colorbar()
plt.show()

