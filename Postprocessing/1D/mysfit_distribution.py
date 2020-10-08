# python 2.7
'''
plot mysfit pdf to see range values 

'''

#-----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

sys.path.append('../Postprocessing/')
import cpso_pp as pp

# ----------------------------------------------------------------------------

run = '1D_MCM_ana_8param'
cpso_path = '/postproc/COLLIN/MTD3/' + run
folder_save = cpso_path + '/Analysis'  
save_plot = True
figname = folder_save + '/mysfit_distribution.png'

# --- load data
t0 = time.clock()
nc = Dataset(cpso_path + '/merged.nc')
energy = np.array(nc.variables['energy'][:])
nc.close()

print "Ellapsed time reading netcdf file:", time.clock() - t0
print''
# ----------------------------------------------------------------------------

plt.figure()
plt.hist(energy)
plt.savefig(figname)
plt.show()
