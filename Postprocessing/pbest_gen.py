# python 2.7
'''
JCollin 01-2020

- added number of particle that can be used for statistics
- added zoom over the 100 last iteration


Script to save best mysifit
Both merged and single runs are supported

RUNDIR_XXX and /home2/datawork/jcollin/MT3D_CPSO/multiple_cpso/merged.nc are set in Makefile

to run :
> make p_best.nc

'''
#-----------------------------------------------------------------------------
import os
import numpy as np
from netCDF4 import Dataset
#import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
RUNDIR = "RUN_DIR_XXX"
nc_file = "nc_file_xxx"
out_file = RUNDIR + '/' + "p_best.nc"
niter_zoom = 0
debug = False
threshold = 40
# ----------------------------------------------------------------------------
nc = Dataset(nc_file)
energy = np.array(nc.variables['energy'][:])
nc.close()

# --- BEST MISFIT
niter = energy.shape[-1]

print "--------------------------------"
print " total number of iterations", niter 

if len(energy.shape) == 2:
	best_fit = np.array(np.min(energy, 0))
	npart = energy.shape[0] * energy.shape[1]
elif len(energy.shape) == 3:
	best_fit = np.array(np.min(np.min(energy, 0),0))
	npart = energy.shape[0] * energy.shape[1] * energy.shape[2]

if debug:
    print niter_zoom
    print best_fit[niter-niter_zoom:].shape

# ---> number of particles for statistics
np_stats = np.sum(energy - np.min(best_fit) <= threshold)
print "Best mysfits", np.min(best_fit)
print "number of particles for statistics ", np_stats 
print "total number of particles", npart 
# --> save date in out_file
if os.path.isfile(out_file):
    os.remove(out_file)

nc = Dataset(out_file, "w", format='NETCDF4')
nc.np_stats = np_stats
nc.createDimension('iter', niter)
nc.createDimension('iter_zoom', niter_zoom)
nc.createVariable('best_fit', 'f8', ('iter'))
nc.createVariable('best_fit_zoom', 'f8', ('iter_zoom'))
nc.variables['best_fit'][:] = best_fit
nc.variables['best_fit_zoom'][:] = best_fit[niter-niter_zoom:] - best_fit[-1]
nc.close()


