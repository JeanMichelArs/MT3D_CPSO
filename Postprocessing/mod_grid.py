# python 2.7
'''
JCollin 01-2020

script to select models in order to avoid oversampling in local minimas
Let's assume we dicretize the parameter space by Delta_m regular spacing 
m_select = {mi | || mi - mj ||_inf >= Delta_m }

TODO:
- add timing + __name__ is main ! + write in hour/ minute / sec format
- put functions in a module
'''
#-----------------------------------------------------------------------------
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cpso_pp as pp
# ----------------------------------------------------------------------------
# be careful if save_netcdf: outfile is removed before creating a new one
run = 'Bolivia_115param_015'
NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run
save_plot = True
folder_save = NCPATH + '/Postprocessing'
save_plot = True
outfile = folder_save + "/m_grid.nc"
save_netcdf = True

# --- create directory to save plots 
if not os.path.exists(folder_save):
    os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(NCPATH + '/merged.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
nc.close()
print "Ellapsed time reading netcdf file:", time.clock() - t0

# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))
m_gbest = models[i_gbest[0], i_gbest[1], : , i_gbest[2]]
m_gbest = m_gbest[0]

# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = pp.NaN_filter(models, energy)
del models, energy 

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 
nruns, popsize, nparam, niter = pp.filt_models.shape
print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)

# --- > regrid parameter space
delta_m = 1e-3 
m_grid, f_grid, rgrid_error = pp.regrid(filt_models, filt_energy, delta_m, center=True)
print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest

# ---> save m_grid, f_grid, r_grid_error, m_gbest, 
#      f_best, delta_m 
if save_netcdf:
    if os.path.isfile(outfile):
        os.remove(outfile)
        print "remove ", outfile
    nc = Dataset(outfile, "w", format='NETCDF4')
    # dimensions: name, size
     
    nc.createDimension('nruns', m_grid.shape[0])
    nc.createDimension('popsize', m_grid.shape[1])
    nc.createDimension('nparam', m_grid.shape[2])
    nc.createDimension('niter', m_grid.shape[3])
    # Variables: name, format, shape
    nc.createVariable('m_grid', 'f8', ('nruns', 'popsize', 'nparam', 'niter'))
    nc.createVariable('f_grid', 'f8', ('nruns', 'popsize', 'niter'))
    nc.createVariable('m_gbest', 'f8', ('nparam'))
    # filling values
    nc.variables['m_grid'][:, :, :, :] = m_grid
    nc.variables['f_grid'][:, :, :] = f_grid
    nc.variables['m_gbest'][:] = m_gbest
    nc.rgrid_error = rgrid_error
    nc.f_gbest = f_gbest
    nc.delta_m = delta_m 
    print outfile, "saved sucessfully" 
    nc.close()
