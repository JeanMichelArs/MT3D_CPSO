# python 2.7
'''
compute <m> = sum(m * exp(F(-m))
        pdf(m_i)
        std(m_i)

works for MT1D candidates 
parameter space exploration can be performed through cpso or mcm algorithm

Main issue is that models for MCM and CPSO have different shapes
CPSO models (popsize, n_dim, max_iter)
MCM models (max_iter, n_dim)

>>> python 1D_select_rms.py

'''

#-----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')

"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
"""

from scipy.interpolate import griddata
import glob

sys.path.append('../Postprocessing/')
sys.path.append('../../Forward_MT/')
import cpso_pp as pp
from forward_1d import MT1D_analytic


# ----------------------------------------------------------------------------
# be careful if save_netcdf: outfile is removed before creating a new one
# cpso_path : cpso output
# conf_dir : configuration files
# folder_save : parameter uncertainty estimates

# ---> inputs
method = 'mcm'
rms = False
nruns = 2

cpso_path = '/postproc/COLLIN/MTD3/MCM_4nz_cst_Error'
conf_dir = '../../Config/1D/model_000'
#cpso_path = '/home/ars/Bureau'
#conf_dir = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/1D/model_000'
data_file = conf_dir + '/000.ro'
model_file = conf_dir + '/mod1D_Bolivia_000'
exploration_file = cpso_path + '/merged.nc'

# ---> outputs
# plots: fig_pdf is a generic name for pdf
folder_save = cpso_path + '/Analysis'  
save_plot = True
fig_pdf = folder_save + '/pdf_m_nruns' + str(nruns) + '_' 
fig_vert = folder_save + '/vert_pro_nruns' + str(nruns) + '.png' 

save_netcdf = True
outfile = folder_save + "/pdf_m_" + str(nruns) + ".nc"

# ---> postproc
n_inter = 40
lower = -2.
upper = 2.
kappa = 1

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(exploration_file)
if method is 'cpso':
    energy = np.array(nc.variables['energy'][:nruns, :, :])
    models =  np.array(nc.variables['models'][:nruns, :, :, :])
    logrhosynth =  np.squeeze(np.log10(np.array(nc.variables['rho_i'][0, :])))
elif method is 'mcm':
    energy = np.array(nc.variables['energy'][:nruns, :])
    models =  np.array(nc.variables['models'][:nruns, :, :])
    logrhosynth =  np.squeeze(np.log10(np.array(nc.variables['rho_i'][0, :])))
else: 
    print "Error method is not defined"

nc.close()
print "Ellapsed time reading netcdf file:", time.clock() - t0
print''

#---------------------------------------------------------------------------
# MT data counting (needed for rms option)
ndata = pp.get_ndata(data_file)
print "number of data for rms: ", "{:e}".format(ndata)
print''
per, Rz, Iz, Erz, rho, Erho, phi, Ephi = np.loadtxt(data_file, unpack=True)

#---------------------------------------------------------------------------
# Read MT1D model
hz, rhosynth = np.loadtxt(model_file, unpack=True)
print''

# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))

# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = pp.NaN_filter(models, energy)
del models, energy 

# check for mcm exloration debug
nparam = filt_models.shape[2]
Err = np.zeros(shape=(nparam,))
for iparam in range(nparam):
    Err[iparam] = np.max(np.abs(filt_models[:, :, iparam] - logrhosynth[iparam]))
if (Err > upper).any(): print "Error models out of window", Err

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 

if method is 'cpso':
    nruns, popsize, nparam, niter = filt_models.shape
    print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)
elif method is 'mcm':
    nruns, niter, nparam = filt_models.shape
    print "number of models to filter: ", "{:e}".format(nruns * niter)
else : print "Error undefined method"

print''

# ---> filter extreme value old...
threshold = np.max(filt_energy) 
m_near, f_near = pp.value_filter(filt_models, filt_energy, threshold)

# check
# Error mcm exploration detected
np.max(np.abs(m_near - logrhosynth))

# --- > regrid parameter space
delta_m = 1e-3 
m_grid, f_grid, rgrid_error = pp.regrid(m_near, f_near, delta_m, center=True)
del m_near, f_near
print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest
print''

print "max distance between m_grid and logrhosynth :"
print np.max(np.abs(m_grid - logrhosynth))

# ---> Xi2 weighted mean model, in log and physical space
m_weight = pp.weighted_mean(m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=True)

# ---> Xi2 weighted STD model, in log and physical space
std_weight = pp.weighted_std(m_weight, m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=True)
print''

# ---- marginal laws centered around solution 
pdf_m, n_bin, x_bin = pp.marginal_law(m_grid, f_grid, logrhosynth, ndata,
                       n_inter=n_inter,lower=lower, upper=upper, kappa=kappa,
                       rms=rms)

print "check marginal law intervals and window for each parameter"
print "should be ", m_grid.shape[0]
print np.sum(n_bin, axis=1)

# ---- Misfit of the Mean Model 
zc, rhoc, phic = MT1D_analytic(hz, 10**m_weight, per)
#---------------------------------------------------
#COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
#---------------------------------------------------
XHI2=(sum((Rz-np.real(zc))**2/Erz**2)+sum((Iz-np.imag(zc))**2/Erz**2))/2
print ''
print 'Magnetotelluric Misfit for Mean Model=>',XHI2
print 'While Number of data => ', ndata*2
print ''

# ---> save m_grid, f_grid, r_grid_error, m_gbest, 
#      f_best, delta_m , xbin, n_bin, pdf_m, m_weight, m_pow
#      kappa, lower, upper
if save_netcdf:
    if os.path.isfile(outfile):
        os.remove(outfile)
        print "remove ", outfile
    nc = Dataset(outfile, "w", format='NETCDF4')
    # dimensions: name, size
    nc.createDimension('popsize', f_grid.shape[0])
    nc.createDimension('nparam', m_grid.shape[1])
    nc.createDimension('n_inter', pdf_m.shape[1])
    # Variables: name, format, shape
    nc.createVariable('m_grid', 'f8', ('popsize', 'nparam'))
    nc.createVariable('f_grid', 'f8', ('popsize'))
    nc.createVariable('m_synth', 'f8', ('nparam'))
    nc.createVariable('m_weight', 'f8', ('nparam'))
    nc.createVariable('std_weight', 'f8', ('nparam'))
    nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('n_bin', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('x_bin', 'f8', ('nparam', 'n_inter'))
    # filling values
    nc.variables['m_grid'][:, :] = m_grid
    nc.variables['f_grid'][:] = f_grid
    nc.variables['m_synth'][:] = logrhosynth
    nc.variables['m_weight'][:] = m_weight
    nc.variables['std_weight'][:] = std_weight
    nc.variables['pdf_m'][:, :] = pdf_m
    nc.variables['x_bin'][:, :] = x_bin
    nc.variables['n_bin'][:, :] = n_bin
    nc.mysfit_mweight = XHI2
    nc.ndata = ndata*2
    nc.rgrid_error = rgrid_error
    nc.f_gbest = f_gbest
    nc.delta_m = delta_m 
    nc.kappa = kappa
    nc.lower = lower
    nc.upper = upper
    print outfile, "saved sucessfully" 
    nc.close()

if save_plot:
    print "plot results"
    # marginal laws
    pp.plot_pdfm(gen_name=fig_pdf, pdf_m=pdf_m, x_bin=x_bin, n_bin=n_bin,
              m_synth=logrhosynth, m_weight=m_weight, std_weight=std_weight,
              transparent=False)
    # vertical profile of <m> vs logrhosynth
    pp.vertical_profile(figname=fig_vert, pdf_m=pdf_m, m_weight=m_weight,
                 logrhosynth=logrhosynth, hz=hz, x_bin=x_bin, 
                 transparent=False, cut_off=1e-3) 

