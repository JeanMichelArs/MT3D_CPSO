# python 2.7
'''
Script to analyze the convergence of pdf

basically use 1D_select_rms.py to compute pdf and loop over number of runs to 
get ||pdf(nrun, iparam) - pdf(irun,iparam)|| with nrun great enough to see a 
convergence

From a quick look at pdf it appears that 20 runs seems to be already enough 
to observe a relevant result


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

sys.path.append('../../Postprocessing/')
import cpso_pp as pp

# ----------------------------------------------------------------------------


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
# ----------------------------------------------------------------------------
# TAKE DATA FROM CPSO
# be careful if save_netcdf: outfile is removed before creating a new one
# cpso_path : cpso output
# conf_dir : configuration files
# folder_save : parameter uncertainty estimates

run = 'Bolivia_1D_8param'
cpso_path = '/postproc/COLLIN/MTD3/' + run
conf_dir = '/home/j/jcollin/MT3D_CPSO/Config/1D'
folder_save = cpso_path + '/Analysis'  
save_plot = True
outfile = folder_save + "/.pdf_convergence.nc"
save_netcdf = True

vec_run = [1, 5, 10, 20, 50, 100, 200]
nparam = 8

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(cpso_path + '/200runs_merged.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
logrhosynth =  np.log10(np.array(nc.variables['rho_i'][0,0,0,:]))
nc.close()

print "Ellapsed time reading netcdf file:", time.clock() - t0
print''


#---------------------------------------------------------------------------
# MT data counting (needed for rms option)
os.chdir(conf_dir)
ndata = 0
for fn in glob.glob('*.ro*'):
	with open(fn) as f:
		ndata=ndata+sum(1 for line in f if line.strip() and not line.startswith('#'))

print "number of data for rms: ", "{:e}".format(ndata)
print''

# ----------------------------------------------------------------------------
# Loop over number of runs (pdf)
n_inter = 40
lower = -2.
upper = 2.
kappa = 1
pdf_m = np.empty(shape=(len(vec_run), nparam, n_inter))
indx = 0

for irun in vec_run:

  # -------------------------------------------------------------------------
  # global data
  i_gbest = np.where(energy[:irun, : , :] == np.min(energy[:irun, :]))
  f_gbest = np.min(energy[:irun, :, :])

  # ----------------------------------------------------------------------------
  # First filter NaN values in case run did not finish 
  filt_models, filt_energy, niter = pp.NaN_filter(models[:irun, : , :, :],
                                                  energy[:irun, : , :])
  # --->  Prefilter models according to parameter space regular subgriding
  # Care must be taken that we do this parameter log 
  nruns, popsize, nparam, niter = filt_models.shape
  print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)
  print''

  threshold = np.max(filt_energy) # np.median(filt_energy)
  m_near, f_near = pp.value_filter(filt_models, filt_energy, threshold)

  # --- > regrid parameter space
  delta_m = 1e-3 
  m_grid, f_grid, rgrid_error = pp.regrid(m_near, f_near, delta_m, center=True)
  del m_near, f_near
  print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
  print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest
  print''

  # ---> Xi2 weighted mean model, in log and physical space
  m_weight = pp.weighted_mean(m_grid, f_grid, ndata, kappa=1, rms=True, log=True)

  # ---> Xi2 weighted STD model, in log and physical space
  std_weight = pp.weighted_std(m_weight, m_grid, f_grid, ndata, kappa=1, rms=True, log=True)

  # ---- marginal laws using RMS
  indx = indx + 1
  pdf_m[indx, :, :], n_bin, x_bin = pp.marginal_law(m_grid, f_grid, logrhosynth, ndata,
                                        n_inter=n_inter,lower=lower, upper=upper,
                                        kappa=kappa, rms=True)

# ----------------------------------------------------------------------------
# Norme L2 Error
# | pdf(nruns, iparam) - pdf(irun, ipara)|_2

pdf_error = np.empty(shape=(len(vec_run), nparam))
for irun in range(len(vec_run)):
    pdf_error[irun, :] = np.linalg.norm(pdf_m[irun, :] - pdf_m[-1, :], 2)

plt.figure()
# ---> plot selection
for iparam in range(nparam):
    plt.plot(vec_run, pdf_error[:, iparam])
    plt.xlabel('number of runs')
plt.show()
