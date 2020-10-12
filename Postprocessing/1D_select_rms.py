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
import glob

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
# be careful if save_netcdf: outfile is removed before creating a new one
run = ''
NCPATH = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/1D'
folder_save = NCPATH +run
save_plot = True
outfile = folder_save + "/mod1D_MargLaw_test.nc"
save_netcdf = True

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(NCPATH + '/mod1D_Bolivia_001.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
logrhosynth =  np.log10(np.array(nc.variables['rho_i'][:])[0,0,:])
nc.close()
print "Ellapsed time reading netcdf file:", time.clock() - t0
print''
#---------------------------------------------------------------------------
# MT data counting (needed for rms option)
os.chdir(NCPATH+'/data/')
ndata=0
for fn in glob.glob('*.ro*'):
	with open(fn) as f:
		ndata=ndata+sum(1 for line in f if line.strip() and not line.startswith('#'))

print "number of data for rms: ", "{:e}".format(ndata)
print''
# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))
#m_gbest = models[i_gbest[0], : , i_gbest[1]]
#m_gbest = m_gbest[0]


# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = pp.NaN_filter(models, energy)
del models, energy 

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 
nruns, popsize, nparam, niter = filt_models.shape
print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)
print''

threshold =np.max(filt_energy)# np.median(filt_energy)
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
mpow_weight = np.log10(pp.weighted_mean(m_grid, ndata, f_grid, kappa=1, rms=True, log=False))
print "Mean-difference between log and physical space :", np.max(np.abs(mpow_weight - m_weight))
print''

# ---> Xi2 weighted STD model, in log and physical space
std_weight = pp.weighted_std(m_weight, m_grid, f_grid, ndata, kappa=1, rms=True, log=True)
stdpow_weight = pp.weighted_std(10**mpow_weight, m_grid, f_grid, ndata, kappa=1, rms=True, log=False)
print "STD-difference between log and physical space :", np.max(np.abs(stdpow_weight - std_weight))
print''

# ---- marginal laws using kappa damping coefficient
n_inter = 40
lower = -2.  
upper = 2. 
kappa = 1

pdf_m, n_bin, x_bin = pp.marginal_law(m_grid, f_grid, logrhosynth, ndata, n_inter=n_inter,lower=lower, upper=upper, kappa=kappa, rms=True)

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
    nc.createVariable('mpow_weight', 'f8', ('nparam'))
    nc.createVariable('std_weight', 'f8', ('nparam'))
    nc.createVariable('stdpow_weight', 'f8', ('nparam'))
    nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('n_bin', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('x_bin', 'f8', ('nparam', 'n_inter'))
    # filling values
    nc.variables['m_grid'][:, :] = m_grid
    nc.variables['f_grid'][:] = f_grid
    nc.variables['m_synth'][:] = logrhosynth
    nc.variables['m_weight'][:] = m_weight
    nc.variables['mpow_weight'][:] = mpow_weight
    nc.variables['std_weight'][:] = std_weight
    nc.variables['stdpow_weight'][:] = stdpow_weight
    nc.variables['pdf_m'][:, :] = pdf_m
    nc.variables['x_bin'][:, :] = x_bin
    nc.variables['n_bin'][:, :] = n_bin
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
    for ipar in range(nparam):
        fig=plt.figure()
        valpar=round(logrhosynth[ipar],2)
        meanpar=round(m_weight[ipar],2)
        Sbin=sum(n_bin[ipar, :])
        hist=np.empty(int(Sbin))
        for j in range(len(n_bin[ipar, :])):
            hist[int(sum(n_bin[ipar, 0:j])):int(sum(n_bin[ipar, 0:j+1]))]=x_bin[ipar, j]
        ax1 = fig.add_subplot(111)
        binplot=np.hstack((x_bin[ipar, :],x_bin[ipar,-1]+np.diff(x_bin[ipar, :])[-1]))-np.diff(x_bin[ipar, :])[0]/2.
        ax1.hist(hist,bins=binplot)
        ax1.set_ylabel('Nbr of model', color='b',fontsize=10)
        ax1.set_xlabel('Resistivity (Log-scale)',fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b',labelsize=8)
        ax1.tick_params(axis='x',labelsize=8)
        ax2 = ax1.twinx()
        ax2.axvline(x=valpar, color='y',label='synth')
        ax2.axvline(x=meanpar, color='g',label='mean')
        ax2.plot(x_bin[ipar, :], pdf_m[ipar, :], 'r')
        ax2.set_ylabel('Probability', color='r',fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
        align_yaxis(ax1, 0, ax2, 0)
        ax2.legend(loc=0,fontsize=10)
        plt.suptitle('Parameter '+str(ipar+1)+' <Rho>:'+str(meanpar)+'$\Omega.m$ (log scale), STD:'+str(round(std_weight[ipar],2)),fontsize=12)
        plt.savefig(folder_save + '/' + 'Parameter_'+str(ipar+1)+'.png')
        plt.clf()



        #plt.subplot(2, 1, 1)
        #plt.plot(x_bin[iparam, :], pdf_m[iparam, :], 'r')
        #plt.axvline(x=logrhosynth[iparam], color='red')
        #plt.axvline(x=m_weight[iparam], color='blue')
        #plt.xlim([-1, 1])
        #plt.subplot(2, 1, 2)
        #plt.plot(x_bin[iparam, :], n_bin[iparam, :])
        #plt.xlim([-1, 1])
        #plt.savefig(folder_save + '/' + "fun_diff"+  str(int(iparam)) )
        #plt.clf()

