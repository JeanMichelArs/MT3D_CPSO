# python 2.7
'''
computes statistics straight from exploration outputs without filter or regrid
it is correct for mc but doubtfull for cpso 

advantage is the fewer computation and lesser memory RAM usage

compute <m> = sum(m * exp(F(-m))
        pdf(m_i)
        std(m_i)

works for MT1D candidates 

>>> python raw_pdf.py

'''

#-----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')

from scipy.interpolate import griddata
import glob

sys.path.append('../')
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
nruns = 64
idd = '002'
cpso_path = '/home2/scratch/jcollin/MT3D_CPSO/mcm_16nz'

conf_dir = '../../Config/1D/model_' + idd
#cpso_path = '/home/ars/Bureau'
#conf_dir = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/1D/model_000'
data_file = conf_dir + '/' + idd + '.ro'
model_file = conf_dir + '/mod1D_Bolivia_' + idd 
exploration_file = cpso_path + '/merged.nc'

# ---> outputs
# plots: fig_pdf is a generic name for pdf
config = 'raw'
folder_save = cpso_path + '/raw'  
save_plot = False
fig_pdf = folder_save + '/pdf_m_nruns' + str(nruns) + '_' + config + '_' 
fig_vert = folder_save + '/vert_pro_nruns' + str(nruns) + '_' + config + '.png' 

save_netcdf = True
outfile = folder_save + "/pdf_m_" + str(nruns) + config + ".nc"

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

# ---> Xi2 weighted mean model, in log and physical space
m_weight = pp.raw_weighted_mean(models, energy, ndata, rms=rms)

"""
# ---> Xi2 weighted STD model, in log and physical space
std_weight = pp.weighted_std(m_weight, m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=True)
print''
"""

# ---- marginal laws centered around solution 
pdf_m, n_bin, x_bin = pp.raw_marginal_law(models, energy, logrhosynth, ndata,
                       n_inter=n_inter,lower=lower, upper=upper)

print "check marginal law intervals and window for each parameter"
print np.sum(n_bin, axis=1)

# ---> save m_grid, f_grid, r_grid_error, m_gbest, 
#      f_best, delta_m , xbin, n_bin, pdf_m, m_weight, m_pow
#      kappa, lower, upper
if save_netcdf:
    if os.path.isfile(outfile):
        os.remove(outfile)
        print "remove ", outfile
    nc = Dataset(outfile, "w", format='NETCDF4')
    # dimensions: name, size
    nc.createDimension('nparam', models.shape[-1])
    nc.createDimension('n_inter', pdf_m.shape[1])
    # Variables: name, format, shape
    nc.createVariable('m_synth', 'f8', ('nparam'))
    nc.createVariable('m_weight', 'f8', ('nparam'))
    #nc.createVariable('std_weight', 'f8', ('nparam'))
    nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('n_bin', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('x_bin', 'f8', ('nparam', 'n_inter'))
    # filling values
    nc.variables['m_synth'][:] = logrhosynth
    nc.variables['m_weight'][:] = m_weight
    #nc.variables['std_weight'][:] = std_weight
    nc.variables['pdf_m'][:, :] = pdf_m
    nc.variables['x_bin'][:, :] = x_bin
    nc.variables['n_bin'][:, :] = n_bin
    #nc.mysfit_mweight = XHI2
    nc.ndata = ndata * 2
    nc.f_gbest = f_gbest
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

