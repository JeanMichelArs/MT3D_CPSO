# python 2.7
'''
jeudi 3 decembre 2020, 14:27:33 (UTC+0100)

Uncertainties estimation based on Tarits 94

- computation on both filtered and raw models
'''
#-----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

sys.path.append('../')
import cpso_pp as pp

# ----------------------------------------------------------------------------
# ---> inputs
config = 'filter'
nruns = 2
nmodels_cut = int(-1)
cpso_path = '/postproc/COLLIN/MTD3/MCM_16nz_cst_Error/raw'
pdf_file = cpso_path + '/pdf_m_' + str(nruns) + config  + '.nc'
# only for raw models
exploration_file = cpso_path + '/../merged.nc'
expl_cut = np.int(10**5)

# ---> outputs
filtered = True
method = 'xhi'
folder_save = cpso_path   
save_plot = True
save_netcdf = True
outfile = folder_save + "/pdf_error_fun_" + method + str(nruns) + config + ".nc"
timing = True
figname = folder_save + '/' + 'pdf_error_fun_' + method + str(nruns) + config + '.png'

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
if filtered:
    nc = Dataset(pdf_file, 'r')
    """
    f_grid = np.array(nc.variables['f_grid'][:nmodels_cut])
    m_grid =  np.array(nc.variables['m_grid'][:nmodels_cut, :])
    """
    f_grid = np.array(nc.variables['f_grid'][:])
    m_grid = np.array(nc.variables['m_grid'][:, :])
    pdf_m = np.array(nc.variables['pdf_m'][:, :])
    m_synth = np.array(nc.variables['m_synth'][:])
    ndata = nc.ndata
    upper = nc.upper
    lower = nc.lower
    nc.close()
    popsize = np.prod(f_grid.shape)
else:
    nc = Dataset(pdf_file)
    pdf_m =  np.array(nc.variables['pdf_m'][:, :])
    m_synth =  np.array(nc.variables['m_synth'][:])
    ndata = nc.ndata
    upper = nc.upper
    lower = nc.lower
    nc.close()
    nc = Dataset(exploration_file, 'r')
    models = np.array(nc.variables['models'][:nruns, :expl_cut, :])
    energy = np.array(nc.variables['energy'][:nruns, :expl_cut])
    nc.close()
    popsize = np.prod(energy.shape)

nparam, n_inter = pdf_m.shape

#---------------------------------------------------------------------------
# ---> compute uncertainties

if filtered:
    pdf_error = pp.pdf_std(m_grid, f_grid, m_synth, ndata, method=method,
                           n_inter=n_inter, lower=lower, upper=upper,
                           timing=timing)
else:
    pdf_error = pp.raw_pdf_std(models, energy, m_synth, ndata, method=method,
                                n_inter=n_inter, lower=lower, upper=upper,
                                timing=timing)
# ---> plot test
plt.figure()
for iparam in range(nparam):
    plt.subplot(211)
    plt.plot(pdf_error[iparam, :])
    plt.ylabel('error')
    plt.subplot(212)
    plt.plot(pdf_m[iparam, :])
    plt.ylabel('pdf_m')

if save_plot:
    plt.savefig(figname)

plt.show()

# --->  
if save_netcdf:
    if os.path.isfile(outfile):
        os.remove(outfile)
        print "remove ", outfile
    nc = Dataset(outfile, "w", format='NETCDF4')
    # dimensions: name, size
    nc.createDimension('popsize', popsize)
    nc.createDimension('nparam', pdf_m.shape[0])
    nc.createDimension('n_inter', pdf_m.shape[1])
    # Variables: name, format, shape
    nc.createVariable('pdf_error', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('m_synth', 'f8', ('nparam'))
    # filling values
    nc.variables['pdf_error'][:, :] = pdf_error
    nc.variables['pdf_m'][:, :] = pdf_m
    nc.variables['m_synth'][:] = m_synth
    nc.lower = lower
    nc.upper = upper
    nc.nruns = nruns
    nc.nmodels_cut = nmodels_cut
    print outfile, "saved sucessfully" 
    nc.close()

