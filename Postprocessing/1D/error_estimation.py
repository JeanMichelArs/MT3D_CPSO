# python 2.7
'''
jeudi 3 decembre 2020, 14:27:33 (UTC+0100)

Uncertainties estimation based on Tarits 94


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
nruns = 50
nmodels_cut = int(-1)
cpso_path = '/postproc/COLLIN/MTD3/MCM_4nz_cst_Error/Analysis'
pdf_file = cpso_path + '/pdf_m_' + str(nruns) + '.nc'

# ---> outputs
method = 'xhi'
folder_save = cpso_path   
save_plot = True
save_netcdf = True
outfile = folder_save + "/pdf_error_fun_" + method + str(nruns) + ".nc"
timing = True
figname = folder_save + '/' + 'pdf_error_fun_' + method + str(nruns) + '.png'

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
nc = Dataset(pdf_file)
f_grid = np.array(nc.variables['f_grid'][:nmodels_cut])
m_grid =  np.array(nc.variables['m_grid'][:nmodels_cut, :])
pdf_m =  np.array(nc.variables['pdf_m'][:, :])
m_synth =  np.array(nc.variables['m_synth'][:])
ndata = nc.ndata
upper = nc.upper
lower = nc.lower
nc.close()

n_inter = pdf_m.shape[1]
#---------------------------------------------------------------------------
# ---> compute uncertainties

# --- > as a function


print method
"""
pdf_error, U_ikk = pp.pdf_std(m_grid, f_grid, m_synth, ndata, method=method,
                              n_inter=n_inter, lower=lower, upper=upper,
                              timing=timing)
"""
# --- > as a script
"""
nparam = m_grid.shape[1]
nmodels = f_grid.shape
pdf_error = np.empty(shape=(nparam, n_inter))
eps = 1e-3
L = upper - lower
p_inter = np.arange(n_inter + 1.) * L / n_inter - L * 0.5
p_inter[0] = p_inter[0] - eps
p_inter[-1] = p_inter[-1] + eps

if timing:
    t0 = time.clock()

if method is 'xhi':
    F = np.exp(-(f_grid - np.min(f_grid)) / 2)
elif method is 'rms':
    F = np.exp(-(np.sqrt(f_grid / ndata))
else:
    print "Error: unkwown method"

E = np.sum(F) / nmodels
V = np.sum(F**2) / nmodels

# here we distribute F over procs
# each proc as a parameter

# no more loop over iparam, iparam is rank

print "E =", E
print "V =", V
for iparam in range(nparam):
    for i_inter in range(n_inter):
        im = m_grid[:, iparam] - m_synth[iparam] >= p_inter[i_inter]
        ip = m_grid[:, iparam] - m_synth[iparam] < p_inter[i_inter + 1]
        i_mod = im & ip
        #n_bin[iparam, i_inter] = np.sum(i_mod)
        if  np.sum(i_mod) >= 1:
            E_ik = np.sum(F[i_mod]) / nmodels
            V_ik = np.sum(F[i_mod]**2) / nmodels
            U_ikk = (V_ik * (E - E_ik)**2 + E_ik**2 * (V - V_ik)) / E**4
            pdf_error[iparam, i_inter] = np.sqrt(U_ikk)
        else:
            pdf_error[iparam, i_inter] = 0
        print '% Error on pdf, i=', iparam, "i_inter", i_inter,
        print 'pdf_error=', pdf_error[iparam, i_inter]

pdf_error = pdf_error / np.sqrt(nmodels)

# Here we gather results over rank 0
"""


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
    nc.createDimension('popsize', f_grid.shape[0])
    nc.createDimension('nparam', m_grid.shape[1])
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
    nc.eps = eps
    nc.nmodels_cut = nmodels_cut
    print outfile, "saved sucessfully" 
    nc.close()

