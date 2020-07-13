"""
script to analyze cpso convergence over the runs

- TODO:
    - plot < m > for several number of runs [1, 5, 10, 20, 50, 75, 100]
    - plot <std> //
"""
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import cpso_pp as pp
# ----------------------------------------------------------------------------

data_dir = "/postproc/COLLIN/MTD3/Bolivia_115param_015"
runs = [1, 5, 10, 20]
nparam = 115

# ---- getting data
m_weight = np.empty((len(runs), nparam))
std_weight = np.empty((len(runs), nparam))
icount = 0
for irun in runs:
    fname = data_dir + '/mselect_mod_nruns' + str(irun) + '.nc'
    nc = Dataset(fname, 'r')
    m_weight[icount, :] = nc.variables['m_weight'][:]
    std_weight[icount, :] = nc.variables['std_weight'][:]
    nc.close()
    icount = icount + 1
    
# ---- plot <m> and <std>
plt.figure(figsize=(10, 8))
for irun in range(len(runs)):
    plt.subplot(2, 1, 1)
    plt.plot(m_weight[irun, :], label=str(runs[irun]))
    plt.ylabel('<m>')
    plt.subplot(2, 1, 2)
    plt.plot(std_weight[irun, :],label=str(runs[irun]))
    plt.xlabel('iparam') 
    plt.ylabel('<std>')

plt.legend()
plt.show()

# ---- norme 2 


