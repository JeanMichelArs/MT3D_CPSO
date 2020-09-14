"""
script to analyze cpso convergence over the runs

- plot < m > for several number of runs [1, 5, 10, 20, 50, 75, 100]
- plot <std> //
- log of error (norme2) E(nruns), for both < m > and < std >

- TODO: 
    - plot error against pbest to see a convergence ?
"""
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import cpso_pp as pp
# ----------------------------------------------------------------------------

data_dir = "/postproc/COLLIN/MTD3/Bolivia_115param_015"
runs = [1, 5, 10, 20, 50, 75, 100]
nparam = 115

# ---- getting data
# -> <m>, <std>
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

# -> global run
# !!! Care must be take as mselect is an old run and stats computation may 
# differ from multi runs
fname = data_dir + '/mselect.nc'
m_gbest = np.empty((nparam,))
m_gweight = np.empty((nparam,))

nc = Dataset(fname, 'r')
m_gbest[:] = nc.variables['m_gbest'][:]
m_gweight[:] = nc.variables['m_weight'][:]
nc.close()

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
figname = data_dir + '/likehood_std_convergence.png'
plt.savefig(figname)
plt.show()

# ---- norme 2 
m_l2 = np.empty(len(runs))
std_l2 = np.empty(len(runs))
for irun in range(len(runs)):
    m_l2[irun] =  np.linalg.norm(m_weight[irun, :] - m_weight[-1, :], ord=2)
    std_l2[irun] = np.linalg.norm(std_weight[-1, :] - std_weight[irun, :],
                                  ord=2)


plt.figure(figsize=(8, 4))
plt.plot(runs, m_l2, 'b', label='<m>')
plt.scatter(runs, m_l2)
plt.plot(runs, std_l2, 'r', label='<std>')
plt.scatter(runs, std_l2)
plt.title('norme 2 error 100 runs')
plt.xlabel('nruns')
plt.savefig(data_dir + '/error_m_std_nruns.png')
plt.show()

# -- loglog
plt.figure(figsize=(8, 4))
plt.loglog(runs, m_l2 , 'b', label='<m>')
plt.loglog(runs, std_l2 , 'r', label='<std>')
plt.loglog(runs, np.array(runs)**-0.5, label='nruns^(-1/2)')
plt.scatter(runs, m_l2)
plt.scatter(runs, std_l2)
plt.grid()
plt.ylabel('log(error)')
plt.xlabel('nruns')
plt.legend()
plt.savefig(data_dir + '/loglog_conv.png')
plt.show()




