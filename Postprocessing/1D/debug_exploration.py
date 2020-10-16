"""

jeudi 15 octobre 2020, 11:58:40 (UTC+0200)

CPSO algorithm tends to over explore boundaries in MT problems, here we try 
to understand whether this is due an algorithm issue or MT cost funtion 
topology 

for example in CPSO_1D_4nz the fourth parameter has a huge number of parameters
at the sup boundary 

TODO:

    - find models at the sup boundary and plot their time-evolution
    !!! 99% of run 1 models are located between m_synth + [1.9, 2.0]
    for parameter 3

"""
# ----------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

sys.path.append("../")
import cpso_pp as pp


# ----------------------------------------------------------------------------

NCPATH = '/postproc/COLLIN/MTD3/Calibre_CPSO_4nz'
ncfile = NCPATH + '/merged.nc'

nc = Dataset(ncfile, 'r')
models = nc.variables['models'][:]
m_synth = np.log10(nc.variables['rho_i'][0, :])
nc.close()

# ---> check cpso exploration
nparam = 4
nruns = 4
iparam = 3
eps = 0.1
print np.max(np.abs(models[:, :, iparam, :] - m_synth[iparam]))


print "number of models close to sup bound"

for irun in np.arange(nruns):
    print "irun :", irun
    print "nmod close bounds ", \
            np.sum(np.abs(models[irun, :, iparam, :] - m_synth[iparam] - 2) < eps)
    print "exploration amplitude :", \
            np.max(models[irun, :, iparam, :]) - np.min(models[irun, :, iparam, :])

# ---> model distribution
irun = 1

plt.plot(np.sort(models[irun, :, iparam, :].reshape(4*400)))
plt.show()

plt.figure()
for iparam in np.arange(nparam):
     plt.subplot(2, 2, iparam + 1)
     plt.pcolormesh(models[irun, :, iparam, :])
     plt.colorbar()

plt.show()
