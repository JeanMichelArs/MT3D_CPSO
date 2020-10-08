"""
vendredi 2 octobre 2020, 09:42:34 (UTC+0200)

Script to debug regridding and intervals of models
an accumulation seems to appear with reagrds to binning technique

"""
# ----------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

sys.path.append('../Postprocessing/')
import cpso_pp as pp
# ----------------------------------------------------------------------------

upper = 2
lower = - upper
delta_mgrid = 1e-2 

# --- test cases
out_big_p = upper + 0.6* delta_mgrid
out_p = upper + delta_mgrid / 2
out_small_p = upper + delta_mgrid / 10

out_big_m = - out_big_p
out_m = - out_p
out_small_m = - out_small_p

in_close_p = upper - delta_mgrid / 10
in_p = upper - delta_mgrid / 2
in_far_p = upper - 2 * delta_mgrid 

in_close_m = - in_close_p
in_m = - in_p 
in_far_m = - in_far_p

models = np.array([out_big_m, out_m, out_small_m, in_close_m, in_m, in_far_m,
                  in_close_p, in_p, in_far_p, out_big_p, out_p, out_small_p])

f_near = models * 5

# ---> regrid ok !
m_grid, f_grid, rgrid_error = pp.regrid(models, f_near, delta_m=delta_mgrid, center=True)

m_non_unique = (models + delta_mgrid/2) // delta_mgrid * delta_mgrid

# ---> marginal_law ?
# is epsilon relevant ?
nparam = 1
n_inter = 40
n_bin = np.empty(shape=(n_inter))
pdf_m = np.empty(shape=(n_inter))
x_bin = np.empty(shape=(n_inter))

L = upper - lower
eps = 1e-3
p_inter = np.arange(n_inter + 1.) * L / n_inter - L * 0.5
p_inter[0] = p_inter[0] - eps
p_inter[-1] = p_inter[-1] + eps

x_bin = (p_inter[:-1] + p_inter[1:]) * 0.5

# ---> binning
for i_inter in range(n_inter):
    x_bin[:] = np.squeeze(p_inter[:-1] + p_inter[1:]) * 0.5 
    im = m_grid >= p_inter[i_inter]
    ip = m_grid < p_inter[i_inter + 1]
    i_mod = im & ip
    n_bin[i_inter] = np.sum(i_mod)

# ---> checking with real data 
"""There is no issue here """

mtfile = '/postproc/COLLIN/MTD3/1D_MCM_ana_8param/Analysis/debug_pdf1.nc'

nc = Dataset(mtfile, 'r')
m_grid = nc.variables['m_grid'][:]
n_bin = nc.variables['n_bin'][:]
x_bin = nc.variables['x_bin'][:]
nc.close()

plt.figure()
for ip in range(8):
    plt.plot(x_bin[ip, :], n_bin[ip, :])

plt.show()

# ---> checking on plot

fig = plt.figure()
plt.plot(x_bin[0, :], n_bin[0, :])

plt.show()






