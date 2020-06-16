# python 2.7
'''
JCollin 01-2020

script to select models in order to avoid oversampling in local minimas
Let's assume we dicretize the parameter space by Delta_m regular spacing 
m_select = {mi | || mi - mj ||_inf >= Delta_m }

'''
#-----------------------------------------------------------------------------
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------

def NaN_filter(models, energy, **kwargs):
    """ remove Nan values, usefull for runs that did not finish """
    nruns, popsize, nparam, nitertot = models.shape
    tmp = energy[energy < 1e20 ]
    niter = len(tmp) // nruns // popsize
    filt_energy = tmp.reshape((nruns, popsize, niter))
    print "run stopped at it ", niter, ' / ', nitertot
    return models[:, :, :, :niter], filt_energy, niter

# ----------------------------------------------------------------------------
def value_filter(models, energy, threshold, **kwargs):
    """ 
    filter models and corresponding energy with low pass filtr on energy
    """
    nruns, popsize, nparam, nitertot = models.shape
    f_best = np.min(energy)
    i_best = np.where(energy == np.min(energy))
    m_best = models[i_best[0], i_best[1], : , i_best[2]]
    "! in some cases minimum may be found multiple times therefore we need to pick one"
    if np.prod(m_best.shape) > nparam:
        print "several(", m_best.shape[0] ,") best fit found; max diff between models:"
        print  np.max(np.abs(np.diff(m_best, axis=0)))
        m_best = m_best[0, :]
    f_near = energy[energy < np.min(energy) + threshold]
    i_near = np.where(energy < np.min(energy) + threshold)
    m_near = models[i_near[0], i_near[1], :, i_near[2]]
    print "number of value filtered models :", f_near.shape, "/", np.prod(energy.shape)  
    return m_near, f_near

# ----------------------------------------------------------------------------
def regrid(m_near, f_near, delta_m, center=True, **kwargs):
    """
    regrid models, and energy to avoid oversampling
    centered = True : models are troncrated on zero-centred grid [...,-delta_m, 0, detla_m,...]
    centrerd = False : models are troncated
    """
    if center:
        m_grid, idx = np.unique((m_near + delta_m/2) // delta_m * delta_m , 
                                 return_index=True, axis=0)
        f_grid = f_near[idx]
    else:
        m_grid, idx = np.unique(m_near // delta_m * delta_m , return_index=True, axis=0)
        m_grid = m_grid + delta_m / 2
        f_grid = f_near[idx]
    rgrid_error = np.abs(np.min(energy) - np.min(f_grid))
    print "regrid error : ", rgrid_error
    return m_grid, f_grid, rgrid_error

# ----------------------------------------------------------------------------
def weighted_mean(m_grid, f_grid, kappa=1, log=True, **kwargs):
    """ mean models weighted by energy and kappa 
    if log = True : stats are performed on models (log10 of real models)
    else : stats are performed on 10**models
    """
    nparam = m_grid.shape[1]
    f_best = np.min(f_grid)
    m_weight = np.empty(shape=(nparam,))
    S = 1 / np.sum(np.exp((f_best - f_grid) / 2 / kappa))
    if log:
        for iparam in range(nparam):
            m_weight[iparam] = np.sum(np.exp((f_best - f_grid) / 2 / kappa) * \
                               m_grid[:, iparam]) * S
    else: 
        for iparam in range(nparam):
            m_weight[iparam] = np.sum(np.exp((f_best - f_grid) / 2 / kappa) * \
                               10**m_grid[:, iparam]) * S
    return m_weight

# ----------------------------------------------------------------------------
def marginal_law(m_grid, f_grid, m_best, n_inter=30, lower=-1, upper=1, 
                 kappa=100, **kwargs):
    """ parameter marginal laws around m_best (is m_best, m_weighted ?)
    for high energy values kappa must be applied
    a good approximation is based on regridding error
    """
    nparam = m_grid.shape[1]
    n_bin = np.empty(shape=(nparam, n_inter))
    pdf_m = np.empty(shape=(nparam, n_inter))
    x_bin = np.empty(shape=(nparam, n_inter))
    f_best = np.min(f_grid)
    lmbda = 0.5 / kappa
    eps = 5 * 1e-3
    for iparam in range(nparam):
       for i_inter in range(n_inter):
            p_inter = lower + i_inter * (upper-lower) / n_inter + m_best[iparam]
            x_bin[iparam, i_inter] = p_inter
            i_mod = np.abs(m_grid[:, iparam] - p_inter + eps) <= (upper - lower) \
                    / n_inter * 0.5
            n_bin[iparam, i_inter] = np.sum(i_mod)
            if  np.sum(i_mod) >= 1:
                pdf_m[iparam, i_inter] = np.sum(np.exp((f_best - f_grid[i_mod]) \
                                         * lmbda)) / \
                                         np.sum(np.exp((f_best - f_grid) * lmbda))
            else:
                pdf_m[iparam, i_inter] = 0
    return pdf_m, n_bin, x_bin


# ----------------------------------------------------------------------------
# ---------------------- MAIN ------------------------------------------------
# ----------------------------------------------------------------------------
run = 'Bolivia_115param_015'
NCPATH = '/postproc/COLLIN/MTD3/' + run
save_plot = True
folder_save = '../RUN/' + run + '/results'

# --- create directory to save plots 
if not os.path.exists(folder_save):
    os.makedirs(folder_save)

# --- load data
nc = Dataset(NCPATH + '/merged_98_99.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
nc.close()

# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))
m_gbest = models[i_gbest[0], i_gbest[1], : , i_gbest[2]]
m_gbest = m_gbest[0]

# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = NaN_filter(models, energy)

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 
nruns, popsize, nparam, nitertot = models.shape
print "number of models to filter: ", nruns * popsize * nparam * niter

threshold = np.median(energy)
m_near, f_near = value_filter(models, energy, threshold)
np.max(np.abs(m_near - m_near))
np.max(np.abs(f_near - f_near))

# --- > regrid parameter space
delta_m = 1e-3 
m_grid, f_grid, rgrid_error = regrid(m_near, f_near, delta_m, center=True)
print "number of particles in subgrid", m_grid.shape[0]
print "Error in energy minimaum after sugrid :", np.min(f_grid) - f_gbest

# ---> Xi2 weighted mean model, in log and physical space
m_weight = weighted_mean(m_grid, f_grid, kappa=1, log=True)
mpow_weight = np.log10(weighted_mean(m_grid, f_grid, kappa=1, log=False))
print "difference between log and physical space :", np.max(np.abs(mpow_weight - m_weight))

# ---- marginal laws using kappa damping coefficient
n_inter = 20
lower = -1.  
upper = 1. 
kappa = np.abs(rgrid_error) 

pdf_m, n_bin, x_bin = marginal_law(m_grid, f_grid, m_gbest, n_inter=n_inter,
                                   lower=lower, upper=upper, kappa=kappa)
for iparam in range(nparam):
    plt.subplot(2, 1, 1)
    plt.plot(x_bin[iparam, :], pdf_m[iparam], 'r')
    plt.axvline(x=m_gbest[iparam], color='red')
    plt.axvline(x=m_weight[iparam], color='blue')
    plt.xlim([-1, 1])
    plt.subplot(2, 1, 2)
    plt.plot(x_bin[iparam, :], n_bin[iparam, :])
    plt.xlim([-1, 1])
    plt.savefig(folder_save + '/' + "fun_diff"+  str(int(iparam)) )
    plt.clf()

