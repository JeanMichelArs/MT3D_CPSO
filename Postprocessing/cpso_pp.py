# python 2.7
'''
JCollin 06-2020

CPSO postprocessing module
'''
#-----------------------------------------------------------------------------
import os
import time
import numpy as np

# ----------------------------------------------------------------------------

def NaN_filter(models, energy, timing=True, **kwargs):
    """ remove Nan values, usefull for runs that did not finish """
    if timing:
        t0 = time.clock()
    nruns, popsize, nparam, nitertot = models.shape
    tmp = energy[energy < 1e20 ]
    niter = len(tmp) // nruns // popsize
    filt_energy = energy[:, :, :niter]
    print "run stopped at it ", niter, ' / ', nitertot
    if timing:
        print "ellapsed time in NaN_filter", time.clock() - t0
    return models[:, :, :, :niter], filt_energy, niter

# ----------------------------------------------------------------------------
def value_filter(models, energy, threshold, timing=True, **kwargs):
    """ 
    filter models and corresponding energy with low pass filtr on energy
    """
    if timing:
        t0 = time.clock()
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
    ncount = f_near.shape[0]
    ntot = np.prod(energy.shape)
    print "number of value filtered models :","{:e}".format(ncount), "/", "{:e}".format(ntot)
    if timing:
        print "ellapsed time in value_filter", time.clock() - t0 
    return m_near, f_near

# ----------------------------------------------------------------------------
def regrid(m_near, f_near, delta_m, center=True, timing=True, **kwargs):
    """
    regrid models, and energy to avoid oversampling
    centered = True : models are troncrated on zero-centred grid [...,-delta_m, 0, detla_m,...]
    centrerd = False : models are troncated
    """
    if timing: 
        t0 = time.clock()
    if center:
        m_grid, idx = np.unique((m_near + delta_m/2) // delta_m * delta_m, 
                                 return_index=True, axis=0)
        f_grid = f_near[idx]
    else:
        m_grid, idx = np.unique(m_near // delta_m * delta_m , return_index=True, axis=0)
        m_grid = m_grid + delta_m / 2
        f_grid = f_near[idx]
    rgrid_error = np.abs(np.min(f_near) - np.min(f_grid))
    print "regrid error : ", rgrid_error
    if timing:
        print "ellapsed time in regrid", time.clock() - t0
    return m_grid, f_grid, rgrid_error

# ----------------------------------------------------------------------------
def weighted_mean(m_grid, f_grid,ndata, kappa=1,rms=False, log=True, timing=True, **kwargs):
    """ mean models weighted by energy and kappa 
    if rms = True : stat are performed on rms instead of Xhi, ndata required!
    if log = True : stats are performed on models (log10 of real models)
    else : stats are performed on 10**models
    """
    if timing:
        t0 = time.clock()
    nparam = m_grid.shape[1]
    f_best = np.min(f_grid)
    m_weight = np.empty(shape=(nparam,))
    if rms==True:
        f_grid=np.sqrt(f_grid/ndata)
        S = 1 / np.sum(np.exp(- f_grid / 2 ))
        if log:
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp(- f_grid / 2 ) * \
                                    m_grid[:, iparam]) * S
        else: 
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp( - f_grid / 2 ) * \
                                    10**m_grid[:, iparam]) * S
    else:    
        S = 1 / np.sum(np.exp((f_best - f_grid) / 2 / kappa))
        if log:
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp((f_best - f_grid) / 2 / kappa) * \
                                   m_grid[:, iparam]) * S
        else: 
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp((f_best - f_grid) / 2 / kappa) * \
                                  10**m_grid[:, iparam]) * S
        if timing:
            print "ellapsed time in weighted_mean", time.clock() - t0
    return m_weight

# ----------------------------------------------------------------------------
def weighted_std(m_weight,m_grid, f_grid, ndata, kappa=1, rms=False, log=True, timing=True, **kwargs):
    """ std models weighted by energy and kappa 
    if rms = True : stat are performed on rms instead of Xhi, ndata required!
    if log = True : stats are performed on models (log10 of real models)
    else : stats are performed on 10**models
    Formulation from Luu et al. 2019
    """
    if timing:
        t0 = time.clock()

    nparam = m_grid.shape[1]
    nmodel=m_grid.shape[0]
    f_best = np.min(f_grid)
    std_weight = np.empty(shape=(nparam,))
    if rms==True:
        f_grid=np.sqrt(f_grid/ndata)
        S = 1 / np.sum(np.exp(- f_grid / 2 ))
        if log:
            for iparam in range(nparam):
                std_weight[iparam]=np.sqrt((nmodel/(nmodel-1))*np.sum(np.exp(- f_grid / 2 )* (m_grid[:, iparam]-m_weight[iparam])**2)*S)
        else: 
            for iparam in range(nparam):
                std_weight[iparam]=np.sqrt((nmodel/(nmodel-1))*np.sum(np.exp(- f_grid / 2 )* (10**m_grid[:, iparam]-m_weight[iparam])**2)*S)
    else:
        S = 1 / np.sum(np.exp((f_best - f_grid) / 2 / kappa))
        if log:
            for iparam in range(nparam):
                std_weight[iparam]=np.sqrt((nmodel/(nmodel-1))*np.sum(np.exp((f_best - f_grid) / 2 / kappa)* (m_grid[:, iparam]-m_weight[iparam])**2)*S)
        else: 
            for iparam in range(nparam):
                std_weight[iparam]=np.sqrt((nmodel/(nmodel-1))*np.sum(np.exp((f_best - f_grid) / 2 / kappa)* (10**m_grid[:, iparam]-m_weight[iparam])**2)*S)

    if timing:
        print "ellapsed time in weighted_std", time.clock() - t0
    return std_weight

# ----------------------------------------------------------------------------
def marginal_law(m_grid, f_grid, m_best, ndata, n_inter=30, lower=-1, upper=1, 
                 kappa=100, rms=False, timing=True, **kwargs):
    """ parameter marginal laws around m_best (is m_best, m_weighted ?)
    if rms = True : stat are performed on rms instead of Xhi, ndata required!
    for high energy values kappa must be applied
    a good approximation is based on regridding error
    """
    if timing:
        t0 = time.clock()
    nparam = m_grid.shape[1]
    n_bin = np.empty(shape=(nparam, n_inter))
    pdf_m = np.empty(shape=(nparam, n_inter))
    x_bin = np.empty(shape=(nparam, n_inter))
    f_best = np.min(f_grid)
    lmbda = 0.5 / kappa
    eps = 5 * 1e-3
    if rms==True:
        f_grid=np.sqrt(f_grid/ndata)
    for iparam in range(nparam):
       for i_inter in range(n_inter):
            p_inter = lower + i_inter * (upper-lower) / n_inter + m_best[iparam]
            x_bin[iparam, i_inter] = p_inter
            i_mod = np.abs(m_grid[:, iparam] - p_inter + eps) <= (upper - lower) \
                    / n_inter * 0.5
            n_bin[iparam, i_inter] = np.sum(i_mod)
            if  np.sum(i_mod) >= 1:
                if rms==True:
                    pdf_m[iparam, i_inter] = np.sum(np.exp((f_best - f_grid[i_mod]) \
                                             * lmbda)) / \
                                             np.sum(np.exp((f_best - f_grid) * lmbda))
                else:
                    pdf_m[iparam, i_inter] = np.sum(np.exp( - f_grid[i_mod] \
                                             * lmbda)) / \
                                             np.sum(np.exp(- f_grid * lmbda))
            else:
                pdf_m[iparam, i_inter] = 0
    if timing:
        print "ellapsed time in marginal_law", time.clock() - t0
    return pdf_m, n_bin, x_bin
# ----------------------------------------------------------------------------

