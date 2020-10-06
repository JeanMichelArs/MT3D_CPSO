# python 2.7
'''
JCollin 06-2020

CPSO postprocessing module
'''
#-----------------------------------------------------------------------------
import os
import time
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from scipy.interpolate import griddata
from netCDF4 import Dataset

# ----------------------------------------------------------------------------

def get_ndata(data_file):
    """
    returns number of observed measurement data of MT field

    parameters:
      - data_file : text file containing observations
      - ndata : number of observations
    """
    ndata = 0
    for fn in glob.glob(data_file):
        with open(fn) as f:
            ndata = ndata + sum(1 for line in f if line.strip() and not line.startswith('#'))
    return ndata

# ----------------------------------------------------------------------------

def NaN_filter(models, energy, timing=True, **kwargs):
    """ remove Nan values, usefull for runs that did not finish
    """
    if timing:
        t0 = time.clock()
    if len(energy.shape) == 2:
        "MCM format"
        nruns, nitertot, nparam = models.shape
        tmp = energy[energy < 1e20 ]
        niter = len(tmp) // nruns
        filt_energy = energy[:, :niter]
        if timing:
            print "ellapsed time in NaN_filter", time.clock() - t0
        return models[:, :niter, :], filt_energy, niter
    elif len(energy.shape) == 3:
        "CPSO Format"
        nruns, popsize, nparam, nitertot = models.shape
        tmp = energy[energy < 1e20 ]
        niter = len(tmp) // nruns // popsize
        filt_energy = energy[:, :, :niter]
        if timing:
            print "ellapsed time in NaN_filter", time.clock() - t0
        return models[:, :, :, :niter], filt_energy, niter
    else:
        print "Error wrong energy shape"
        return None
    

# ----------------------------------------------------------------------------
def value_filter(models, energy, threshold, timing=True, **kwargs):
    """ 
    filter models and corresponding energy with low pass filtr on energy
    May seem unecessary for unr that finished but it shaped the data
   in accordance with other postprocessing functions
    """
    if timing:
        t0 = time.clock()
    # ---> cpso 
    if len(energy.shape) == 3:
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
    # ---> mcm
    elif len(energy.shape) == 2:
        nruns, nitertot, nparam = models.shape
        f_best = np.min(energy)
        i_best = np.where(energy == np.min(energy))
        m_best = models[i_best[0], i_best[1], :]
        "! in some cases minimum may be found multiple times therefore we need to pick one"
        if np.prod(m_best.shape) > nparam:
            print "several(", m_best.shape[0] ,") best fit found; max diff between models:"
            print  np.max(np.abs(np.diff(m_best, axis=0)))
            m_best = m_best[0, :]
        f_near = energy[energy < np.min(energy) + threshold]
        i_near = np.where(energy < np.min(energy) + threshold)
        m_near = models[i_near[0], i_near[1], :]
    else:
        print "Error unsupported energy shape"
        return None
    # ---> 
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
def weighted_mean(m_grid, f_grid, ndata, kappa=1,rms=False, log=True, 
                  timing=True, **kwargs):
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
        f_grid = np.sqrt(f_grid / ndata)
        S = 1 / np.sum(np.exp(-f_grid))
        if log:
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp(-f_grid) * \
                                    m_grid[:, iparam]) * S
        else: 
            for iparam in range(nparam):
                m_weight[iparam] = np.sum(np.exp(-f_grid) * \
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
def weighted_std(m_weight, m_grid, f_grid, ndata, kappa=1, rms=False, log=True,
                 timing=True, **kwargs):
    """ std models weighted by energy and kappa 
    if rms = True : stat are performed on rms instead of Xhi, ndata required!
    if log = True : stats are performed on models (log10 of real models)
    else : stats are performed on 10**models
    Formulation from Luu et al. 2019
    """
    if timing:
        t0 = time.clock()

    nparam = m_grid.shape[1]
    nmodel = m_grid.shape[0]
    f_best = np.min(f_grid)
    std_weight = np.empty(shape=(nparam,))
    if rms==True:
        f_grid = np.sqrt(f_grid / ndata)
        S = 1 / np.sum(np.exp(-f_grid))
        if log:
            for iparam in range(nparam):
                std_weight[iparam] = np.sqrt((nmodel/(nmodel-1)) \
                     * np.sum(np.exp(-f_grid) * (m_grid[:, iparam] \
                                                 - m_weight[iparam])**2) * S)
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
                 kappa=1, rms=False, timing=True, **kwargs):
    """ parameter marginal laws around m_best 
    if rms = True : stat are performed on rms instead of Xhi, ndata required!
    kappa is deprecated, it used to be a damping coefficient
    
    inputs: 
      - m_grid : moedls interpolated on a regular grid
      - f_grid : cost function evalutation at m_grid
      - ndata : number of measurements
      - n_inter : number of intervals for probability density function
      - upper, lower : windows to perform statistics around m_best (should be
        the same than evolutionnary algorithm)

    outputs :
      - pdf_m : marginal laws for each parameter 
      - n_bin : number of regrided models in each intervals
      - x_bin : interval center for plot purpose
    """
    if timing:
        t0 = time.clock()
    nparam = m_grid.shape[1]
    n_bin = np.empty(shape=(nparam, n_inter))
    pdf_m = np.empty(shape=(nparam, n_inter))
    x_bin = np.empty(shape=(nparam, n_inter))
    f_best = np.min(f_grid)
    lmbda = 0.5 / kappa
    eps = 1e-3
    L = upper - lower
    p_inter = np.arange(n_inter + 1.) * L / n_inter - L * 0.5
    p_inter[0] = p_inter[0] - eps
    p_inter[-1] = p_inter[-1] + eps
    if rms==True:
        f_grid = np.sqrt(f_grid / ndata)
        S = 1 / np.sum(np.exp(- f_grid))
    else:
        S =  1 / np.sum(np.exp((f_best - f_grid) * 0.5))
    
    for iparam in range(nparam):
        for i_inter in range(n_inter):
            x_bin[iparam, :] = np.squeeze(p_inter[:-1] + p_inter[1:]) * 0.5 \
                               + m_best[iparam]
            im = m_grid[:, iparam] - m_best[iparam] >= p_inter[i_inter]
            ip = m_grid[:, iparam] - m_best[iparam] < p_inter[i_inter + 1]
            i_mod = im & ip
            n_bin[iparam, i_inter] = np.sum(i_mod)
            if  np.sum(i_mod) >= 1:
                if rms==True:
                    pdf_m[iparam, i_inter] = np.sum(np.exp(- f_grid[i_mod])) \
                                             * S
                else:
                    pdf_m[iparam, i_inter] = np.sum(np.exp((f_best - f_grid[i_mod]) \
                                             * lmbda)) * S
            else:
                pdf_m[iparam, i_inter] = 0
        print '% Error on pdf, i=', iparam, 'E=', (1-np.sum(pdf_m[iparam, :]))*100 
        x_bin[:, 0] = x_bin[:, 0] + eps * 0.5
        x_bin[:, -1] = x_bin[:, -1] - eps * 0.5

    if timing:
        print "ellapsed time in marginal_law", time.clock() - t0
    return pdf_m, n_bin, x_bin


# ----------------------------------------------------------------------------
def old_marginal_law(m_grid, f_grid, m_best, ndata, n_inter=30, lower=-1, upper=1, 
                 kappa=100, rms=False, timing=True, **kwargs):
    """ DEPRECATED WRONG INTERVALS  
    parameter marginal laws around m_best (is m_best, m_weighted ?)
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
    eps = 5 * 1e-6
    if rms==True:
        f_grid = np.sqrt(f_grid / ndata)
    for iparam in range(nparam):
       for i_inter in range(n_inter):
            p_inter = lower + i_inter * (upper-lower) / n_inter + m_best[iparam]
            x_bin[iparam, i_inter] = np.squeeze(p_inter)
            i_mod = np.abs(m_grid[:, iparam] - p_inter + eps) <= (upper - lower) \
                    / n_inter * 0.5
            n_bin[iparam, i_inter] = np.sum(i_mod)
            if  np.sum(i_mod) >= 1:
                if rms==True:
                    pdf_m[iparam, i_inter] = np.sum(np.exp( - f_grid[i_mod] /2)) / \
                                             np.sum(np.exp(- f_grid /2))
                else:
                    pdf_m[iparam, i_inter] = np.sum(np.exp((f_best - f_grid[i_mod]) \
                                             * lmbda)) / \
                                             np.sum(np.exp((f_best - f_grid) * lmbda))
            else:
                pdf_m[iparam, i_inter] = 0
    if timing:
        print "ellapsed time in marginal_law", time.clock() - t0
    return pdf_m, n_bin, x_bin


# ----------------------------------------------------------------------------
def vertical_profile(figname, pdf_m=None, m_weight=None, logrhosynth=None,
        hz=None, x_bin=None, cut_off=1e-3, transparent=True, **kwargs):
    """  MEAN MODEL & PDF """
    fig = plt.figure()
    nparam = hz.shape[0]
    dz = np.zeros(nparam + 1)
    for k in range(nparam + 1):
        dz[k] = np.sum(hz[0:k])
    # SHADED LAW
    cmm = cm.plasma
    norm = BoundaryNorm(np.arange(0,1,0.01), ncolors=cmm.N, clip=True)
    for ipar in range(nparam):
        ddz = np.linspace(-dz[ipar+1], -dz[ipar], 10)
        ddx = np.linspace(np.min(x_bin[ipar, :]), np.max(x_bin[ipar, :]), 80)
        nn = len(ddz)
        ll = pdf_m[ipar, :]
        ll[ll < cut_off] = 'nan'
        ml = griddata(x_bin[ipar, :], ll, ddx, method='linear')
        shaded = np.tile(ml, (nn, 1))
        mx, mz = np.meshgrid(ddx, ddz)
        pcol = plt.pcolormesh(mx, mz, shaded, alpha=0.7, cmap=cmm, norm=norm,
                              antialiased=True, linewidth=0.0, rasterized=True)
        pcol.set_edgecolor('Face')

    c = plt.colorbar()
    c.set_label(label='PDF', fontsize=10)
    c.ax.tick_params(axis='x', labelsize=8)
    mean_mod = np.hstack((m_weight,m_weight[-1]))
    synth_mod = np.hstack((logrhosynth, logrhosynth[-1]))
    plt.step(mean_mod, -dz, linewidth=2, color='g', label='mean')
    plt.step(synth_mod, -dz, linewidth=2, color='y', label='synth')
    plt.legend(loc=0, fontsize=10)
    plt.ylabel('Depth', fontsize=10)
    plt.xlabel('Resistivity (Log-scale)', fontsize=10)
    plt.xlim(0, 5.5)
    #plt.show()
    plt.savefig(figname, transparent=transparent)
    return None

# ----------------------------------------------------------------------------

