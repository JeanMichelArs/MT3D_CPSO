#!/usr/bin/python2.7
""" ----------------------------------------------------------------------------
 MCM MT1D with analytic forward modelling for comparison to Tarits 
 Monte Carlo Code

 >>> mpirun -np 4 python MT1D_analytic_MCM.py &> mcm.log

 Each mpi proc runs a separate MonteCarlo exploration startin from random Xstart
 Results are stored in a netcdf file for each proc with models, energy ...
 files maybe concatenated using 
 
 >>> make merge 


 TODO: Double check for Error I believe that error is recomputed at ervy run
 when it should be read from a constant file

 --------------------------------------------------------------------------- """

import numpy as np
import linecache
import matplotlib.pyplot as plt
from scipy.interpolate import griddata,interp1d,Rbf
import utm
from stochopy import MonteCarlo, Evolutionary
import time
from mpi4py import MPI
import os
import seaborn as sns
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
def MT1D_analytic(thick, rho, per):
    """ 
    Forward Modelling MT1D using analytic function 
    - inputs: 
      thick : layer thickness
      rho : resisistivity
      per : period
    - outputs
    """
    if len(thick) == len(rho):
        thick = thick[0:-1]
    nlay = len(rho)
    frequencies = 1 / per
    amu = 4 * np.pi * 10**(-7) #Magnetic Permeability (H/m)
    Z = np.empty(len(per), dtype=complex)
    arho = np.empty(len(per))
    phase = np.empty(len(per))   
    for iff, frq in enumerate(frequencies):
        nlay = len(rho)
        w =  2 * np.pi * frq       
        imp = list(range(nlay))
        #compute basement impedance
        imp[nlay-1] = np.sqrt(w * amu * rho[nlay-1] * 1j)
        for j in range(nlay-2, -1, -1):
            rholay = rho[j]
            thicklay = thick[j]
            # 3. Compute apparent rholay from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt((w * amu * (1/rholay))*1j)
            wj = dj * rholay
            ej = np.exp(-2*thicklay*dj)
            # The next step is to calculate the reflection coeficient (F6)
            # and impedance (F7) using the current layer intrinsic impedance 
            # and the prior computer layer impedance j+1.
            belowImp = imp[j+1]
            rj = (wj - belowImp)/(wj + belowImp)
            re = rj*ej 
            Zj = wj * ((1 - re)/(1 + re))
            imp[j] = Zj
        # Finally you can compute the apparent rholay F8 and phase F9 and 
        #  print the resulting data!
        Z[iff] = imp[0]
        absZ = abs(Z[iff])
        arho[iff] = (absZ * absZ)/(amu * w)
        phase[iff] = np.arctan2(np.imag(Z[iff]), np.real(Z[iff]))*180/np.pi
        #if convert to microvolt/m/ntesla
        Z[iff] = Z[iff] / np.sqrt(amu * amu * 10**6)
    return Z, arho, phase

# ---------------------------------------------------------------------------
def F(X):
    """
    COST FUNCTION
    input : X resistivity model X = [m_i]i=1,M in log10 space
    output : Err**2 
    """
    rest = 10**X
    cost = XHI2(rest)
    return cost
# ----------------------------------------------------------------------------
def XHI2(X):
    """ COMPUTE MT1D Mysfit USING IMPEDANCE TENSOR COMPONENTS """
    zc, rhoc, phic = MT1D_analytic(hz, X, per)
    RXHI2 = (np.sum((np.real(z) - np.real(zc))**2 / Erz**2) \
           + np.sum((np.imag(z) - np.imag(zc))**2 / Erz**2)) * 0.5
    """
    print ''
    print 'Magnetotelluric Misfit XHI2=>', RXHI2
    print ''
    """
    return RXHI2

# ----------------------------------------------------------------------------
def check_exploration(models=None, logrhosynth=None, cst_upper=None,
                      cst_lower=None):
    """
    Ensure that model exploration is within parameter window

    Parameters
    ----------
    models : ndarray
        models after exploration in log10 space
    rhosynth : np.array
        center of exploration, solution estimate when possible
    cst_upper, cst_lower : int
        inf and sup limits around rhosynth for exploration
    """
    if np.min(models - logrhosynth) < -cst_lower \
    or np.max(models - logrhosynth) > cst_upper:
        print
        print "Error model exploration out of window"
        print "min", np.min(models - logrhosynth)
        print "max", np.max(models - logrhosynth)
        print
    return None

# ----------------------------------------------------------------------------
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

# Initialize TIME
starttime = time
tt0 = time.clock()
if rank==0:
    print 'job running on ',nprocs,' processors'
    tstart = time.clock()

# ----------------------------------------------------------------------------
# Parameters
# ---> inputs
idd = '000'
conf_path = '../Config/1D/model_' + idd 
filed = conf_path + '/mod1D_Bolivia_' + idd  
error_file = conf_path + '/' + idd + '.ro'
# ---> probabilistic param
cst_lower = 2 
cst_upper = 2
max_iter = 500 

# outputs
folderout = '/postproc/COLLIN/MTD3/MCM_8nz_cst_Error'
outfile = folderout + '/mcm_exploration_' + str(rank) + '.nc'

# ----------------------------------------------------------------------------
# End of parameters

if (rank == 0) and (not os.path.exists(folderout)):
    os.makedirs(folderout)

# DECLARE VARIABLE FOR MPI
#-------------------------
rho = None
hz = None
per = None
z = None
Erz = None
nz = None
rhosynth = None

if rank==0:
    # INITIALIZE RESISTIVITY MODEL & MACKIE INPUT
    #--------------------------------------------
    #Read 1D model
    hz, rhosynth = np.loadtxt(filed, unpack=True)
    nz = len(hz)
    # ---> read MT DATA
    per, Rz, Iz, Erz, rho, Erho, phi, Ephi = np.loadtxt(error_file, unpack=True)
    z = Rz + 1j * Iz


# SHARE MPI VARIABLE
#-------------------
hz = comm.bcast(hz, root=0)
per = comm.bcast(per, root=0)
z = comm.bcast(z, root=0)
Erz = comm.bcast(Erz, root=0)
nz = comm.bcast(nz, root=0)
rhosynth = comm.bcast(rhosynth, root=0)

# ---------------------------------------------------------------------------
def F(X):
    """
    COST FUNCTION
    input : X resistivity model X = [m_i]i=1,M in log10 space
    output : Err**2 
    """
    rest = 10**X
    cost = XHI2(rest)
    return cost
# ----------------------------------------------------------------------------
def XHI2(X):
    """ COMPUTE MT1D Mysfit USING IMPEDANCE TENSOR COMPONENTS """
    zc, rhoc, phic = MT1D_analytic(hz, X, per)
    RXHI2 = (np.sum((np.real(z) - np.real(zc))**2 / Erz**2) \
           + np.sum((np.imag(z) - np.imag(zc))**2 / Erz**2)) * 0.5
    """
    print ''
    print 'Magnetotelluric Misfit XHI2=>', RXHI2
    print ''
    """
    return RXHI2

# ----------------------------------------------------------------------------
# ---->  MINIMISATION OF THE COST FUNCTION USING MCM ALGORITHM 

n_dim = nz
Xstart = None
if rank==0:
    Xstart = np.zeros((n_dim))
    Xstart[:] = np.log10(rhosynth) + np.random.uniform(low=-cst_lower,
                                           high=cst_upper, size=n_dim)

Xstart = comm.bcast(Xstart, root=0)
lower = np.log10(rhosynth) - np.ones(n_dim) * cst_lower 
upper = np.log10(rhosynth) + np.ones(n_dim) * cst_upper 


# Initialize SOLVER
mc = MonteCarlo(F, lower = lower, upper = upper, max_iter = max_iter)

# SOLVE
xopt, gfit = mc.sample(sampler = "pure")

# check
check_exploration(models=mc.models, logrhosynth=np.log10(rhosynth),
                  cst_upper=cst_upper, cst_lower=cst_lower)

# ---> Each process writes outputs

if os.path.isfile(outfile):
    os.remove(outfile)
nparam = n_dim
# ---> maybe a check dimension and return 0 or 1 
nc = Dataset(outfile, "w", format='NETCDF4')
# dimensions: name, size
nc.createDimension('max_iter', max_iter) 
nc.createDimension('nz', nz)
nc.createDimension('nparam', len(xopt))
# Variables: name, format, shape
nc.createVariable('hz', 'f8', ('nz',))
nc.createVariable('rho_i', 'f8', ('nz',)) 
nc.createVariable('xopt', 'f8', ('nparam'))
nc.createVariable('log_xopt', 'f8', ('nparam'))
nc.createVariable('models', 'f8', ('max_iter', 'nparam'))
nc.createVariable('energy', 'f8', ('max_iter'))  
nc.createVariable('rho_opt', 'f8', ('nz')) 
# Filling values
nc.variables['rho_i'][:] = rhosynth 
nc.variables['hz'][:] = hz
nc.variables['rho_opt'][:] = 10**xopt
nc.variables['xopt'][:] = xopt
nc.variables['models'][:, :] = mc.models[:, :]
nc.variables['energy'][:] = mc.energy[:]
nc.close()

print
print "total ellapsed time :", time.clock() - tt0
print

# --------------------------------------

