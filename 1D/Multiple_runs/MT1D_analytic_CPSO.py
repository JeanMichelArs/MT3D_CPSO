#!/usr/bin/python2.7
"""
----------------------------------------------------------------------------
 CPSO MT1D with analytic forward modelling for comparison to Tarits 
 Monte Carlo Code

 For multiple runs python takes an input argument 
 
 >>> mpirun -np $nprocs python  MT1D_analytic_CPSO.py $irun
 
 where irun is an integer referring to the number of the run
 output files will be written in outdir

 ----------------------------------------------------------------------------
"""

import sys
sys.path.append('../../Forward_MT/')
import numpy as np
import linecache
import matplotlib.pyplot as plt
#import mackie3d
from scipy.interpolate import griddata,interp1d,Rbf
import utm
from stochopy import MonteCarlo, Evolutionary
from time import time
from mpi4py import MPI
import os
import seaborn as sns
from netCDF4 import Dataset


# ----------------------------------------------------------------------------
def MT1D_analytic(thick,rho,per):
    if len(thick)==len(rho):
        thick=thick[0:-1]

    nlay=len(rho)
    frequencies = 1/per
    amu=4*np.pi*10**(-7) #Magnetic Permeability (H/m)
    Z=np.empty(len(per),dtype=complex)
    arho=np.empty(len(per))
    phase=np.empty(len(per))   
    for iff,frq in enumerate(frequencies):
        nlay=len(rho)
        w =  2*np.pi*frq       
        imp = list(range(nlay))
        #compute basement impedance
        imp[nlay-1] = np.sqrt(w*amu*rho[nlay-1]*1j)
        for j in range(nlay-2,-1,-1):
            rholay = rho[j]
            thicklay = thick[j]
            # 3. Compute apparent rholay from top layer impedance
            #Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt((w * amu * (1/rholay))*1j)
            wj = dj * rholay
            ej = np.exp(-2*thicklay*dj)
            #The next step is to calculate the reflection coeficient (F6) and impedance (F7) using the current layer intrinsic impedance and the prior computer layer impedance j+1.
            belowImp = imp[j+1]
            rj = (wj - belowImp)/(wj + belowImp)
            re = rj*ej 
            Zj = wj * ((1 - re)/(1 + re))
            imp[j] = Zj
    
        #Finally you can compute the apparent rholay F8 and phase F9 and print the resulting data!
        Z[iff] = imp[0]
        absZ = abs(Z[iff])
        arho[iff] = (absZ * absZ)/(amu * w)
        phase[iff] = np.arctan2(np.imag(Z[iff]), np.real(Z[iff]))*180/np.pi
        #if convert to microvolt/m/ntesla
        Z[iff]=Z[iff]/np.sqrt(amu*amu*10**6)

    return Z,arho,phase


# ----------------------------------------------------------------------------

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

# Initialize TIME
starttime = time()

if rank==0:
    print 'job running on ', nproc, ' processors'

# ---> inputs
conf_path = '../../Config/1D/model_001'
filed = conf_path + '/mod1D_Bolivia_001'

# ---> cpso parameters
cst_lower = 2 
cst_upper = 2
popsize = 8
max_iter =  100 * popsize 

# ----> outputs
outdir = '/postproc/COLLIN/MTD3/Calibre_CPSO_8nz_pop8'
irun = sys.argv[1]


if (rank==0) and (not os.path.exists(outdir)):
    os.makedirs(outdir)


# DECLARE VARIABLE FOR MPI
#-------------------------
rho = None
hz = None
per = None
mod1D = None
z = None
Erz = None
nz = None
rhosynth = None
FLAGS = None

if rank==0:
    # INITIALIZE RESISTIVITY MODEL & MACKIE INPUT
    #--------------------------------------------
    #Read 1D model
    hz, rhosynth = np.loadtxt(filed, unpack=True)
    nz = len(hz)
    # READ MT1D DATA
    #---------------------
    print ' '
    print ' #################'
    print ' -----------------'
    print ' READ MT1D DATA'
    print ' -----------------'
    print ' #################'
    print ' '
    idd = '001'
    error_file = conf_path + '/' + idd + '.ro'
    per, Rz, Iz, Erz, rho, Erho, phi, Ephi = np.loadtxt(error_file, unpack=True)
    z = Rz + 1j * Iz


# SHARE MPI VARIABLE
#-------------------
hz = comm.bcast(hz, root=0)
per = comm.bcast(per, root=0)
mod1D = comm.bcast(mod1D, root=0)
z = comm.bcast(z, root=0)
Erz = comm.bcast(Erz, root=0)
nz = comm.bcast(nz, root=0)
rhosynth = comm.bcast(rhosynth, root=0)
FLAGS = comm.bcast(FLAGS, root=0)

#---------------------------------#
#                                 #
# DEFINITION OF THE COST FUNCTION #
#                                 #
#---------------------------------#

#COST FUNCTION
def F(X):
    rest=10**X
    cost=XHI2(rest)
    return cost

#MT MISFIT
def XHI2(X):
     # COMPUTE MT1D RESP
    zc, rhoc, phic = MT1D_analytic(hz, X, per)
    #---------------------------------------------------
    #COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
    #---------------------------------------------------
    XHI2=(sum((np.real(z)-np.real(zc))**2/Erz**2)+sum((np.imag(z)-np.imag(zc))**2/Erz**2))/2
    print ''
    print 'Magnetotelluric Misfit XHI2=>',XHI2
    print ''
    return XHI2


#-----------------------------------#
#                                   #
# MINIMISATION OF THE COST FUNCTION #
# USING CPSO ALGORITHM (K.Luu 2018) #
#                                   #
#-----------------------------------#

# changing to run directory
os.chdir(outdir)

n_dim = nz

Xstart = None
if rank==0:
    Xstart=np.zeros((popsize,n_dim))
    for i in range(popsize):
        Xstart[i, :] = np.log10(rhosynth) + np.random.uniform(low=-cst_lower, high=cst_upper, size=n_dim)

Xstart = comm.bcast(Xstart, root=0)

lower = np.log10(rhosynth) - np.ones(n_dim) * cst_lower 
upper = np.log10(rhosynth) + np.ones(n_dim) * cst_upper 

# Initialize SOLVER
ea = Evolutionary(F, lower = lower, upper = upper, popsize = popsize, max_iter = max_iter, mpi = True, snap = True)

# SOLVE
xopt,gfit=ea.optimize(solver = "cpso", xstart = Xstart , sync = True)

if rank==0:
    print ' '
    print ' #################'
    print ' -----------------'
    print ' COMPUTE MT1D BEST'
    print ' -----------------'
    print ' #################'
    print ' '
    print 'RHO_BEST:',10**xopt
    zc,rhoc,phic=MT1D_analytic(hz,10**xopt,per)
    #---------------------------------------------------
    #COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
    #---------------------------------------------------
    XHI2=(sum((np.real(z)-np.real(zc))**2/Erz**2)+sum((np.imag(z)-np.imag(zc))**2/Erz**2))/2
    print ''
    print 'Magnetotelluric Best Misfit =>',XHI2
    print ''
    #PLOT IMPEDANCE TENSOR
    #---------------------
    plt.figure(figsize=(14,15))
    plt.subplot(211)
    plt.errorbar(per,rho,yerr=Erho, label="$Z_{1D}^{data}$",fmt='o',markersize=10,elinewidth=2.5,color='blue')
    plt.plot(per,rhoc,label="$Z_{1D}^{resp}$",linewidth=3,color='red')
    plt.ylim(5,300)
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    plt.xlabel('period (sec)',fontsize=40)
    plt.ylabel('Roa (Ohm.m)',fontsize=40)
    xtick=10**np.arange(round(np.log10(min(per)),0),round(np.log10(max(per)),0)+1)#np.array([1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
    ytick=np.array([1,1e1,1e2,1e3])
    plt.xticks(xtick,fontsize=30)
    plt.yticks(ytick,fontsize=30)
    plt.grid(True,which="both",ls="-")
    plt.legend(prop={'size':30},loc=9,ncol=4)
    plt.subplot(212)
    plt.errorbar(per,phi,yerr=Ephi,label="$Z_{1D}^{data}$",fmt='o',markersize=10,elinewidth=2.5,color='blue')
    plt.plot(per,phic,label="$Z_{1D}^{resp}$",linewidth=3,color='red')
    plt.ylim(-180.,180,)
    xtick=10**np.arange(round(np.log10(min(per)),0),round(np.log10(max(per)),0)+1)#np.array([1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
    ytick=np.array([-180,-135,-90,-45,0,45,90,135,180])
    plt.xticks(xtick,fontsize=30)
    plt.yticks(ytick,fontsize=30)
    plt.xscale('log', nonposx='clip')
    plt.xlabel('period (sec)',fontsize=40)
    plt.ylabel('phase(deg)',fontsize=40)
    plt.suptitle('site '+idd,fontsize=40)
    plt.grid(True,which="both",ls="-")
    plt.savefig('resp'+idd+'.eps',dpi=500, bbox_inches='tight')
    # Print time
    print("Elapsed time: %.2f seconds" % (time() - starttime))


# ---> Write data in netcdf file
if rank==0:
    outfile = outdir + '/test_' + irun + '.nc'
    nparam = n_dim
    print "Writting in ", outfile
    print "models shape:", np.shape(ea.models)
    print ""
    print "xopt", np.shape(xopt)
    print "models", np.shape(ea.models)
    print "energy", np.shape(ea.energy)
    nc = Dataset(outfile, "w", format='NETCDF4')
    # ---> dimensions: name, size
    nc.createDimension('max_iter', max_iter) 
    nc.createDimension('nz', nz)
    nc.createDimension('nparam', len(xopt))
    nc.createDimension('popsize', np.shape(ea.models)[0])
    # ---> Variables: name, format, shape
    nc.createVariable('hz', 'f8', ('nz'))
    nc.createVariable('rho_i', 'f8', ('nz')) 
    nc.createVariable('xopt', 'f8', ('nparam'))
    nc.createVariable('log_xopt', 'f8', ('nparam'))
    nc.createVariable('models', 'f8', ('popsize', 'nparam', 'max_iter'))
    nc.createVariable('energy', 'f8', ('popsize', 'max_iter'))  
    nc.createVariable('rho_opt', 'f8', ('nz')) 
    # ---> FILLING VALUES
    nc.variables['hz'][:] = hz
    nc.variables['rho_i'][:] = rhosynth
    nc.variables['rho_opt'][:] = 10**xopt
    nc.variables['xopt'][:] = xopt
    nc.variables['models'][:, :, :] = ea.models[:, :, :]
    nc.variables['energy'][:, :] = ea.energy[:, :]
    nc.close()

