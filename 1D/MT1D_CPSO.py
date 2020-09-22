#!/usr/bin/python2.7
# ----------------------------------------------------------------------------
# CPSO for MT1D problems 
#
# ----------------------------------------------------------------------------

import sys
sys.path.append('/data/users/j/jcollin/MT3D_CPSO/Analyse/')
import numpy as np
import linecache
import matplotlib.pyplot as plt
from scipy.interpolate import griddata,interp1d,Rbf
import utm
from stochopy import MonteCarlo, Evolutionary
from time import time
from mpi4py import MPI
import os
import seaborn as sns
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
def MT1D_analytic(thick, rho, per):
    if len(thick) == len(rho):
        thick = thick[0:-1]

    nlay = len(rho)
    frequencies = 1 / per
    amu = 4 * np.pi * 10**(-7) #Magnetic Permeability (H/m)
    Z = np.empty(len(per), dtype=complex)
    arho = np.empty(len(per))
    phase = np.empty(len(per))   
    for iff,frq in enumerate(frequencies):
        nlay =len(rho)
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
#---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

# Initialize TIME
starttime = time()

if rank==0:
    print 'job running on ',nproc,' processors'

cst_lower = 2 
cst_upper = 2

# DECLARE VARIABLE FOR MPI
#-------------------------
filed = None
popsize = None
max_iter = None
rho = None
hx = None
hy = None
hz = None
per = None
mod1D = None
z = None
Erz = None
nz = None
rhosynth = None
FLAGS=None

if rank==0:
    # INITIALIZE RESISTIVITY MODEL & MACKIE INPUT
    #--------------------------------------------
    #Read 1D model
    filed='mod1D_Bolivia_001'#raw_input("Initial 1D MT model:")
    hz,rhosynth=np.loadtxt(filed,unpack=True)
    nz=len(hz)
    hx=np.array([10000])#np.ones([5])*1000
    hy=np.array([10000])#np.ones([5])*1000
    nx=1
    ny=1
    # CPSO parameter
    popsize=8#input("Size of model SWARM:")
    max_iter=100*nz#input("Number of iteration:")
    """
    #Periods
    per=np.loadtxt('per',skiprows=1)
    nper=len(per)
    # COMPUTE MT1D DATA
    #---------------------
    print ' '
    print ' #################'
    print ' -----------------'
    print ' COMPUTE MT1D DATA'
    print ' -----------------'
    print ' #################'
    print ' '
    z,rho,phi=MT1D_analytic(hz,rhosynth,per)
    # NOISE & ERROR LEVEL IN %
    #------------------------#
    noise=0.005
    error=0.05
    #------------------------#
    # Adding noise to Z
    Erz=error*np.abs(z)
    Rz=np.real(z)+np.random.normal(0,noise,len(z))*np.abs(z)
    Iz=np.imag(z)+np.random.normal(0,noise,len(z))*np.abs(z)
    # Rho & Phi
    rho=rho+np.random.normal(0,noise,len(z))*abs(rho)
    Erho=error*abs(rho)
    phi=phi+np.random.normal(0,noise,len(z))*np.abs(phi)
    Ephi=error*np.abs(phi)
    # WRITE DATA FILE FORMAT *.ro11, *.ro12, *.ro21, *.ro22
    #-----------------------------------------------------
    idd='001'    
    file='data/'+idd+'.ro'
    #WRITE DATA
    np.savetxt(file,np.transpose(np.vstack((per,Rz,Iz,Erz,rho,Erho,phi,Ephi))),fmt= '%16.10f',delimiter='     ')
    """
    # READ MT1D DATA
    #---------------------
    print ' '
    print ' #################'
    print ' -----------------'
    print ' READ MT1D DATA'
    print ' -----------------'
    print ' #################'
    print ' '
    idd='001'
    per,Rz,Iz,Erz,rho,Erho,phi,Ephi=np.loadtxt(idd+'.ro',unpack=True)
    z=Rz+1j*Iz


# SHARE MPI VARIABLE
#-------------------
hx = comm.bcast(hx, root=0)
hy = comm.bcast(hy, root=0)
hz = comm.bcast(hz, root=0)
per = comm.bcast(per, root=0)
mod1D = comm.bcast(mod1D, root=0)
z = comm.bcast(z, root=0)
Erz = comm.bcast(Erz, root=0)
nz = comm.bcast(nz, root=0)
rhosynth = comm.bcast(rhosynth, root=0)
popsize = comm.bcast(popsize, root=0)
max_iter = comm.bcast(max_iter, root=0)
FLAGS=comm.bcast(FLAGS, root=0)

#---------------------------------#
#                                 #
# DEFINITION OF THE COST FUNCTION #
#                                 #
#---------------------------------#

#COST FUNCTION
def F(X):
    rest = 10**X
    cost = XHI2(rest)
    return cost

# ----------------------------------------------------------------------------
#MT MISFIT
def XHI2(X):
     # COMPUTE MT1D RESP
    #---------------------
    print ' '
    print ' #################'
    print ' -----------------'
    print ' COMPUTE MT1D RESP'
    print ' -----------------'
    print ' #################'
    print ' '
    zc, rhoc, phic = MT1D_analytic(hz, X, per)
    print 
    #---------------------------------------------------
    #COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
    #---------------------------------------------------
    XHI2 = (sum((np.real(z) - np.real(zc))**2 / Erz**2) + \
            sum((np.imag(z) - np.imag(zc))**2 / Erz**2)) * 0.5
    print ''
    print 'Magnetotelluric Misfit XHI2=>', XHI2
    print ''
    return XHI2


#-----------------------------------#
#                                   #
# MINIMISATION OF THE COST FUNCTION #
# USING CPSO ALGORITHM (K.Luu 2018) #
#                                   #
#-----------------------------------#

n_dim = nz

Xstart = None
if rank == 0:
    Xstart = np.zeros((popsize, n_dim))
    for i in range(popsize):
        Xstart[i, :] = np.log10(rhosynth) + np.random.uniform(low=-cst_lower,
                                                  high=cst_upper, size=n_dim)

Xstart = comm.bcast(Xstart, root=0)

lower = np.log10(rhosynth) - np.ones(n_dim) * cst_lower 
upper = np.log10(rhosynth) + np.ones(n_dim) * cst_upper 

# Initialize SOLVER
ea = Evolutionary(F, lower = lower, upper = upper, popsize = popsize, 
                  max_iter = max_iter, mpi = True, snap = True)

# SOLVE
xopt, gfit = ea.optimize(solver = "cpso", xstart = Xstart , sync = True)


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


if rank==0:
    outfile=filed+'.nc'
    n_jobs=1
    nparam=n_dim
    print "Writting in ", outfile
    print "models shape:", np.shape(ea.models)
    print ""
    print "hx:", hx
    print "xopt", np.shape(xopt)
    print "models", np.shape(ea.models)
    print "energy", np.shape(ea.energy)
 
    # ---> maybe a check dimension and return 0 or 1 
    nc = Dataset(outfile, "w", format='NETCDF4')
    # dimensions: name, size
    nc.createDimension('max_iter', max_iter) 
    nc.createDimension('nx', nx)
    nc.createDimension('ny', ny)
    nc.createDimension('nz', nz)
    nc.createDimension('nparam', len(xopt))
    nc.createDimension('popsize', np.shape(ea.models)[0])
    nc.createDimension('n_jobs', n_jobs)
    # Variables: name, format, shape
    nc.createVariable('hx', 'f8', ('nx'))
    nc.createVariable('hy', 'f8', ('ny'))
    nc.createVariable('hz', 'f8', ('nz'))
    nc.createVariable('rho_i', 'f8', ('nx','ny','nz')) 
    nc.createVariable('xopt', 'f8', ('nparam'))
    nc.createVariable('log_xopt', 'f8', ('nparam'))
    nc.createVariable('models', 'f8', ('n_jobs','popsize', 'nparam', 'max_iter'))
    nc.createVariable('energy', 'f8', ('n_jobs','popsize', 'max_iter'))  
    nc.createVariable('rho_opt', 'f8', ('nx', 'ny', 'nz', 'n_jobs')) 

    
    # FILLING VALUES
    # ---> Only if i_job == 1
    nc.variables['hx'][:] = hx
    nc.variables['hy'][:] = hy
    nc.variables['hz'][:] = hz
    # ---> model_i useless
    # ---> 10**Xi[model_i-1] distrib ini
    # ---> 10**Xopt[model_i-1] !! distrib_3D_opt (n_job,nx,ny,nz)
    nc.variables['rho_i'][0,0,:] = rhosynth
        
    # ----> modify [it_start:it_end]
    # ----> xopt and log_xopt are erased after each jobi
    # ----> Removed last 
    nc.variables['rho_opt'][0, 0, :, 0] = 10**xopt
    nc.variables['xopt'][:] = xopt
    nc.variables['models'][0,:, :, :] = ea.models[:, :, :]
    nc.variables['energy'][0,:, :] = ea.energy[:, :]
    nc.close()

"""
#-----------------------------------------------#
#                                               #
#    COMPUTE FUNCTION DENSITY PROBABILITY       #
#                                               #
#-----------------------------------------------#

nparam=len(np.unique(model))
model_inv=np.zeros((popsize*max_iter,nparam))
fit_inv=np.zeros(popsize*max_iter)
cpt=0
for i in range(max_iter):
    for j in range(popsize):
        model_inv[cpt,:]=np.around(ea.models[j,:,i],1)
        fit_inv[cpt]=ea.energy[j,i]/1e6
        cpt=cpt+1
    
#   print ea.models.shape, ea.energy.shape
#   print ea.models
#   print ea.energy
#   print model_inv, fit_inv
JPPD=np.exp(-fit_inv/2)/sum(np.exp(-fit_inv/2))
#   print JPPD
PMM=np.zeros(nparam)
for i in range(nparam):
    PMM[i]=sum(model_inv[:,i]*JPPD[:])
    
Cm=np.zeros((nparam,nparam))
Mm=PMM
Mm.shape=(nparam,1)
Mm_t=np.transpose(Mm)
for i in range(popsize*max_iter):
    m=model_inv[i,:]
    m.shape=(nparam,1)
    m_t=np.transpose(m)
    ma=(m-Mm)
    mb=np.transpose(ma)
    #Cm=Cm+np.dot(m,m_t)*JPPD[i]-np.dot(Mm,Mm_t)*JPPD[i]
    Cm=Cm+np.dot(ma,mb)*JPPD[i]
    
StD=np.zeros(nparam)
for i in range(nparam):
    StD[i]=np.sqrt(Cm[i,i])
    
print StD


#-----------------------------------#
#                                   #
#  WRITING BEST RESISTIVITY MODEL   #
#                                   #
#-----------------------------------#

xopt=np.around(xopt,1)          
Xd=10**xopt[model-1]
model_i=np.zeros((nx,ny,nz))
cpt=1
for k in np.arange(nz):
    for j in np.arange(ny):
        for i in np.arange(nx):
            model_i[i,j,k]=cpt
            cpt=cpt+1


filefmt=open('3DRHO_BEST.rslt',"w")
filefmt.write(str(nx)+'     '+str(ny)+'     '+str(nz))
filefmt.write('\n')
for i in np.arange(nx):
    filefmt.write(str(hx[i])+'      ')

filefmt.write('\n')
for j in np.arange(ny):
    filefmt.write(str(hy[j])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(hz[k])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(k+1))
    filefmt.write('\n')
    for j in np.arange(ny):
        for i in np.arange(nx):
            filefmt.write(str(int(model_i[i,j,k]))+'       ')

        filefmt.write('\n')


filefmt.write('0.00000000000      ')
for i in np.arange(nx*ny*nz):
    filefmt.write(str(float(Xd[i]))+'     ')

filefmt.write('\n')
filefmt.close()


#-----------------------------------#
#                                   #
#  WRITING MEAN RESISTIVITY MODEL   #
#                                   #
#-----------------------------------#

Xd=10**PMM[model-1]
model_i=np.zeros((nx,ny,nz))
cpt=1
for k in np.arange(nz):
    for j in np.arange(ny):
        for i in np.arange(nx):
            model_i[i,j,k]=cpt
            cpt=cpt+1


filefmt=open('3DRHO_MEAN.rslt',"w")
filefmt.write(str(nx)+'     '+str(ny)+'     '+str(nz))
filefmt.write('\n')
for i in np.arange(nx):
    filefmt.write(str(hx[i])+'      ')

filefmt.write('\n')
for j in np.arange(ny):
    filefmt.write(str(hy[j])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(hz[k])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(k+1))
    filefmt.write('\n')
    for j in np.arange(ny):
        for i in np.arange(nx):
            filefmt.write(str(int(model_i[i,j,k]))+'       ')

        filefmt.write('\n')


filefmt.write('0.00000000000      ')
for i in np.arange(nx*ny*nz):
    filefmt.write(str(float(Xd[i]))+'     ')

filefmt.write('\n')
filefmt.close()  


#-----------------------------------#
#                                   #
#  WRITING STD RESISTIVITY MODEL   #
#                                   #
#-----------------------------------#

Xd=10**StD[model-1]
model_i=np.zeros((nx,ny,nz))
cpt=1
for k in np.arange(nz):
    for j in np.arange(ny):
        for i in np.arange(nx):
            model_i[i,j,k]=cpt
            cpt=cpt+1


filefmt=open('3DRHO_STD.rslt',"w")
filefmt.write(str(nx)+'     '+str(ny)+'     '+str(nz))
filefmt.write('\n')
for i in np.arange(nx):
    filefmt.write(str(hx[i])+'      ')

filefmt.write('\n')
for j in np.arange(ny):
    filefmt.write(str(hy[j])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(hz[k])+'      ')

filefmt.write('\n')
for k in np.arange(nz):
    filefmt.write(str(k+1))
    filefmt.write('\n')
    for j in np.arange(ny):
        for i in np.arange(nx):
            filefmt.write(str(int(model_i[i,j,k]))+'       ')

        filefmt.write('\n')


filefmt.write('0.00000000000      ')
for i in np.arange(nx*ny*nz):
    filefmt.write(str(float(Xd[i]))+'     ')

filefmt.write('\n')
filefmt.close()  


#COMPUTE F with the MEAN solution array
#-------------------------------------
F(PMM)
print (ea)
print "Elapsed time: %.2f seconds" % (time() - starttime)
    

#PLOT MISFIT EVOLUTION
#-----------------------
mean_energy=np.zeros(max_iter)
best_energy=np.zeros(max_iter)
for i in range(max_iter):
    mean_energy[i]=sum(ea.energy[:,i])/popsize
    best_energy[i]=np.min(ea.energy[:,i])

np.savetxt('Mean_Ener.txt',mean_energy)
np.savetxt('Best_Ener.txt',best_energy)
plt.plot(np.arange(max_iter)+1,mean_energy,c='b',linewidth=2,label='Mean F(m)')
plt.plot(np.arange(max_iter)+1,best_energy,c='r',linewidth=2,label='Min F(m)')
plt.xlabel('Iterations',fontsize=15)
plt.ylabel('F(m)',fontsize=15)
xtick=np.arange(1,max_iter+1,10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(1,max_iter+0.1)
# plt.yscale('log', nonposy='clip')
plt.legend()
plt.savefig('MISFIT_Evol.png')
plt.show()

"""
