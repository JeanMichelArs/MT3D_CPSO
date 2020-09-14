#!/usr/bin/python2.7
# ----------------------------------------------------------------------------
# CPSO for MT1D problems 
#
# For multiple runs python takes an input argument 
# > mpirun -np $nprocs python MT1D_CPSO_chained.py $irun
#
# where irun is an integer referring to the number of the run
# output files will be written in outdir
# ----------------------------------------------------------------------------

import sys
sys.path.append('../../Forward_MT/')
import numpy as np
import linecache
import matplotlib.pyplot as plt
import mackie3d
from scipy.interpolate import griddata,interp1d,Rbf
import utm
from stochopy import MonteCarlo, Evolutionary
from time import time
from mpi4py import MPI
import os
import seaborn as sns
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
outdir = '/postproc/COLLIN/MTD3/Bolivia_1D_8param/test'
irun = sys.argv[1]

# ----------------------------------------------------------------------------
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
perpy = None
mod1D = None
z = None
Erz = None
nz = None
rhosynth = None

if rank==0:
    # INITIALIZE RESISTIVITY MODEL & MACKIE INPUT
    #--------------------------------------------
    #Read 1D model
    conf_path = '../../Config/1D'
    filed = conf_path + '/mod1D_Bolivia_001'#raw_input("Initial 1D MT model:")
    hz,rhosynth = np.loadtxt(filed,unpack=True)
    nz = len(hz)
    hx = np.array([10000])#np.ones([5])*1000
    hy = np.array([10000])#np.ones([5])*1000
    nx = 1
    ny = 1
    # CPSO parameter
    popsize = 18 #input("Size of model SWARM:")
    max_iter = 100 *nz #input("Number of iteration:")
    #Periods
    perpy = np.loadtxt(conf_path + '/per',skiprows=1)
    nper = len(perpy)
    #Create 1D bottom
    mod1D = np.loadtxt(conf_path + '/ini1d',skiprows=1)
    n1d = len(mod1D)
    # FORTRAN FORMAT ARRAYS
    mod1D = np.asfortranarray(mod1D)
    perpy = np.asfortranarray(perpy)
    hx = np.asfortranarray(hx)
    hy = np.asfortranarray(hy)
    hz = np.asfortranarray(hz)
    # COMPUTE MT1D DATA
    #---------------------
    print ' '
    print ' #################'
    print ' -----------------'
    print ' COMPUTE MT1D RESP'
    print ' -----------------'
    print ' #################'
    print ' '
    mod1D = np.asfortranarray(mod1D)
    perpy = np.asfortranarray(perpy)
    resistpy = np.asfortranarray(rhosynth)
    hx = np.asfortranarray(hx)
    hy = np.asfortranarray(hy)
    hz = np.asfortranarray(hz)
    # CALL MACKIE
    hxsurf, hysurf, hzsurf, exsurf, eysurf = mackie3d.mtd3(mod1D, perpy,
                                                           resistpy,hx,hy,hz)
    ###########################
    #                         #
    #       MT1D TENSOR       #
    #                         #
    ###########################
    fmu = 4 * np.pi * 10**(-4)
    amu = 4 * np.pi * 10**(-7)
    # NOISE & ERROR LEVEL IN %
    #------------------------#
    noise = 0.005
    error = 0.05
    #------------------------#
    per = perpy
    hxi = hxsurf[:,:,0]
    hyi = hysurf[:,:,0]
    hzi = hzsurf[:,:,0]
    exi = exsurf[:,:,0]
    eyi = eysurf[:,:,0]
    # COMPUTE IMPEDANCE TENSOR
    #-------------------------
    # mackie en -iomegat, donc partie conjuguee des champs pour etre
    # meme cadran que 2d, data,...
    hxc = np.conj(hxi)   # (per,pol,site)
    hyc = np.conj(hyi)
    hzc = np.conj(hzi)
    exc = np.conj(exi)
    eyc = np.conj(eyi)
    # Determinant
    det = -1*(hxc[:,0] * hyc[:,1] - hxc[:,1] * hyc[:,0])
    # ANTI-DIAG TERM => Zyx
    z = hyc[:,1] * eyc[:,0] - hyc[:,0] * eyc[:,1]
    z = -z / (det * fmu)
    # Adding noise to Zyx
    Erz = error * np.abs(z) #np.random.normal(0,error,len(zyx))*np.abs(zyx)+1e-6
    Rz = np.real(z) + np.random.normal(0, noise, len(z)) * np.abs(z)
    Iz = np.imag(z) + np.random.normal(0, noise, len(z)) * np.abs(z)
    # Roayx & Phiyx
    zt = np.abs(z)
    rho = zt * zt * amu * 10**6 / (2. * np.pi / per)
    rho = 10**(np.log10(rho) + np.random.normal(0, noise * np.mean(abs(np.log10(rho))),len(z)))
    Erho = 10**(np.random.normal(0, error * np.mean(abs(np.log10(rho))), len(z)))
    phi = np.arctan2(np.imag(z), np.real(z)) * 180 / np.pi
    phi = phi + np.random.normal(0, noise * np.mean(abs(phi)), len(z))
    Ephi = np.random.normal(0, error * np.mean(abs(phi)), len(z))
    # WRITE DATA FILE FORMAT *.ro11, *.ro12, *.ro21, *.ro22
    #-----------------------------------------------------
    idd = '001'    
    file = conf_path + '/' + idd + '.ro'
    #WRITE DATA
    np.savetxt(file, np.transpose((per, Rz, Iz, Erz, rho, Erho, phi, Ephi)), 
               fmt= '%16.10f',delimiter='     ')


# SHARE MPI VARIABLE
#-------------------
hx = comm.bcast(hx, root=0)
hy = comm.bcast(hy, root=0)
hz = comm.bcast(hz, root=0)
perpy = comm.bcast(perpy, root=0)
mod1D = comm.bcast(mod1D, root=0)
z = comm.bcast(z, root=0)
Erz = comm.bcast(Erz, root=0)
nz = comm.bcast(nz, root=0)
rhosynth = comm.bcast(rhosynth, root=0)
popsize = comm.bcast(popsize, root=0)
max_iter = comm.bcast(max_iter, root=0)

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
    ###rest=np.zeros(nx*ny*nz)
    ###print model.shape, rest.shape
    ###print np.unique(model)
    ###for i in range(len(rest)):
    ###    rest[i]=10**X[model[i]-1]
    ###




#MT MISFIT
def XHI2(X):
    # FORTRAN FORMAT ARRAYS
    X = np.asfortranarray(X)
    print ' '
    print ' ################'
    print ' ----------------'
    print ' CALL MACKIE F2PY'
    print ' ----------------'
    print ' ################'
    print ' '
    #print mod1D.shape, perpy.shape, X.shape, hx.shape,hy.shape,hz.shape
    hxsurf,hysurf,hzsurf,exsurf,eysurf=mackie3d.mtd3(mod1D,perpy,X,hx,hy,hz)
    ###########################
    #                         #
    #       MT MISFIT         #
    #                         #
    ###########################
    fmu=4*np.pi*10**(-4)
    amu=4*np.pi*10**(-7)
    per_c=perpy
    XHI2=0  
    #------------------------#
    perd=perpy
    hxi=hxsurf[:,:,0]
    hyi=hysurf[:,:,0]
    hzi=hzsurf[:,:,0]
    exi=exsurf[:,:,0]
    eyi=eysurf[:,:,0]
    # COMPUTE IMPEDANCE TENSOR
    #-------------------------
    # mackie en -iomegat, donc partie conjuguee des champs pour etre
    # meme cadran que 2d, data,...
    hxc=np.conj(hxi)   # (per,pol,site)
    hyc=np.conj(hyi)
    hzc=np.conj(hzi)
    exc=np.conj(exi)
    eyc=np.conj(eyi)
    # Determinant
    det=-1*((hxc[:,0]*hyc[:,1])-(hxc[:,1]*hyc[:,0]))
    # ANTI-DIAG TERM => Zyx
    zc=(hyc[:,1]*eyc[:,0])-(hyc[:,0]*eyc[:,1])
    zc=-zc/(det*fmu)
    # Roayx & Phiyx
    zt=np.abs(zc)
    rhoc=zt*zt*amu*10**6/(2.*np.pi/perd)
    phic=np.arctan2(np.imag(zc),np.real(zc))*180/np.pi
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
    hxsurf,hysurf,hzsurf,exsurf,eysurf=mackie3d.mtd3(mod1D,perpy,10**xopt,hx,hy,hz)
    ###########################
    #                         #
    #       MT MISFIT         #
    #                         #
    ###########################
    fmu=4*np.pi*10**(-4)
    amu=4*np.pi*10**(-7)
    per_c=perpy
    XHI2=0  
    #------------------------#
    perd=perpy
    hxi=hxsurf[:,:,0]
    hyi=hysurf[:,:,0]
    hzi=hzsurf[:,:,0]
    exi=exsurf[:,:,0]
    eyi=eysurf[:,:,0]
    # COMPUTE IMPEDANCE TENSOR
    #-------------------------
    # mackie en -iomegat, donc partie conjuguee des champs pour etre
    # meme cadran que 2d, data,...
    hxc=np.conj(hxi)   # (per,pol,site)
    hyc=np.conj(hyi)
    hzc=np.conj(hzi)
    exc=np.conj(exi)
    eyc=np.conj(eyi)
    # Determinant
    det=-1*((hxc[:,0]*hyc[:,1])-(hxc[:,1]*hyc[:,0]))
    # ANTI-DIAG TERM => Zyx
    zc=(hyc[:,1]*eyc[:,0])-(hyc[:,0]*eyc[:,1])
    zc=-zc/(det*fmu)
    # Roayx & Phiyx
    zt=np.abs(zc)
    rhoc=zt*zt*amu*10**6/(2.*np.pi/per)
    phic=np.arctan2(np.imag(zc),np.real(zc))*180/np.pi
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
    outfile = outdir + irun + '.nc'
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
