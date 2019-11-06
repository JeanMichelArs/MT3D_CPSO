#!/usr/bin/python2.7
'''
- max_iter is no longer found in input file but is defined in this script
- I/O modification using netcdf file instead


'''


import numpy as np
import linecache
import matplotlib.pyplot as plt
#import mackie3d
from scipy.interpolate import griddata,interp1d,Rbf
#import utm
from stochopy import MonteCarlo, Evolutionary
from time import time
from mpi4py import MPI
import os
from netCDF4 import Dataset
#import seaborn as sns

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()


# Initialize TIME
starttime = time()

if rank==0:
    mysfit = 'norme'
    config = '4_param'
    method = 'cpso'
    outfile = method + '_' + mysfit + '_' + config + '.nc'
    print "Output file : ", outfile  
    print 'job running on ',nproc,' processors'

#------------------------
# Chained Variables n_jobs and max_iter have to be edited in mk_files
# Automaically replaced via mk_files 
# index of current job, total number of jobs and total/local maximum iteration 
i_job = 1
n_jobs = 2
max_iter = 50
n_iter_job = 25
it_start = (i_job - 1) * n_iter_job
it_end = np.min([i_job*n_iter_job, max_iter])


# DECLARE VARIABLE FOR MPI
#-------------------------
rho=None
nx=None
ny=None
nz=None
hx=None
hy=None
hz=None
perpy=None
nper=None
mod1D=None
n1d=None
refmtx=None
refmty=None
angm=None
mum=None
lambdam=None
dx=None
dy=None
dz=None
lab=None
site_id=None
ImpT_data=None
RoPh_data=None
popsize=None
#max_iter=None
model=None
Xi=None
tmpDir=None



#if rank==0:
#    tmpDir = os.environ.get('TMPDIR') + '/'



if rank==0:
    # INITIALIZE RESISTIVITY MODEL & MACKIE INPUT
    #--------------------------------------------
    
    #Read 3D model
    file=raw_input("Initial 3D MT model:")
    ld=linecache.getline(file, 1)
    lx=linecache.getline(file, 2)
    ly=linecache.getline(file, 3)
    lz=linecache.getline(file, 4)
    dim=ld.split()
    nx=int(dim[0])
    ny=int(dim[1])
    nz=int(dim[2])
    #vecteur de taille de maille
    hx=np.hstack(float(lx.split()[i]) for i in np.arange(nx) )
    hy=np.hstack(float(ly.split()[i]) for i in np.arange(ny) )
    hz=np.hstack(float(lz.split()[i]) for i in np.arange(nz) )
    #valeur modele
    ind=4+(nz)+(nz-1)*ny+(ny)
    l=linecache.getline(file, ind+1)
    Xi=np.log10(np.hstack(float(l.split()[i+1]) for i in np.arange(len(l.split())-1) ))
    ### print 'Xi',Xi.shape,Xi
    #Indice
    ind=5
    model=np.zeros((nx*ny*nz),dtype=int)
    model_i=np.zeros((nx,ny,nz),dtype=int)
    for k in np.arange(nz):
        ind=ind+1
        for i in np.arange(ny):
            ind=4+(k+1)+(k)*ny+(i+1)
            l=linecache.getline(file, ind)
            L=np.vstack(float(l.split()[i]) for i in np.arange(nx) )
            L.shape=(nx,)
            model[k*nx*ny+i*nx:k*nx*ny+i*nx+nx]=L
            model_i[:,i,k]=L
    
    ### print 'model:',model.shape,model,'unique',np.unique(model)
    rho=10**Xi[model-1]
    
    #Periods
    perpy=np.loadtxt('per',skiprows=1)
    nper=len(perpy)
    
    
    #Create 1D bottom
    mod1D=np.loadtxt('ini1d',skiprows=1)
    n1d = len(mod1D)
    
    # FORTRAN FORMAT ARRAYS
    mod1D=np.asfortranarray(mod1D)
    perpy=np.asfortranarray(perpy)
    hx=np.asfortranarray(hx)
    hy=np.asfortranarray(hy)
    hz=np.asfortranarray(hz)
    
    
    (refmty,refmtx)=input("UTM references of the MT model (lower left corner, east/north):")     #Model lower left corner, east/north
    
    angm=input("Rotation angle of the MT model (clockwise)")                #clockwise
    
    angm=angm*np.pi/180
    

    #ROTATION MESH
    if angm>0:
    	refmtxx=np.cos(angm)*refmtx+np.sin(angm)*refmty
    	refmtyy=np.cos(angm)*refmty-np.sin(angm)*refmtx
    	refmtx=refmtxx
    	refmty=refmtyy
    
    #NODES MT mesh 
    dx=np.ones(nx+1)
    for i in range(nx+1):
        dx[i]=sum(hx[0:i])+refmtx
            
    dy=np.ones(ny+1)
    for i in range(ny+1):
        dy[i]=sum(hy[0:i])+refmty
    
    dz=np.ones(nz+1)
    for i in range(nz+1):
        dz[i]=sum(hz[0:i])
    
    
    # MT SOUNDING POSITION FILE
    MTsoundingpos=str(raw_input("Magnetotelluric sounding position file (labm,xMT,yMT,zMT,rotMT) :"))
    lab,ysound,xsound,alt_sound,rot=np.loadtxt(MTsoundingpos,unpack=True)
    nsound=len(lab)
    
    
    #ROTATION MT SOUNDING
    if angm>0:
    	dxx=np.cos(angm)*xsound+np.sin(angm)*ysound
    	dyy=np.cos(angm)*ysound-np.sin(angm)*xsound
    	xsound=dxx
    	ysound=dyy
    
    # MT SOUNDING POSITION ON MESH
    sound_i=np.zeros((nsound),dtype=int)
    sound_j=np.zeros((nsound),dtype=int)
    for i in range(nsound):
        sound_i[i]=int(min(np.where(dx-xsound[i]>0)[0]))
        sound_j[i]=int(min(np.where(dy-ysound[i]>0)[0]))
    
    # site index cell
    site_id=(sound_j-1)*nx+sound_i-1
    
    
    # LOAD MT DATA
    #--------------
    
    ImpT_data=np.zeros((13,50,len(lab)))  # MAXIMUM 50 PERIODS
    RoPh_data=np.zeros((17,50,len(lab)))
    
    for n in range(len(lab)):
        nm=int(lab[n])
        if nm<=9.:
                idd='00'+str(nm)
        if nm>9 and nm<100:
                idd='0'+str(nm)
        if nm>=100:
                idd=str(nm)
                
        filexx='data/'+idd+'.ro11'
        filexy='data/'+idd+'.ro12'
        fileyx='data/'+idd+'.ro21'
        fileyy='data/'+idd+'.ro22'
        # read file
        perd,RZxx,IZxx,ErZxx,Roaxx,ErRoaxx,Phixx,ErPhixx=np.loadtxt(filexx,unpack=True)
        perd,RZxy,IZxy,ErZxy,Roaxy,ErRoaxy,Phixy,ErPhixy=np.loadtxt(filexy,unpack=True)
        perd,RZyx,IZyx,ErZyx,Roayx,ErRoayx,Phiyx,ErPhiyx=np.loadtxt(fileyx,unpack=True)
        perd,RZyy,IZyy,ErZyy,Roayy,ErRoayy,Phiyy,ErPhiyy=np.loadtxt(fileyy,unpack=True)
        # Gather data by component (Impedance Tensor matrix & RoaPhi Matrix)
        ImpT_data[:,0:len(perd),n]=(perd,RZxx,IZxx,ErZxx,RZxy,IZxy,ErZxy,RZyx,IZyx,ErZyx,RZyy,IZyy,ErZyy)
        RoPh_data[:,0:len(perd),n]=(perd,Roaxx,ErRoaxx,Phixx,ErPhixx,Roaxy,ErRoaxy,Phixy,ErPhixy,
                     Roayx,ErRoayx,Phiyx,ErPhiyx,Roayy,ErRoayy,Phiyy,ErPhiyy)
    
    popsize=input("Size of model SWARM:")
    #max_iter=input("Number of iteration:")
    
#Share variable with all MPI node
#---------------------------------
rho=comm.bcast(rho, root=0)
nx=comm.bcast(nx, root=0)
ny=comm.bcast(ny, root=0)
nz=comm.bcast(nz, root=0)
hx=comm.bcast(hx, root=0)
hy=comm.bcast(hy, root=0)
hz=comm.bcast(hz, root=0)
perpy=comm.bcast(perpy, root=0)
nper=comm.bcast(nper, root=0)
mod1D=comm.bcast(mod1D, root=0)
n1d=comm.bcast(n1d, root=0)
refmtx=comm.bcast(refmtx, root=0)
refmty=comm.bcast(refmty, root=0)
angm=comm.bcast(angm, root=0)
mum=comm.bcast(mum, root=0)
lambdam=comm.bcast(lambdam, root=0)
dx=comm.bcast(dx, root=0)
dy=comm.bcast(dy, root=0)
dz=comm.bcast(dz, root=0)
lab=comm.bcast(lab, root=0)
site_id=comm.bcast(site_id, root=0)
ImpT_data=comm.bcast(ImpT_data, root=0)
RoPh_data=comm.bcast(RoPh_data, root=0)
popsize=comm.bcast(popsize, root=0)
max_iter=comm.bcast(max_iter, root=0)
model=comm.bcast(model, root=0)
Xi=comm.bcast(Xi, root=0)
tmpDir=comm.bcast(tmpDir, root=0)

#---------------------------------#
#                                 #
# DEFINITION OF THE COST FUNCTION #
#                                 #
#---------------------------------#

#COST FUNCTION
def F(X):
    starttime = time()
    #X=np.around(X,1) # Arrondie au dixieme pour discretiser l'espace des parametres 
    rest=10**X[model-1]
    cost=XHI2(rest) - 1
    print("Elapsed time: %.2f seconds" % (time() - starttime))
    return cost



#MT MISFIT
def XHI2(X):
    # FORTRAN FORMAT ARRAYS
    X=np.asfortranarray(X)
    if rank==0:
        print ' '
        print ' ################'
        print ' ----------------'
        print ' CALL MACKIE F2PY'
        print ' ----------------'
        print ' ################'
        print ' '
    '''
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
    RMS1r=0
    RMS1p=0
    for n in range(len(lab)):
        ss=site_id[n]
        per_data=ImpT_data[0,:,n]
        aa=np.where(per_data!=0)[0]
        jj=np.where(per_data[aa]-min(per_c)>0)[0]
        cc=np.where(per_data[aa[jj]]-max(per_c)<0)[0]
        ii=aa[jj[cc]]
        perd=per_data[ii]
        hxi=np.zeros((len(ii),2),dtype=complex)
        hyi=np.zeros((len(ii),2),dtype=complex)
        hzi=np.zeros((len(ii),2),dtype=complex)
        exi=np.zeros((len(ii),2),dtype=complex)
        eyi=np.zeros((len(ii),2),dtype=complex)
    # RESPONSE (E & H fields) INTERPOLATION TO DATA PERIODS FOR EACH POLARIZATION
    #----------------------------------------------------------------------------
        for pol in range(2):
            #Raw E & H fields
            hxr=hxsurf[:,pol,ss]
            hyr=hysurf[:,pol,ss]
            hzr=hzsurf[:,pol,ss]
            exr=exsurf[:,pol,ss]
            eyr=eysurf[:,pol,ss]
            #Interpolated E & H fields / Real & Imaginary part separetely
            #Hx
            rhxi=griddata(per_c,np.real(hxr),perd,method='linear')
            ihxi=griddata(per_c,np.imag(hxr),perd,method='linear')
            hxi[:,pol]=rhxi+1j*ihxi
            #Hy
            rhyi=griddata(per_c,np.real(hyr),perd,method='linear')
            ihyi=griddata(per_c,np.imag(hyr),perd,method='linear')
            hyi[:,pol]=rhyi+1j*ihyi
            #Hz
            rhzi=griddata(per_c,np.real(hzr),perd,method='linear')
            ihzi=griddata(per_c,np.imag(hzr),perd,method='linear')
            hzi[:,pol]=rhzi+1j*ihzi  
            #Ex
            rexi=griddata(per_c,np.real(exr),perd,method='linear')
            iexi=griddata(per_c,np.imag(exr),perd,method='linear')
            exi[:,pol]=rexi+1j*iexi
            #Ey
            reyi=griddata(per_c,np.real(eyr),perd,method='linear')
            ieyi=griddata(per_c,np.imag(eyr),perd,method='linear')
            eyi[:,pol]=reyi+1j*ieyi

    # COMPUTE IMPEDANCE TENSOR
    #-------------------------
        FLAGS=-1
        per=perd
        # mackie en -iomegat, donc partie conjuguee des champs pour etre
        # meme cadran que 2d, data,...
        hxc=np.conj(hxi)   # (per,pol,site)
        hyc=np.conj(hyi)
        hzc=np.conj(hzi)
        exc=np.conj(exi)
        eyc=np.conj(eyi)
        # Determinant
        det=FLAGS*((hxc[:,0]*hyc[:,1])-(hxc[:,1]*hyc[:,0]))
        ##############
        ##############
        # DIAG TERM => Zxx
        zxx=(hyc[:,1]*exc[:,0])-(hyc[:,0]*exc[:,1])
        zxx=-zxx/(det*fmu)
        # Roaxx & Phixx
        zt=np.abs(zxx)
        rhoxx=zt*zt*amu*10**6/(2.*np.pi/per)
        phixx=np.arctan2(np.imag(zxx),np.real(zxx))*180/np.pi
        # DIAG TERM => Zyy
        zyy=(hxc[:,0]*eyc[:,1])-(hxc[:,1]*eyc[:,0])
        zyy=-zyy/(det*fmu)
        # Roayy & Phiyy
        zt=np.abs(zyy)
        rhoyy=zt*zt*amu*10**6/(2.*np.pi/per)
        phiyy=np.arctan2(np.imag(zyy),np.real(zyy))*180/np.pi
        # ANTI-DIAG TERM => Zxy
        zxy=(hxc[:,0]*exc[:,1])-(hxc[:,1]*exc[:,0])
        zxy=-zxy/(det*fmu)
        # Roaxy & Phixy
        zt=np.abs(zxy)
        rhoxy=zt*zt*amu*10**6/(2.*np.pi/per)
        phixy=np.arctan2(np.imag(zxy),np.real(zxy))*180/np.pi
        # ANTI-DIAG TERM => Zyx
        zyx=(hyc[:,1]*eyc[:,0])-(hyc[:,0]*eyc[:,1])
        zyx=-zyx/(det*fmu)
        # Roayx & Phiyx
        zt=np.abs(zyx)
        rhoyx=zt*zt*amu*10**6/(2.*np.pi/per)
        phiyx=np.arctan2(np.imag(zyx),np.real(zyx))*180/np.pi
    # COMPUTE MISFIT
    #----------------
        #FIELD DATA
        #---------
        # Roa & PHASE
        rhoxx_d=RoPh_data[1,ii,n]
        Erhoxx_d=RoPh_data[2,ii,n]
        phixx_d=RoPh_data[3,ii,n]
        Ephixx_d=RoPh_data[4,ii,n]
        rhoxy_d=RoPh_data[5,ii,n]
        Erhoxy_d=RoPh_data[6,ii,n]
        phixy_d=RoPh_data[7,ii,n]
        Ephixy_d=RoPh_data[8,ii,n]
        rhoyx_d=RoPh_data[9,ii,n]
        Erhoyx_d=RoPh_data[10,ii,n]
        phiyx_d=RoPh_data[11,ii,n]
        Ephiyx_d=RoPh_data[12,ii,n]
        rhoyy_d=RoPh_data[13,ii,n]
        Erhoyy_d=RoPh_data[14,ii,n]
        phiyy_d=RoPh_data[15,ii,n]
        Ephiyy_d=RoPh_data[16,ii,n]
        # IMPEDANCE TENSOR
        zxxd=ImpT_data[1,ii,n]+1j*ImpT_data[2,ii,n]
        Ezxxd=ImpT_data[3,ii,n]
        zxyd=ImpT_data[4,ii,n]+1j*ImpT_data[5,ii,n]
        Ezxyd=ImpT_data[6,ii,n]
        zyxd=ImpT_data[7,ii,n]+1j*ImpT_data[8,ii,n]
        Ezyxd=ImpT_data[9,ii,n]
        zyyd=ImpT_data[10,ii,n]+1j*ImpT_data[11,ii,n]
        Ezyyd=ImpT_data[12,ii,n]
        #MODEL RESPONSE
        #--------------
        # Roa & PHASE
        rhoxx_ci=rhoxx[:]
        phixx_ci=phixx[:]
        rhoxy_ci=rhoxy[:]
        phixy_ci=phixy[:]
        rhoyx_ci=rhoyx[:]
        phiyx_ci=phiyx[:]
        rhoyy_ci=rhoyy[:]
        phiyy_ci=phiyy[:]
        # IMPEDANCE TENSOR
        zxxc=zxx
        zyyc=zyy
        zxyc=zxy
        zyxc=zyx
        # WRITE RESP+DATA file (Rhoa et Phase)
        #-------------------------------------
        #if rank==0:
        #    nm=lab[n]
        #    if nm<10.:
        #            idd='00'+str(int(nm))
        #    if nm>99.:
        #            idd=str(int(nm))
        #    if (10<=nm and nm<=90):
        #            idd='0'+str(int(nm))
        #            
        #    resp_f=open(tmpDir+'RhoPhi_'+idd+'.resp',"w")
        #    resp_f.write('per  Rhoxxd  Err_Rhoxxd  Rhoxxc   Phixxd  Err_Phixxd  Phixxc ... same for xy, yx, yy')
        #    resp_f.write('\n')
        #    for i in range(len(ii)):
        #         resp_f.write(str(perd[i])+'     '+str(rhoxx_d[i])+'        '+str(Erhoxx_d[i])+'        '+str(rhoxx_ci[i])+'        '+str(phixx_d[i])+'        '+str(Ephixx_d[i])+'        '+str(phixx_ci[  i])+'        ')
        #         resp_f.write(str(rhoxy_d[i])+'        '+str(Erhoxy_d[i])+'        '+str(rhoxy_ci[i])+'        '+str(phixy_d[i])+'        '+str(Ephixy_d[i])+'        '+str(phixy_ci[i])+'        ')
        #         resp_f.write(str(rhoyx_d[i])+'        '+str(Erhoyx_d[i])+'        '+str(rhoyx_ci[i])+'        '+str(phiyx_d[i])+'        '+str(Ephiyx_d[i])+'        '+str(phiyx_ci[i])+'        ')
        #         resp_f.write(str(rhoyy_d[i])+'        '+str(Erhoyy_d[i])+'        '+str(rhoyy_ci[i])+'        '+str(phiyy_d[i])+'        '+str(Ephiyy_d[i])+'        '+str(phiyy_ci[i])+'        ')
        #         resp_f.write('\n')
        #
        #    resp_f.close()
            
        #-----------------------------------
        #COMPUTE RMS USING Roa & PHASE
        #-----------------------------------
        #  RMS1r=RMS1r+sum((np.log10(rhoxx_d)-np.log10(rhoxx_ci))**2)+sum((np.log10(rhoxy_d)-np.log10(rhoxy_ci))**2)+ \
        #           sum((np.log10(rhoyx_d)-np.log10(rhoyx_ci))**2)+sum((np.log10(rhoyy_d)-np.log10(rhoyy_ci))**2)
        RMS1r=RMS1r+sum((rhoxx_d-rhoxx_ci)**2)+sum((rhoxy_d-rhoxy_ci)**2)+ \
                 sum((rhoyx_d-rhoyx_ci)**2)+sum((rhoyy_d-rhoyy_ci)**2)
        RMS1p=RMS1p+sum((phixx_d-phixx_ci)**2)+sum((phixy_d-phixy_ci)**2)+ \
                 sum((phiyx_d-phiyx_ci)**2)+ sum((phiyy_d-phiyy_ci)**2)
        RMS1r=RMS1r/(len(ii)*4)
        RMS1p=RMS1p/(len(ii)*4)
        #---------------------------------------------------
        #COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
        #---------------------------------------------------
        ### if len(np.where(Ezxxd==0)[0])!=0:
        ###     print len(np.where(Ezxxd==0)[0]),'zxx',nm
        ### elif len(np.where(Ezxyd==0)[0])!=0:
        ###     print len(np.where(Ezxyd==0)[0]),'zxy',nm
        ### elif len(np.where(Ezyxd==0)[0])!=0:
        ###     print len(np.where(Ezyxd==0)[0]),'zyx',nm
        ### elif len(np.where(Ezyyd==0)[0])!=0:
        ###     print len(np.where(Ezyyd==0)[0]),'zyy',nm
        ### else:
        ###     print 'pas de pb'
        ### 
        XHI2=XHI2+sum((np.real(zxxd)-np.real(zxxc))**2/Ezxxd**2)+sum((np.imag(zxxd)-np.imag(zxxc))**2/Ezxxd**2)+ \
            sum((np.real(zxyd)-np.real(zxyc))**2/Ezxyd**2)+sum((np.imag(zxyd)-np.imag(zxyc))**2/Ezxyd**2)+ \
            sum((np.real(zyxd)-np.real(zyxc))**2/Ezyxd**2)+sum((np.imag(zyxd)-np.imag(zyxc))**2/Ezyxd**2)+ \
            sum((np.real(zyyd)-np.real(zyyc))**2/Ezyyd**2)+sum((np.imag(zyyd)-np.imag(zyyc))**2/Ezyyd**2)   
    print ''
    print 'Magnetotelluric Misfit XHI2=>',XHI2
    print 'RMS Resistivity apparent => RMS1r=',np.sqrt(RMS1r/len(lab)),'Ohm.m'
    print 'RMS Phase => RMS1p=',np.sqrt(RMS1p/len(lab)),' degree'
    print ''
    '''
    XHI2 = np.linalg.norm(X)
    return XHI2



#-----------------------------------#
#                                   #
# MINIMISATION OF THE COST FUNCTION #
# USING CPSO ALGORITHM (K.Luu 2018) #
#                                   #
#-----------------------------------#

# ----- Inialization of cpso ------
# Xstart: 
# - initial model at first job
# - last model from previous job afterwards

# ---> stochopy interface modification for restart
# - mod_best : None or ndarray, optionnal default None
# Individual best position of the population Used for restart simulation
# - gmod_best : None or ndarray Gobal best position of swarm
# Both mod_best and gmod_best are non_standardized

n_dim = None
Xstart = None
gmod_best = None
mod_best = None

if rank==0:
    n_dim = len(np.unique(model))
    Xstart=np.zeros((popsize,len(Xi)))
    
    if i_job==1:
        # First job
        for i in range(popsize):
            Xstart[i, :] = Xi
            gmod_best = None
            mod_best = None
    else:
	# from restart
        nc = Dataset(outfile, "r", format="NETCDF4")  
        Xstart = nc.variables['models'][:, :, it_start-1]
        # finding best models 
        idx = np.argmin(nc.variables['energy'][:, :it_start-1], axis=1)
        nparam = len(nc.dimensions['nparam'])
        mod_best = np.zeros([popsize, nparam])
        # ---> might be a long loop...
        for bee in range(popsize):
            mod_best[bee, :] = nc.variables['models'][:, bee, idx[bee]]
        gmod_best = np.array(nc.variables['xopt'][:])
        nc.close

# Communication
Xtstart = comm.bcast(Xstart, root=0)
gmod_best = comm.bcast(gmod_best, root=0)
mod_best = comm.bcast(mod_best, root=0)

# RESTART
# ---> lower and upper should remain the exact same regardless of jobs
# ---> may be better to keep them in netcdf file or a coefficient
# ---> in netcdf file
lower=Xi-np.ones(n_dim)*2
upper=Xi+np.ones(n_dim)*2

# Initialize SOLVER
ea = Evolutionary(F, lower = lower, upper = upper, popsize = popsize,
                  max_iter = n_iter_job, mpi = True, snap = True)

# SOLVE
xopt,gfit=ea.optimize(solver = "cpso", xstart = Xstart , sync = True,
                      mod_best = mod_best, gmod_best = gmod_best)


  ### #-----------------------------------------------#
  ### #                                               #
  ### #    COMPUTE FUNCTION DENSITY PROBABILITY       #
  ### #                                               #
  ### #-----------------------------------------------#
  ### if rank==0:
  ###     nparam=len(np.unique(model))
  ###     model_inv=np.zeros((popsize*max_iter,nparam))
  ###     fit_inv=np.zeros(popsize*max_iter)
  ###     cpt=0
  ###     for i in range(max_iter):
  ###         for j in range(popsize):
  ###             model_inv[cpt,:]=np.around(ea.models[j,:,i],1)
  ###             fit_inv[cpt]=ea.energy[j,i]/1e6
  ###             cpt=cpt+1
  ###     
  ###     #   print ea.models.shape, ea.energy.shape
  ###     #   print ea.models
  ###     #   print ea.energy
  ###     #   print model_inv, fit_inv
  ###     JPPD=np.exp(-fit_inv/2)/sum(np.exp(-fit_inv/2))
  ###     #   print JPPD
  ###     PMM=np.zeros(nparam)
  ###     for i in range(nparam):
  ###         PMM[i]=sum(model_inv[:,i]*JPPD[:])
  ###     
  ###     Cm=np.zeros((nparam,nparam))
  ###     Mm=PMM
  ###     Mm.shape=(nparam,1)
  ###     Mm_t=np.transpose(Mm)
  ###     for i in range(popsize*max_iter):
  ###         m=model_inv[i,:]
  ###         m.shape=(nparam,1)
  ###         m_t=np.transpose(m)
  ###         ma=(m-Mm)
  ###         mb=np.transpose(ma)
  ###         #Cm=Cm+np.dot(m,m_t)*JPPD[i]-np.dot(Mm,Mm_t)*JPPD[i]
  ###         Cm=Cm+np.dot(ma,mb)*JPPD[i]
  ###     
  ###     StD=np.zeros(nparam)
  ###     for i in range(nparam):
  ###         StD[i]=np.sqrt(Cm[i,i])
  ###     
  ###     print StD
  ### 
  ### 
#-----------------------------------#
#                                   #
#  WRITING BEST RESISTIVITY MODEL   #
#                                   #
#-----------------------------------#
# JC Keep this to write XOPT
#

print model.shape

#----------------- RESTART netcdf IO ------------------
if rank==0:
    print "Writting in ", outfile
    print "models shape:", np.shape(ea.models)
    print ""
    print "hx:", hx 
    # ---> maybe a check dimension and return 0 or 1 
    if not os.path.isfile(outfile):
        nc = Dataset(outfile, "w", format='NETCDF4')
        # dimensions: name, size
        nc.createDimension('iter', max_iter) 
        nc.createDimension('nx', nx)
        nc.createDimension('ny', ny)
        nc.createDimension('nz', nz)
        nc.createDimension('nparam', len(xopt))
        nc.createDimension('popsize', np.shape(ea.models)[0])
        # Variables: name, format, shape
        nc.createVariable('hx', 'f8', ('nx'))
        nc.createVariable('hy', 'f8', ('ny'))
        nc.createVariable('hz', 'f8', ('nz'))
        nc.createVariable('model_i', 'f8', ('nx','ny','nz')) 
        nc.createVariable('xopt', 'f8', ('nparam'))
        nc.createVariable('log_xopt', 'f8', ('nparam'))
        nc.createVariable('models', 'f8', ('popsize', 'nparam', 'iter'))
        nc.createVariable('energy', 'f8', ('popsize', 'iter'))    
    else:
        nc = Dataset(outfile, 'a')
    
    # FILLING VALUES
    # ---> Only if i_job == 1
    if i_job==1:
        nc.variables['hx'][:] = hx
        nc.variables['hy'][:] = hy
        nc.variables['hz'][:] = hz
        nc.variables['model_i'][:,:,:] = model_i

    # ----> modify [it_start:it_end]
    # ----> xopt and log_xopt are erased after each job
    nc.variables['xopt'][:] = 10**xopt
    nc.variables['log_xopt'][:] = xopt
    nc.variables['models'][:, :, it_start:it_end] = ea.models
    nc.variables['energy'][:, it_start:it_end] = ea.energy
    nc.close()

# ----------------- END OF IO ---------------------------
'''
if rank==0:
    #xopt=np.around(xopt,1)          
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
    for i in np.arange(len(xopt)):
        filefmt.write(str(float(10**xopt[i]))+'     ')
    
    filefmt.write('\n')
    filefmt.close()

'''
  ###  #-----------------------------------#
  ###  #                                   #
  ###  #  WRITING MEAN RESISTIVITY MODEL   #
  ###  #                                   #
  ###  #-----------------------------------#
  ###  if rank==0:
  ###      Xd=10**PMM[model-1]
  ###      model_i=np.zeros((nx,ny,nz))
  ###      cpt=1
  ###      for k in np.arange(nz):
  ###          for j in np.arange(ny):
  ###              for i in np.arange(nx):
  ###                  model_i[i,j,k]=cpt
  ###                  cpt=cpt+1
  ###      
  ###      
  ###      filefmt=open('3DRHO_MEAN.rslt',"w")
  ###      filefmt.write(str(nx)+'     '+str(ny)+'     '+str(nz))
  ###      filefmt.write('\n')
  ###      for i in np.arange(nx):
  ###          filefmt.write(str(hx[i])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for j in np.arange(ny):
  ###          filefmt.write(str(hy[j])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for k in np.arange(nz):
  ###          filefmt.write(str(hz[k])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for k in np.arange(nz):
  ###          filefmt.write(str(k+1))
  ###          filefmt.write('\n')
  ###          for j in np.arange(ny):
  ###              for i in np.arange(nx):
  ###                  filefmt.write(str(int(model_i[i,j,k]))+'       ')
  ###      
  ###              filefmt.write('\n')
  ###      
  ###      
  ###      filefmt.write('0.00000000000      ')
  ###      for i in np.arange(nx*ny*nz):
  ###          filefmt.write(str(float(Xd[i]))+'     ')
  ###      
  ###      filefmt.write('\n')
  ###      filefmt.close()  
  ###  
  ###  
  ###  #-----------------------------------#
  ###  #                                   #
  ###  #  WRITING STD RESISTIVITY MODEL   #
  ###  #                                   #
  ###  #-----------------------------------#
  ###  if rank==0:
  ###      Xd=10**StD[model-1]
  ###      model_i=np.zeros((nx,ny,nz))
  ###      cpt=1
  ###      for k in np.arange(nz):
  ###          for j in np.arange(ny):
  ###              for i in np.arange(nx):
  ###                  model_i[i,j,k]=cpt
  ###                  cpt=cpt+1
  ###      
  ###      
  ###      filefmt=open('3DRHO_STD.rslt',"w")
  ###      filefmt.write(str(nx)+'     '+str(ny)+'     '+str(nz))
  ###      filefmt.write('\n')
  ###      for i in np.arange(nx):
  ###          filefmt.write(str(hx[i])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for j in np.arange(ny):
  ###          filefmt.write(str(hy[j])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for k in np.arange(nz):
  ###          filefmt.write(str(hz[k])+'      ')
  ###      
  ###      filefmt.write('\n')
  ###      for k in np.arange(nz):
  ###          filefmt.write(str(k+1))
  ###          filefmt.write('\n')
  ###          for j in np.arange(ny):
  ###              for i in np.arange(nx):
  ###                  filefmt.write(str(int(model_i[i,j,k]))+'       ')
  ###      
  ###              filefmt.write('\n')
  ###      
  ###      
  ###      filefmt.write('0.00000000000      ')
  ###      for i in np.arange(nx*ny*nz):
  ###          filefmt.write(str(float(Xd[i]))+'     ')
  ###      
  ###      filefmt.write('\n')
  ###      filefmt.close()  
  ###  
  ###  
  ###  #COMPUTE F with the MEAN solution array
  ###  #-------------------------------------
  ###  if rank==0:
  ###      F(PMM)
  ###      print (ea)
  ###      print "Elapsed time: %.2f seconds" % (time() - starttime)
  ###      
  ###  
  ###  #PLOT MISFIT EVOLUTION
  ###  #-----------------------
  ###  if rank==0:
  ###      mean_energy=np.zeros(max_iter)
  ###      best_energy=np.zeros(max_iter)
  ###      for i in range(max_iter):
  ###          mean_energy[i]=sum(ea.energy[:,i])/popsize
  ###          best_energy[i]=np.min(ea.energy[:,i])
  ###  
  ###      np.savetxt('Mean_Ener.txt',mean_energy)
  ###      np.savetxt('Best_Ener.txt',best_energy)
  ###      plt.plot(np.arange(max_iter)+1,mean_energy,c='b',linewidth=2,label='Mean F(m)')
  ###      plt.plot(np.arange(max_iter)+1,best_energy,c='r',linewidth=2,label='Min F(m)')
  ###      plt.xlabel('Iterations',fontsize=15)
  ###      plt.ylabel('F(m)',fontsize=15)
  ###      xtick=np.arange(1,max_iter+1,10)
  ###      plt.xticks(fontsize=12)
  ###      plt.yticks(fontsize=12)
  ###      plt.xlim(1,max_iter+0.1)
  ###      # plt.yscale('log', nonposy='clip')
  ###      plt.legend()
  ###      plt.savefig('MISFIT_Evol.png')
  ###      plt.show()
  ###  
