from netCDF4 import Dataset
import numpy as np
import linecache
from shutil import copyfile
import mackie3d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import utm
import os
import shutil


# MACKIE FILE PATH
mackiepath='/home/ars/Documents/CODE_DEV/MT_3D/MT3D_Mackie/F2PY/Mackie_F2PY_silent/mackie3d.so'


# READ MODEL PARAMETRIZATION
#---------------------------
file = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/chained_datar/3D_cross/4_param/model_3Dcross_4.ini'
ld = linecache.getline(file, 1)
lx = linecache.getline(file, 2)
ly = linecache.getline(file, 3)
lz = linecache.getline(file, 4)
dim=ld.split()
nx=int(dim[0])
ny=int(dim[1])
nz=int(dim[2])
#valeur modele
ind=4+(nz)+(nz-1)*ny+(ny)
l=linecache.getline(file, ind+1)
Xi=np.log10(np.hstack(float(l.split()[i+1]) for i in np.arange(len(l.split())-1) ))
# Meshing vector
hx=np.hstack(float(lx.split()[i]) for i in np.arange(nx) )
hy=np.hstack(float(ly.split()[i]) for i in np.arange(ny) )
hz=np.hstack(float(lz.split()[i]) for i in np.arange(nz) )
#Model parametrization
ind=5
modelp=np.zeros((nx*ny*nz),dtype=int)
for k in np.arange(nz):
    ind=ind+1
    for i in np.arange(ny):
        ind=4+(k+1)+(k)*ny+(i+1)
        l=linecache.getline(file, ind)
        L=np.vstack(float(l.split()[i]) for i in np.arange(nx) )
        L.shape=(nx,)
        modelp[k*nx*ny+i*nx:k*nx*ny+i*nx+nx]=L


# READ NC FILE
#------------
outfile='/home/ars/Documents/CODE_TEST/MT3D_CPSO/chained_datar/3D_cross/4_param/cpso_mackie_4_param.nc'

ids=outfile.rfind('/')
path=outfile[:ids+1]

nc = Dataset(outfile, "r", format="NETCDF4")
#models history
models = nc.variables['models'][:, :,:]
#models misfit history
energy = nc.variables['energy'][:,:]
nc.close()

popsize=models.shape[0]
nparam=models.shape[1]
max_iter=models.shape[2]

pbest=np.amin(energy,axis=0)
pmean=np.mean(energy,axis=0)
mrg=np.std(energy,axis=0)

plt.plot(pbest,label='best xhi2')
plt.plot(pmean,'--r',label='mean xhi2')
plt.fill_between(np.arange(max_iter), pmean-mrg, pmean+mrg,facecolor='yellow', alpha=0.5 , label='StD xhi2')
plt.xlabel('CPSO Iterations',fontsize=12)
plt.ylabel('XHI2',fontsize=12)
#plt.yscale('log', nonposy='clip')
plt.grid(True,which="both",ls="--")
plt.title('Convergence')
plt.legend()
plt.savefig(path+'convergence.png',dpi=500, bbox_inches='tight')
plt.close()


# COMPUTE PROBABILITY DENSITY FUNCTION
#-------------------------------------

FF=1e-12

model_inv=np.zeros((popsize*max_iter,nparam))
fit_inv=np.zeros(popsize*max_iter)

cpt=0
for i in range(max_iter):
    for j in range(popsize):
        model_inv[cpt,:]=np.around(models[j,:,i],1)
        fit_inv[cpt]=energy[j,i]*FF
        cpt=cpt+1

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

print 'StD on Parameter:',StD



# COMPUTE MARGINAL POSTERIOR PROBABILITY FUNCTION
#-------------------------------------------------
exists = os.path.isdir(path+'marginal_prob/')
if exists:
    shutil.rmtree(path+'marginal_prob/')

os.mkdir(path+'marginal_prob/')

Iparam = 2  # Interval param en log Rho
Lower = Xi - Iparam
Upper = Xi + Iparam
Ninter = 20
Interv = np.linspace(Lower, Upper, Ninter)

for k in range(nparam):
    mrgl = np.zeros((Ninter-1, 2))
    for nn in range(Ninter-1):
        cI = (Interv[nn, k] + Interv[nn+1, k]) / 2
        dI = cI - Interv[nn, k]
        dist = np.abs(models[:, k, :] - cI)
        ii = np.where( dist <= dI)
        lsample = len(ii[0])
        if lsample > 0:
            prob = sum(np.exp(-energy[ii[0], ii[1]]*FF/2)) / sum(np.exp(-fit_inv/2))
        else:
            prob='nan'

        mrgl[nn, :] = np.array([cI, prob])
        plt.scatter(cI, prob)
        
    plt.plot(mrgl[:,0],mrgl[:,1],'--r')
    """
    dr=np.linspace(min(mrgl[:,0]),max(mrgl[:,0]),50)
    mi=griddata(mrgl[:,0],mrgl[:,1],dr,method='linear')
    plt.plot(dr,mi,'--r')
    """
    #xtick=np.arange(round(min(Interv[:,k]),1),round(max(Interv[:,k]),1)+0.1,0.2)
    ytick=np.arange(0,1.1,.1)
    #plt.xticks(xtick,fontsize=10)
    plt.yticks(ytick,fontsize=10)
    plt.xlabel('Resistivity,log scale',fontsize=12)
    plt.ylabel('Marginal Probability',fontsize=12)
    plt.grid(True,which="both",ls="--")
    plt.title('Marginal PDF_'+str(int(k+1))+' Mean='+str(round(PMM[k,0],3))+' StD='+str(round(StD[k],3)))
    plt.savefig(path+'marginal_prob/mpdf_param'+str(int(k+1))+'.png',dpi=500, bbox_inches='tight')
    plt.clf()
    plt.close()
        
        
      


# WRITTING MEAN AND STD MODEL
#-----------------------------

for i in range(2):
    if i==0:
        Xd=10**PMM[modelp-1]
        filew=path+'3DRHO_MEAN.rslt'
    else:
        Xd=StD[modelp-1]
        filew=path+'3DRHO_StD.rslt'

    model_i=np.zeros((nx,ny,nz))
    cpt=1
    for k in np.arange(nz):
        for j in np.arange(ny):
            for i in np.arange(nx):
                model_i[i,j,k]=cpt
                cpt=cpt+1
                
    filefmt=open(filew,"w")
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


# COMPUTE MEAN MODEL RESPONSES
#-----------------------------
#input

refmty=0
refmtx=0

#node
dx=np.ones(nx+1)
for i in range(nx+1):
    dx[i]=sum(hx[0:i])+refmtx
        
dy=np.ones(ny+1)
for i in range(ny+1):
    dy[i]=sum(hy[0:i])+refmty

dz=np.ones(nz+1)
for i in range(nz+1):
    dz[i]=sum(hz[0:i])
    
angm=0
posf=path+'mt_sounding.pos'

#Periods
perpy=np.loadtxt(path+'per',skiprows=1)
nper=len(perpy)

#Create 1D bottom
mod1D=np.loadtxt(path+'ini1d',skiprows=1)
n1d = len(mod1D)

# MT SOUNDING POSITION FILE
lab,ysound,xsound,alt_sound,rot=np.loadtxt(posf,unpack=True)
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
            
    filexx=path+'data/'+idd+'.ro11'
    filexy=path+'data/'+idd+'.ro12'
    fileyx=path+'data/'+idd+'.ro21'
    fileyy=path+'data/'+idd+'.ro22'
    # read file
    perd,RZxx,IZxx,ErZxx,Roaxx,ErRoaxx,Phixx,ErPhixx=np.loadtxt(filexx,unpack=True)
    perd,RZxy,IZxy,ErZxy,Roaxy,ErRoaxy,Phixy,ErPhixy=np.loadtxt(filexy,unpack=True)
    perd,RZyx,IZyx,ErZyx,Roayx,ErRoayx,Phiyx,ErPhiyx=np.loadtxt(fileyx,unpack=True)
    perd,RZyy,IZyy,ErZyy,Roayy,ErRoayy,Phiyy,ErPhiyy=np.loadtxt(fileyy,unpack=True)
    # Gather data by component (Impedance Tensor matrix & RoaPhi Matrix)
    ImpT_data[:,0:len(perd),n]=(perd,RZxx,IZxx,ErZxx,RZxy,IZxy,ErZxy,RZyx,IZyx,ErZyx,RZyy,IZyy,ErZyy)
    RoPh_data[:,0:len(perd),n]=(perd,Roaxx,ErRoaxx,Phixx,ErPhixx,Roaxy,ErRoaxy,Phixy,ErPhixy,
                 Roayx,ErRoayx,Phiyx,ErPhiyx,Roayy,ErRoayy,Phiyy,ErPhiyy)


def XHI2(X):
    # FORTRAN FORMAT ARRAYS
    X=np.asfortranarray(X)
    print ' '
    print ' ################'
    print ' ----------------'
    print ' CALL MACKIE F2PY'
    print ' ----------------'
    print ' ################'
    print ' '
    hxsurf,hysurf,hzsurf,exsurf,eysurf=mackie3d.mtd3(mod1D,perpy,X,hx,hy,hz)
    ###########################
    #       MT MISFIT         #
    ###########################
    fmu=4*np.pi*10**(-4)
    amu=4*np.pi*10**(-7)
    per_c=perpy
    XHI2=0
    RMS1r=np.zeros(nsound)
    RMS1p=np.zeros(nsound)
    for n in range(nsound):
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
        per=perd
        # mackie en -iomegat, donc partie conjuguee des champs pour etre
        # meme cadran que 2d, data,...
        hxc=np.conj(hxi)   # (per,pol,site)
        hyc=np.conj(hyi)
        hzc=np.conj(hzi)
        exc=np.conj(exi)
        eyc=np.conj(eyi)
        # Determinant
        FLAGS=-1
        det=FLAGS*((hxc[:,0]*hyc[:,1])-(hxc[:,1]*hyc[:,0]))
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
        ##############
        ##############
        ### # Determinant
        ### det=-1*((hxc[:,0]*hyc[:,1])-(hxc[:,1]*hyc[:,0]))
        ### # DIAG TERM => Zxx
        ### zxx=(hyc[:,1]*exc[:,0])-(hyc[:,0]*exc[:,1])
        ### zxx=-zxx/(det*fmu)
        ### zt=np.abs(zxx)
        ### rhoxx=zt*zt*amu*10**6/(2.*np.pi/per)
        ### phixx=np.arctan2(np.imag(zxx),np.real(zxx))*180/np.pi
        ### # DIAG TERM => Zyy
        ### zyy=(hxc[:,0]*eyc[:,1])-(hxc[:,1]*eyc[:,0])
        ### zyy=-zyy/(det*fmu)
        ### zt=np.abs(zyy)
        ### rhoyy=zt*zt*amu*10**6/(2.*np.pi/per)
        ### phiyy=np.arctan2(np.imag(zyy),np.real(zyy))*180/np.pi
        ### # ANTI-DIAG TERM => Zxy
        ### zxy=(hxc[:,0]*exc[:,1])-(hxc[:,1]*exc[:,0])
        ### zxy=-zxy/(det*fmu)
        ### zt=np.abs(zxy)
        ### rhoxy=zt*zt*amu*10**6/(2.*np.pi/per)
        ### phixy=np.arctan2(np.imag(zxy),np.real(zxy))*180/np.pi
        ### # ANTI-DIAG TERM => Zyx
        ### zyx=(hyc[:,1]*eyc[:,0])-(hyc[:,0]*eyc[:,1])
        ### zyx=-zyx/(det*fmu)
        ### zt=np.abs(zyx)
        ### rhoyx=zt*zt*amu*10**6/(2.*np.pi/per)
        ### phiyx=np.arctan2(np.imag(zyx),np.real(zyx))*180/np.pi
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
        nm=lab[n]
        if nm<10.:
                idd='00'+str(int(nm))
        if nm>99.:
                idd=str(int(nm))
        if (10<=nm and nm<=90):
                idd='0'+str(int(nm))
                
        resp_f=open(path+'RhoPhi_'+idd+'.resp',"w")
        resp_f.write('per  Rhoxxd  Err_Rhoxxd  Rhoxxc   Phixxd  Err_Phixxd  Phixxc ... same for xy, yx, yy')
        resp_f.write('\n')
        for i in range(len(ii)):
             resp_f.write(str(perd[i])+'     '+str(rhoxx_d[i])+'        '+str(Erhoxx_d[i])+'        '+str(rhoxx_ci[i])+'        '+str(phixx_d[i])+'        '+str(Ephixx_d[i])+'        '+str(phixx_ci[  i])+'        ')
             resp_f.write(str(rhoxy_d[i])+'        '+str(Erhoxy_d[i])+'        '+str(rhoxy_ci[i])+'        '+str(phixy_d[i])+'        '+str(Ephixy_d[i])+'        '+str(phixy_ci[i])+'        ')
             resp_f.write(str(rhoyx_d[i])+'        '+str(Erhoyx_d[i])+'        '+str(rhoyx_ci[i])+'        '+str(phiyx_d[i])+'        '+str(Ephiyx_d[i])+'        '+str(phiyx_ci[i])+'        ')
             resp_f.write(str(rhoyy_d[i])+'        '+str(Erhoyy_d[i])+'        '+str(rhoyy_ci[i])+'        '+str(phiyy_d[i])+'        '+str(Ephiyy_d[i])+'        '+str(phiyy_ci[i])+'        ')
             resp_f.write('\n')
    
        resp_f.close()
        # PLOT RESPONSE MODEL AND DATA
        resp=np.vstack((perd,rhoxx_d,Erhoxx_d,rhoxx_ci,phixx_d,Ephixx_d,phixx_ci,
                                          rhoxy_d,Erhoxy_d,rhoxy_ci,phixy_d,Ephixy_d,phixy_ci,
                                          rhoyx_d,Erhoyx_d,rhoyx_ci,phiyx_d,Ephiyx_d,phiyx_ci,
                                          rhoyy_d,Erhoyy_d,rhoyy_ci,phiyy_d,Ephiyy_d,phiyy_ci ))
        plt.figure(figsize=(14,15))
        plt.subplot(211)
        plt.errorbar(resp[0,:],resp[1,:],resp[2,:],fmt='o',markersize=10,elinewidth=2.5,color='pink')
        plt.plot(resp[0,:],resp[3,:],label="XX",linewidth=3,color='pink')
        plt.errorbar(resp[0,:],resp[7,:],resp[8,:],fmt='o',markersize=10,elinewidth=2.5,color='red')
        plt.plot(resp[0,:],resp[9,:],label="XY",linewidth=3,color='red')
        plt.errorbar(resp[0,:],resp[13,:],resp[14,:],fmt='o',markersize=10,elinewidth=2.5,color='blue')
        plt.plot(resp[0,:],resp[15,:],label="YX",linewidth=3,color='blue')
        plt.errorbar(resp[0,:],resp[19,:],resp[20,:],fmt='o',markersize=10,elinewidth=2.5,color='cyan')
        plt.plot(resp[0,:],resp[21,:],label="YY",linewidth=3,color='cyan')
        plt.ylim(5,300)
        plt.yscale('log', nonposy='clip')
        plt.xscale('log', nonposx='clip')
        plt.xlabel('period (sec)',fontsize=40)
        plt.ylabel('Roa (Ohm.m)',fontsize=40)
        xtick=np.array([1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
        ytick=np.array([1e-1,1,1e1,1e2,1e3])
        plt.xticks(xtick,fontsize=30)
        plt.yticks(ytick,fontsize=30)
        plt.grid(True,which="both",ls="-")
        plt.legend(prop={'size':30},loc=9,ncol=4)
        plt.subplot(212)
        plt.errorbar(resp[0,:],resp[4,:],resp[5,:],fmt='o',markersize=10,elinewidth=2.5,color='pink')
        plt.plot(resp[0,:],resp[6,:],label="XX",linewidth=3,color='pink')
        plt.errorbar(resp[0,:],resp[10,:],resp[11,:],fmt='o',markersize=10,elinewidth=2.5,color='red')
        plt.plot(resp[0,:],resp[12,:],label="XY",linewidth=3,color='red')
        plt.errorbar(resp[0,:],resp[16,:],resp[17,:],fmt='o',markersize=10,elinewidth=2.5,color='blue')
        plt.plot(resp[0,:],resp[18,:],label="YX",linewidth=3,color='blue')
        plt.errorbar(resp[0,:],resp[22,:],resp[23,:],fmt='o',markersize=10,elinewidth=2.5,color='cyan')
        plt.plot(resp[0,:],resp[24,:],label="YY",linewidth=3,color='cyan')
        plt.ylim(-180.,180,)
        xtick=np.array([1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
        ytick=np.array([-180,-135,-90,-45,0,45,90,135,180])
        plt.xticks(xtick,fontsize=30)
        plt.yticks(ytick,fontsize=30)
        plt.xscale('log', nonposx='clip')
        plt.xlabel('period (sec)',fontsize=40)
        plt.ylabel('phase(deg)',fontsize=40)
        plt.suptitle('site '+idd,fontsize=40)
        plt.grid(True,which="both",ls="-")
        plt.savefig(path+'resp'+idd+'.eps',dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()
        #-----------------------------------
        #COMPUTE MT MISFIT USING Roa & PHASE
        #-----------------------------------
        RMS1r[n]=sum((np.log10(rhoxx_d)-np.log10(rhoxx_ci))**2)+sum((np.log10(rhoxy_d)-np.log10(rhoxy_ci))**2)+ \
                 sum((np.log10(rhoyx_d)-np.log10(rhoyx_ci))**2)+sum((np.log10(rhoyy_d)-np.log10(rhoyy_ci))**2)
        RMS1p[n]=sum((phixx_d-phixx_ci)**2)+sum((phixy_d-phixy_ci)**2)+ \
                 sum((phiyx_d-phiyx_ci)**2)+ sum((phiyy_d-phiyy_ci)**2)
        RMS1r[n]=RMS1r[n]/(len(ii)*4)
        RMS1p[n]=RMS1p[n]/(len(ii)*4)
        #---------------------------------------------------
        #COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
        #---------------------------------------------------
        XHI2=XHI2+sum((np.real(zxxd)-np.real(zxxc))**2/Ezxxd**2)+sum((np.imag(zxxd)-np.imag(zxxc))**2/Ezxxd**2)+ \
            sum((np.real(zxyd)-np.real(zxyc))**2/Ezxyd**2)+sum((np.imag(zxyd)-np.imag(zxyc))**2/Ezxyd**2)+ \
            sum((np.real(zyxd)-np.real(zyxc))**2/Ezyxd**2)+sum((np.imag(zyxd)-np.imag(zyxc))**2/Ezyxd**2)+ \
            sum((np.real(zyyd)-np.real(zyyc))**2/Ezyyd**2)+sum((np.imag(zyyd)-np.imag(zyyc))**2/Ezyyd**2)

    RMSrr=np.sqrt(sum(RMS1r)/nsound)
    RMSpp=np.sqrt(sum(RMS1p)/nsound)
    print ''
    print 'Magnetotelluric Misfit XHI2=>',XHI2
    print 'RMS Resistivity apparent => RMS1r=',RMSrr,'Ohm.m, Log-scale'
    print 'RMS Phase => RMS1p=',RMSpp,' degree'
    print ''


# COMPUTE
XHI2(10**PMM[modelp-1])

