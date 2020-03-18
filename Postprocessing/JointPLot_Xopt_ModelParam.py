from netCDF4 import Dataset
import numpy as np
import linecache
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
import os
import seaborn as sns
import pandas


sns.set_style("dark")
sns.set_context("talk")


# READ INITIAL MODEL
#-------------------
file = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/sensi_analysis/bolivia_24param/MT3D_BOLIVIA_IMAGIRb.fmt'
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
Xo=np.log10(np.hstack(float(l.split()[i+1]) for i in np.arange(len(l.split())-1) ))
# Meshing vector
hx=np.hstack(float(lx.split()[i]) for i in np.arange(nx) )
hy=np.hstack(float(ly.split()[i]) for i in np.arange(ny) )
hz=np.hstack(float(lz.split()[i]) for i in np.arange(nz) )
#Model parametrization
ind=5
model=np.zeros((nx*ny*nz),dtype=int)
for k in np.arange(nz):
    ind=ind+1
    for i in np.arange(ny):
        ind=4+(k+1)+(k)*ny+(i+1)
        l=linecache.getline(file, ind)
        L=np.vstack(float(l.split()[i]) for i in np.arange(nx) )
        L.shape=(nx,)
        model[k*nx*ny+i*nx:k*nx*ny+i*nx+nx]=L


# READ PARAMETER MODEL
#-------------------
file = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/sensi_analysis/bolivia_24param/parameter_model.ini'
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
Xip=np.log10(np.hstack(float(l.split()[i+1]) for i in np.arange(len(l.split())-1) ))
nparam=len(Xip)
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


# CENTER CELL POSITION
#----------------------
refN=622901.875
refE=7513822.0

ang=0 #0  #  angle de rotation de la grille
angg=ang*np.pi/180

ctx=np.ones(len(hx))
for i in np.arange(len(hx)):
    ctx[i]=(sum(hx[0:i])+sum(hx[0:i+1]))/2
    
cty=np.ones(len(hy))
for i in np.arange(len(hy)):
    cty[i]=(sum(hy[0:i])+sum(hy[0:i+1]))/2

ctz=np.ones(len(hz))
for i in np.arange(len(hz)):
    ctz[i]=(sum(hz[0:i])+sum(hz[0:i+1]))/2

Xang=np.zeros(nx*ny*nz)
Yang=np.zeros(nx*ny*nz)
Zang=np.zeros(nx*ny*nz)
cpt=0
for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                Xang[cpt]=refN-cty[j]*np.sin(angg)+ctx[i]*np.cos(angg)
                Yang[cpt]=refE+cty[j]*np.cos(angg)+ctx[i]*np.sin(angg)
                Zang[cpt]=-ctz[k]
                cpt=cpt+1

# READ NC FILE
#------------
outfile='/home/ars/Documents/CODE_TEST/MT3D_CPSO/sensi_analysis/bolivia_24param/merged.nc'

ids=outfile.rfind('/')
path=outfile[:ids+1]

nc = Dataset(outfile, "r", format="NETCDF4")
#models history
models = nc.variables['models'][:,:, :,:]
nrun=models.shape[0]
popsize=models.shape[1]
nparam=models.shape[2]
max_iter=models.shape[3]
models=np.reshape(models,(popsize*nrun,nparam,max_iter))  # each run after the others

#models misfit history
energy = nc.variables['energy'][:,:,:]
energy=np.reshape(energy,(popsize*nrun,max_iter))
popsize=popsize*nrun
nc.close()

# Best & Mean Models
pbest=np.amin(energy,axis=0)
pmean=np.mean(energy,axis=0)

# find best model
aa=np.argmin(pbest)
bb=np.argmin(energy[:,aa])
mbest=models[bb,:,aa]

# Update initial model with mbest
ini_up=10**(Xo[model-1]+mbest[modelp-1])


# JOINTPLOT
#----------
idstruc=np.unique(modelp)
nstruc=len(idstruc)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax3 = fig.add_subplot(221)

# Xopt & size structure (sqrt((minX-maxX)**2+(minY-maxY)**2+(minZ-maxZ)**2)
size=np.zeros(nstruc)
for i in idstruc:
    xx=max(Xang[modelp==i])-min(Xang[modelp==i])
    yy=max(Yang[modelp==i])-min(Yang[modelp==i])
    zz=max(Zang[modelp==i])-min(Zang[modelp==i])
    size[i-1]=np.sqrt(xx**2+yy**2+zz**2)/1000

# Xopt & mean rho structure 
meanrho=np.zeros(nstruc)
stdrho=np.zeros(nstruc)
for i in idstruc:
    meanrho[i-1]=np.mean(Xo[model-1][modelp==i])
    stdrho[i-1]=np.std(Xo[model-1][modelp==i])
    
data = pandas.DataFrame({"Optimum Parameter variation (log)": mbest,
                        "Parameter Size (km)": size,
                        "Parameter Mean(log)": meanrho,
                        "Parameter Std(log)": stdrho})

sns.jointplot("Optimum Parameter variation (log)", "Parameter Size (km)", data=data, kind="scatter",color="g").plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.savefig(path+'JointPlot_Xopt_Size.png', format='png',dpi=300)

sns.jointplot("Optimum Parameter variation (log)", "Parameter Mean(log)", data=data, kind="scatter",color="g").plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.savefig(path+'JointPlot_Xopt_Mean.png', format='png',dpi=300)

sns.jointplot("Optimum Parameter variation (log)", "Parameter Std(log)", data=data, kind="scatter",color="g").plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.savefig(path+'JointPlot_Xopt_Std.png', format='png',dpi=300)

# Xopt & Position (X,Y,Z) structure (center struc)
tx=np.arange(refE,refE+sum(hy)+min(hy)/5,min(hy)/5)
ty=np.arange(refN,refN+sum(hy)+min(hx)/5,min(hx)/5)
mx,my=np.meshgrid(tx,ty)

modelxopt=np.zeros(len(modelp))
for i in idstruc:
    modelxopt[modelp==i]=mbest[i-1]
    
fig, axes = plt.subplots(nz/4, 4, figsize=(12,8 ))
contourval=np.arange(-1,1.05,0.05)

for k in range(nz):
    a=k/4
    b=k-(a)*4
    ax1=axes[a,b]
    val=modelxopt[k*nx*ny:(k+1)*nx*ny]
    X=Yang[k*nx*ny:(k+1)*nx*ny]
    Y=Xang[k*nx*ny:(k+1)*nx*ny]
    valb=griddata((X, Y), val, (mx, my), method='nearest')
    im=ax1.contourf(mx/1000, my/1000, valb,contourval,cmap=cm.PiYG)
    if b==0:
        ax1.tick_params(axis="y", labelsize=10)
        #ax1.set_yticklabels(np.array([623,627,631]),fontsize=10)
    else:
        ax1.tick_params(axis="y", labelsize=0)
        #ax1.set_yticklabels(np.array([623,627,631]),fontsize=0)

    if a==nz/4-1:
        ax1.tick_params(axis="x", labelsize=10)
        #ax1.set_xticklabels(np.array([7514,7519,7524]),fontsize=10)
    else:
        ax1.tick_params(axis="x", labelsize=0)
        #ax1.set_xticklabels(np.array([7514,7519,7524]),fontsize=0)
                           
    ax1.set_title('layer '+str(k+1),fontsize=13)
    plt.axis('equal')

v= np.arange(-1,1.2,0.2)
c=fig.colorbar(im, ticks=v, ax=axes.ravel().tolist(), orientation='horizontal',pad=0.1)
c.set_label(label='Parameter Optimum Variation (log scale)',fontsize=15)
c.ax.tick_params(axis='x', labelsize=13)
plt.savefig(path+'Parameter_Variation_Map.png', format='png',dpi=300)



    

