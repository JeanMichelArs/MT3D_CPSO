from netCDF4 import Dataset
import numpy as np
import linecache
import matplotlib.pyplot as plt
import os
import seaborn as sns


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


pbest=np.amin(energy,axis=0)
pmean=np.mean(energy,axis=0)
mrg=np.std(energy,axis=0)

plt.plot(pbest,label='best xhi2')
plt.plot(pmean,'--r',label='mean xhi2')
plt.fill_between(np.arange(max_iter), pmean-mrg, pmean+mrg,facecolor='yellow', alpha=0.5 , label='StD xhi2')
plt.xlabel('CPSO Iterations')
plt.ylabel('XHI2')
plt.grid(True,which="both",ls="--")
plt.title('Convergence')
plt.legend()
plt.show()


# find best model
aa=np.argmin(pbest)
bb=np.argmin(energy[:,aa])
mbest=models[bb,:,aa]

# Update initial model with mbest
ini_up=10**(Xo[model-1]+mbest[modelp-1])

model=np.reshape(model,(nx,ny,nz),order='F')
            
filem=path+'update_model.ini'  
filefmt=open(filem,"w")
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
            filefmt.write(str(int(model[i,j,k]))+'       ')
            
        filefmt.write('\n')
        
filefmt.write('0.00000000000      ')
for i in np.arange(nx*ny*nz):
    filefmt.write(str((ini_up[i]))+'     ')
filefmt.write('\n')
filefmt.close()
