from netCDF4 import Dataset
import numpy as np
import linecache
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import os
import seaborn as sns
import pandas
import matplotlib.colors as colors
import matplotlib.cm as cmx

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)



sns.set_style("dark")
sns.set_context("talk")

# be careful if save_netcdf: outfile is removed before creating a new one
run = 'Bolivia_115param_015' ### run = 'Bolivia_115param_015'
NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run
model_ini=NCPATH+'/MT3D_BOLIVIA_IMAGIRb.fmt'
model_par=NCPATH+'/parameter_model.ini'
folder_save = NCPATH + '/Postprocessing'
ncfile=NCPATH+'/Postprocessing/mselect_mod.nc'

refN=622901.875
refE=7513822.0


# READ INITIAL MODEL
#-------------------
ld = linecache.getline(model_ini, 1)
lx = linecache.getline(model_ini, 2)
ly = linecache.getline(model_ini, 3)
lz = linecache.getline(model_ini, 4)
dim=ld.split()
nx=int(dim[0])
ny=int(dim[1])
nz=int(dim[2])
#valeur modele
ind=4+(nz)+(nz-1)*ny+(ny)
l=linecache.getline(model_ini, ind+1)
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
        l=linecache.getline(model_ini, ind)
        L=np.vstack(float(l.split()[i]) for i in np.arange(nx) )
        L.shape=(nx,)
        model[k*nx*ny+i*nx:k*nx*ny+i*nx+nx]=L


# READ PARAMETER MODEL
#-------------------
ld = linecache.getline(model_par, 1)
lx = linecache.getline(model_par, 2)
ly = linecache.getline(model_par, 3)
lz = linecache.getline(model_par, 4)
dim=ld.split()
nx=int(dim[0])
ny=int(dim[1])
nz=int(dim[2])
#valeur modele
ind=4+(nz)+(nz-1)*ny+(ny)
l=linecache.getline(model_par, ind+1)
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
        l=linecache.getline(model_par, ind)
        L=np.vstack(float(l.split()[i]) for i in np.arange(nx) )
        L.shape=(nx,)
        modelp[k*nx*ny+i*nx:k*nx*ny+i*nx+nx]=L


# CENTER CELL POSITION
#----------------------
ang=0 #0  #  angle de rotation de la grille
angg=ang*np.pi/180

dx=np.zeros(nx+1)
for i in np.arange(nx):
    dx[i+1]=sum(hx[0:i+1])
    
dy=np.zeros(ny+1)
for i in np.arange(ny):
    dy[i+1]=sum(hy[0:i+1])

dz=np.zeros(nz+1)
for i in np.arange(nz):
    dz[i+1]=sum(hz[0:i+1])

dx=dx+refN
dy=dy+refE

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
nc = Dataset(ncfile, "r", format="NETCDF4")

m_grid=nc.variables['m_grid'][:, :] 
f_grid=nc.variables['f_grid'][:] 
m_gbest=nc.variables['m_gbest'][:] 
m_weight=nc.variables['m_weight'][:] 
mpow_weight=nc.variables['mpow_weight'][:] 
std_weight=nc.variables['std_weight'][:]
stdpow_weight=nc.variables['stdpow_weight'][:] 
pdf_m=nc.variables['pdf_m'][:, :] 
x_bin=nc.variables['x_bin'][:, :] 
n_bin=nc.variables['n_bin'][:, :] 
rgrid_error=nc.rgrid_error 
f_gbest=nc.f_gbest 
delta_m =nc.delta_m  
kappa=nc.kappa 
lower=nc.lower 
upper=nc.upper
nparam=len(m_weight)

"""
#PLOT 3D PARAM
#-------------
#color
values = range(nparam)
jet_r = cm = plt.get_cmap('jet_r') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet_r)

modelpp=np.reshape(modelp,(nx,ny,nz))

vy, vx, vz = np.meshgrid(dy,dx,dz)

for ipar in np.arange(nparam):
    colorVal = scalarMap.to_rgba(values[ipar]) # color param
    valpar=np.mean(Xo[modelp==ipar+1])  # initial value param
    meanpar=round(m_weight[ipar]+valpar,2)
    Sbin=sum(n_bin[ipar, :])
    hist=np.empty(int(Sbin))
    for j in range(len(n_bin[ipar, :])):
        hist[int(sum(n_bin[ipar, 0:j])):int(sum(n_bin[ipar, 0:j+1]))]=x_bin[ipar, j]+valpar-np.diff(x_bin[ipar, :])[0]/2
        
    par=np.full((nx,ny,nz),False)
    par[modelpp == ipar+1]=True
    fig = plt.figure(figsize=(11,4.5))
    ax = fig.add_subplot(121, projection="3d")
    ax.voxels(vx, vy, vz,par,facecolors=colorVal,edgecolor='k',linewidth=0.5)
    ax.set_xlabel('North',fontsize=8)
    ax.set_ylabel('East',fontsize=8)
    ax.set_zlabel('Depth',fontsize=8)
    ax.tick_params(labelsize=8)
    ax.azim=45
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax1 = fig.add_subplot(122)
    ax1.hist(hist,bins=x_bin[ipar, :]+valpar-np.diff(x_bin[ipar, :])[0]/2)
    ax1.set_ylabel('Nbr of model', color='b',fontsize=10)
    ax1.set_xlabel('Resistivity (Log-scale)',fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b',labelsize=8)
    ax1.tick_params(axis='x',labelsize=8)
    ax2 = ax1.twinx()
    ax2.axvline(x=m_gbest[ipar]+valpar, color='yellow',label='best')
    ax2.axvline(x=meanpar, color='g',label='mean')
    ax2.plot(x_bin[ipar, :]+valpar, pdf_m[ipar, :], 'r')
    ax2.set_ylabel('Probability', color='r',fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
    align_yaxis(ax1, 0, ax2, 0)
    ax2.legend(loc=0,fontsize=10)
    plt.suptitle('Parameter '+str(ipar+1)+' <Rho>:'+str(meanpar)+'$\Omega.m$ (log scale), STD:'+str(round(std_weight[ipar],2)),fontsize=12)
    plt.savefig(folder_save + '/' + 'Parameter '+str(ipar+1)+'.png')
    #plt.show()

"""
# JOINTPLOT
#----------

# Xopt & size structure (sqrt((minX-maxX)**2+(minY-maxY)**2+(minZ-maxZ)**2)
size=np.empty(nparam)
for ipar in np.arange(nparam):
    xx=max(Xang[modelp==ipar+1])-min(Xang[modelp==ipar+1])
    yy=max(Yang[modelp==ipar+1])-min(Yang[modelp==ipar+1])
    zz=max(Zang[modelp==ipar+1])-min(Zang[modelp==ipar+1])
    size[ipar]=np.sqrt(xx**2+yy**2+zz**2)/1000

# Xopt & mean rho structure
f_m_weight=m_weight
f_m_gbest=m_gbest
for ipar in np.arange(nparam):
    valpar=np.mean(Xo[modelp==ipar+1])  # initial value param
    f_m_weight[ipar]=m_weight[ipar]+valpar
    f_m_gbest[ipar]=m_gbest[ipar]+valpar
    
data = pandas.DataFrame({"Best Rho (log)":f_m_gbest,
                        "Parameter Size (km)": size,
                        "<Rho>(log)": f_m_weight,
                        "Std(log)": np.around(std_weight,1)})

ylabels=np.arange(-1,1,0.25)
sns.set()
sns.set_context("paper")
g = sns.PairGrid(data)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
g.axes[3,0].set_yticklabels(ylabels)
plt.savefig(folder_save + '/' +'Pairplot_Size_Mean_Best_Std.png', format='png',dpi=300)
plt.show()




    

