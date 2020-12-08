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
run = '' ### run = 'Bolivia_115param_015'
NCPATH = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/1D/' + run
folder_save = NCPATH + 'pdf_compare_plot/'
ncfile=NCPATH+'50runs_mod1D_MargLaw_test.nc'
mcmcfile=NCPATH +'Pascal_lois_2/res.res'
mbest_centered=False

# --- create directory to save plots 
if not os.path.exists(folder_save):
    os.makedirs(folder_save)


# READ NC FILE
#------------
nc = Dataset(ncfile, "r", format="NETCDF4")

print nc.variables

m_grid=nc.variables['m_grid'][:, :] 
f_grid=nc.variables['f_grid'][:] 
logrhosynth=nc.variables['m_synth'][:] 
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

#READ MCMC FILE
#--------------
ninter=int(linecache.getline(mcmcfile, 1))
mcmc_pdf=np.empty((nparam,ninter,5))

idl=2
for ipar in range(nparam):
    idl=idl+1
    for iinter in range(ninter):
        idl=idl+1
        idl_val=linecache.getline(mcmcfile, idl).split()
        mcmc_pdf[ipar,iinter,:]=[np.log10(float(idl_val[0])),float(idl_val[1]),float(idl_val[2]),float(idl_val[3]),float(idl_val[4])]
        

#PLOT 
#-------------

for ipar in np.arange(nparam):
    fig=plt.figure()
    valpar=round(logrhosynth[ipar],2)
    meanpar=round(m_weight[ipar],2)
    Sbin=sum(n_bin[ipar, :])
    hist=np.empty(int(Sbin))
    for j in range(len(n_bin[ipar, :])):
        hist[int(sum(n_bin[ipar, 0:j])):int(sum(n_bin[ipar, 0:j+1]))]=x_bin[ipar, j]-np.diff(x_bin[ipar, :])[0]/2
    ax1 = fig.add_subplot(111)
    ax1.hist(hist,bins=x_bin[ipar, :]-np.diff(x_bin[ipar, :])[0]/2,label='Sampling')
    ax1.set_ylabel('Nbr of model', color='b',fontsize=10)
    ax1.set_xlabel('Resistivity (Log-scale)',fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b',labelsize=8)
    ax1.tick_params(axis='x',labelsize=8)
    ax2 = ax1.twinx()
    ax2.axvline(x=valpar, color='y',label='Synth')
    ax2.axvline(x=meanpar, color='g',label='Mean')
    ax2.plot(x_bin[ipar, :], pdf_m[ipar, :], 'r',label='CPSO pdf')
    #mcmc plot
    plt.plot(mcmc_pdf[ipar,:,0],mcmc_pdf[ipar,:,1],color='brown',label='MCMC pdf')
    plt.fill_between(mcmc_pdf[ipar,:,0],mcmc_pdf[ipar,:,3],mcmc_pdf[ipar,:,4],color='orange', alpha=0.5,label='MCMC pdf_error')
    ax2.set_ylabel('Probability', color='r',fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
    align_yaxis(ax1, 0, ax2, 0)
    ax2.legend(loc=0,fontsize=10)
    plt.suptitle('Parameter '+str(ipar+1)+' <Rho>:'+str(meanpar)+'$\Omega.m$ (log scale), STD:'+str(round(std_weight[ipar],2)),fontsize=12)
    plt.savefig(folder_save + '/' + 'compare_'+str(ipar+1)+'.png')
    #plt.show()
    plt.clf()


"""
    if mbest_centered==False:
        x_bin[ipar, :]=x_bin[ipar, :]-m_gbest[ipar]
    valpar=np.mean(Xo[modelp==ipar+1])  # initial value param
    idc=max(values[(values-valpar)<=0])
    colorVal = scalarMap.to_rgba(idc) # color param    
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
    ax2.axvline(x=valpar, color='purple',label='ini')
    ax2.axvline(x=m_gbest[ipar]+valpar, color='yellow',label='best')
    ax2.axvline(x=meanpar, color='g',label='mean')
    ax2.plot(x_bin[ipar, :]+valpar, pdf_m[ipar, :], 'r')
    ax2.set_ylabel('Probability', color='r',fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
    align_yaxis(ax1, 0, ax2, 0)
    ax2.legend(loc=0,fontsize=10)
    plt.suptitle('Parameter '+str(ipar+1)+' <Rho>:'+str(meanpar)+'$\Omega.m$ (log scale), STD:'+str(round(std_weight[ipar],2)),fontsize=12)
    plt.savefig(folder_save + '/' + 'Parameter_'+str(ipar+1)+'.png')



"""


    

