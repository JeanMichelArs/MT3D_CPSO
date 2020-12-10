from netCDF4 import Dataset
import numpy as np
import linecache
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

sns.set_style("dark")
sns.set_context("paper")

# be careful if save_netcdf: outfile is removed before creating a new one
run = 'Bolivia_115param_015'
NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run + '/convergence_rms'
folder_save = NCPATH + '/2Dmap_convergence'

nruns_range=np.array([1,5,10,20,50,75,100],dtype=int)
iterpercent_range=np.array([10,30,50,65,85,100],dtype=int)#np.array([10,30,50,65,75,85,100],dtype=int)

len_nruns=len(nruns_range)
len_iterp=len(iterpercent_range)

# READ FULL NC FILE
#-------------------
ncfile=NCPATH+"/100runs_merged/mselect_mod_nruns100_iterpercent100.nc"
#'/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/Bolivia_115param_015/Postprocessing_rms/mselect_mod.nc'
#NCPATH+"/100runs_merged/mselect_mod_nruns" + str(100)+'_iterpercent'+str(100) + ".nc"

nc = Dataset(ncfile, "r", format="NETCDF4")
m_weight=nc.variables['m_weight'][:] 
std_weight=nc.variables['std_weight'][:]
nparam=len(m_weight)
nc.close()


# READ PARTIAL ITER NC FILE
#---------------------------
mean_cv=np.empty((len_iterp+1,len_nruns+1,nparam))
std_cv=np.empty((len_iterp+1,len_nruns+1,nparam))

for rr in range(len_nruns): 
	for itt in range(len_iterp): 
		ncfile=NCPATH+"/"+str(nruns_range[rr])+"runs_merged/mselect_mod_nruns"+str(nruns_range[rr])+'_iterpercent'+str(iterpercent_range[itt])+'.nc'
		nc = Dataset(ncfile, "r", format="NETCDF4")
		m_part=nc.variables['m_weight'][:] 
		std_part=nc.variables['std_weight'][:]
		mean_cv[itt,rr,:]=m_weight-m_part
		std_cv[itt,rr,:]=std_weight-std_part
		nc.close()

mean_cv[-1,:,:]=mean_cv[len_iterp-1,:,:]
mean_cv[:,-1,:]=mean_cv[:,len_nruns-1,:]
std_cv[-1,:,:]=std_cv[len_iterp-1,:,:]
std_cv[:,-1,:]=std_cv[:,len_nruns-1,:]

# PLOT & SAVE
#-------------
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

mruns,miter=np.meshgrid(np.hstack((0,nruns_range)),np.hstack((0,iterpercent_range)))

for ipar in range(nparam):
	fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,6))
	ax=axes[0]
	val=round(np.amax(abs(mean_cv[:,:,ipar])),1)
	im=ax.pcolormesh(mruns,miter,mean_cv[:,:,ipar],cmap=cm.seismic, vmin=-val, vmax=val)
	ax.set_ylabel('% Total iter')
	ax.set_xlabel('Merged runs')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='5%', pad=0.5)
	c=fig.colorbar(im, cax=cax, orientation='horizontal')
	c.set_label(label='<m>-<m(r,i)>')
	c.ax.tick_params(axis='x')
	ax=axes[1]
        val=round(np.amax(abs(std_cv[:,:,ipar])),1)
	om=ax.pcolormesh(mruns,miter,std_cv[:,:,ipar],cmap=cm.seismic, vmin=-val, vmax=val)
	ax.set_ylabel('% Total iter')
	ax.set_xlabel('Merged runs')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('bottom', size='5%', pad=0.5)
	c=fig.colorbar(om, cax=cax, orientation='horizontal')
	c.set_label(label='<std>-<std(r,i)>')
	c.ax.tick_params(axis='x')
	plt.suptitle('Convergence parameter '+str(ipar+1))
	plt.savefig(folder_save+'/2Dconv_param'+str(ipar+1)+'.png')
	#plt.show()
	plt.close()



# Norme l2
mean_cv=mean_cv**2
mean_cv=np.sqrt(np.sum(mean_cv,axis=2))
std_cv=std_cv**2
std_cv=np.sqrt(np.sum(std_cv,axis=2))


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,6))

ax=axes[0]
val=round(np.amax(abs(mean_cv)),1)
im=ax.pcolormesh(mruns,miter,mean_cv,cmap=cm.seismic, vmin=-val,vmax=val)
ax.set_ylabel('% Total iter')
ax.set_xlabel('Merged runs')

divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='5%', pad=0.5)
c=fig.colorbar(im, cax=cax, orientation='horizontal')
c.set_label(label='$N_{L2}$(<m>-<m(r,i)>)')
c.ax.tick_params(axis='x')


ax=axes[1]
val=round(np.amax(abs(std_cv)),1)
om=ax.pcolormesh(mruns,miter,std_cv,cmap=cm.seismic, vmin=-val,vmax=val)
ax.set_ylabel('% Total iter')
ax.set_xlabel('Merged runs')
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='5%', pad=0.5)
c=fig.colorbar(om, cax=cax, orientation='horizontal')
c.set_label(label='$N_{L2}$(<std>-<std(r,i)>)')
c.ax.tick_params(axis='x')


plt.suptitle('Convergence Norme L2')
plt.savefig(folder_save+'/2Dmap_conv_L2.png')
#plt.show()