from netCDF4 import Dataset
import numpy as np
import linecache
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
import os
import seaborn as sns
import pandas
import matplotlib.colors as mcolors
import matplotlib.cm as cm

sns.set_style("dark")
sns.set_context("paper")

# be careful if save_netcdf: outfile is removed before creating a new one
run = 'Bolivia_115param_015'
NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run + '/convergence_rms/5runs_merged'
folder_save = NCPATH #+ '/Postprocessing_rms'

nrun=5
iterpercent=np.array([10,30,50,65,75,85,100])#np.array([10,25,50,75,100])
npart=len(iterpercent)

# READ FULL NC FILE
#-------------------
ncfile=NCPATH+'/mselect_mod_nruns'+str(nrun)+'_iterpercent100.nc'

nc = Dataset(ncfile, "r", format="NETCDF4")
m_weight=nc.variables['m_weight'][:] 
std_weight=nc.variables['std_weight'][:]
nparam=len(m_weight)
nc.close()


# READ PARTIAL ITER NC FILE
#---------------------------
mean_cv=np.empty((npart,nparam))
std_cv=np.empty((npart,nparam))

nValues = iterpercent
normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
colormap = cm.jet_r


for itt in range(npart):
	ncfile=NCPATH+'/mselect_mod_nruns'+str(nrun)+'_iterpercent'+str(iterpercent[itt])+'.nc'
	nc = Dataset(ncfile, "r", format="NETCDF4")
	m_part=nc.variables['m_weight'][:] 
	std_part=nc.variables['std_weight'][:]
	mean_cv[itt,:]=m_weight-m_part
	std_cv[itt,:]=std_weight-std_part
	nc.close()

fig, axes = plt.subplots(nrows=2, ncols=2)
for i in range(npart):
	ax=axes[0,0]
	ax.plot(np.arange(nparam)+1,mean_cv[i,:], color=colormap(normalize(iterpercent[i])))
	if i==0:
		ax.set_ylabel('<m>-<m(niter)>')
	ax=axes[1,0]
	ax.plot(np.arange(nparam)+1,std_cv[i,:], color=colormap(normalize(iterpercent[i])))
	if i==0:
		ax.set_ylabel('<std>-<std(niter)>')
		ax.set_xlabel('iparam')

mean_cv=mean_cv**2
mean_cv=np.sqrt(np.sum(mean_cv,axis=1))
std_cv=std_cv**2
std_cv=np.sqrt(np.sum(std_cv,axis=1))

ax=axes[0,1]
ax.plot(iterpercent,mean_cv)
ax=axes[1,1]
ax.plot(iterpercent,std_cv)
ax.set_xlabel('% Total iter')

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(iterpercent)
c=fig.colorbar(scalarmappaple,ax=axes.ravel().tolist(), orientation='horizontal')
c.set_label(label='% Total iteration')
c.ax.tick_params(axis='x')

plt.suptitle('Iter Convergence Left:param, Right:Norme L2')
plt.savefig(NCPATH+'/nruns100_iter_conv.png')
plt.show()
