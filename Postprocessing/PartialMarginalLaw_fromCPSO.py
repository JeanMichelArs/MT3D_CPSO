# python 2.7
'''
JMARS 08/2020

script to compute 'partial' marginal law for few parameters from cpso sampling


'''

#-----------------------------------------------------------------------------
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

import cpso_pp as pp

# ----------------------------------------------------------------------------
# be careful if save_netcdf: outfile is removed before creating a new one
run = ''#'Bolivia_115param_015'
NCPATH = '/home/ars/Documents/CODE_TEST/MT3D_CPSO/sensi_analysis/bolivia_24param'#'/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run
folder_save = NCPATH + '/Postprocessing_rms/Partial_Laws'
save_plot = True
"""
run = 'Bolivia_115param_015'
NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run
folder_save = NCPATH + '/Postprocessing_rms/Partial_Laws'
save_plot = True
"""

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(NCPATH + '/merged.nc')
energy = np.array(nc.variables['energy'][:])
models =  np.array(nc.variables['models'][:])
nc.close()
print "Ellapsed time reading netcdf file:", time.clock() - t0

#---------------------------------------------------------------------------
# MT data counting (needed for rms option)
os.chdir(NCPATH+'/data/')
ndata=0
for fn in glob.glob('*.ro*'):
	with open(fn) as f:
		ndata=ndata+sum(1 for line in f if line.strip() and not line.startswith('#'))

# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))
m_gbest = models[i_gbest[0], i_gbest[1], : , i_gbest[2]]
m_gbest = m_gbest[0]

# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = pp.NaN_filter(models, energy)
del models, energy 

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 
nruns, popsize, nparam, niter = filt_models.shape
print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)

threshold =np.max(filt_energy)# np.median(filt_energy)
m_near, f_near = pp.value_filter(filt_models, filt_energy, threshold)

# --- > regrid parameter space
delta_m = 1e-3 
m_grid, f_grid, rgrid_error = pp.regrid(m_near, f_near, delta_m, center=True)
del m_near, f_near
print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest


# -------------------------------------------------------------------------
# Select models depending on parameter
idparam=np.array([1,4,7,11,15,17,21,24])
eps=0.22
ref_ini=np.ones((nparam-1))*eps
print ref_ini

nval=41
delta_range=np.around(np.linspace(-1.0,1.0,nval),3)
eps_delta=np.mean(np.diff(delta_range))


for ipar in idparam:
	l_id=list(np.arange((nparam),dtype=int))
	l_id.remove(ipar-1)
	ii=np.array(l_id)
	idsup=(m_grid[:,ii] <= ref_ini).all(axis=1)
	rs=np.where(idsup==True)[0]
	idinf=(m_grid[:,ii][rs,:] >= -ref_ini).all(axis=1)
	ri=np.where(idinf==True)[0]
	idmodel=rs[ri]
	m_part=m_grid[idmodel,:]
	f_part=f_grid[idmodel]
	print m_part.shape
	# Compute partial law
	f_part=np.sqrt(f_part/ndata)
	raw_pdf_part=np.zeros(len(idmodel))
	for i_mod in range(len(idmodel)):
		raw_pdf_part[i_mod]=np.exp( - f_part[i_mod] /2)/ np.sum(np.exp(- f_part /2))
		plt.scatter(m_part[i_mod,ipar-1],raw_pdf_part[i_mod],c='r')

	raw_law_part=np.transpose(np.vstack((m_part[:,ipar-1],raw_pdf_part)))
	np.savetxt(folder_save+'/raw_law_part'+str(ipar)+'.txt',raw_law_part,fmt='%6.3f %12.8f')

	pdf_part=np.full(nval,'nan',dtype=float)
	for dd in range(nval):
		delta_i=delta_range[dd]
		rk=abs(m_part[:,ipar-1]-delta_i)<=eps_delta/2
		if np.sum(rk)>=1:
			pdf_part[dd]=np.sum(np.exp( - f_part[rk] /2)) / np.sum(np.exp(- f_part /2))
			plt.scatter(delta_i,pdf_part[dd],c='b',marker='*')
	
	law_part=np.transpose(np.vstack((delta_range,pdf_part)))
	np.savetxt(folder_save+'/range_law_part'+str(ipar)+'.txt',law_part,fmt='%6.3f %12.8f')


	if save_plot:
		plt.savefig(folder_save+'/'+str(ipar)+'.png')
		plt.close()
	


