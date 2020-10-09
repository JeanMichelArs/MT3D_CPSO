# python 2.7
'''
JCollin 01-2020
Jmars 10-2020

Compute several post-processing of cpso3D experience 
for specific number of runs & % of total iterations available

=> Needed to compute 2D map convergence of the parameters
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
nruns_range=np.array([100])#np.array([1,5,10,20,50,75,100],dtype=int)
iterpercent_range=np.array([85])#np.array([10,30,50,65,75,85],dtype=int)

for nruns in nruns_range:
    for iterpercent in iterpercent_range:
        print ''
        print 'Computing stat for '+str(iterpercent) +' percent of total iteration of '+ str(nruns)+" runs merged"
        print '--------------------------------------------------------------------------------------------------'
        # ----------------------------------------------------------------------------
        # be careful if save_netcdf: outfile is removed before creating a new one
        run = 'Bolivia_115param_015'
        NCPATH = '/home2/datawork/sflora/MT3D_CPSO/sensi_analysis/' + run
        folder_save = NCPATH + '/convergence_rms/'+str(nruns)+'runs_merged'
        save_plot = False
        outfile = folder_save + "/mselect_mod_nruns" + str(nruns)+'_iterpercent'+str(iterpercent) + ".nc"
        save_netcdf = True              
        # --- create directory to save plots 
        if not os.path.exists(folder_save):
             os.makedirs(folder_save)

        # --- load data
        t0 = time.clock()
        nc = Dataset(NCPATH + '/merged.nc')
        energy = np.array(nc.variables['energy'][:nruns, :, :])
        models =  np.array(nc.variables['models'][:nruns, :, :, :])
        nc.close()
        print "Ellapsed time reading netcdf file:", time.clock() - t0
        #---------------------------------------------------------------------------
        # MT data counting (needed for rms option)
        os.chdir(NCPATH+'/data/')
        ndata=0
        for fn in glob.glob('*.ro*'):
            with open(fn) as f:
                ndata=ndata+sum(1 for line in f if line.strip() and not line.startswith('#'))
                
            # global data           
            i_gbest = np.where(energy == np.min(energy))
            f_gbest = np.min(energy)
            i_gbest = np.where(energy == np.min(energy))
            m_gbest = models[i_gbest[0], i_gbest[1], : , i_gbest[2]]
            m_gbest = m_gbest[0]            
            # ----------------------------------------------------------------------------
            # First filter NaN values in case run did not finish            
            filt_models, filt_energy, niter = pp.NaN_filter(models, energy,iterpercent)
            del models, energy              
            # --->  Prefilter models according to parameter space regular subgriding
            # !!! Care must be taken that we do this parameter log 
            nruns, popsize, nparam, niter = filt_models.shape
            print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)            
            threshold =np.max(filt_energy)# np.median(filt_energy)
            m_near, f_near = pp.value_filter(filt_models, filt_energy, threshold)
            del filt_models, filt_energy            
            # --- > regrid parameter space
            delta_m = 1e-3
            m_grid, f_grid, rgrid_error = pp.regrid(m_near, f_near, delta_m, center=True)
            del m_near, f_near
            print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
            print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest               
            # ---> Xi2 weighted mean model, in log and physical space
            m_weight = pp.weighted_mean(m_grid, f_grid, ndata, kappa=1, rms=True, log=True)
            # ---> Xi2 weighted STD model, in log and physical space
            # ---- marginal laws using kappa damping coefficient
            n_inter = 20
            lower = -1.
            upper = 1.
            kappa = 1               
            pdf_m, n_bin, x_bin = pp.marginal_law(m_grid, f_grid, m_gbest*0,
                    ndata, n_inter=n_inter,lower=lower, upper=upper,
                    kappa=kappa, rms=True)
                
            # ---> save m_grid, f_grid, r_grid_error, m_gbest, 
            #      f_best, delta_m , xbin, n_bin, pdf_m, m_weight
            #      kappa, lower, upper
            if save_netcdf:
                if os.path.isfile(outfile):
                    os.remove(outfile)
                   print "remove ", outfile
                nc = Dataset(outfile, "w", format='NETCDF4')
                # dimensions: name, size
                nc.createDimension('popsize', f_grid.shape[0])
                nc.createDimension('nparam', m_grid.shape[1])
                nc.createDimension('n_inter', pdf_m.shape[1])
                # Variables: name, format, shape
                nc.createVariable('m_grid', 'f8', ('popsize', 'nparam'))
                nc.createVariable('f_grid', 'f8', ('popsize'))
                nc.createVariable('m_gbest', 'f8', ('nparam'))
                nc.createVariable('m_weight', 'f8', ('nparam'))
                nc.createVariable('std_weight', 'f8', ('nparam'))
                nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
                nc.createVariable('n_bin', 'f8', ('nparam', 'n_inter'))
                nc.createVariable('x_bin', 'f8', ('nparam', 'n_inter'))
                # filling values
                nc.variables['m_grid'][:, :] = m_grid
                nc.variables['f_grid'][:] = f_grid
                nc.variables['m_gbest'][:] = m_gbest
                nc.variables['m_weight'][:] = m_weight
                nc.variables['std_weight'][:] = std_weight
                nc.variables['pdf_m'][:, :] = pdf_m
                nc.variables['x_bin'][:, :] = x_bin
                nc.variables['n_bin'][:, :] = n_bin
                nc.rgrid_error = rgrid_error
                nc.f_gbest = f_gbest
                nc.delta_m = delta_m
                nc.kappa = kappa
                nc.lower = lower
                nc.upper = upper
                print outfile, "saved sucessfully"
                nc.close()
