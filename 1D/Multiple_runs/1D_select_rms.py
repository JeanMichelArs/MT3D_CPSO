# python 2.7
'''
compute <m> = sum(m * exp(F(-m))
        pdf(m_i)
        std(m_i)

works for MT1D candidates 
parameter space exploration can be performed through cpso or mcm algorithm

Main issue is that models for MCM and CPSO have different shapes
CPSO models (popsize, n_dim, max_iter)
MCM models (max_iter, n_dim)

>>> python 1D_select_rms.py

'''

#-----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

sys.path.append('../../Postprocessing/')
import cpso_pp as pp
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

#-----------------------------------------------------------------------------
def MT1D_analytic(thick,rho,per):
    if len(thick)==len(rho):
        thick=thick[0:-1]

    nlay=len(rho)
    frequencies = 1/per
    amu=4*np.pi*10**(-7) #Magnetic Permeability (H/m)
    Z=np.empty(len(per),dtype=complex)
    arho=np.empty(len(per))
    phase=np.empty(len(per))   
    for iff,frq in enumerate(frequencies):
        nlay=len(rho)
        w =  2*np.pi*frq       
        imp = list(range(nlay))
        #compute basement impedance
        imp[nlay-1] = np.sqrt(w*amu*rho[nlay-1]*1j)
        for j in range(nlay-2,-1,-1):
            rholay = rho[j]
            thicklay = thick[j]
            # 3. Compute apparent rholay from top layer impedance
            #Step 2. Iterate from bottom layer to top(not the basement) 
            # Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt((w * amu * (1/rholay))*1j)
            wj = dj * rholay
            ej = np.exp(-2*thicklay*dj)
            #The next step is to calculate the reflection coeficient (F6) and impedance (F7) using the current layer intrinsic impedance and the prior computer layer impedance j+1.
            belowImp = imp[j+1]
            rj = (wj - belowImp)/(wj + belowImp)
            re = rj*ej 
            Zj = wj * ((1 - re)/(1 + re))
            imp[j] = Zj
    
        #Finally you can compute the apparent rholay F8 and phase F9 and print the resulting data!
        Z[iff] = imp[0]
        absZ = abs(Z[iff])
        arho[iff] = (absZ * absZ)/(amu * w)
        phase[iff] = np.arctan2(np.imag(Z[iff]), np.real(Z[iff]))*180/np.pi
        #if convert to microvolt/m/ntesla
        Z[iff]=Z[iff]/np.sqrt(amu*amu*10**6)

    return Z,arho,phase
# ----------------------------------------------------------------------------
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
# ----------------------------------------------------------------------------
# be careful if save_netcdf: outfile is removed before creating a new one
# cpso_path : cpso output
# conf_dir : configuration files
# folder_save : parameter uncertainty estimates

method = 'mcm'
rms = False
nruns = 25 

cpso_path = '/postproc/COLLIN/MTD3/MCM_8nz_cst_Error'
conf_dir = '../../Config/1D/model_001'
data_file = conf_dir + '/001.ro'
model_file = conf_dir + '/mod1D_Bolivia_001'
folder_save = cpso_path + '/Analysis'  
save_plot = True
outfile = folder_save + "/pdf_m_" + str(nruns) + ".nc"
save_netcdf = True
# ---> postproc
n_inter = 40
lower = -2.
upper = 2.
kappa = 1

# --- create directory to save plots 
if not os.path.exists(folder_save):
	os.makedirs(folder_save)

# --- load data
t0 = time.clock()
nc = Dataset(cpso_path + '/merged.nc')
if method is 'cpso':
    energy = np.array(nc.variables['energy'][:nruns, :, :])
    models =  np.array(nc.variables['models'][:nruns, :, :, :])
    logrhosynth =  np.squeeze(np.log10(np.array(nc.variables['rho_i'][0, :])))
elif method is 'mcm':
    energy = np.array(nc.variables['energy'][:nruns, :])
    models =  np.array(nc.variables['models'][:nruns, :, :])
    logrhosynth =  np.squeeze(np.log10(np.array(nc.variables['rho_i'][0, :])))
else: 
    print "Error method is not defined"

nc.close()
print "Ellapsed time reading netcdf file:", time.clock() - t0
print''

#---------------------------------------------------------------------------
# MT data counting (needed for rms option)
ndata = pp.get_ndata(data_file)
print "number of data for rms: ", "{:e}".format(ndata)
print''
per, Rz, Iz, Erz, rho, Erho, phi, Ephi = np.loadtxt(data_file, unpack=True)

#---------------------------------------------------------------------------
# Read MT1D model
hz, rhosynth = np.loadtxt(model_file, unpack=True)
print''

# ---------------------------------------------------------------------------
# global data

i_gbest = np.where(energy == np.min(energy))
f_gbest = np.min(energy)
i_gbest = np.where(energy == np.min(energy))

# ----------------------------------------------------------------------------
# First filter NaN values in case run did not finish 

filt_models, filt_energy, niter = pp.NaN_filter(models, energy)
del models, energy 

# check for mcm exloration debug
nparam = filt_models.shape[2]
Err = np.zeros(shape=(nparam,))
for iparam in range(nparam):
    Err[iparam] = np.max(np.abs(filt_models[:, :, iparam] - logrhosynth[iparam]))
if (Err > upper).any(): print "Error models out of window", Err

# --->  Prefilter models according to parameter space regular subgriding
# !!! Care must be taken that we do this parameter log 

if method is 'cpso':
    nruns, popsize, nparam, niter = filt_models.shape
    print "number of models to filter: ", "{:e}".format(nruns * popsize * niter)
elif method is 'mcm':
    nruns, niter, nparam = filt_models.shape
    print "number of models to filter: ", "{:e}".format(nruns * niter)
else : print "Error undefined method"

print''

# ---> filter extreme value old...
threshold = np.max(filt_energy) 
m_near, f_near = pp.value_filter(filt_models, filt_energy, threshold)

# check
# Error mcm exploration detected
np.max(np.abs(m_near - logrhosynth))

# --- > regrid parameter space
delta_m = 1e-3 
m_grid, f_grid, rgrid_error = pp.regrid(m_near, f_near, delta_m, center=True)
del m_near, f_near
print "number of particles in subgrid", "{:e}".format(m_grid.shape[0])
print "Error in energy minimum after subgrid :", np.min(f_grid) - f_gbest
print''

print "max distance between m_grid and logrhosynth :"
print np.max(np.abs(m_grid - logrhosynth))

# ---> Xi2 weighted mean model, in log and physical space
m_weight = pp.weighted_mean(m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=True)
mpow_weight = np.log10(pp.weighted_mean(m_grid, ndata, f_grid, kappa=kappa, rms=rms, log=False))
print "Mean-difference between log and physical space :", np.max(np.abs(mpow_weight - m_weight))
print''

# ---> Xi2 weighted STD model, in log and physical space
std_weight = pp.weighted_std(m_weight, m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=True)
stdpow_weight = pp.weighted_std(10**mpow_weight, m_grid, f_grid, ndata, kappa=kappa, rms=rms, log=False)
print "STD-difference between log and physical space :", np.max(np.abs(stdpow_weight - std_weight))
print''

# ---- marginal laws centered around solution 
pdf_m, n_bin, x_bin = pp.marginal_law(m_grid, f_grid, logrhosynth, ndata,
                       n_inter=n_inter,lower=lower, upper=upper, kappa=kappa,
                       rms=rms)

print "check marginal law intervals and window for each parameter"
print "should be ", m_grid.shape[0]
print np.sum(n_bin, axis=1)

# ---- Misfit of the Mean Model 
zc, rhoc, phic = MT1D_analytic(hz, 10**m_weight, per)
#---------------------------------------------------
#COMPUTE MT MISFIT USING IMPEDANCE TENSOR COMPONENTS
#---------------------------------------------------
XHI2=(sum((Rz-np.real(zc))**2/Erz**2)+sum((Iz-np.imag(zc))**2/Erz**2))/2
print ''
print 'Magnetotelluric Misfit for Mean Model=>',XHI2
print 'While Number of data => ', ndata*2
print ''

# ---> save m_grid, f_grid, r_grid_error, m_gbest, 
#      f_best, delta_m , xbin, n_bin, pdf_m, m_weight, m_pow
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
    nc.createVariable('m_synth', 'f8', ('nparam'))
    nc.createVariable('m_weight', 'f8', ('nparam'))
    nc.createVariable('mpow_weight', 'f8', ('nparam'))
    nc.createVariable('std_weight', 'f8', ('nparam'))
    nc.createVariable('stdpow_weight', 'f8', ('nparam'))
    nc.createVariable('pdf_m', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('n_bin', 'f8', ('nparam', 'n_inter'))
    nc.createVariable('x_bin', 'f8', ('nparam', 'n_inter'))
    # filling values
    nc.variables['m_grid'][:, :] = m_grid
    nc.variables['f_grid'][:] = f_grid
    nc.variables['m_synth'][:] = logrhosynth
    nc.variables['m_weight'][:] = m_weight
    nc.variables['mpow_weight'][:] = mpow_weight
    nc.variables['std_weight'][:] = std_weight
    nc.variables['stdpow_weight'][:] = stdpow_weight
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

if save_plot:
    print "plot results"
    for ipar in range(nparam):
        # Marginal Laws
        fig = plt.figure()
        valpar = round(logrhosynth[ipar], 2)
        meanpar = round(m_weight[ipar], 2)
        Sbin = sum(n_bin[ipar, :])
        hist = np.empty(int(Sbin))
        for j in range(len(n_bin[ipar, :])):
            hist[int(sum(n_bin[ipar, 0:j])):int(sum(n_bin[ipar, 0:j+1]))]=x_bin[ipar, j]-np.diff(x_bin[ipar, :])[0]/2
        ax1 = fig.add_subplot(111)
        ax1.hist(hist,bins=x_bin[ipar, :]-np.diff(x_bin[ipar, :])[0]/2)
        ax1.set_ylabel('Nbr of model', color='b',fontsize=10)
        ax1.set_xlabel('Resistivity (Log-scale)',fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b',labelsize=8)
        ax1.tick_params(axis='x',labelsize=8)
        ax2 = ax1.twinx()
        ax2.axvline(x=valpar, color='y',label='synth')
        ax2.axvline(x=meanpar, color='g',label='mean')
        ax2.plot(x_bin[ipar, :], pdf_m[ipar, :], 'r')
        ax2.set_ylabel('Probability', color='r',fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r',labelsize=8)
        align_yaxis(ax1, 0, ax2, 0)
        ax2.legend(loc=0,fontsize=10)
        plt.suptitle('Parameter '+str(ipar+1)+' <Rho>:'+str(meanpar)+'$\Omega.m$ (log scale), STD:'+str(round(std_weight[ipar],2)),fontsize=12)
        plt.savefig(folder_save + '/' + 'nruns' + str(nruns) +'_Parameter_'+str(ipar+1)+'.png',transparent=True)
        plt.clf()

    # MEAN MODEL & PDF
    fig = plt.figure()
    dz=np.zeros(nparam+1)
    for k in range(nparam+1):
        dz[k]=sum(hz[0:k])
    # SHADED LAW
    cmm=cm.plasma
    norm = BoundaryNorm(np.arange(0,1,0.01), ncolors=cmm.N, clip=True)
    for ipar in range(nparam):
        ddz=np.linspace(-dz[ipar+1],-dz[ipar],10)
        ddx=np.linspace(min(x_bin[ipar, :]),max(x_bin[ipar, :]),80)
        nn=len(ddz)
        ll=pdf_m[ipar, :]
        ll[ll<1e-3]='nan'
        ml=griddata(x_bin[ipar, :],ll,ddx,method='linear')
        shaded=np.tile(ml,(nn,1))
        mx,mz=np.meshgrid(ddx,ddz)
        pcol =plt.pcolormesh(mx,mz,shaded,alpha=0.7,cmap=cmm, norm=norm,antialiased=True, linewidth=0.0,rasterized=True)
        pcol.set_edgecolor('Face')
        #plt.contourf(mx,mz,shaded,np.arange(0,1,0.001),cmap=cmm,alpha=0.5)

    c=plt.colorbar()
    #c=plt.colorbar(ticks=np.arange(0,1.1,0.1),location='bottom',pad=0.1)
    c.set_label(label='PDF',fontsize=10)
    c.ax.tick_params(axis='x', labelsize=8)
    #
    mean_mod=np.hstack((m_weight,m_weight[-1]))
    synth_mod=np.hstack((logrhosynth,logrhosynth[-1]))
    plt.step(mean_mod,-dz,linewidth=2,color='g',label='mean')
    plt.step(synth_mod,-dz,linewidth=2,color='y',label='synth')
    plt.legend(loc=0,fontsize=10)
    plt.ylabel('Depth',fontsize=10)
    plt.xlabel('Resistivity (Log-scale)',fontsize=10)
    plt.xlim(0,5.5)
    plt.savefig(folder_save + '/PosteriorModel_'+str(ipar+1)+'.png',transparent=True)
        

