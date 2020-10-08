"""
mardi 6 octobre 2020, 17:59:14 (UTC+0200)

Script to plot vertical profile of <m> vs solution with std

"""
# ----------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from scipy.interpolate import griddata
from netCDF4 import Dataset

sys.path.append('../Postprocessing/')
import cpso_pp as pp

# ----------------------------------------------------------------------------
stats_path = '/postproc/COLLIN/MTD3/Calibre_CPSO_16nz/Analysis/'
stats_file = stats_path + 'pdf_m_nruns100.nc'
figname = stats_path + 'vertical_profile_100runs.png'

conf_dir = '../../Config/1D/model_002'
model_file = conf_dir + '/mod1D_Bolivia_002'


# ----------------------------------------------------------------------------
nc = Dataset(stats_file, 'r')
m_weight = nc.variables['m_weight'][:]
n_bin = nc.variables['n_bin'][:]
x_bin = nc.variables['x_bin'][:]
pdf_m = nc.variables['pdf_m'][:]
nc.close()

hz, rhosynth = np.loadtxt(model_file, unpack=True)

# ---> plot that shit

pp.vertical_profile(figname=figname, pdf_m=pdf_m, m_weight=m_weight,
                 logrhosynth=np.log10(rhosynth), hz=hz, x_bin=x_bin,
                 transparent=False, cut_off=1e-2) 


